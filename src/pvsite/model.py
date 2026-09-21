"""The decision problem.

Select a subset of the screened candidate sites to build, subject to a capacity
target, a capital budget and the spare capacity at each substation.

Three objectives, all genuinely in conflict:

    1. maximise 25-year energy yield      - wants high irradiation, which is south
    2. minimise total capital cost        - wants cheap land and short grid runs
    3. minimise portfolio risk            - wants sites spread across substations
                                            and geography, so one fault, one grid
                                            outage or one dust storm cannot take
                                            out the whole portfolio

Objective 3 is the one that usually gets left out, and it is the one that makes the
answer defensible to a lender. A portfolio that puts 200 MW behind a single
substation is cheaper and sunnier, and one transformer failure removes all of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import haversine_km

# Engineering and cost assumptions, stated in one place so they can be argued with.
PR = 0.87                     # performance ratio, as built at Nir
DEGRADATION_PER_YEAR = 0.005  # 0.5%/yr linear
LIFETIME_YEARS = 25
EPC_EUR_PER_KW = 620.0        # modules, mounting, inverters, install
GRID_EUR_PER_KM_MW = 780.0    # line plus bay, scaled by capacity
ROAD_EUR_PER_KM = 42000.0     # site access construction
SLOPE_EUR_PER_MW_DEG = 1450.0 # earthworks penalty


@dataclass(frozen=True)
class ProblemConfig:
    target_capacity_mw: float = 220.0
    capacity_tolerance: float = 0.06     # +/- fraction of target
    budget_meur: float = 190.0
    min_separation_km: float = 12.0
    max_share_per_substation: float = 0.30  # of selected capacity


def site_economics(sites: pd.DataFrame) -> pd.DataFrame:
    """Per-site yield and capital cost. Vectorised; no optimisation here."""
    out = sites.copy()
    mw = out["capacity_mw"].to_numpy()

    # Lifetime energy, MWh, with linear degradation.
    annual_mwh = out["ghi_kwh_m2_yr"].to_numpy() * mw * PR
    years = np.arange(LIFETIME_YEARS)
    lifetime_factor = float(np.sum(1.0 - DEGRADATION_PER_YEAR * years))
    out["annual_mwh"] = np.round(annual_mwh, 0)
    out["lifetime_mwh"] = np.round(annual_mwh * lifetime_factor, 0)

    epc = EPC_EUR_PER_KW * mw * 1000.0
    land = out["land_eur_ha"].to_numpy() * out["area_ha"].to_numpy()
    grid = GRID_EUR_PER_KM_MW * out["grid_km"].to_numpy() * mw
    road = ROAD_EUR_PER_KM * out["road_km"].to_numpy()
    terrain = SLOPE_EUR_PER_MW_DEG * mw * out["slope_deg"].to_numpy()

    out["capex_eur"] = np.round(epc + land + grid + road + terrain, 0)
    out["capex_eur_per_kw"] = np.round(out["capex_eur"] / (mw * 1000.0), 1)
    out["lcoe_proxy"] = np.round(
        out["capex_eur"] / np.maximum(out["lifetime_mwh"], 1.0), 2
    )
    return out


def separation_matrix(sites: pd.DataFrame) -> np.ndarray:
    """Pairwise great-circle distance in km."""
    lat = sites["lat"].to_numpy()
    lon = sites["lon"].to_numpy()
    n = lat.size
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = haversine_km(lat, lon, float(lat[i]), float(lon[i]))
    return d


class Problem:
    """Evaluates a binary selection vector against the three objectives."""

    def __init__(self, sites: pd.DataFrame, cfg: ProblemConfig | None = None):
        self.cfg = cfg or ProblemConfig()
        self.sites = site_economics(sites)
        self.n = len(self.sites)

        self.mw = self.sites["capacity_mw"].to_numpy(dtype=float)
        self.lifetime_mwh = self.sites["lifetime_mwh"].to_numpy(dtype=float)
        self.capex = self.sites["capex_eur"].to_numpy(dtype=float)
        self.substation = self.sites["substation"].to_numpy(dtype=int)
        self.sub_capacity = self.sites["substation_capacity_mw"].to_numpy(dtype=float)
        self.n_subs = int(self.substation.max()) + 1
        self.dist = separation_matrix(self.sites)

        self._max_mwh = float(self.lifetime_mwh.sum())
        self._max_capex = float(self.capex.sum())

    # ---------------------------------------------------------------- objectives

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a population.

        Parameters
        ----------
        x : (pop, n) binary array

        Returns
        -------
        f : (pop, 3) objectives, all minimised
        g : (pop,) total constraint violation, 0 when feasible
        """
        x = np.atleast_2d(x).astype(bool)
        pop = x.shape[0]

        cap = x @ self.mw
        energy = x @ self.lifetime_mwh
        cost = x @ self.capex

        f1 = -energy / self._max_mwh          # maximise energy
        f2 = cost / self._max_capex           # minimise capex
        f3 = np.empty(pop)
        g = np.zeros(pop)

        target = self.cfg.target_capacity_mw
        tol = self.cfg.capacity_tolerance
        lo, hi = target * (1 - tol), target * (1 + tol)

        for k in range(pop):
            sel = np.flatnonzero(x[k])
            f3[k] = self._risk(sel, cap[k])
            g[k] = self._violation(sel, cap[k], cost[k], lo, hi)

        return np.column_stack([f1, f2, f3]), g

    def _risk(self, sel: np.ndarray, total_mw: float) -> float:
        """Concentration risk: Herfindahl over substations, plus geographic clustering."""
        if sel.size == 0 or total_mw <= 0:
            return 1.0
        share = np.bincount(
            self.substation[sel], weights=self.mw[sel], minlength=self.n_subs
        ) / total_mw
        herfindahl = float(np.sum(share**2))

        if sel.size > 1:
            sub = self.dist[np.ix_(sel, sel)]
            iu = np.triu_indices(sel.size, k=1)
            mean_sep = float(sub[iu].mean())
            spread = 1.0 - min(mean_sep / 180.0, 1.0)
        else:
            spread = 1.0
        return 0.65 * herfindahl + 0.35 * spread

    def _violation(
        self, sel: np.ndarray, cap: float, cost: float, lo: float, hi: float
    ) -> float:
        v = 0.0
        if cap < lo:
            v += (lo - cap) / lo
        elif cap > hi:
            v += (cap - hi) / hi

        budget = self.cfg.budget_meur * 1e6
        if cost > budget:
            v += (cost - budget) / budget

        if sel.size:
            # Substation spare capacity.
            per_sub = np.bincount(
                self.substation[sel], weights=self.mw[sel], minlength=self.n_subs
            )
            head = np.zeros(self.n_subs)
            for s in range(self.n_subs):
                mask = self.substation == s
                if mask.any():
                    head[s] = self.sub_capacity[mask][0]
            over = np.clip(per_sub - head, 0, None)
            v += float(over.sum()) / max(cap, 1.0)

            # Concentration cap.
            if cap > 0:
                share = per_sub / cap
                v += float(np.clip(share - self.cfg.max_share_per_substation, 0, None).sum())

            # Minimum separation between selected sites.
            if sel.size > 1:
                sub = self.dist[np.ix_(sel, sel)]
                iu = np.triu_indices(sel.size, k=1)
                close = np.clip(self.cfg.min_separation_km - sub[iu], 0, None)
                v += float(close.sum()) / (self.cfg.min_separation_km * max(sel.size, 1))
        return float(v)

    # ---------------------------------------------------------------- reporting

    def describe(self, x: np.ndarray) -> dict:
        """Human-readable summary of one solution."""
        sel = np.flatnonzero(np.asarray(x).astype(bool))
        cap = float(self.mw[sel].sum())
        energy = float(self.lifetime_mwh[sel].sum())
        cost = float(self.capex[sel].sum())
        f, g = self.evaluate(np.asarray(x).reshape(1, -1))
        per_sub = np.bincount(
            self.substation[sel], weights=self.mw[sel], minlength=self.n_subs
        )
        return {
            "sites": int(sel.size),
            "capacity_mw": round(cap, 1),
            "lifetime_gwh": round(energy / 1000.0, 1),
            "capex_meur": round(cost / 1e6, 1),
            "eur_per_mwh": round(cost / max(energy, 1.0), 2),
            "risk": round(float(f[0, 2]), 4),
            "feasible": bool(g[0] <= 1e-9),
            "violation": round(float(g[0]), 5),
            "max_substation_share": round(float(per_sub.max() / max(cap, 1)), 3),
            "site_ids": self.sites["site_id"].to_numpy()[sel].tolist(),
        }
