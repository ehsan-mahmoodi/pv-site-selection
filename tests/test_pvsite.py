"""Tests. Run with:  PYTHONPATH=src python tests/test_pvsite.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvsite import (  # noqa: E402
    GAConfig, NSGA2, Problem, ProblemConfig, RegionSpec, generate, site_economics,
)
from pvsite.candidates import haversine_km  # noqa: E402
from pvsite.nsga2 import (  # noqa: E402
    _dominates, crowding_distance, fast_non_dominated_sort,
)


# ---------------------------------------------------------------- candidates

def test_generate_is_deterministic():
    a = generate(n=60, seed=42)
    b = generate(n=60, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_candidates_stay_inside_the_region():
    r = RegionSpec()
    s = generate(r, n=300, seed=1)
    assert s["lat"].between(r.lat_min, r.lat_max).all()
    assert s["lon"].between(r.lon_min, r.lon_max).all()


def test_attributes_are_physically_plausible():
    s = generate(n=400, seed=2)
    assert s["ghi_kwh_m2_yr"].between(1700, 2350).all()
    assert (s["capacity_mw"] > 0).all()
    assert (s["grid_km"] >= 0).all()
    assert (s["land_eur_ha"] >= 1800).all()
    assert (s["slope_deg"] <= 18.0).all()


def test_irradiation_increases_towards_the_south():
    s = generate(n=500, seed=3)
    south = s[s["lat"] < s["lat"].median()]["ghi_kwh_m2_yr"].mean()
    north = s[s["lat"] >= s["lat"].median()]["ghi_kwh_m2_yr"].mean()
    assert south > north + 40.0


def test_haversine_against_a_known_distance():
    # Auckland to Wellington is about 494 km great circle.
    d = haversine_km(np.array([-36.85]), np.array([174.76]), -41.29, 174.78)
    assert abs(float(d[0]) - 494.0) < 15.0


def test_haversine_is_zero_at_the_same_point():
    d = haversine_km(np.array([32.0]), np.array([54.0]), 32.0, 54.0)
    assert float(d[0]) < 1e-9


# ---------------------------------------------------------------- economics

def test_site_economics_adds_expected_columns():
    s = site_economics(generate(n=50, seed=4))
    for col in ["annual_mwh", "lifetime_mwh", "capex_eur", "lcoe_proxy"]:
        assert col in s.columns
    assert (s["capex_eur"] > 0).all()
    assert (s["lifetime_mwh"] > s["annual_mwh"]).all()


def test_further_from_grid_costs_more_all_else_equal():
    base = generate(n=2, seed=5).iloc[[0, 0]].reset_index(drop=True)
    base.loc[0, "grid_km"] = 5.0
    base.loc[1, "grid_km"] = 90.0
    econ = site_economics(base)
    assert econ.loc[1, "capex_eur"] > econ.loc[0, "capex_eur"]


# ---------------------------------------------------------------- domination

def test_dominates_is_antisymmetric():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    assert _dominates(a, 0.0, b, 0.0) == 1
    assert _dominates(b, 0.0, a, 0.0) == -1


def test_feasible_beats_infeasible_regardless_of_objectives():
    good = np.array([9.0, 9.0, 9.0])   # terrible objectives, but feasible
    bad = np.array([0.0, 0.0, 0.0])    # perfect objectives, infeasible
    assert _dominates(good, 0.0, bad, 0.5) == 1


def test_non_dominated_sort_finds_the_true_front():
    f = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0], [5.0, 6.0], [3.0, 4.0]])
    g = np.zeros(5)
    fronts = fast_non_dominated_sort(f, g)
    assert set(fronts[0].tolist()) == {0, 1, 2}


def test_crowding_distance_marks_the_extremes_infinite():
    f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    d = crowding_distance(f)
    assert np.isinf(d[0]) and np.isinf(d[2])
    assert np.isfinite(d[1])


# ---------------------------------------------------------------- problem

def test_evaluate_returns_the_right_shapes():
    sites = generate(n=80, seed=6)
    p = Problem(sites, ProblemConfig(target_capacity_mw=120.0))
    x = np.zeros((4, p.n), dtype=bool)
    x[:, :4] = True
    f, g = p.evaluate(x)
    assert f.shape == (4, 3)
    assert g.shape == (4,)


def test_empty_portfolio_is_infeasible():
    sites = generate(n=60, seed=7)
    p = Problem(sites, ProblemConfig(target_capacity_mw=150.0))
    f, g = p.evaluate(np.zeros((1, p.n), dtype=bool))
    assert g[0] > 0


def test_concentrated_portfolio_scores_worse_risk_than_a_spread_one():
    sites = generate(n=200, seed=8)
    p = Problem(sites, ProblemConfig())
    same = np.flatnonzero(p.substation == p.substation[0])[:5]
    spread = [int(np.flatnonzero(p.substation == s)[0])
              for s in range(min(5, p.n_subs))]
    xa = np.zeros(p.n, dtype=bool); xa[same] = True
    xb = np.zeros(p.n, dtype=bool); xb[spread] = True
    fa, _ = p.evaluate(xa.reshape(1, -1))
    fb, _ = p.evaluate(xb.reshape(1, -1))
    assert fa[0, 2] > fb[0, 2], "one substation should be riskier than five"


def test_budget_breach_is_reported_as_a_violation():
    sites = generate(n=120, seed=9)
    tight = Problem(sites, ProblemConfig(target_capacity_mw=220.0, budget_meur=1.0))
    x = np.zeros(tight.n, dtype=bool)
    order = np.argsort(-tight.mw)[:6]
    x[order] = True
    _, g = tight.evaluate(x.reshape(1, -1))
    assert g[0] > 0


# ---------------------------------------------------------------- end to end

def test_optimiser_reaches_feasibility_and_a_real_front():
    sites = generate(n=300, seed=20260921)
    p = Problem(sites, ProblemConfig(target_capacity_mw=200.0))
    ga = NSGA2(p, GAConfig(pop_size=60, generations=60, seed=11))
    res = ga.run()

    # 60 generations on 300 candidates is a deliberately short run: the point is
    # that the repair operator plus constrained domination gets a useful share of
    # the population into the feasible region, not that it converges fully. The
    # 200-generation run in examples/make_figures.py reaches 100%.
    feasible = res["violation"] <= 1e-9
    assert feasible.sum() >= 10, (
        f"only {feasible.sum()}/{len(feasible)} feasible; "
        "the repair operator is not doing its job"
    )

    front_f = res["front_f"][res["violation"][res["front_index"]] <= 1e-9]
    assert len(front_f) >= 5
    # A real front trades off: the objectives must not all be identical.
    assert np.ptp(front_f[:, 0]) > 1e-6
    assert np.ptp(front_f[:, 2]) > 1e-6


def test_every_feasible_solution_respects_the_capacity_band():
    sites = generate(n=250, seed=12)
    cfg = ProblemConfig(target_capacity_mw=180.0)
    p = Problem(sites, cfg)
    ga = NSGA2(p, GAConfig(pop_size=50, generations=40, seed=13))
    res = ga.run()
    lo = cfg.target_capacity_mw * (1 - cfg.capacity_tolerance)
    hi = cfg.target_capacity_mw * (1 + cfg.capacity_tolerance)
    for i in np.flatnonzero(res["violation"] <= 1e-9):
        cap = float(p.mw[res["population"][i]].sum())
        assert lo - 1e-6 <= cap <= hi + 1e-6, f"capacity {cap} outside [{lo},{hi}]"


def test_front_is_actually_non_dominated():
    sites = generate(n=200, seed=14)
    p = Problem(sites, ProblemConfig(target_capacity_mw=160.0))
    ga = NSGA2(p, GAConfig(pop_size=40, generations=30, seed=15))
    res = ga.run()
    f = res["front_f"]
    g = res["violation"][res["front_index"]]
    for i in range(len(f)):
        for j in range(len(f)):
            if i != j:
                assert _dominates(f[j], g[j], f[i], g[i]) != 1, (
                    f"solution {j} dominates {i} but both are on the front"
                )


def test_results_are_reproducible():
    sites = generate(n=150, seed=16)
    p = Problem(sites, ProblemConfig(target_capacity_mw=150.0))
    a = NSGA2(p, GAConfig(pop_size=40, generations=25, seed=99)).run()
    b = NSGA2(p, GAConfig(pop_size=40, generations=25, seed=99)).run()
    np.testing.assert_allclose(a["front_f"], b["front_f"])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
