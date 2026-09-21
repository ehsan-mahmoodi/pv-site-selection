"""NSGA-II for binary decision vectors.

Written out rather than imported from pymoo so the demonstrator runs with numpy
alone, and because the constraint handling here is problem-specific: a repair
operator pushes infeasible portfolios back towards the capacity target instead of
discarding them, which matters when a random binary vector over 500 sites is almost
never feasible by chance.

Reference: Deb, Pratap, Agarwal & Meyarivan (2002), "A fast and elitist
multiobjective genetic algorithm: NSGA-II".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 120
    generations: int = 220
    crossover_prob: float = 0.9
    mutation_rate: float | None = None  # defaults to 1/n
    tournament_size: int = 2
    seed: int = 20260921


def fast_non_dominated_sort(f: np.ndarray, g: np.ndarray) -> list[np.ndarray]:
    """Rank by constrained domination. Feasible always beats infeasible."""
    n = f.shape[0]
    dominates = [[] for _ in range(n)]
    n_dominated = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            d = _dominates(f[i], g[i], f[j], g[j])
            if d == 1:
                dominates[i].append(j)
                n_dominated[j] += 1
            elif d == -1:
                dominates[j].append(i)
                n_dominated[i] += 1

    for i in range(n):
        if n_dominated[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt: list[int] = []
        for i in fronts[k]:
            for j in dominates[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [np.array(fr, dtype=int) for fr in fronts[:-1]]


def _dominates(fa: np.ndarray, ga: float, fb: np.ndarray, gb: float) -> int:
    """1 if a dominates b, -1 if b dominates a, 0 otherwise."""
    fa_ok, fb_ok = ga <= 1e-9, gb <= 1e-9
    if fa_ok and not fb_ok:
        return 1
    if fb_ok and not fa_ok:
        return -1
    if not fa_ok and not fb_ok:
        if ga < gb:
            return 1
        if gb < ga:
            return -1
        return 0
    le = np.all(fa <= fb)
    lt = np.any(fa < fb)
    if le and lt:
        return 1
    ge = np.all(fb <= fa)
    gt = np.any(fb < fa)
    if ge and gt:
        return -1
    return 0


def crowding_distance(f: np.ndarray) -> np.ndarray:
    """Deb's crowding distance, to keep the front spread out."""
    n, m = f.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(f[:, k])
        lo, hi = f[order[0], k], f[order[-1], k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if hi - lo < 1e-12:
            continue
        dist[order[1:-1]] += (f[order[2:], k] - f[order[:-2], k]) / (hi - lo)
    return dist


class NSGA2:
    def __init__(self, problem, cfg: GAConfig | None = None):
        self.problem = problem
        self.cfg = cfg or GAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n = problem.n
        self.mut = self.cfg.mutation_rate or (1.0 / self.n)
        self.history: list[dict] = []

    # ------------------------------------------------------------ operators

    def _seed_population(self) -> np.ndarray:
        """Start from portfolios already near the capacity target."""
        pop = np.zeros((self.cfg.pop_size, self.n), dtype=bool)
        mean_mw = float(self.problem.mw.mean())
        k = max(1, int(self.problem.cfg.target_capacity_mw / mean_mw))
        for i in range(self.cfg.pop_size):
            size = max(1, k + int(self.rng.integers(-3, 4)))
            idx = self.rng.choice(self.n, size=min(size, self.n), replace=False)
            pop[i, idx] = True
        return pop

    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Add or drop sites until the capacity band is met."""
        cfg = self.problem.cfg
        lo = cfg.target_capacity_mw * (1 - cfg.capacity_tolerance)
        hi = cfg.target_capacity_mw * (1 + cfg.capacity_tolerance)
        mw = self.problem.mw
        cap = float(mw[x].sum())

        guard = 0
        while cap < lo and guard < self.n:
            off = np.flatnonzero(~x)
            if off.size == 0:
                break
            pick = off[self.rng.integers(off.size)]
            x[pick] = True
            cap += mw[pick]
            guard += 1
        guard = 0
        while cap > hi and guard < self.n:
            on = np.flatnonzero(x)
            if on.size <= 1:
                break
            pick = on[self.rng.integers(on.size)]
            x[pick] = False
            cap -= mw[pick]
            guard += 1
        return x

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.cfg.crossover_prob:
            return a.copy(), b.copy()
        mask = self.rng.random(self.n) < 0.5
        c1 = np.where(mask, a, b)
        c2 = np.where(mask, b, a)
        return c1, c2

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        flip = self.rng.random(self.n) < self.mut
        x = np.logical_xor(x, flip)
        return x

    def _tournament(self, rank: np.ndarray, crowd: np.ndarray) -> int:
        cand = self.rng.integers(0, self.cfg.pop_size, self.cfg.tournament_size)
        best = cand[0]
        for c in cand[1:]:
            if rank[c] < rank[best] or (
                rank[c] == rank[best] and crowd[c] > crowd[best]
            ):
                best = c
        return int(best)

    # ------------------------------------------------------------ main loop

    def run(self, verbose: bool = False) -> dict:
        pop = self._seed_population()
        for i in range(pop.shape[0]):
            pop[i] = self._repair(pop[i])
        f, g = self.problem.evaluate(pop)

        for gen in range(self.cfg.generations):
            fronts = fast_non_dominated_sort(f, g)
            rank = np.empty(pop.shape[0], dtype=int)
            crowd = np.empty(pop.shape[0])
            for r, fr in enumerate(fronts):
                rank[fr] = r
                crowd[fr] = crowding_distance(f[fr])

            children = np.empty_like(pop)
            for i in range(0, self.cfg.pop_size, 2):
                p1 = pop[self._tournament(rank, crowd)]
                p2 = pop[self._tournament(rank, crowd)]
                c1, c2 = self._crossover(p1, p2)
                children[i] = self._repair(self._mutate(c1))
                if i + 1 < self.cfg.pop_size:
                    children[i + 1] = self._repair(self._mutate(c2))

            cf, cg = self.problem.evaluate(children)
            merged = np.vstack([pop, children])
            mf = np.vstack([f, cf])
            mg = np.concatenate([g, cg])

            fronts = fast_non_dominated_sort(mf, mg)
            keep: list[int] = []
            for fr in fronts:
                if len(keep) + fr.size <= self.cfg.pop_size:
                    keep.extend(fr.tolist())
                else:
                    room = self.cfg.pop_size - len(keep)
                    cd = crowding_distance(mf[fr])
                    keep.extend(fr[np.argsort(-cd)[:room]].tolist())
                    break
            idx = np.array(keep, dtype=int)
            pop, f, g = merged[idx], mf[idx], mg[idx]

            feasible = int((g <= 1e-9).sum())
            self.history.append(
                {
                    "generation": gen,
                    "feasible": feasible,
                    "best_energy": float(-f[:, 0].min()),
                    "best_cost": float(f[:, 1].min()),
                    "best_risk": float(f[:, 2].min()),
                }
            )
            if verbose and (gen % 20 == 0 or gen == self.cfg.generations - 1):
                print(
                    f"  gen {gen:4d}  feasible {feasible:3d}/{self.cfg.pop_size}"
                    f"  energy {-f[:,0].min():.4f}"
                    f"  cost {f[:,1].min():.4f}"
                    f"  risk {f[:,2].min():.4f}"
                )

        fronts = fast_non_dominated_sort(f, g)
        front = fronts[0]
        return {
            "population": pop,
            "objectives": f,
            "violation": g,
            "front_index": front,
            "front_x": pop[front],
            "front_f": f[front],
            "history": self.history,
        }
