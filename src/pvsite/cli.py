"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .candidates import RegionSpec, generate
from .model import Problem, ProblemConfig, site_economics
from .nsga2 import GAConfig, NSGA2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvsite",
        description="Multi-objective site selection for a PV portfolio.",
    )
    p.add_argument("--version", action="version", version=f"pvsite {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    scr = sub.add_parser("screen", help="generate the candidate site pre-study")
    scr.add_argument("--n", type=int, default=500)
    scr.add_argument("--seed", type=int, default=20260921)
    scr.add_argument("--out", type=Path, default=Path("outputs"))

    opt = sub.add_parser("optimise", help="run NSGA-II over a candidate csv")
    opt.add_argument("--sites", type=Path, required=True)
    opt.add_argument("--target-mw", type=float, default=220.0)
    opt.add_argument("--budget-meur", type=float, default=190.0)
    opt.add_argument("--pop", type=int, default=120)
    opt.add_argument("--generations", type=int, default=220)
    opt.add_argument("--seed", type=int, default=20260921)
    opt.add_argument("--out", type=Path, default=Path("outputs"))

    demo = sub.add_parser("demo", help="screen then optimise, end to end")
    demo.add_argument("--n", type=int, default=500)
    demo.add_argument("--target-mw", type=float, default=220.0)
    demo.add_argument("--budget-meur", type=float, default=190.0)
    demo.add_argument("--pop", type=int, default=120)
    demo.add_argument("--generations", type=int, default=220)
    demo.add_argument("--seed", type=int, default=20260921)
    demo.add_argument("--out", type=Path, default=Path("outputs"))
    return p


def _knees(problem: Problem, front_x: np.ndarray, front_f: np.ndarray) -> dict:
    """Three defensible picks from the front, plus the balanced compromise."""
    lo = front_f.min(axis=0)
    hi = front_f.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    norm = (front_f - lo) / span

    picks = {
        "max_energy": int(np.argmin(front_f[:, 0])),
        "min_cost": int(np.argmin(front_f[:, 1])),
        "min_risk": int(np.argmin(front_f[:, 2])),
        "balanced": int(np.argmin(np.linalg.norm(norm, axis=1))),
    }
    return {name: problem.describe(front_x[i]) for name, i in picks.items()}


def cmd_screen(args) -> int:
    sites = site_economics(generate(RegionSpec(), n=args.n, seed=args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.out / "candidates.csv", index=False)
    print(f"screened {len(sites)} candidate sites -> {args.out/'candidates.csv'}")
    print(sites[["ghi_kwh_m2_yr", "capacity_mw", "grid_km", "capex_eur_per_kw",
                 "lcoe_proxy"]].describe().round(1).to_string())
    return 0


def _optimise(sites: pd.DataFrame, args) -> int:
    cfg = ProblemConfig(
        target_capacity_mw=args.target_mw, budget_meur=args.budget_meur
    )
    problem = Problem(sites, cfg)
    print(f"{problem.n} candidates, {problem.mw.sum():,.0f} MW available, "
          f"target {cfg.target_capacity_mw:.0f} MW, budget EUR {cfg.budget_meur:.0f}M")

    ga = NSGA2(problem, GAConfig(pop_size=args.pop, generations=args.generations,
                                 seed=args.seed))
    res = ga.run(verbose=True)

    front_x, front_f = res["front_x"], res["front_f"]
    feasible = res["violation"][res["front_index"]] <= 1e-9
    front_x, front_f = front_x[feasible], front_f[feasible]
    print(f"\nPareto front: {len(front_f)} feasible non-dominated portfolios")

    picks = _knees(problem, front_x, front_f)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(front_f)):
        d = problem.describe(front_x[i])
        d.pop("site_ids")
        rows.append(d)
    pd.DataFrame(rows).to_csv(args.out / "pareto_front.csv", index=False)
    (args.out / "selected_portfolios.json").write_text(
        json.dumps(picks, indent=2), encoding="utf-8"
    )
    pd.DataFrame(res["history"]).to_csv(args.out / "convergence.csv", index=False)

    print()
    table = pd.DataFrame(
        [{"portfolio": k, **{kk: vv for kk, vv in v.items() if kk != "site_ids"}}
         for k, v in picks.items()]
    )
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))

    np.save(args.out / "front_x.npy", front_x)
    return 0


def cmd_optimise(args) -> int:
    sites = pd.read_csv(args.sites)
    return _optimise(sites, args)


def cmd_demo(args) -> int:
    sites = generate(RegionSpec(), n=args.n, seed=args.seed)
    print(f"pre-study screened {len(sites)} candidate points across "
          f"{RegionSpec().name}\n")
    return _optimise(sites, args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"screen": cmd_screen, "optimise": cmd_optimise, "demo": cmd_demo}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
