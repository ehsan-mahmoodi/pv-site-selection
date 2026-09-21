"""Regenerate every figure in outputs/ from a single deterministic run."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvsite import GAConfig, NSGA2, Problem, ProblemConfig, RegionSpec, generate
from pvsite.cli import _knees
from pvsite.model import site_economics
from pvsite.plots import overview, tradeoff_table

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    sites = generate(RegionSpec(), n=500, seed=20260921)
    problem = Problem(sites, ProblemConfig())
    ga = NSGA2(problem, GAConfig(pop_size=120, generations=200, seed=20260921))
    res = ga.run()

    feasible = res["violation"][res["front_index"]] <= 1e-9
    front_x = res["front_x"][feasible]
    rows = []
    for i in range(front_x.shape[0]):
        d = problem.describe(front_x[i])
        d.pop("site_ids")
        rows.append(d)
    front = pd.DataFrame(rows)
    picks = _knees(problem, front_x, res["front_f"][feasible])
    history = pd.DataFrame(res["history"])

    OUT.mkdir(parents=True, exist_ok=True)
    site_economics(sites).to_csv(OUT / "candidates.csv", index=False)
    front.to_csv(OUT / "pareto_front.csv", index=False)
    history.to_csv(OUT / "convergence.csv", index=False)
    (OUT / "selected_portfolios.json").write_text(json.dumps(picks, indent=2),
                                                  encoding="utf-8")

    for theme in ("dark", "light"):
        sfx = "" if theme == "dark" else "-light"
        overview(site_economics(sites), front, picks, history, problem, front_x,
                 OUT / f"portfolio-overview{sfx}.png", theme)
        tradeoff_table(picks, OUT / f"tradeoff-table{sfx}.png", theme)

    print(f"front: {len(front)} portfolios")
    for k, v in picks.items():
        print(f"  {k:11s} {v['sites']} sites  {v['capacity_mw']:.0f} MW  "
              f"EUR {v['capex_meur']:.1f}M  risk {v['risk']:.4f}")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
