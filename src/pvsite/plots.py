"""Figures for the site selection study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DARK = {
    "bg": "#0d1117", "panel": "#161b22", "ink": "#e6edf3", "mut": "#8b949e",
    "acc": "#58a6ff", "grn": "#3fb950", "amb": "#d29922", "red": "#f85149",
    "vio": "#bc8cff", "grid": "#30363d",
}
LIGHT = {
    "bg": "#ffffff", "panel": "#f6f8fa", "ink": "#24292f", "mut": "#57606a",
    "acc": "#0969da", "grn": "#1a7f37", "amb": "#9a6700", "red": "#cf222e",
    "vio": "#8250df", "grid": "#d0d7de",
}


def _style(ax, c, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(c["panel"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["mut"], labelsize=8)
    ax.grid(True, color=c["grid"], alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=c["ink"], fontsize=11, fontweight="bold", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=c["mut"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=c["mut"], fontsize=9)


def overview(
    sites: pd.DataFrame,
    front: pd.DataFrame,
    picks: dict,
    history: pd.DataFrame,
    problem,
    front_x: np.ndarray,
    out: Path,
    theme: str = "dark",
) -> Path:
    c = DARK if theme == "dark" else LIGHT
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2), facecolor=c["bg"])

    # 1 - the region and the chosen portfolio
    ax = axes[0]
    ax.scatter(sites["lon"], sites["lat"], s=np.clip(sites["capacity_mw"] / 4, 3, 30),
               c=sites["ghi_kwh_m2_yr"], cmap="viridis", alpha=0.55, linewidths=0)
    chosen = set(picks["balanced"]["site_ids"])
    sel = sites[sites["site_id"].isin(chosen)]
    ax.scatter(sel["lon"], sel["lat"], s=140, facecolors="none",
               edgecolors=c["red"], linewidths=1.8, label="selected")
    _style(ax, c, "500 screened sites", "longitude", "latitude")
    leg = ax.legend(fontsize=7, facecolor=c["panel"], edgecolor=c["grid"])
    for t in leg.get_texts():
        t.set_color(c["mut"])

    # 2 - the Pareto front, cost against energy, coloured by risk
    ax = axes[1]
    sc = ax.scatter(front["capex_meur"], front["lifetime_gwh"], c=front["risk"],
                    cmap="plasma", s=26, alpha=0.9, linewidths=0)
    for name, style in [("max_energy", "^"), ("min_cost", "v"),
                        ("min_risk", "s"), ("balanced", "o")]:
        d = picks[name]
        ax.scatter(d["capex_meur"], d["lifetime_gwh"], marker=style, s=110,
                   facecolors="none", edgecolors=c["ink"], linewidths=1.5)
        ax.annotate(name.replace("_", " "), (d["capex_meur"], d["lifetime_gwh"]),
                    textcoords="offset points", xytext=(6, 5),
                    color=c["ink"], fontsize=7)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046)
    cb.set_label("concentration risk", color=c["mut"], fontsize=8)
    cb.ax.tick_params(colors=c["mut"], labelsize=7)
    cb.outline.set_edgecolor(c["grid"])
    _style(ax, c, "Pareto front", "capex, EUR million", "lifetime energy, GWh")

    # 3 - what risk costs: cheapest portfolio available at each level of risk
    ax = axes[2]
    order = front.sort_values("risk")
    ax.scatter(order["risk"], order["capex_meur"], s=14, color=c["amb"],
               alpha=0.45, linewidths=0)
    # Lower envelope: running minimum capex as risk tolerance loosens.
    envelope = order["capex_meur"].cummin()
    ax.plot(order["risk"], envelope, color=c["amb"], lw=2.0,
            label="cheapest at this risk")
    lo, hi = picks["min_risk"], picks["min_cost"]
    ax.annotate(
        f"holding risk at {lo['risk']:.3f} instead of {hi['risk']:.3f}\n"
        f"costs EUR {lo['capex_meur'] - hi['capex_meur']:.0f}M more",
        xy=(lo["risk"], lo["capex_meur"]), xytext=(0.30, 0.80),
        textcoords="axes fraction", color=c["ink"], fontsize=8,
        arrowprops=dict(arrowstyle="->", color=c["mut"], lw=1),
    )
    _style(ax, c, "The price of diversification", "concentration risk",
           "capex, EUR million")
    leg = ax.legend(fontsize=7, facecolor=c["panel"], edgecolor=c["grid"],
                    loc="lower right")
    for t in leg.get_texts():
        t.set_color(c["mut"])

    # 4 - convergence
    ax = axes[3]
    ax.plot(history["generation"], history["feasible"], color=c["grn"], lw=1.6,
            label="feasible in population")
    ax.set_ylim(0, history["feasible"].max() * 1.1)
    _style(ax, c, "Convergence", "generation", "feasible solutions")
    ax2 = ax.twinx()
    ax2.plot(history["generation"], history["best_risk"], color=c["vio"], lw=1.4,
             label="best risk")
    ax2.tick_params(colors=c["mut"], labelsize=8)
    ax2.set_ylabel("best risk", color=c["mut"], fontsize=9)
    for s in ax2.spines.values():
        s.set_color(c["grid"])
    lines = ax.get_lines() + ax2.get_lines()
    leg = ax.legend(lines, [l.get_label() for l in lines], fontsize=7,
                    facecolor=c["panel"], edgecolor=c["grid"], loc="center right")
    for t in leg.get_texts():
        t.set_color(c["mut"])

    fig.suptitle(
        "PV portfolio siting - 500 candidates, 220 MW target, three competing objectives",
        color=c["ink"], fontsize=12, fontweight="bold", x=0.006, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=c["bg"])
    plt.close(fig)
    return out


def tradeoff_table(picks: dict, out: Path, theme: str = "dark") -> Path:
    """The four candidate portfolios as a rendered comparison."""
    c = DARK if theme == "dark" else LIGHT
    names = ["max_energy", "balanced", "min_cost", "min_risk"]
    rows = ["sites", "capacity_mw", "lifetime_gwh", "capex_meur",
            "eur_per_mwh", "risk", "max_substation_share"]
    labels = ["Sites", "Capacity MW", "Lifetime GWh", "Capex EURm",
              "EUR / MWh", "Risk index", "Largest substation share"]

    fig, ax = plt.subplots(figsize=(9.0, 3.4), facecolor=c["bg"])
    ax.set_facecolor(c["bg"])
    ax.axis("off")
    cell = [[f"{picks[n][r]:,}" if isinstance(picks[n][r], (int, float)) else
             str(picks[n][r]) for n in names] for r in rows]
    tbl = ax.table(
        cellText=cell, rowLabels=labels,
        colLabels=[n.replace("_", " ") for n in names],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    for (r, col), cl in tbl.get_celld().items():
        cl.set_edgecolor(c["grid"])
        cl.set_facecolor(c["panel"] if r > 0 else c["bg"])
        cl.get_text().set_color(c["ink"] if r > 0 else c["acc"])
        if col == -1:
            cl.get_text().set_color(c["mut"])
            cl.get_text().set_ha("right")
    ax.set_title("Four defensible answers, not one",
                 color=c["ink"], fontsize=11, fontweight="bold", loc="left", pad=16)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=c["bg"])
    plt.close(fig)
    return out
