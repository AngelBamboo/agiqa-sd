"""Publication figures for the benchmark paper.

Everything is drawn from the summary records, so a rerun of the matrix updates
every figure without touching this file. English labels: the target is IEEE TMM.

Usage::

    python -m bench.figures
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from bench import leaderboard, stability

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "bench")

FAMILY_COLOR = {
    "classical": "#6c757d",
    "cnn": "#0072b2",
    "transformer": "#009e73",
    "clip": "#d55e00",
    "linear": "#cc79a7",
}
FAMILY_LABEL = {
    "classical": "Classical statistics",
    "cnn": "CNN regressor",
    "transformer": "Transformer",
    "clip": "CLIP / vision-language",
    "linear": "Linear readout",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _rows(dataset):
    records = [r for r in leaderboard.load_records(dataset) if "pooled" in r]
    return leaderboard.to_rows(records)


def scatter_pooled_within(dataset="agiqa3k", name="fig1_pooled_vs_within"):
    rows = _rows(dataset)
    rows = [r for r in rows if isinstance(r["pooled"], float) and isinstance(r["within"], float)]
    # A dumbbell rather than a scatter: many models share almost the same
    # coordinates, so points sit on top of one another. One row per model makes
    # the pooled-within gap directly readable and keeps every label legible.
    rows = sorted(rows, key=lambda r: r["pooled"])
    fig, ax = plt.subplots(figsize=(3.5, 4.4))
    for y, r in enumerate(rows):
        color = FAMILY_COLOR.get(r["family"], "#333333")
        ax.plot([r["within"], r["pooled"]], [y, y], color="#bbbbbb", lw=1.0, zorder=1)
        ax.scatter(r["within"], y, s=26, facecolor="white", edgecolor=color,
                   linewidth=1.0, zorder=3)
        ax.scatter(r["pooled"], y, s=26, color=color, edgecolor="white",
                   linewidth=0.5, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=6.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("SRCC")
    ax.grid(axis="x", alpha=0.15, lw=0.4)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="#555555",
                   markeredgecolor="#555555", label="Pooled"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#555555", label="Within-generator"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6)
    _save(fig, name)


def rank_shift(dataset="agiqa3k", name="fig2_rank_shift"):
    rows = _rows(dataset)
    rows = [r for r in rows if isinstance(r["pooled"], float) and isinstance(r["within"], float)]
    leaderboard.rank_analysis(rows)
    rows.sort(key=lambda r: r["rank_pooled"])
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    for r in rows:
        color = FAMILY_COLOR.get(r["family"], "#333333")
        ax.plot([0, 1], [r["rank_pooled"], r["rank_within"]], color=color, lw=0.8, alpha=0.85)
        ax.scatter([0, 1], [r["rank_pooled"], r["rank_within"]], s=10, color=color, zorder=3)
        ax.text(-0.04, r["rank_pooled"], r["model"], ha="right", va="center", fontsize=5.5)
        ax.text(1.04, r["rank_within"], r["model"], ha="left", va="center", fontsize=5.5)
    n = len(rows)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pooled", "Within-generator"])
    ax.set_ylabel("Rank (1 = best)")
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0)
    _save(fig, name)


def _cross_taus(rows):
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], {})[r["model"]] = r
    names = sorted(by)
    pooled = np.full((len(names), len(names)), np.nan)
    within = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                pooled[i, j] = within[i, j] = 1.0
                continue
            common = [m for m in set(by[a]) & set(by[b])
                      if isinstance(by[a][m]["pooled"], float) and isinstance(by[b][m]["pooled"], float)
                      and isinstance(by[a][m]["within"], float) and isinstance(by[b][m]["within"], float)]
            if len(common) < 3:
                continue
            pooled[i, j] = stats.kendalltau([by[a][m]["pooled"] for m in common],
                                            [by[b][m]["pooled"] for m in common]).statistic
            within[i, j] = stats.kendalltau([by[a][m]["within"] for m in common],
                                            [by[b][m]["within"] for m in common]).statistic
    return names, pooled, within


def cross_dataset(name="fig3_cross_dataset"):
    rows = leaderboard.to_rows([r for r in leaderboard.load_records() if "pooled" in r])
    names, pooled, within = _cross_taus(rows)
    labels = [n.replace("agiqa", "AGIQA-").replace("aigciqa", "AIGCIQA") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for ax, mat, title in zip(axes, (pooled, within), ("Pooled leaderboard", "Within-generator leaderboard")):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Kendall tau")
    _save(fig, name)


def generator_heatmap(dataset="agiqa3k", name="fig4_generator_stability"):
    data = stability.load(dataset)
    result = stability.analyse(data[dataset])
    gens = result["generators"]
    mat = np.array(result["matrix"])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(mat, vmin=-0.2, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(gens)))
    ax.set_yticks(range(len(gens)))
    ax.set_xticklabels(gens, rotation=30, ha="right")
    ax.set_yticklabels(gens)
    for i in range(len(gens)):
        for j in range(len(gens)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Kendall tau")
    ax.set_title(f"{dataset}: model ranking by generator", fontsize=8)
    _save(fig, name)


def main():
    scatter_pooled_within("agiqa3k")
    rank_shift("agiqa3k")
    cross_dataset()
    generator_heatmap("agiqa3k")
    generator_heatmap("aigciqa2023", "fig5_generator_stability_2023")


if __name__ == "__main__":
    main()
