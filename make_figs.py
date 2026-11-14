"""Publication figures for the source-confound paper."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import agiqa.config as cfg

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

OUT = cfg.FIGURES
os.makedirs(OUT, exist_ok=True)

C_MAIN = "#3b6ea5"
C_ACC = "#bf616a"
C_GREY = "#9aa4b2"


def load_rows():
    return list(csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8")))


def fig1_dataset():
    rows = load_rows()
    gens = ["AttnGAN", "glide", "DALLE2", "sd1.5", "midjourney", "xl2.2"]
    data = [[float(r["mos_quality"]) for r in rows if r["name"].split("_")[0] == g] for g in gens]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bp = ax.boxplot(data, labels=gens, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", lw=1.2),
                    flierprops=dict(marker=".", ms=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_MAIN)
        patch.set_alpha(0.85)
    ax.set_ylabel("MOS (perceptual quality)")
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(np.mean([np.mean(d) for d in data]), ls="--", lw=0.8, color=C_GREY)
    fig.savefig(os.path.join(OUT, "fig1_mos_by_generator.png"))
    plt.close(fig)


def fig2_protocol():
    """Random-split vs LOGO, per generator."""
    d = json.load(open(os.path.join(cfg.RUNS, "aigciqa2023_protocols.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(cfg.RUNS, "aigciqa2023_protocols.json")) else None
    logo = {"AttnGAN": (0.382, 0.013), "glide": (0.680, 0.013), "xl2.2": (0.372, 0.019),
            "DALLE2": (0.488, 0.022), "midjourney": (0.700, 0.005), "sd1.5": None}
    names = [k for k, v in logo.items() if v]
    means = [logo[k][0] for k in names]
    errs = [logo[k][1] for k in names]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=errs, capsize=2.5, color=C_MAIN, width=0.6, label="leave-one-generator-out")
    ax.axhline(0.833, ls="--", color=C_ACC, lw=1.2, label="random split (0.833)")
    ax.axhline(0.288, ls=":", color=C_GREY, lw=1.4, label="source-only control (0.288)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("SRCC")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.8)
    fig.savefig(os.path.join(OUT, "fig2_protocol_gap.png"))
    plt.close(fig)


def fig3_decomposition():
    models = ["ResNet-18", "CLIP + MLP", "CoOp"]
    overall = [0.830, 0.833, 0.862]
    within = [0.530, 0.535, 0.549]
    between = [1.000, 1.000, 1.000]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(models))
    w = 0.26
    ax.bar(x - w, overall, w, label="overall", color=C_GREY)
    ax.bar(x, within, w, label="within-generator", color=C_MAIN)
    ax.bar(x + w, between, w, label="between-generator", color=C_ACC)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("SRCC")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=7,
              handlelength=1.2, borderaxespad=0.2)
    fig.savefig(os.path.join(OUT, "fig3_decomposition.png"))
    plt.close(fig)


def fig4_source_only():
    gens = ["GLIDE", "Midjourney", "SDXL", "AttnGAN", "DALL-E 2"]
    src = [0.517, 0.433, 0.205, 0.148, 0.138]
    mod = [0.680, 0.700, 0.372, 0.382, 0.488]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(gens))
    w = 0.36
    ax.bar(x - w / 2, mod, w, label="trained model", color=C_MAIN)
    ax.bar(x + w / 2, src, w, label="source-only control", color=C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=20)
    ax.set_ylabel("SRCC (held-out generator)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, "fig4_source_only.png"))
    plt.close(fig)


def fig5_ceiling():
    gens = ["Midjourney", "GLIDE", "SD 1.5", "DALL-E 2", "AttnGAN", "SDXL"]
    achieved = [0.785, 0.799, 0.578, 0.494, 0.360, 0.167]
    ceiling = [0.792, 0.676, 0.726, 0.683, 0.443, 0.590]
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_GREY)
    ax.scatter(ceiling, achieved, s=42, color=C_MAIN, zorder=3)
    for x, y, g in zip(ceiling, achieved, gens):
        ax.annotate(g, (x, y), textcoords="offset points", xytext=(4, -3), fontsize=7)
    ax.set_xlabel("estimated reliability ceiling")
    ax.set_ylabel("achieved within-generator SRCC")
    ax.set_xlim(0.35, 0.85)
    ax.set_ylim(0.1, 0.85)
    fig.savefig(os.path.join(OUT, "fig5_ceiling.png"))
    plt.close(fig)


def fig6_alpha():
    a = [100, 300, 1000, 3000, 10000, 30000, 100000]
    rand = [0.840, 0.846, 0.849, 0.845, 0.832, 0.810, 0.786]
    logo = [0.561, 0.571, 0.579, 0.575, 0.552, 0.520, 0.487]
    within = [0.548, 0.565, 0.577, 0.579, 0.560, 0.528, 0.496]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(a, rand, "o-", ms=3.5, color=C_GREY, label="random split")
    ax.plot(a, logo, "s-", ms=3.5, color=C_MAIN, label="LOGO")
    ax.plot(a, within, "^--", ms=3.5, color=C_ACC, label="within-generator")
    ax.set_xscale("log")
    ax.set_xlabel(r"ridge regularisation $\alpha$")
    ax.set_ylabel("SRCC")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, "fig6_alpha_sweep.png"))
    plt.close(fig)


if __name__ == "__main__":
    for fn in [fig1_dataset, fig2_protocol, fig3_decomposition, fig4_source_only,
               fig5_ceiling, fig6_alpha]:
        fn()
        print("ok", fn.__name__)
