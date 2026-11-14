"""Publication figures for the source-confound paper.

Every panel is drawn from the archived records in results/ (plus the dataset CSV
for the MOS distribution), so re-running this script after new experiments is
enough to refresh the figures -- no numbers are hard-coded here.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

# paper order and display names
GENS = ["sd1.5", "midjourney", "glide", "AttnGAN", "xl2.2", "DALLE2"]
LABEL = {"sd1.5": "SD 1.5", "midjourney": "Midjourney", "glide": "GLIDE",
         "AttnGAN": "AttnGAN", "xl2.2": "SDXL", "DALLE2": "DALL-E 2"}


def load_rows():
    return list(csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8")))


def run_record(name):
    path = os.path.join(cfg.RUNS, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing record {path}; run the experiment that produces it first")
    return json.load(open(path, encoding="utf-8"))


def logo_folds(prefix):
    """Per-generator (mean, std) of the three seeds of a LOGO experiment."""
    out = {}
    for g in GENS:
        vals = [run_record(f"{prefix}_logo_{g}_s{s}.json")["srcc"] for s in range(3)]
        out[g] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    return out


def split_stats(pred_csv):
    """Overall, within-generator and between-generator SRCC of a prediction file."""
    rows = list(csv.DictReader(open(pred_csv, encoding="utf-8")))
    pred = np.array([float(r["pred"]) for r in rows])
    mos = np.array([float(r["mos"]) for r in rows])
    gens = np.array([r["generator"] for r in rows])
    within = []
    gm, pm = [], []
    for g in sorted(set(gens)):
        sel = gens == g
        within.append(stats.spearmanr(pred[sel], mos[sel]).statistic)
        gm.append(mos[sel].mean())
        pm.append(pred[sel].mean())
    return (float(stats.spearmanr(pred, mos).statistic), float(np.mean(within)),
            float(stats.spearmanr(pm, gm).statistic))


def fig1_dataset():
    rows = load_rows()
    data = [[float(r["mos_quality"]) for r in rows if r["name"].split("_")[0] == g] for g in GENS]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bp = ax.boxplot(data, tick_labels=[LABEL[g] for g in GENS], patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", lw=1.2),
                    flierprops=dict(marker=".", ms=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_MAIN)
        patch.set_alpha(0.85)
    ax.set_ylabel("MOS (perceptual quality)")
    ax.tick_params(axis="x", rotation=25)
    ax.axhline(np.mean([np.mean(d) for d in data]), ls="--", lw=0.8, color=C_GREY)
    fig.savefig(os.path.join(OUT, "fig1_mos_by_generator.png"))
    plt.close(fig)


def fig2_protocol():
    """Random-split vs LOGO, per generator, for the CLIP readout."""
    folds = logo_folds("clip")
    rnd = [run_record(f"clip_random_all_s{s}.json")["srcc"] for s in range(3)]
    within_random = split_stats(os.path.join(cfg.PREDICTIONS, "preds_clip_random.csv"))[1]
    control = [run_record(f"source_only_{g}.json")["srcc"] for g in GENS]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(GENS))
    ax.bar(x, [folds[g][0] for g in GENS], yerr=[folds[g][1] for g in GENS],
           capsize=2.5, color=C_MAIN, width=0.6, label="leave-one-generator-out")
    ax.axhline(within_random, ls="--", color=C_ACC, lw=1.2,
               label=f"within-source, random split ({within_random:.3f})")
    ax.axhline(np.mean(control), ls=":", color=C_GREY, lw=1.4,
               label=f"source-only control ({np.mean(control):.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[g] for g in GENS], rotation=25, ha="right")
    ax.set_ylabel("SRCC")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.8)
    fig.savefig(os.path.join(OUT, "fig2_protocol_gap.png"))
    plt.close(fig)


def fig3_decomposition():
    models = ["ResNet-18", "CLIP + MLP", "CoOp"]
    files = ["preds_random_s0.csv", "preds_clip_random.csv", "preds_coop.csv"]
    stats_by_model = [split_stats(os.path.join(cfg.PREDICTIONS, f)) for f in files]
    overall = [s[0] for s in stats_by_model]
    within = [s[1] for s in stats_by_model]
    between = [s[2] for s in stats_by_model]

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
    control = [run_record(f"source_only_{g}.json")["srcc"] for g in GENS]
    trained = [np.mean([run_record(f"clip_logo_{g}_s{s}.json")["srcc"] for s in range(3)])
               for g in GENS]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(GENS))
    w = 0.36
    ax.bar(x - w / 2, trained, w, label="trained model", color=C_MAIN)
    ax.bar(x + w / 2, control, w, label="source-only control", color=C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[g] for g in GENS], rotation=25, ha="right")
    ax.set_ylabel("SRCC (held-out generator)")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(OUT, "fig4_source_only.png"))
    plt.close(fig)


def fig5_alpha():
    sweep = run_record("alpha_sweep.json")
    alphas = sorted(int(a) for a in sweep)
    rand = [sweep[str(a)]["random"] for a in alphas]
    logo = [sweep[str(a)]["logo"] for a in alphas]
    within = [sweep[str(a)]["within"] for a in alphas]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(alphas, rand, "o-", ms=3.5, color=C_GREY, label="random split")
    ax.plot(alphas, logo, "s-", ms=3.5, color=C_MAIN, label="LOGO")
    ax.plot(alphas, within, "^--", ms=3.5, color=C_ACC, label="within-generator")
    ax.set_xscale("log")
    ax.set_xlabel(r"ridge regularisation $\alpha$")
    ax.set_ylabel("SRCC")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, "fig6_alpha_sweep.png"))
    plt.close(fig)


if __name__ == "__main__":
    for fn in [fig1_dataset, fig2_protocol, fig3_decomposition, fig4_source_only, fig5_alpha]:
        fn()
        print("ok", fn.__name__)
