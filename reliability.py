"""Split-half reliability of the five-rater MOS in AIGCIQA2023.

The five individual ratings released with the database are split into a group of
two and a group of three; the two group means are correlated across images and
the result is corrected to the full five-rater mean with the Spearman-Brown
formula

    rho_5 = 2 * r_2v3 / (1 + r_2v3)

The square root of that reliability is the ceiling on the correlation any model
can reach with the five-rater mean, given that the images themselves are fixed.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

import agiqa.config as cfg

CSV = os.path.join(cfg.DATA_2023, "AIGIQA2023plus.csv")


def load():
    """Returns generator labels and an (n_images, 5) matrix of integer ratings."""
    rows = None
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            rows = list(csv.reader(open(CSV, encoding=enc)))[1:]
            break
        except UnicodeDecodeError:
            continue
    rows = [r for r in rows if len(r) >= 8]
    gens = [r[2] for r in rows]
    scores = np.array([[float(x) for x in r[3:8]] for r in rows])
    # the release contains a small number of out-of-range entries (a "11" on a
    # 1-5 scale); clip them to the scale rather than dropping the image
    n_bad = int((scores > 5).sum())
    if n_bad:
        print(f"clipped {n_bad} out-of-range rating(s) to the 1-5 scale")
        scores = np.clip(scores, 0, 5)
    return gens, scores


def spearman_brown(r):
    return 2 * r / (1 + r)


def cronbach_alpha(X):
    """Reliability of the mean of the columns of X, without a split.

    Split-half depends on how the five raters are divided; alpha does not, so it
    is reported alongside the 2-vs-3 estimate as a partition-free check.
    """
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reverse", action="store_true",
                    help="swap the two/three split (three first) as a robustness check")
    args = ap.parse_args()

    gens, scores = load()
    gens = np.array(gens)
    if args.reverse:
        a, b = scores[:, :3].mean(1), scores[:, 3:].mean(1)
    else:
        a, b = scores[:, :2].mean(1), scores[:, 2:].mean(1)

    out = {}
    print(f"{'generator':18s} {'n':>5s} {'2v3 r':>7s} {'rho_5':>7s} {'alpha':>7s} {'ceil_a':>7s}")
    for g in sorted(set(gens)):
        sel = gens == g
        if sel.sum() < 30:
            continue
        r = float(stats.pearsonr(a[sel], b[sel]).statistic)
        rho = float(spearman_brown(r))
        al = float(cronbach_alpha(scores[sel]))
        out[g] = {"n": int(sel.sum()), "r_2v3": r, "rho_5": rho, "ceiling": float(np.sqrt(rho)),
                  "alpha": al, "ceiling_alpha": float(np.sqrt(al))}
        print(f"{g:18s} {sel.sum():5d} {r:7.3f} {rho:7.3f} {al:7.3f} {np.sqrt(al):7.3f}")

    rhos = [v["rho_5"] for v in out.values()]
    print(f"\nreliability {min(rhos):.2f}-{max(rhos):.2f} (mean {np.mean(rhos):.2f}); "
          f"ceiling {np.sqrt(min(rhos)):.2f}-{np.sqrt(max(rhos)):.2f}")

    name = "reliability_reverse.json" if args.reverse else "reliability.json"
    json.dump(out, open(os.path.join(cfg.RUNS, name), "w", encoding="utf-8"), indent=2)
    print("wrote", name)


if __name__ == "__main__":
    main()
