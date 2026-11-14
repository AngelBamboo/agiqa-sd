"""Attainable ceiling on correlation, from the released per-image rating spread.

AGIQA-3K reports how many people took part in the subjective experiment (21) and
releases, for every image, the mean and the standard deviation of its quality
ratings. Neither database releases the individual ratings.

Under a one-way random-effects model with k raters per image,

    MSB = k * Var(image means)          (between-image mean square)
    MSW = mean over images of SD_i^2    (within-image mean square)

and the reliability of the k-rater mean is (MSB - MSW) / MSB. Its square root is
the ceiling on the correlation any predictor of the images can reach with that
mean, given a linear, additive measurement-error model. The relation is exact
for Pearson correlation and approximate for Spearman.

    python reliability.py
    python reliability.py --k 11,21,31     # sensitivity to the rater count
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import agiqa.config as cfg


def icc_k(mos, sd, k):
    """Reliability of the mean of k raters, one-way random effects."""
    msb = k * np.var(mos, ddof=1)
    msw = np.mean(np.asarray(sd, dtype=float) ** 2)
    if msb <= msw:
        return float("nan")
    return float((msb - msw) / msb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="21", help="raters per image; 21 is the number "
                                              "reported for AGIQA-3K")
    args = ap.parse_args()
    ks = [int(v) for v in args.k.split(",")]

    path = os.path.join(cfg.DATA_3K, "data.csv")
    df = pd.read_csv(path)
    df["gen"] = df["name"].str.split("_").str[0]

    out = {}
    print(f"{'generator':11s} {'n':>5s} {'MOS':>5s} {'rater SD':>8s} "
          + " ".join(f"{'rho@k=' + str(k):>9s}" for k in ks)
          + " " + " ".join(f"{'ceil@' + str(k):>8s}" for k in ks))
    for g, s in df.groupby("gen"):
        rels = {k: icc_k(s["mos_quality"].values, s["std_quality"].values, k) for k in ks}
        out[g] = {"n": int(len(s)), "mos_mean": float(s["mos_quality"].mean()),
                  "rater_sd": float(s["std_quality"].mean()),
                  **{f"rho_k{k}": rels[k] for k in ks},
                  **{f"ceiling_k{k}": float(np.sqrt(rels[k])) for k in ks}}
        print(f"{g:11s} {len(s):5d} {s['mos_quality'].mean():5.2f} "
              f"{s['std_quality'].mean():8.3f} "
              + " ".join(f"{rels[k]:9.3f}" for k in ks)
              + " " + " ".join(f"{np.sqrt(rels[k]):8.3f}" for k in ks))

    main_k = ks[1] if len(ks) > 1 else ks[0]
    rels = [v[f"rho_k{main_k}"] for v in out.values()]
    print(f"\nk={main_k}: reliability {min(rels):.3f}-{max(rels):.3f}, "
          f"ceiling {np.sqrt(min(rels)):.3f}-{np.sqrt(max(rels)):.3f}")

    json.dump({"raters": ks, "per_generator": out},
              open(os.path.join(cfg.RUNS, "reliability.json"), "w", encoding="utf-8"), indent=2)
    print("wrote reliability.json")


if __name__ == "__main__":
    main()
