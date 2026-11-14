"""A predictor that knows only the generator label.

For a random 80/20 split, take the per-generator mean MOS of the *training*
set and assign it to every test image of that generator. The test labels are
used only to score the result, never to build the means, and no image-level
quality regressor is fitted. Whatever pooled SRCC this achieves is what the
source label alone is worth under the same protocol as the trained models.

    python source_mean_baseline.py --preds results/predictions/preds_random_s0.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agiqa.config as cfg

DATA = os.environ.get("AGIQA_3K", cfg.DATA_3K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True,
                    help="prediction CSV for the test split (only its image names are used)")
    ap.add_argument("--data", default=os.path.join(DATA, "data.csv"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    test_rows = list(csv.DictReader(open(args.preds, encoding="utf-8")))
    test_names = {r["name"] for r in test_rows}
    full = list(csv.DictReader(open(args.data, encoding="utf-8")))

    train = [r for r in full if r["name"] not in test_names]
    if not train:
        sys.exit("no training images left: is the prediction file the whole database?")

    means = {}
    for r in train:
        g = r["name"].split("_")[0]
        means.setdefault(g, []).append(float(r["mos_quality"]))
    means = {g: float(np.mean(v)) for g, v in means.items()}
    fallback = float(np.mean([float(r["mos_quality"]) for r in train]))

    print(f"{len(test_rows)} test images, {len(train)} training images, "
          f"{len(means)} generators")
    for g in sorted(means):
        print(f"  {g:11s} train mean {means[g]:.3f}")

    pred = np.array([means.get(r["generator"], fallback) for r in test_rows])
    gt = np.array([float(r["mos"]) for r in test_rows])
    gen = np.array([r["generator"] for r in test_rows])

    pooled = float(stats.spearmanr(pred, gt).statistic)
    gmean_gt = [gt[gen == g].mean() for g in sorted(set(gen))]
    gmean_pr = [pred[gen == g].mean() for g in sorted(set(gen))]
    between = float(stats.spearmanr(gmean_pr, gmean_gt).statistic)
    print(f"\npooled SRCC {pooled:.3f}   between-generator SRCC {between:.3f}   "
          f"within-generator SRCC undefined (constant inside each generator)")

    out = args.out or os.path.join(
        cfg.PREDICTIONS, "preds_sourcemean_" + os.path.basename(args.preds))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "generator", "mos", "pred"])
        for r, p in zip(test_rows, pred):
            w.writerow([r["name"], r["generator"], r["mos"], f"{p:.6f}"])
    print("wrote", out)


if __name__ == "__main__":
    main()
