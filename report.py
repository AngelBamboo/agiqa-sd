"""Report a quality model the way Section 7.2 of the paper recommends.

Give it one or more prediction files and it prints, for each of them, the
pooled SRCC, the within-source SRCC (equal-weight mean over generators) and the
between-source SRCC; if you pass one file per held-out generator it also prints
the equal-weight mean across files, which is the leave-one-generator-out figure.

Input: a CSV with a ``pred`` column and either a ``mos`` or ``gt`` column plus a
``generator`` column. Anything else in the file is ignored, so the per-image
predictions in results/predictions work as they are.

    python report.py results/predictions/preds_clip_random.csv
    python report.py results/predictions/preds_clip_logo_*.csv
    python report.py --json results/predictions/preds_ridge_logo_3k_*.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats


def read_predictions(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    for col in ("pred", "generator"):
        if col not in rows[0]:
            raise SystemExit(f"{path}: need a '{col}' column")
    gt_col = "mos" if "mos" in rows[0] else ("gt" if "gt" in rows[0] else None)
    if gt_col is None:
        raise SystemExit(f"{path}: need a 'mos' or 'gt' column")
    pred = np.array([float(r["pred"]) for r in rows])
    gt = np.array([float(r[gt_col]) for r in rows])
    gen = np.array([r["generator"] for r in rows])
    return pred, gt, gen


def summarise(pred, gt, gen, min_per_generator=30):
    """Pooled, within-source and between-source SRCC of one prediction file."""
    pooled = float(stats.spearmanr(pred, gt).statistic)
    within, gmean_gt, gmean_pred = [], [], []
    for g in sorted(set(gen)):
        sel = gen == g
        if sel.sum() < min_per_generator:
            continue
        if np.std(pred[sel]) > 1e-12:
            within.append(float(stats.spearmanr(pred[sel], gt[sel]).statistic))
        gmean_gt.append(float(gt[sel].mean()))
        gmean_pred.append(float(pred[sel].mean()))
    between = (float(stats.spearmanr(gmean_pred, gmean_gt).statistic)
               if len(gmean_gt) > 2 else float("nan"))
    return {"n": int(len(gt)), "generators": int(len(set(gen))),
            "pooled": pooled, "within": float(np.mean(within)) if within else float("nan"),
            "between": between}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--label", default="", help="name used in the JSON output")
    args = ap.parse_args()

    per_file = []
    for path in args.files:
        pred, gt, gen = read_predictions(path)
        res = summarise(pred, gt, gen)
        res["file"] = os.path.basename(path)
        per_file.append(res)

    if not args.json:
        print(f"{'file':44s} {'n':>6s} {'pooled':>7s} {'within':>7s} {'between':>8s}")
        for r in per_file:
            print(f"{r['file']:44s} {r['n']:6d} {r['pooled']:7.3f} "
                  f"{r['within']:7.3f} {r['between']:8.3f}")

    out = {"files": per_file}
    if len(per_file) > 1:
        means = {}
        for k in ("pooled", "within", "between"):
            vals = [r[k] for r in per_file if not np.isnan(r[k])]
            means[k] = float(np.mean(vals)) if vals else float("nan")
        out["equal_weight_mean"] = means
        if not args.json:
            btw = "n/a" if np.isnan(means["between"]) else f"{means['between']:.3f}"
            print(f"\nequal-weight mean over {len(per_file)} files: "
                  f"pooled {means['pooled']:.3f}  within {means['within']:.3f}  "
                  f"between {btw}")
            print("(when each file is one held-out generator, the 'within' mean is "
                  "the leave-one-generator-out figure)")

    if args.json:
        out["label"] = args.label or os.path.commonprefix([f["file"] for f in per_file])
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
