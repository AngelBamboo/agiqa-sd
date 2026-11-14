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


def pair_counts(x, y, chunk=256):
    """Concordant, discordant and tied pairs of (x, y), over i < j."""
    n = len(x)
    conc = disc = tied = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = np.sign(x[start:stop, None] - x[None, :])
        dy = np.sign(y[start:stop, None] - y[None, :])
        keep = np.triu(np.ones((stop - start, n), dtype=bool), k=start + 1)
        s = (dx * dy)[keep]
        conc += int((s > 0).sum())
        disc += int((s < 0).sum())
        tied += int((s == 0).sum())
    return conc, disc, tied


def pair_decomposition(pred, gt, gen, min_per_generator=30):
    """Split Kendall's tau into the same-source and cross-source pair sets.

    Kendall's tau is a (pair-count weighted) average of the sign agreement over
    pairs, so the pairs can be partitioned by whether the two images share a
    generator without approximation. Spearman's rho has no such exact
    decomposition, which is why this diagnostic is reported in tau.
    """
    c_all, d_all, t_all = pair_counts(pred, gt)
    c_in = d_in = t_in = 0
    for g in sorted(set(gen)):
        sel = gen == g
        if sel.sum() < min_per_generator:
            continue
        c, d, t = pair_counts(pred[sel], gt[sel])
        c_in += c
        d_in += d
        t_in += t
    c_out, d_out, t_out = c_all - c_in, d_all - d_in, t_all - t_in

    def tau(c, d, t):
        den = c + d + t
        return float((c - d) / den) if den else float("nan")

    n_in, n_out = c_in + d_in + t_in, c_out + d_out + t_out
    return {"tau_all": tau(c_all, d_all, t_all),
            "tau_within": tau(c_in, d_in, t_in),
            "tau_cross": tau(c_out, d_out, t_out),
            "pair_share_cross": float(n_out / (n_in + n_out)),
            "cross_share_of_net": float((c_out - d_out) / (c_all - d_all))
            if (c_all - d_all) else float("nan"),
            "n_pairs_same": int(n_in), "n_pairs_cross": int(n_out)}


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
    ap.add_argument("--pairs", action="store_true",
                    help="also split Kendall's tau into same-source and cross-source pairs")
    args = ap.parse_args()

    per_file = []
    for path in args.files:
        pred, gt, gen = read_predictions(path)
        res = summarise(pred, gt, gen)
        if args.pairs:
            res.update(pair_decomposition(pred, gt, gen))
        res["file"] = os.path.basename(path)
        per_file.append(res)

    if not args.json:
        if args.pairs:
            print(f"{'file':40s} {'tau':>6s} {'same':>6s} {'cross':>6s} "
                  f"{'pairs%':>6s} {'net%':>6s}")
            for r in per_file:
                print(f"{r['file']:40s} {r['tau_all']:6.3f} {r['tau_within']:6.3f} "
                      f"{r['tau_cross']:6.3f} {100 * r['pair_share_cross']:6.1f} "
                      f"{100 * r['cross_share_of_net']:6.1f}")
            print("(pairs% = share of test pairs that join two generators; "
                  "net% = share of the net concordant-minus-discordant count "
                  "contributed by those pairs)")
        else:
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
