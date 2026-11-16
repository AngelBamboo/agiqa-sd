"""Stratified bootstrap confidence intervals for the three correlation axes.

The databases have six generators and a few hundred images per generator, so a
point estimate of SRCC carries visible uncertainty. Resampling images *within*
each generator keeps the source composition fixed and answers the question a
reviewer will ask: is the pooled-within gap larger than the sampling noise?

Usage::

    python -m bench.bootstrap --dataset agiqa3k --reps 300
"""
import argparse
import glob
import json
import os

import numpy as np

import agiqa.metrics as M
from bench import datasets

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
SCORES = os.path.join(BENCH, "scores")
SUMMARY = os.path.join(BENCH, "summary")
OUT = os.path.join(BENCH, "bootstrap")


def axes(scores, mos, gens):
    pooled = M.srcc(scores, mos)
    per = []
    for g in sorted(set(gens)):
        sel = gens == g
        if sel.sum() >= 30 and np.std(scores[sel]) > 1e-9:
            per.append(M.srcc(scores[sel], mos[sel]))
    within = float(np.mean(per)) if per else float("nan")
    g_mos = [mos[gens == g].mean() for g in sorted(set(gens))]
    g_score = [scores[gens == g].mean() for g in sorted(set(gens))]
    between = M.srcc(g_score, g_mos) if len(g_mos) >= 3 else float("nan")
    return pooled, within, between


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(datasets.LOADERS))
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    items = datasets.load(args.dataset)
    mos = np.array([it["mos"] for it in items])
    gens = np.array([it["generator"] for it in items])
    rng = np.random.RandomState(0)

    for path in sorted(glob.glob(os.path.join(SCORES, args.dataset, "*.npy"))):
        model = os.path.basename(path)[:-4]
        scores = np.load(path)
        keep = np.isfinite(scores)
        if keep.sum() < 100:
            continue
        s, m, g = scores[keep], mos[keep], gens[keep]

        point = axes(s, m, g)
        draws = []
        strata = [np.where(g == name)[0] for name in sorted(set(g))]
        for _ in range(args.reps):
            idx = np.concatenate([rng.choice(st, size=len(st), replace=True) for st in strata])
            draws.append(axes(s[idx], m[idx], g[idx]))
        draws = np.array(draws, dtype=float)
        record = {
            "dataset": args.dataset,
            "model": model,
            "n": int(keep.sum()),
            "reps": args.reps,
            "pooled": {"point": point[0], "lo": float(np.nanpercentile(draws[:, 0], 2.5)),
                       "hi": float(np.nanpercentile(draws[:, 0], 97.5))},
            "within": {"point": point[1], "lo": float(np.nanpercentile(draws[:, 1], 2.5)),
                       "hi": float(np.nanpercentile(draws[:, 1], 97.5))},
            "between": {"point": point[2], "lo": float(np.nanpercentile(draws[:, 2], 2.5)),
                        "hi": float(np.nanpercentile(draws[:, 2], 97.5))},
        }
        json.dump(record, open(os.path.join(OUT, f"{args.dataset}__{model}.json"), "w",
                               encoding="utf-8"), indent=2)
        print(f"{model:12s} pooled {point[0]:+.3f} [{record['pooled']['lo']:+.3f}, "
              f"{record['pooled']['hi']:+.3f}]  within {point[1]:+.3f} "
              f"[{record['within']['lo']:+.3f}, {record['within']['hi']:+.3f}]")


if __name__ == "__main__":
    main()
