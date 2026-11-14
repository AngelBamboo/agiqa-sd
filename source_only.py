import os
"""Source-only control: predict quality from inferred generator identity alone.

For each test image, find the nearest training generator (centroid similarity in
CLIP space) and answer with that generator's mean MOS. No quality model, no
image-quality features. If this control approaches the performance of a trained
regressor, the regressor is doing source inference.
"""
import argparse
import csv
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agiqa.data import make_split
import agiqa.config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=cfg.CLIP_3K)
    ap.add_argument("--holdout", default="AttnGAN")
    ap.add_argument("--out", default="", help="JSON path; 'auto' writes one record per fold into AGIQA_RUNS")
    args = ap.parse_args()

    names = open(args.feats.replace(".npy", "_names.txt"), encoding="utf-8").read().split("\n")
    X = np.load(args.feats)
    meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))}
    rows = [{"name": n, "generator": n.split("_")[0], "mos": float(meta[n]["mos_quality"]),
             "prompt": meta[n]["prompt"]} for n in names]

    tr, te = make_split(rows, "logo", holdout=args.holdout)

    # Centroids of the generators seen in training
    cents, means = {}, {}
    for g in sorted({rows[i]["generator"] for i in tr}):
        idx = [i for i in tr if rows[i]["generator"] == g]
        cents[g] = X[idx].mean(0)
        means[g] = np.mean([rows[i]["mos"] for i in idx])
    C = np.stack([cents[g] for g in cents])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    gnames = list(cents)

    pred, gt = [], []
    for i in te:
        sim = X[i] @ C.T
        pred.append(means[gnames[int(np.argmax(sim))]])
        gt.append(rows[i]["mos"])
    pred, gt = np.array(pred), np.array(gt)

    res = {"holdout": args.holdout, "n": len(te),
           "srcc": float(stats.spearmanr(pred, gt).statistic),
           "plcc": float(stats.pearsonr(pred, gt).statistic),
           "unique_pred": int(len(np.unique(pred)))}
    print(json.dumps(res, indent=2))

    if args.out:
        path = (os.path.join(cfg.RUNS, f"source_only_{args.holdout}.json")
                if args.out == "auto" else args.out)
        json.dump(res, open(path, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
