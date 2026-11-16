"""Pooled SRCC of a predictor that infers the generator, then returns its mean.

The predictor never estimates quality from the image. Each test image is
assigned to the nearest training-generator centroid in CLIP ViT-B/32 feature
space, and is scored with that generator's training-set mean MOS. The result
is the part of pooled SRCC that source recognition alone can buy, under the
same stratified split used for the label-only baseline.

    python scripts/source_inference_baseline.py
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import datasets  # noqa: E402


def split_indices(generators, seed=0, frac=0.8):
    """Generator-stratified 80/20 split, identical to the label-only baseline."""
    rng = np.random.RandomState(seed)
    test = []
    for g in sorted(set(generators)):
        idx = np.where(generators == g)[0]
        rng.shuffle(idx)
        test += list(idx[int(len(idx) * frac):])
    test = np.array(sorted(test))
    train = np.setdiff1d(np.arange(len(generators)), test)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agiqa3k")
    ap.add_argument("--features", default=os.path.join(ROOT, "features", "agiqa3k_ViT-B-32_openai.npy"))
    ap.add_argument("--names", default=os.path.join(ROOT, "features", "agiqa3k_ViT-B-32_openai_names.txt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench", "source_inference.json"))
    args = ap.parse_args()

    items = datasets.load(args.dataset)
    feats = np.load(args.features)
    names = open(args.names, encoding="utf-8").read().split("\n")
    position = {n: i for i, n in enumerate(names)}
    order = [position[it["key"]] for it in items]
    feats = feats[order]

    gens = np.array([it["generator"] for it in items])
    mos = np.array([it["mos"] for it in items])
    train, test = split_indices(gens, args.seed)

    centroids, means = {}, {}
    for g in sorted(set(gens)):
        idx = train[gens[train] == g]
        c = feats[idx].mean(axis=0)
        centroids[g] = c / np.linalg.norm(c)
        means[g] = mos[idx].mean()
    labels = sorted(centroids)
    C = np.stack([centroids[g] for g in labels])
    assign = np.array(labels)[np.argmax(feats[test] @ C.T, axis=1)]

    pred = np.array([means[g] for g in assign])
    pooled = float(stats.spearmanr(pred, mos[test]).statistic)
    accuracy = float((assign == gens[test]).mean())

    record = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_test": int(len(test)),
        "source_accuracy": accuracy,
        "pooled_srcc": pooled,
        "within_mean": float(np.mean([
            stats.spearmanr(pred[gens[test] == g], mos[test][gens[test] == g]).statistic
            for g in labels if (gens[test] == g).sum() > 2])),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
