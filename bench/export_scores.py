"""Export per-image scores as CSV so they can be released without .npy files.

The score arrays under ``results/bench/scores`` are aligned with the dataset row
order. This script joins them back to the item keys, generators and MOS and
writes one small CSV per model and database under ``results/bench/scores_csv``.
"""
import csv
import glob
import os

import numpy as np

from bench import datasets

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
SCORES = os.path.join(BENCH, "scores")
OUT = os.path.join(BENCH, "scores_csv")


def main():
    for ds in datasets.LOADERS:
        items = datasets.load(ds)
        src = os.path.join(SCORES, ds)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(OUT, ds)
        os.makedirs(dst, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(src, "*.npy"))):
            model = os.path.basename(path)[:-4]
            scores = np.load(path)
            with open(os.path.join(dst, f"{model}.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "generator", "mos", "score"])
                for item, score in zip(items, scores):
                    writer.writerow([item["key"], item["generator"], item["mos"],
                                     "" if not np.isfinite(score) else f"{score:.6f}"])
        print("wrote", dst)


if __name__ == "__main__":
    main()
