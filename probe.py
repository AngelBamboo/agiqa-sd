"""How much generator information do the frozen CLIP features carry?

A linear probe on the same features the quality heads use, five-fold stratified
accuracy plus a normalised confusion matrix. If source is decodable, a quality
model built on these features has an easy shortcut available.
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import agiqa.config as cfg
from nested_ridge import load

GEN = ["sd1.5", "midjourney", "glide", "AttnGAN", "xl2.2", "DALLE2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=2000)
    args = ap.parse_args()

    X, y, g, prompts = load("3k")
    labels = sorted(set(g))
    yi = np.array([labels.index(v) for v in g])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)

    acc, bacc, cm = [], [], np.zeros((len(labels), len(labels)), dtype=int)
    for tr, te in skf.split(X, yi):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=args.max_iter, C=1.0).fit(sc.transform(X[tr]), yi[tr])
        pred = clf.predict(sc.transform(X[te]))
        acc.append((pred == yi[te]).mean())
        bacc.append(balanced_accuracy_score(yi[te], pred))
        cm += confusion_matrix(yi[te], pred, labels=range(len(labels)))

    major = max(np.bincount(yi)) / len(yi)
    res = {"accuracy": float(np.mean(acc)), "accuracy_std": float(np.std(acc, ddof=1)),
           "balanced_accuracy": float(np.mean(bacc)),
           "majority_class_baseline": float(major), "chance": 1.0 / len(labels),
           "labels": labels, "confusion": cm.tolist()}
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=2))
    print("\nrows = true, cols = predicted, order:", labels)
    for name, row in zip(labels, cm):
        print(f"{name:11s} " + " ".join(f"{v:5d}" for v in row))

    json.dump(res, open(os.path.join(cfg.RUNS, "source_probe.json"), "w", encoding="utf-8"),
              indent=2)
    print("wrote source_probe.json")


if __name__ == "__main__":
    main()
