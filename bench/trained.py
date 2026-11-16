"""Fold the trained baselines' per-image predictions into benchmark records.

Two prediction sets are combined, because they answer different questions:

* the **random-split** test predictions give the pooled / within-generator /
  between-generator decomposition under the protocol the literature uses;
* the six **leave-one-generator-out** folds give the source-disjoint number.
  Its within-generator mean is the benchmark's "can the model rank images from
  a generator it has never seen" axis.

The concatenated LOGO pool is deliberately *not* reported as a pooled SRCC:
each fold is a different model with its own output scale, so pooling the six
folds would measure cross-fold calibration as much as quality. That artefact is
itself worth a paragraph in the paper, and is why the record keeps the fold mean
separate.

Usage::

    python -m bench.trained --dataset agiqa3k
    python -m bench.trained --dataset aigciqa2023
"""
import argparse
import csv
import json
import os

import numpy as np

from bench.runner import SUMMARY, decompose

PRED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "predictions")

GENS_3K = ["AttnGAN", "DALLE2", "glide", "midjourney", "sd1.5", "xl2.2"]
GENS_2023 = ["Controlnet", "DALLE", "Glide", "Lafite", "Unidiffuser", "stable-diffusion"]

# dataset -> model -> (random csv, logo pattern, generators, family)
PATTERNS = {
    "agiqa3k": {
        "resnet18": ("preds_random_s0.csv", "preds_resnet_logo_{gen}_s0.csv", GENS_3K, "cnn"),
        "clip_mlp": ("preds_clip_random.csv", "preds_clip_logo_{gen}_s0.csv", GENS_3K, "clip"),
        "coop": ("preds_coop_random_all_s0.csv", "preds_coop_logo_{gen}_s0.csv", GENS_3K, "clip"),
        "ridge": ("preds_ridge_random_3k.csv", "preds_ridge_logo_3k_{gen}.csv", GENS_3K, "linear"),
    },
    "aigciqa2023": {
        "clip_mlp": ("preds_clip_2023_random_s0.csv", "preds_clip_2023_logo_{gen}_s0.csv", GENS_2023, "clip"),
        "ridge": (None, "preds_ridge_logo_2023_{gen}.csv", GENS_2023, "linear"),
    },
}


def read_csv(path):
    if not path or not os.path.exists(os.path.join(PRED, path)):
        return None
    return list(csv.DictReader(open(os.path.join(PRED, path), encoding="utf-8")))


def read_folds(pattern, gens):
    rows = []
    for gen in gens:
        part = read_csv(pattern.format(gen=gen))
        if part is None:
            return None
        rows += part
    return rows


def items_of(rows):
    return [{"generator": r["generator"], "mos": float(r["mos"])} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PATTERNS))
    args = ap.parse_args()

    os.makedirs(SUMMARY, exist_ok=True)
    for model, (random_csv, logo_pat, gens, family) in PATTERNS[args.dataset].items():
        random_rows = read_csv(random_csv)
        folds = read_folds(logo_pat, gens)
        if random_rows is None and folds is None:
            print(f"skip {model}: no predictions")
            continue

        record = {
            "dataset": args.dataset,
            "model": model,
            "family": family,
            "meta": {"kind": "trained"},
        }
        if random_rows is not None:
            scores = np.array([float(r["pred"]) for r in random_rows])
            record["random"] = decompose(scores, items_of(random_rows))
            record["meta"]["n_random"] = len(random_rows)
        if folds is not None:
            fold_srcc = {}
            for gen in gens:
                part = [r for r in folds if r["generator"] == gen]
                fold_srcc[gen] = decompose(
                    np.array([float(r["pred"]) for r in part]), items_of(part)
                )["pooled"]["srcc"]
            record["logo"] = {
                "within_mean": float(np.mean(list(fold_srcc.values()))),
                "per_generator": fold_srcc,
                "n_folds": len(gens),
            }
            record["meta"]["n_logo"] = len(folds)

        # leaderboard-facing aliases: the literature number is the random-split
        # pool; the source-disjoint number is the LOGO fold mean.
        if "random" in record:
            record["pooled"] = record["random"]["pooled"]
            record["within_mean"] = record["random"]["within_mean"]
            record["within_weighted"] = record["random"]["within_weighted"]
            record["between"] = record["random"]["between"]
            record["per_generator"] = record["random"]["per_generator"]
        if "logo" in record:
            record["source_disjoint"] = record["logo"]["within_mean"]

        out = os.path.join(SUMMARY, f"{args.dataset}__{model}_trained.json")
        json.dump(record, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else " n/a "
        print(f"{model:10s} pooled {fmt(record.get('pooled', {}).get('srcc') if 'pooled' in record else None)}"
              f"  within {fmt(record.get('within_mean'))}  between {fmt(record.get('between'))}"
              f"  LOGO {fmt(record.get('source_disjoint'))}")


if __name__ == "__main__":
    main()
