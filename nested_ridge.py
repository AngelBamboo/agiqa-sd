"""Ridge readout with regularisation chosen without touching the test source.

For each leave-one-generator-out fold, the coefficient is selected on a
prompt-grouped validation split drawn from the *training* generators only, then
refit on the full training set and evaluated once on the held-out generator.
This removes the selection bias of picking alpha on the test source.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import agiqa.config as cfg

ALPHAS = [30, 100, 300, 1000, 3000, 10000, 30000]


def load():
    names_path = os.path.join(cfg.FEATURES, "agiqa3k_ViT-B-32_openai_names.txt")
    if not os.path.exists(names_path):
        sys.exit("Run extract_features.py first: it writes the feature matrix and "
                 "the matching image-order file next to it.")
    names = open(names_path, encoding="utf-8").read().split("\n")
    X = np.load(cfg.CLIP_3K)
    meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))}
    y = np.array([float(meta[n]["mos_quality"]) for n in names])
    g = np.array([n.split("_")[0] for n in names])
    prompts = np.array([meta[n]["prompt"] for n in names])
    return X, y, g, prompts


def prompt_split(idx, prompts, seed=0, frac=0.2):
    rng = np.random.RandomState(seed)
    keys = sorted({hashlib.md5(prompts[i].lower().encode()).hexdigest() for i in idx})
    rng.shuffle(keys)
    cut = int(len(keys) * (1 - frac))
    val_keys = set(keys[cut:])
    val = np.array([i for i in idx if hashlib.md5(prompts[i].lower().encode()).hexdigest() in val_keys])
    tr = np.array([i for i in idx if i not in set(val)])
    return tr, val


def fit_predict(Xtr, ytr, Xte, alpha):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def main():
    X, y, g, prompts = load()
    holdouts = ["sd1.5", "AttnGAN", "glide", "xl2.2", "DALLE2", "midjourney"]
    results = {}
    for hold in holdouts:
        tr_idx = np.where(g != hold)[0]
        te_idx = np.where(g == hold)[0]

        # inner selection over several inner splits for stability
        scores = {a: [] for a in ALPHAS}
        for seed in range(3):
            itr, iva = prompt_split(tr_idx, prompts, seed=seed)
            for a in ALPHAS:
                p = fit_predict(X[itr], y[itr], X[iva], a)
                scores[a].append(stats.spearmanr(p, y[iva]).statistic)
        mean_scores = {a: float(np.mean(v)) for a, v in scores.items()}
        best = max(mean_scores, key=mean_scores.get)

        pred = fit_predict(X[tr_idx], y[tr_idx], X[te_idx], best)
        srcc = float(stats.spearmanr(pred, y[te_idx]).statistic)
        results[hold] = {"n_test": int(len(te_idx)), "alpha_selected": best,
                         "inner_srcc": mean_scores[best], "srcc": srcc}
        print(f"{hold:11s} n={len(te_idx):4d} alpha*={best:6d} "
              f"inner={mean_scores[best]:.3f} test={srcc:.3f}", flush=True)

    vals = [v["srcc"] for v in results.values()]
    print(f"\nnested ridge LOGO mean {np.mean(vals):.3f}")
    json.dump(results, open(os.path.join(cfg.RUNS, "nested_ridge.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
