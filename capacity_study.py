import os
"""Does model capacity buy performance or source shortcuts?

Compares regularised linear readouts against an MLP on identical frozen CLIP
features, across the random and leave-one-generator-out protocols.
"""
import argparse
import csv
import json

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import sys
import agiqa.config as cfg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_3k():
    names = open(os.path.join(cfg.FEATURES, "agiqa3k_ViT-B-32_openai_names.txt"),
                 encoding="utf-8").read().split("\n")
    X = np.load(cfg.CLIP_3K)
    meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))}
    rows = [{"name": n, "generator": n.split("_")[0],
             "mos": float(meta[n]["mos_quality"])} for n in names]
    return X, rows


def split(rows, protocol, holdout=None, seed=0):
    rng = np.random.RandomState(seed)
    n = len(rows)
    if protocol == "random":
        idx = rng.permutation(n)
        cut = int(n * 0.8)
        return np.sort(idx[:cut]), np.sort(idx[cut:])
    tr = [i for i, r in enumerate(rows) if r["generator"] != holdout]
    te = [i for i, r in enumerate(rows) if r["generator"] == holdout]
    return np.array(tr), np.array(te)


def decompose(pred, gt, gens):
    per = {}
    for g in sorted(set(gens)):
        sel = np.array(gens) == g
        if sel.sum() > 30 and np.std(pred[sel]) > 1e-9:
            per[g] = float(stats.spearmanr(pred[sel], gt[sel]).statistic)
    gm = [gt[np.array(gens) == g].mean() for g in sorted(set(gens))]
    gp = [pred[np.array(gens) == g].mean() for g in sorted(set(gens))]
    return {"overall": float(stats.spearmanr(pred, gt).statistic),
            "within": float(np.mean(list(per.values()))) if per else None,
            "between": float(stats.spearmanr(gp, gm).statistic)}


MODELS = {
    "ridge_1": lambda: Ridge(alpha=1.0),
    "ridge_100": lambda: Ridge(alpha=100.0),
    "ridge_1000": lambda: Ridge(alpha=1000.0),
    "mlp": lambda: MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=800,
                                early_stopping=False, random_state=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=os.path.join(cfg.RUNS, "capacity_study.json"))
    args = ap.parse_args()

    X, rows = load_3k()
    y = np.array([r["mos"] for r in rows])
    gens = [r["generator"] for r in rows]
    seeds = [int(s) for s in args.seeds.split(",")]

    results = {}
    for mname, mk in MODELS.items():
        for protocol in ["random", "logo"]:
            holds = [None] if protocol == "random" else sorted(set(gens))
            per_hold = {}
            for h in holds:
                stats_acc = []
                for s in seeds:
                    tr, te = split(rows, protocol, holdout=h, seed=s)
                    sc = StandardScaler().fit(X[tr])
                    m = mk()
                    m.fit(sc.transform(X[tr]), y[tr])
                    pred = m.predict(sc.transform(X[te]))
                    stats_acc.append(decompose(pred, y[te], [gens[i] for i in te]))
                key = h or "all"
                per_hold[key] = {
                    "overall": float(np.mean([a["overall"] for a in stats_acc])),
                    "within": float(np.nanmean([a["within"] for a in stats_acc])),
                    "between": float(np.nanmean([a["between"] for a in stats_acc])),
                }
            results[f"{mname}/{protocol}"] = per_hold
            vals = [v["overall"] for v in per_hold.values()]
            print(f"{mname:12s} {protocol:7s} overall {np.mean(vals):.3f} "
                  f"within {np.mean([v['within'] for v in per_hold.values()]):.3f}", flush=True)

    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
