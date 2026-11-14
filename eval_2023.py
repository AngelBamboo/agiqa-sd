import os
"""Protocol comparison on AIGCIQA2023, to test whether the inflation tracks the
between/within generator variance ratio rather than being dataset-specific."""
import argparse
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import sys
import agiqa.config as cfg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load():
    feats = np.load(cfg.CLIP_2023)
    rows = list(csv.DictReader(open(os.path.join(cfg.DATA_2023, "data.csv"), encoding="utf-8")))
    return feats, rows


def split(rows, protocol, holdout=None, seed=0, test_ratio=0.2):
    rng = np.random.RandomState(seed)
    n = len(rows)
    if protocol == "random":
        idx = rng.permutation(n)
        cut = int(n * (1 - test_ratio))
        return np.sort(idx[:cut]), np.sort(idx[cut:])
    if protocol == "logo":
        tr = [i for i, r in enumerate(rows) if r["generator"] != holdout]
        te = [i for i, r in enumerate(rows) if r["generator"] == holdout]
        return np.array(tr), np.array(te)
    raise ValueError(protocol)


def fit_eval(X, y, tr, te, seed=0, epochs=200):
    torch.manual_seed(seed)
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd).float().cuda()
    ym, ys = y[tr].mean(), y[tr].std() + 1e-8
    yn = torch.from_numpy((y - ym) / ys).float().cuda()
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(256, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(epochs):
        idx = torch.randint(0, len(tr), (256,))
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xn[tr[idx]]).squeeze(1), yn[tr[idx]])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xn[te]).squeeze(1).cpu().numpy()
    return pred, y[te]


def decompose(pred, gt, gens):
    d = {}
    for g in sorted(set(gens)):
        sel = np.array(gens) == g
        if sel.sum() > 30 and np.std(pred[sel]) > 1e-9:
            d[g] = float(stats.spearmanr(pred[sel], gt[sel]).statistic)
    gm = [gt[np.array(gens) == g].mean() for g in sorted(set(gens))]
    gp = [pred[np.array(gens) == g].mean() for g in sorted(set(gens))]
    return {"overall": float(stats.spearmanr(pred, gt).statistic),
            "within_mean": float(np.mean(list(d.values()))) if d else None,
            "between": float(stats.spearmanr(gp, gm).statistic),
            "per_generator": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X, rows = load()
    y = np.array([float(r["mos"]) for r in rows])
    gens = [r["generator"] for r in rows]

    out = {}
    tr, te = split(rows, "random", seed=args.seed)
    pred, gt = fit_eval(X, y, tr, te, seed=args.seed)
    out["random"] = decompose(pred, gt, [gens[i] for i in te])
    print("random", json.dumps({k: v for k, v in out["random"].items() if k != "per_generator"}))

    logos = {}
    for h in sorted(set(gens)):
        tr, te = split(rows, "logo", holdout=h)
        pred, gt = fit_eval(X, y, tr, te, seed=args.seed)
        r = decompose(pred, gt, [gens[i] for i in te])
        logos[h] = r
        print(f"logo {h:18s} srcc {r['overall']:.3f}")
    out["logo"] = logos
    out["logo_mean"] = float(np.mean([v["overall"] for v in logos.values()]))
    print("logo mean", round(out["logo_mean"], 3))
    json.dump(out, open(os.path.join(cfg.RUNS, "aigciqa2023_protocols.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
