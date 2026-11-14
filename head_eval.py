"""Fit quality heads on cached frozen features.

Runs the same protocol matrix as the end-to-end model but in seconds, so the
generator-confound analysis can be repeated across several backbones.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agiqa.data import GENERATORS, make_split
import agiqa.config as cfg


def load_3k(feat_path):
    names = open(feat_path.replace(".npy", "_names.txt"), encoding="utf-8").read().split("\n")
    feats = np.load(feat_path)
    meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))}
    rows = []
    for i, n in enumerate(names):
        m = meta[n]
        rows.append({"name": n, "idx": i, "generator": n.split("_")[0],
                     "mos": float(m["mos_quality"]), "prompt": m["prompt"]})
    return feats, rows


def train_head(X, y, epochs=200, lr=1e-2, hidden=256, seed=0, gens=None,
               wsr=0.0, batch=256, weights=None):
    """Fits a regression head.

    wsr > 0 adds a within-source ranking term: pairs are drawn only from the
    same generator, so the model cannot gain by recognising the source and must
    use quality-relevant differences instead.
    """
    torch.manual_seed(seed)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd).float().cuda()
    yn = torch.from_numpy((y - y.mean()) / (y.std() + 1e-8)).float().cuda()
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(hidden, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)

    groups = None
    if wsr > 0 and gens is not None:
        groups = {}
        for i, g in enumerate(gens):
            groups.setdefault(g, []).append(i)
        groups = {g: torch.tensor(v) for g, v in groups.items() if len(v) > 8}
        gkeys = list(groups)

    if weights is not None:
        wt = torch.from_numpy(weights / weights.mean()).float().cuda()
    else:
        wt = torch.ones(len(yn), device='cuda')

    n = len(yn)
    for ep in range(epochs):
        idx = torch.randint(0, n, (batch,))
        q = net(Xn[idx]).squeeze(1)
        loss = ((q - yn[idx]) ** 2 * wt[idx]).mean()
        if groups:
            g = gkeys[np.random.randint(len(gkeys))]
            pool = groups[g]
            sel = pool[torch.randint(0, len(pool), (batch,))]
            qs = net(Xn[sel]).squeeze(1)
            ys = yn[sel]
            dq = qs[:, None] - qs[None, :]
            dy = ys[:, None] - ys[None, :]
            mask = dy.abs() > 1e-3
            if mask.any():
                loss = loss + wsr * torch.relu(0.1 - torch.sign(dy) * dq)[mask].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, mu, sd


def predict(net, mu, sd, X):
    with torch.no_grad():
        xb = torch.from_numpy((X - mu) / sd).float().cuda()
        return net(xb).squeeze(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=cfg.CLIP_3K)
    ap.add_argument("--protocol", default="random", choices=["random", "prompt", "logo"])
    ap.add_argument("--holdout", default="AttnGAN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="clip")
    ap.add_argument("--dump", default="")
    ap.add_argument("--wsr", type=float, default=0.0, help="within-source ranking weight")
    ap.add_argument("--reliability", action="store_true", help="weight by 1/annotation-variance")
    args = ap.parse_args()

    X, rows = load_3k(args.feats)
    tr, te = make_split(rows, args.protocol, seed=args.seed, holdout=args.holdout)
    y = np.array([r["mos"] for r in rows], dtype=np.float32)

    gens_tr = [rows[i]["generator"] for i in tr]
    wts = None
    if args.reliability:
        meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))}
        stds = np.array([float(meta[rows[i]["name"]]["std_quality"]) for i in tr])
        wts = 1.0 / (stds ** 2 + 1e-3)
    net, mu, sd = train_head(X[tr], y[tr], seed=args.seed, gens=gens_tr, wsr=args.wsr,
                             weights=wts)
    pred = predict(net, mu, sd, X[te])
    gt = y[te]
    res = {"backend": args.tag, "protocol": args.protocol,
           "holdout": args.holdout if args.protocol == "logo" else None,
           "seed": args.seed, "wsr": args.wsr, "reliability": args.reliability, "n_train": len(tr), "n_test": len(te),
           "srcc": float(stats.spearmanr(pred, gt).statistic),
           "plcc": float(stats.pearsonr(pred, gt).statistic)}

    if args.dump:
        with open(args.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "generator", "mos", "pred"])
            for k, i in enumerate(te):
                w.writerow([rows[i]["name"], rows[i]["generator"], rows[i]["mos"], float(pred[k])])

    print(json.dumps(res, indent=2))
    out = os.path.join(cfg.RUNS, f"{args.tag}_{args.protocol}_"
                                           f"{args.holdout if args.protocol == 'logo' else 'all'}_s{args.seed}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
