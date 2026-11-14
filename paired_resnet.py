"""Seen-source versus unseen-source on the end-to-end ResNet-18.

The frozen-feature version of this control is in paired_source.py. This one
repeats it with the trained backbone, so the conclusion does not depend on the
features being frozen: for each generator the same 20% of its images are scored
twice, once by a model trained without that generator and once by a model
trained with the other 80% of it added to the training set.

    python paired_resnet.py --seeds 0 --epochs 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agiqa.data import AGIQADataset, GENERATORS, load_metadata
from agiqa.models import QualityNet
import agiqa.config as cfg

DATA_ROOT = os.environ.get("AGIQA_ROOT", cfg.DATA_3K)


def train_and_predict(rows, tr_idx, te_idx, seed, epochs, size=224, batch=32, lr=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = AGIQADataset(rows, tr_idx, size, train=True)
    test_set = AGIQADataset(rows, te_idx, size, train=False)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model = QualityNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = mse(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).float().cpu().numpy())
            gts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = load_metadata(DATA_ROOT)
    gen = np.array([r["generator"] for r in rows])
    out = {}
    print(f"{'generator':11s} {'n_test':>7s} {'unseen':>7s} {'seen':>7s} {'diff':>7s} {'sec':>6s}")
    for g in GENERATORS:
        idx_g = np.where(gen == g)[0]
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(idx_g))
        cut = int(len(idx_g) * (1 - args.test_ratio))
        rest_g, test_idx = idx_g[perm[:cut]], idx_g[perm[cut:]]
        other = np.where(gen != g)[0]

        un_scores, se_scores, t0 = [], [], time.time()
        for seed in seeds:
            p_un, gt = train_and_predict(rows, other, test_idx, seed, args.epochs)
            p_se, _ = train_and_predict(rows, np.concatenate([other, rest_g]),
                                        test_idx, seed, args.epochs)
            un_scores.append(stats.spearmanr(p_un, gt).statistic)
            se_scores.append(stats.spearmanr(p_se, gt).statistic)

        un, se = float(np.mean(un_scores)), float(np.mean(se_scores))
        out[g] = {"n_test": int(len(test_idx)), "unseen": un, "seen": se, "diff": se - un}
        print(f"{g:11s} {len(test_idx):7d} {un:7.3f} {se:7.3f} {se - un:+7.3f} "
              f"{time.time() - t0:6.0f}", flush=True)

    diffs = [v["diff"] for v in out.values()]
    out["_mean"] = {"unseen": float(np.mean([v["unseen"] for v in out.values()])),
                    "seen": float(np.mean([v["seen"] for v in out.values()])),
                    "diff": float(np.mean(diffs))}
    print(f"\nmean: unseen {out['_mean']['unseen']:.3f}  seen {out['_mean']['seen']:.3f}  "
          f"diff {out['_mean']['diff']:+.3f}")

    name = args.out or "paired_resnet.json"
    json.dump(out, open(os.path.join(cfg.RUNS, name), "w", encoding="utf-8"), indent=2)
    print("wrote", name)


if __name__ == "__main__":
    main()
