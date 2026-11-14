"""Train and evaluate a quality regressor on AGIQA-3K.

Examples:
    python train.py --protocol random --epochs 30
    python train.py --protocol logo --holdout AttnGAN --epochs 30
    python train.py --protocol logo --holdout AttnGAN --adv 0.5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agiqa.data import AGIQADataset, GENERATORS, load_metadata, make_split
from agiqa.metrics import all_metrics
from agiqa.models import QualityNet, count_params
import agiqa.config as cfg

DATA_ROOT = os.environ.get("AGIQA_ROOT", cfg.DATA_3K)


def evaluate(model, loader, device, ttc=1):
    """ttc > 1 averages predictions over shifted crops of the same image."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if ttc <= 1:
                out = model(x)
            else:
                acc = model(x)
                for s in range(1, ttc):
                    sh = 8 * s
                    acc = acc + model(torch.roll(x, shifts=sh, dims=3))
                out = acc / ttc
            preds.append(out.float().cpu().numpy())
            gts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="random", choices=["random", "prompt", "logo"])
    ap.add_argument("--holdout", default="AttnGAN")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adv", type=float, default=0.0, help="gradient-reversal weight")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="CSV path for per-image predictions")
    ap.add_argument("--save_model", default="", help="path to save the trained weights")
    ap.add_argument("--strong_aug", action="store_true")
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--local", action="store_true", help="train on local crops")
    ap.add_argument("--ttc", type=int, default=1, help="test-time crops to average")
    ap.add_argument("--rank_weight", type=float, default=0.0)
    ap.add_argument("--normalize", default="global", choices=["global", "source"],
                    help="target normalisation; 'source' removes generator offsets")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_metadata(DATA_ROOT)
    tr_idx, te_idx = make_split(rows, args.protocol, seed=args.seed, holdout=args.holdout)
    train_set = AGIQADataset(rows, tr_idx, args.size, train=True,
                             return_generator=args.adv > 0,
                             normalize=args.normalize,
                             strong_aug=args.strong_aug, local=args.local)
    test_set = AGIQADataset(rows, te_idx, args.size, train=False)

    if args.balanced:
        from agiqa.data import GENERATORS
        counts = np.array([sum(1 for i in tr_idx if rows[i]["generator"] == g)
                           for g in GENERATORS], dtype=np.float64)
        w_per_gen = 1.0 / np.maximum(counts, 1)
        sample_w = [w_per_gen[GENERATORS.index(rows[i]["generator"])] for i in tr_idx]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_w, num_samples=len(tr_idx), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler,
                                  num_workers=0, drop_last=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                  num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model = QualityNet(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def ranking_loss(q, y, margin=0.1):
        """Pairwise margin ranking over all pairs in the batch."""
        n = q.shape[0]
        if n < 2:
            return q.sum() * 0
        dq = q[:, None] - q[None, :]
        dy = y[:, None] - y[None, :]
        mask = dy.abs() > 1e-3
        if mask.sum() == 0:
            return q.sum() * 0
        sign = torch.sign(dy)
        return torch.relu(margin - sign * dq)[mask].mean()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        model.grl_lambda = args.adv * min(1.0, ep / max(1, args.epochs * 0.3))
        total = 0.0
        for batch in train_loader:
            if args.adv > 0:
                x, y, g = batch
                g = g.to(device)
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            q = model(x, return_logits=args.adv > 0)
            if args.adv > 0:
                q, g_logits = q
                loss = mse(q, y) + 0.1 * ce(g_logits, g)
            else:
                loss = mse(q, y)
            if args.rank_weight > 0:
                loss = loss + args.rank_weight * ranking_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep+1:3d} loss {total/len(train_loader):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)

    pred, gt = evaluate(model, test_loader, device, args.ttc)
    m = all_metrics(pred, gt)

    if args.dump:
        import csv as _csv
        os.makedirs(os.path.dirname(args.dump), exist_ok=True)
        with open(args.dump, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["name", "generator", "mos", "pred"])
            for i, row in enumerate(test_set.rows):
                w.writerow([row["name"], row["generator"], row["mos"], float(pred[i])])
    result = {
        "protocol": args.protocol,
        "holdout": args.holdout if args.protocol == "logo" else None,
        "adv": args.adv,
        "balanced": args.balanced,
        "rank_weight": args.rank_weight,
        "strong_aug": args.strong_aug,
        "normalize": args.normalize,
        "local": args.local,
        "ttc": args.ttc,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "params": count_params(model),
        **m,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2))

    out = args.out or os.path.join(
        cfg.RUNS, f"{args.protocol}_{result['holdout'] or 'all'}_adv{args.adv}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
