"""Evaluate a model trained on AGIQA-3K against AGIQA-1K (unseen generators).

AGIQA-1K contains deepai and dreamStudio images, neither of which appears in
AGIQA-3K, so this is a clean cross-database + cross-generator test.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agiqa.data import MEAN, STD
from agiqa.metrics import all_metrics
from agiqa.models import QualityNet
import agiqa.config as cfg


class OneK(Dataset):
    def __init__(self, root, size=224):
        self.items = []
        with open(os.path.join(root, "AIGC_all.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = os.path.join(root, "file", r["Image"])
                if os.path.exists(p):
                    self.items.append((p, float(r["Score"]), r["Image"].split("_")[0]))
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, s, g = self.items[i]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1), torch.tensor(s), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default=cfg.DATA_1K)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QualityNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    ds = OneK(args.root, args.size)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds, gts, gens = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            preds.append(model(x.to(device)).float().cpu().numpy())
            gts.append(y.numpy())
            gens += list(g)
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    m = all_metrics(preds, gts)

    from scipy import stats
    per = {}
    gens = np.array(gens)
    for g in sorted(set(gens)):
        sel = gens == g
        if sel.sum() > 20:
            per[g] = {"n": int(sel.sum()),
                      "srcc": float(stats.spearmanr(gts[sel], preds[sel]).statistic)}
    res = {"dataset": "AGIQA-1K", "n": len(ds), **m, "per_generator": per}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
