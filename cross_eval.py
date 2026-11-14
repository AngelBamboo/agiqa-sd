"""Evaluate a model trained on AGIQA-3K against another database, without finetuning.

AGIQA-1K contains deepai and dreamStudio images, neither of which appears in
AGIQA-3K, so this is a clean cross-database + cross-generator test.
AIGCIQA2023 uses six text-to-image models, also disjoint from AGIQA-3K.

    python train.py --protocol random --seed 0 --save_model weights/3k_rn18.pt
    python cross_eval.py --weights weights/3k_rn18.pt --dataset 1k
    python cross_eval.py --weights weights/3k_rn18.pt --dataset 2023
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


class TargetDB(Dataset):
    """A scoring-only view of an external database: image, MOS, generator label."""

    def __init__(self, root, dataset, size=224):
        self.items = []
        if dataset == "1k":
            csv_path, img_dir, name_col = "AIGC_all.csv", "file", "Image"
            with open(os.path.join(root, csv_path), encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    p = os.path.join(root, img_dir, r[name_col])
                    if os.path.exists(p):
                        self.items.append((p, float(r["Score"]),
                                           r[name_col].split("_")[0]))
        else:
            csv_path, img_dir = "data.csv", "allimg"
            with open(os.path.join(root, csv_path), encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    p = os.path.join(root, img_dir, r["name"])
                    if os.path.exists(p):
                        self.items.append((p, float(r["mos"]), r["generator"]))
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
    ap.add_argument("--dataset", default="1k", choices=["1k", "2023"],
                    help="which database to evaluate on")
    ap.add_argument("--root", default="")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="write per-image predictions to this CSV")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QualityNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    root = args.root or (cfg.DATA_1K if args.dataset == "1k" else cfg.DATA_2023)
    ds = TargetDB(root, args.dataset, args.size)
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
    if args.dump:
        with open(args.dump, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "generator", "mos", "pred"])
            for (path, _, _), p, t, gg in zip(ds.items, preds, gts, gens):
                w.writerow([os.path.basename(path), gg, float(t), float(p)])
        print("wrote", args.dump)

    label = "AGIQA-1K" if args.dataset == "1k" else "AIGCIQA2023"
    res = {"dataset": label, "n": len(ds), **m, "per_generator": per,
           "mean_over_generators": float(np.mean([v["srcc"] for v in per.values()]))
           if per else None}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
