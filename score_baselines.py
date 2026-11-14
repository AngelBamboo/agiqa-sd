"""Score AGIQA-3K with published no-reference IQA metrics (via pyiqa).

These metrics need no training, so they can be evaluated on the same splits as
the trained models. If the generator confound is a property of the benchmark
rather than of one architecture, it should show up here too.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy import stats
import agiqa.config as cfg


def collect():
    rows = list(csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8")))
    return rows


def load(path, size=512):
    img = Image.open(path).convert("RGB")
    if max(img.size) > size:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def decompose(scores, gens, mos):
    scores = np.asarray(scores)
    overall = stats.spearmanr(scores, mos).statistic if np.std(scores) > 1e-9 else np.nan
    per = {}
    for g in sorted(set(gens)):
        sel = np.array(gens) == g
        if sel.sum() > 30 and np.std(scores[sel]) > 1e-9:
            per[g] = float(stats.spearmanr(scores[sel], np.array(mos)[sel]).statistic)
    g_m = [np.mean(np.array(mos)[np.array(gens) == g]) for g in sorted(set(gens))]
    g_p = [np.mean(scores[np.array(gens) == g]) for g in sorted(set(gens))]
    between = stats.spearmanr(g_p, g_m).statistic if np.std(g_p) > 1e-9 else np.nan
    return {"overall": float(overall) if overall == overall else None,
            "within_mean": float(np.mean(list(per.values()))) if per else None,
            "between": float(between) if between == between else None,
            "per_generator": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="niqe,brisque,musiq,clipiqa,topiq_nr")
    ap.add_argument("--out", default=os.path.join(cfg.RUNS, "baselines_pyiqa.json"))
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    import pyiqa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = collect()
    if args.max_images:
        rng = np.random.RandomState(0)
        rows = [rows[i] for i in rng.choice(len(rows), args.max_images, replace=False)]

    results = {}
    for name in args.metrics.split(","):
        name = name.strip()
        try:
            metric = pyiqa.create_metric(name, device=device)
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        scores = []
        for r in rows:
            img = load(os.path.join(os.path.join(cfg.DATA_3K, "images"), r["name"])).to(device)
            with torch.no_grad():
                scores.append(float(metric(img)))
        gens = [r["name"].split("_")[0] for r in rows]
        mos = [float(r["mos_quality"]) for r in rows]
        # pyiqa metrics are "lower is better" for some; Spearman sign encodes direction
        results[name] = decompose(scores, gens, mos)
        r = results[name]
        fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else "  n/a"
        print(f"{name:10s} overall {fmt(r['overall'])}  within {fmt(r['within_mean'])}  between {fmt(r['between'])}")

    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
