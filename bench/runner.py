"""Score one model on one database and record the three correlation axes.

Usage::

    python -m bench.runner --dataset agiqa3k --models niqe,musiq,clipiqa
    python -m bench.runner --dataset agiqa3k --models all --max_images 200

The per-image scores are cached under ``results/bench/scores``; the summary is
written to ``results/bench/summary``. Re-running a model whose cache exists is a
no-op unless ``--force`` is given, so an interrupted matrix can be resumed.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy import stats

import agiqa.config as cfg
import agiqa.metrics as M
from bench import datasets, registry

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
SCORES = os.path.join(BENCH, "scores")
SUMMARY = os.path.join(BENCH, "summary")


def _load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    if max(img.size) > size:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def decompose(scores, items):
    """Pooled / within-generator / between-generator correlations.

    The within-generator value is the equal-weight mean over generators, which
    keeps it comparable with the leave-one-generator-out mean. Generators with
    fewer than 30 images or a constant prediction are skipped and listed.
    """
    scores = np.asarray(scores, dtype=float)
    mos = np.array([it["mos"] for it in items], dtype=float)
    gens = np.array([it["generator"] for it in items])

    pooled = M.all_metrics(scores, mos)
    per = {}
    skipped = []
    for g in sorted(set(gens)):
        sel = gens == g
        if sel.sum() < 30 or np.std(scores[sel]) < 1e-9:
            skipped.append(g)
            continue
        per[g] = {
            "n": int(sel.sum()),
            "srcc": M.srcc(scores[sel], mos[sel]),
            "plcc": M.plcc(scores[sel], mos[sel]),
            "krcc": M.krcc(scores[sel], mos[sel]),
            "mos_mean": float(mos[sel].mean()),
            "score_mean": float(scores[sel].mean()),
        }

    g_mos = [np.mean(mos[gens == g]) for g in sorted(set(gens))]
    g_score = [np.mean(scores[gens == g]) for g in sorted(set(gens))]
    between = M.srcc(g_score, g_mos) if len(g_mos) >= 3 else float("nan")

    weights = np.array([per[g]["n"] for g in per], dtype=float) if per else np.array([])
    within_vals = np.array([per[g]["srcc"] for g in per], dtype=float) if per else np.array([])
    within_weighted = float(np.average(within_vals, weights=weights)) if len(per) > 1 else float("nan")

    return {
        "n": int(len(items)),
        "n_generators": int(len(set(gens))),
        "pooled": pooled,
        "within_mean": float(np.mean(within_vals)) if len(per) else float("nan"),
        "within_weighted": within_weighted,
        "between": float(between) if between == between else None,
        "per_generator": per,
        "skipped_generators": skipped,
    }


def score_pyiqa(name, items, device, max_images=0, size=512):
    import pyiqa

    metric = pyiqa.create_metric(name, device=device)
    lower = registry.lower_better(name, metric)

    idx = np.arange(len(items))
    if max_images:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(items), min(max_images, len(items)), replace=False))

    scores = np.full(len(items), np.nan, dtype=float)
    failures = []
    t0 = time.time()
    for count, i in enumerate(idx, 1):
        try:
            x = _load_image(items[i]["path"], size).to(device)
            with torch.no_grad():
                value = float(metric(x))
            if not np.isfinite(value):
                raise ValueError("metric returned a non-finite score")
            scores[i] = -value if lower else value
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop a model
            failures.append({"key": items[i]["key"], "error": f"{type(exc).__name__}: {exc}"[:120]})
        if count % 500 == 0:
            rate = count / (time.time() - t0)
            print(f"  {name}: {count}/{len(idx)}  {rate:.1f} img/s", flush=True)
    return scores, idx, {"lower_better": lower, "n_scored": int(len(idx)),
                         "n_failed": len(failures), "failures": failures[:20],
                         "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(datasets.LOADERS))
    ap.add_argument("--models", default="all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(SCORES, args.dataset), exist_ok=True)
    os.makedirs(SUMMARY, exist_ok=True)

    items = datasets.load(args.dataset)
    print(f"{args.dataset}: {len(items)} images, {len(datasets.generators(items))} generators", flush=True)

    names = registry.default_models() if args.models == "all" else [m.strip() for m in args.models.split(",")]
    for name in names:
        out_json = os.path.join(SUMMARY, f"{args.dataset}__{name}.json")
        out_npy = os.path.join(SCORES, args.dataset, f"{name}.npy")
        if os.path.exists(out_json) and not args.force:
            print(f"skip {name}: cached", flush=True)
            continue
        try:
            scores, idx, meta = score_pyiqa(name, items, args.device, args.max_images)
        except Exception as exc:  # noqa: BLE001 - one broken metric must not stop the matrix
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        np.save(out_npy, scores)
        keep = idx[~np.isnan(scores[idx])]
        record = {
            "dataset": args.dataset,
            "model": name,
            "family": registry.PYIQA_MODELS.get(name, ("unknown", None, ""))[0],
            "meta": meta,
            **decompose(scores[keep], [items[i] for i in keep]),
        }
        json.dump(record, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else " n/a "
        print(f"{name:12s} pooled {fmt(record['pooled']['srcc'])}  "
              f"within {fmt(record['within_mean'])}  between {fmt(record['between'])}", flush=True)


if __name__ == "__main__":
    main()
