"""Export the benchmark records as CSV and Markdown tables for the supplement.

Outputs, under ``results/bench/tables``:

* ``<dataset>.csv``      one row per model with the three axes and ranks;
* ``per_generator.csv``  model by generator SRCC for every database;
* ``bootstrap.csv``      point estimate and 95% interval per model and axis.

The Markdown summary is written to ``paper/SUPPLEMENTARY.md``.
"""
import csv
import glob
import json
import os

import numpy as np

from bench import registry

DISPLAY = {
    "niqe": "NIQE", "brisque": "BRISQUE", "piqe": "PIQE",
    "cnniqa": "CNNIQA", "dbcnn": "DBCNN", "nima": "NIMA", "hyperiqa": "HyperIQA",
    "topiq_nr": "TOPIQ-NR", "arniqa": "ARNIQA",
    "tres": "TReS", "musiq": "MUSIQ", "maniqa": "MANIQA",
    "clipiqa": "CLIP-IQA", "clipiqa+": "CLIP-IQA+", "qualiclip": "QualiCLIP", "liqe": "LIQE",
    "resnet18": "ResNet-18", "clip_mlp": "CLIP+MLP", "coop": "CoOp", "ridge": "ridge",
}

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
SUMMARY = os.path.join(BENCH, "summary")
TABLES = os.path.join(BENCH, "tables")
PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "paper")
PAPER = os.path.abspath(PAPER)


def records(dataset):
    out = []
    for path in sorted(glob.glob(os.path.join(SUMMARY, f"{dataset}__*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        if "pooled" not in rec:
            continue
        out.append(rec)
    return out


def rank(recs):
    recs = sorted(recs, key=lambda r: -r["pooled"]["srcc"])
    for i, r in enumerate(recs, 1):
        r["rank_pooled"] = i
    recs = sorted(recs, key=lambda r: -r["within_mean"])
    for i, r in enumerate(recs, 1):
        r["rank_within"] = i
    return sorted(recs, key=lambda r: r["rank_pooled"])


def model_pool_lines():
    out = ["## Table S1. Model pool", "",
           "| Model | Family | Evaluation | Note |",
           "|---|---|---|---|"]
    for key, (family, training_free, note) in registry.PYIQA_MODELS.items():
        out.append(f"| {DISPLAY.get(key, key)} | {family} | training-free, all three databases | {note} |")
    for key, (family, note) in registry.TRAINED_MODELS.items():
        out.append(f"| {DISPLAY.get(key, key)} | {family} | trained, AGIQA-3K and AIGCIQA2023 | {note} |")
    out.append("")
    return out


def bootstrap_lines(models=("nima", "tres", "clipiqa", "niqe")):
    out = ["## Table S2. Stratified bootstrap intervals on AGIQA-3K (200 resamples)", "",
           "| Model | Pooled SRCC | 95% CI | Within SRCC | 95% CI |",
           "|---|---:|---:|---:|---:|"]
    for name in models:
        d = json.load(open(os.path.join(BENCH, "bootstrap", f"agiqa3k__{name}.json"), encoding="utf-8"))
        p, w = d["pooled"], d["within"]
        out.append(f"| {DISPLAY.get(name, name)} | {p['point']:.3f} | [{p['lo']:.3f}, {p['hi']:.3f}] | "
                   f"{w['point']:.3f} | [{w['lo']:.3f}, {w['hi']:.3f}] |")
    out.append("")
    return out


def main():
    os.makedirs(TABLES, exist_ok=True)
    lines = ["# Supplementary material", "",
             "Generated from `results/bench`; every number is recomputable with `python -m bench.leaderboard`.", ""]
    lines += model_pool_lines()
    lines += bootstrap_lines()

    for ds in ("agiqa3k", "aigciqa2023", "agiqa1k"):
        recs = rank(records(ds))
        rows = []
        for r in recs:
            rows.append({
                "model": r["model"],
                "family": r.get("family", ""),
                "pooled_srcc": r["pooled"]["srcc"],
                "pooled_plcc": r["pooled"]["plcc"],
                "within_srcc": r["within_mean"],
                "between_srcc": r.get("between"),
                "logo_srcc": r.get("source_disjoint"),
                "rank_pooled": r["rank_pooled"],
                "rank_within": r["rank_within"],
            })
        with open(os.path.join(TABLES, f"{ds}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows])

        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| Model | Family | Pooled | Within | Between | LOGO | Rank (pooled) | Rank (within) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            fmt = lambda v: "" if v is None else f"{v:+.3f}"
            lines.append(f"| {row['model']} | {row['family']} | {fmt(row['pooled_srcc'])} | "
                         f"{fmt(row['within_srcc'])} | {fmt(row['between_srcc'])} | "
                         f"{fmt(row['logo_srcc'])} | {row['rank_pooled']} | {row['rank_within']} |")
        lines.append("")

    # per-generator SRCC
    pg_rows = []
    for ds in ("agiqa3k", "aigciqa2023", "agiqa1k"):
        for rec in records(ds):
            for gen, val in (rec.get("per_generator") or {}).items():
                pg_rows.append({"dataset": ds, "model": rec["model"], "generator": gen,
                                "srcc": round(val["srcc"], 4) if isinstance(val, dict) else round(val, 4),
                                "n": val.get("n") if isinstance(val, dict) else None})
    with open(os.path.join(TABLES, "per_generator.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "generator", "srcc", "n"])
        writer.writeheader()
        writer.writerows(pg_rows)

    # bootstrap intervals
    boot_rows = []
    for path in sorted(glob.glob(os.path.join(BENCH, "bootstrap", "*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        for axis in ("pooled", "within", "between"):
            if axis in d:
                boot_rows.append({"dataset": d["dataset"], "model": d["model"], "axis": axis,
                                  "point": round(d[axis]["point"], 4),
                                  "lo": round(d[axis]["lo"], 4),
                                  "hi": round(d[axis]["hi"], 4)})
    with open(os.path.join(TABLES, "bootstrap.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "axis", "point", "lo", "hi"])
        writer.writeheader()
        writer.writerows(boot_rows)

    lines += ["## Bootstrap intervals", "",
              "Point estimates and 95% intervals from 200 stratified resamples. Full table: `results/bench/tables/bootstrap.csv`.", ""]
    lines += ["## Per-generator SRCC", "",
              "Full table: `results/bench/tables/per_generator.csv`.", ""]

    os.makedirs(PAPER, exist_ok=True)
    open(os.path.join(PAPER, "SUPPLEMENTARY.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", TABLES, "and paper/SUPPLEMENTARY.md")


if __name__ == "__main__":
    main()
