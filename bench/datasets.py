"""Unified loaders for the three public AGIQA databases.

Each loader returns a list of items in a fixed order (the order of the released
CSV after dropping rows whose image file is missing). The order is the contract
between the scorer, the cached score arrays and the analysis: every per-image
score file is a 1-D array aligned with ``load(name)``.
"""
import csv
import os

_DEFAULT_ROOTS = {
    "agiqa3k": ["D:/AGIQA/data", "data/agiqa3k"],
    "aigciqa2023": ["D:/AGIQA/data2023", "data/aigciqa2023"],
    "agiqa1k": ["D:/AGIQA/data1k", "data/agiqa1k"],
}
_ENV = {
    "agiqa3k": "AGIQA_3K",
    "aigciqa2023": "AGIQA_2023",
    "agiqa1k": "AGIQA_1K",
}


def root(name):
    """Resolve a database root from the environment, then the known locations."""
    env = os.environ.get(_ENV[name])
    candidates = [env] if env else []
    candidates += _DEFAULT_ROOTS[name]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"{name}: set {_ENV[name]} or place the release under {_DEFAULT_ROOTS[name][-1]}"
    )


def _rows(csv_path):
    with open(csv_path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_agiqa3k():
    base = root("agiqa3k")
    img_dir = os.path.join(base, "images")
    items = []
    for row in _rows(os.path.join(base, "data.csv")):
        path = os.path.join(img_dir, row["name"])
        if not os.path.exists(path):
            continue
        items.append({
            "key": row["name"],
            "path": path,
            "generator": row["name"].split("_")[0],
            "prompt": row["prompt"].strip(),
            "mos": float(row["mos_quality"]),
            "std": float(row["std_quality"]) if row.get("std_quality") else None,
        })
    return items


def load_aigciqa2023():
    base = root("aigciqa2023")
    img_dir = os.path.join(base, "allimg")
    items = []
    for row in _rows(os.path.join(base, "data.csv")):
        path = os.path.join(img_dir, row["name"])
        if not os.path.exists(path):
            continue
        items.append({
            "key": row["name"],
            "path": path,
            "generator": row["generator"],
            "prompt": row["prompt"].strip(),
            "mos": float(row["mos"]),
            "std": float(row["std"]) if row.get("std") else None,
        })
    return items


def load_agiqa1k():
    """AGIQA-1K ships one CSV with image names and scores; no prompt column."""
    base = root("agiqa1k")
    img_dir = os.path.join(base, "file")
    items = []
    for row in _rows(os.path.join(base, "AIGC_all.csv")):
        path = os.path.join(img_dir, row["Image"])
        if not os.path.exists(path):
            continue
        items.append({
            "key": row["Image"],
            "path": path,
            "generator": row["Image"].split("_")[0],
            "prompt": "",
            "mos": float(row["Score"]),
            "std": None,
        })
    return items


LOADERS = {
    "agiqa3k": load_agiqa3k,
    "aigciqa2023": load_aigciqa2023,
    "agiqa1k": load_agiqa1k,
}


def load(name):
    return LOADERS[name]()


def generators(items):
    seen = []
    for item in items:
        if item["generator"] not in seen:
            seen.append(item["generator"])
    return seen
