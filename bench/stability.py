"""Third axis: is the model ranking stable across generators of one database?

The leaderboard question so far compared pooled and within-source rankings. This
module asks a different one: if you evaluate the same models separately on each
generator, do they rank the same way? A low rank correlation between generators
means there is no single best quality metric -- the winner depends on which
generator your test set happens to contain.

Usage::

    python -m bench.stability
    python -m bench.stability --out logs/stability.md
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
SUMMARY = os.path.join(BENCH, "summary")


def load(dataset=None):
    by = {}
    for path in sorted(glob.glob(os.path.join(SUMMARY, "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        if dataset and rec["dataset"] != dataset:
            continue
        per = rec.get("per_generator") or {}
        if per:
            by.setdefault(rec["dataset"], {})[rec["model"]] = {
                g: v.get("srcc") if isinstance(v, dict) else v for g, v in per.items()
            }
    return by


def analyse(per_model):
    """Pairwise Kendall tau between generator-specific model rankings."""
    models = sorted(per_model)
    gens = sorted({g for m in per_model.values() for g in m})
    matrix = np.full((len(gens), len(gens)), np.nan)
    pairs = []
    for i, a in enumerate(gens):
        for j, b in enumerate(gens):
            if j <= i:
                continue
            common = [m for m in models
                      if isinstance(per_model[m].get(a), float)
                      and isinstance(per_model[m].get(b), float)]
            if len(common) < 3:
                continue
            x = [per_model[m][a] for m in common]
            y = [per_model[m][b] for m in common]
            tau = stats.kendalltau(x, y).statistic
            matrix[i, j] = matrix[j, i] = tau
            pairs.append((a, b, len(common), float(tau)))
    np.fill_diagonal(matrix, 1.0)

    # who wins where
    winners = {}
    for g in gens:
        ranked = sorted(
            ((m, per_model[m][g]) for m in models if isinstance(per_model[m].get(g), float)),
            key=lambda kv: -kv[1],
        )
        winners[g] = ranked[:1][0][0] if ranked else None

    off_diag = [p[3] for p in pairs]
    return {
        "generators": gens,
        "matrix": matrix.tolist() if gens else [],
        "pairs": pairs,
        "mean_tau": float(np.mean(off_diag)) if off_diag else float("nan"),
        "min_tau": float(np.min(off_diag)) if off_diag else float("nan"),
        "max_tau": float(np.max(off_diag)) if off_diag else float("nan"),
        "winners": winners,
        "distinct_winners": len({w for w in winners.values() if w}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BENCH, "STABILITY.md"))
    args = ap.parse_args()

    data = load()
    chunks = ["# Per-generator leaderboard stability", ""]
    all_out = {}
    for ds in sorted(data):
        result = analyse(data[ds])
        all_out[ds] = result
        chunks.append(f"## {ds}")
        chunks.append("")
        chunks.append(f"Pairwise Kendall tau between generator rankings: mean "
                      f"{result['mean_tau']:+.3f}, range {result['min_tau']:+.3f} to "
                      f"{result['max_tau']:+.3f}.")
        chunks.append(f"Distinct winners across {len(result['generators'])} generators: "
                      f"{result['distinct_winners']}.")
        chunks.append("")
        chunks.append("| generator | best model |")
        chunks.append("|---|---|")
        for g, w in result["winners"].items():
            chunks.append(f"| {g} | {w} |")
        chunks.append("")
        chunks.append("| | " + " | ".join(result["generators"]) + " |")
        chunks.append("|---|" + "---:|" * len(result["generators"]))
        for i, g in enumerate(result["generators"]):
            vals = ["—" if np.isnan(v) else f"{v:+.2f}" for v in result["matrix"][i]]
            chunks.append(f"| **{g}** | " + " | ".join(vals) + " |")
        chunks.append("")

    text = "\n".join(chunks)
    open(args.out, "w", encoding="utf-8").write(text)
    json.dump(all_out, open(os.path.join(BENCH, "stability.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
