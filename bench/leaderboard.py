"""Aggregate the per-model records into leaderboards and measure re-ranking.

Two questions drive this module:

1. Does the pooled leaderboard order models the same way as the within-source
   leaderboard? The Kendall tau between the two rankings answers it.
2. Across models, is pooled SRCC explained by between-generator ordering or by
   within-generator ranking? Regressing one on the other answers that.

Usage::

    python -m bench.leaderboard                 # print and write Markdown
    python -m bench.leaderboard --dataset agiqa3k --csv out.csv
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench")
SUMMARY = os.path.join(BENCH, "summary")


def load_records(dataset=None):
    records = []
    for path in sorted(glob.glob(os.path.join(SUMMARY, "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        if dataset and rec["dataset"] != dataset:
            continue
        records.append(rec)
    return records


def to_rows(records):
    rows = []
    for rec in records:
        rows.append({
            "dataset": rec["dataset"],
            "model": rec["model"],
            "family": rec.get("family", ""),
            "pooled": rec["pooled"]["srcc"],
            "plcc": rec["pooled"]["plcc"],
            "krcc": rec["pooled"]["krcc"],
            "within": rec["within_mean"],
            "within_weighted": rec.get("within_weighted"),
            "between": rec["between"],
            "source_disjoint": rec.get("source_disjoint"),
        })
    return rows


def rank_analysis(rows):
    """Ranks, rank shifts and the rank correlation for one dataset."""
    ranked = sorted(rows, key=lambda r: (r["pooled"] is None, -(r["pooled"] or 0)))
    for i, row in enumerate(ranked, 1):
        row["rank_pooled"] = i
    ranked_within = sorted(rows, key=lambda r: (r["within"] is None, -(r["within"] or 0)))
    for i, row in enumerate(ranked_within, 1):
        row["rank_within"] = i
    for row in rows:
        row["shift"] = row["rank_pooled"] - row["rank_within"]

    valid = [r for r in rows if isinstance(r["pooled"], (int, float))
             and isinstance(r["within"], (int, float))
             and r["pooled"] == r["pooled"] and r["within"] == r["within"]]
    tau = stats.kendalltau([r["pooled"] for r in valid], [r["within"] for r in valid])
    flips = sum(1 for i in range(len(valid)) for j in range(i + 1, len(valid))
                if (valid[i]["pooled"] - valid[j]["pooled"]) *
                   (valid[i]["within"] - valid[j]["within"]) < 0)
    return {
        "kendall_tau": float(tau.statistic),
        "kendall_p": float(tau.pvalue),
        "pairs": int(len(valid) * (len(valid) - 1) / 2),
        "flipped_pairs": int(flips),
    }


def cross_model_correlations(rows):
    """How much of each model's pooled SRCC is between-source ordering?"""
    valid = [r for r in rows if isinstance(r["between"], (int, float)) and r["between"] == r["between"]]
    if len(valid) < 3:
        return {"n_models": len(valid)}
    pooled = np.array([r["pooled"] for r in valid])
    within = np.array([r["within"] for r in valid])
    between = np.array([r["between"] for r in valid])
    out = {
        "n_models": len(valid),
        "pooled_vs_within": float(stats.pearsonr(pooled, within).statistic),
        "pooled_vs_between": float(stats.pearsonr(pooled, between).statistic),
        "within_vs_between": float(stats.pearsonr(within, between).statistic),
    }
    # a two-predictor fit says which axis carries the pooled number
    X = np.column_stack([within, between, np.ones(len(valid))])
    beta, *_ = np.linalg.lstsq(X, pooled, rcond=None)
    pred = X @ beta
    ss_res = float(((pooled - pred) ** 2).sum())
    ss_tot = float(((pooled - pooled.mean()) ** 2).sum())
    out["ols_beta_within"] = float(beta[0])
    out["ols_beta_between"] = float(beta[1])
    out["ols_r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


def _f3(value):
    return f"{value:+.3f}" if isinstance(value, (int, float)) and value == value else "—"


def markdown(rows, stats_, dataset):
    lines = [
        f"### {dataset}",
        "",
        "| rank (pooled) | model | family | pooled SRCC | within SRCC | between SRCC | LOGO SRCC | rank (within) | shift |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["rank_pooled"]):
        logo = f"{row['source_disjoint']:+.3f}" if row["source_disjoint"] is not None else "—"
        lines.append(
            f"| {row['rank_pooled']} | {row['model']} | {row['family']} | "
            f"{_f3(row['pooled'])} | {_f3(row['within'])} | "
            f"{_f3(row['between'])} | {logo} | {row['rank_within']} | {row['shift']:+d} |"
        )
    lines += [
        "",
        f"Kendall tau between the two leaderboards: {stats_['kendall_tau']:+.3f} "
        f"(p = {stats_['kendall_p']:.3g}); {stats_['flipped_pairs']} of "
        f"{stats_['pairs']} model pairs change order.",
        "",
    ]
    return "\n".join(lines)


def cross_dataset(rows):
    """Does a leaderboard learned on one database transfer to another?"""
    by = {}
    for row in rows:
        by.setdefault(row["dataset"], {})[row["model"]] = row
    names = sorted(by)
    lines = [
        "### Cross-dataset stability",
        "",
        "| pair | common models | tau (pooled) | tau (within) |",
        "|---|---:|---:|---:|",
    ]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = by[names[i]], by[names[j]]
            common = sorted(set(left) & set(right))
            common = [m for m in common
                      if isinstance(left[m]["pooled"], float) and isinstance(right[m]["pooled"], float)
                      and isinstance(left[m]["within"], float) and isinstance(right[m]["within"], float)]
            if len(common) < 3:
                continue
            tp = stats.kendalltau([left[m]["pooled"] for m in common],
                                  [right[m]["pooled"] for m in common]).statistic
            tw = stats.kendalltau([left[m]["within"] for m in common],
                                  [right[m]["within"] for m in common]).statistic
            lines.append(f"| {names[i]} vs {names[j]} | {len(common)} | {tp:+.3f} | {tw:+.3f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=os.path.join(BENCH, "LEADERBOARD.md"))
    args = ap.parse_args()

    records = load_records(args.dataset)
    # A trained model whose random-split predictions are missing has only the
    # LOGO record; it can still be reported in the source-disjoint table, but it
    # cannot enter a pooled leaderboard. Keep it out of the ranking and say so.
    missing_pooled = [f"{r['dataset']}/{r['model']}" for r in records if "pooled" not in r]
    records = [r for r in records if "pooled" in r]
    if missing_pooled:
        print("no pooled record (skipped in ranking): " + ", ".join(missing_pooled))
    if not records:
        raise SystemExit("no summary records; run bench.runner first")
    rows = to_rows(records)

    datasets = sorted({r["dataset"] for r in rows})
    chunks = []
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        st = rank_analysis(sub)
        corr = cross_model_correlations(sub)
        chunks.append(markdown(sub, st, ds))
        if corr.get("n_models", 0) >= 3:
            chunks.append(
                "Cross-model: pooled~within r = {:+.3f}, pooled~between r = {:+.3f}; "
                "OLS pooled = {:+.3f}*within {:+.3f}*between + c (R^2 = {:.3f}).\n".format(
                    corr["pooled_vs_within"], corr["pooled_vs_between"],
                    corr["ols_beta_within"], corr["ols_beta_between"], corr["ols_r2"])
            )

    chunks.append(cross_dataset(rows))
    text = "\n".join(chunks)
    open(args.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
