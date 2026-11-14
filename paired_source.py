"""Seen-source versus unseen-source, on the same test images.

The protocol gap in the paper compares a pooled random-split SRCC with a
leave-one-generator-out mean. Those two numbers are computed over different
ranking ranges, so their difference mixes two effects: losing the between-source
ordering, and losing whatever the model knew about the held-out source.

This script isolates the second effect. For each generator g it holds out the
same 20% of g's images in both conditions and fits the same head twice:

  unseen  training uses the other five generators only
  seen    training uses the other five generators plus the remaining 80% of g

The test images are identical, so the paired difference is the effect of having
seen the source. ``--match_size`` subsamples the seen training set down to the
size of the unseen one, so training-set size is held constant as well.

    python paired_source.py --model mlp --seeds 0,1,2
    python paired_source.py --model ridge
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

import agiqa.config as cfg
from agiqa.data import GENERATORS
from head_eval import load_3k, predict, train_head


def source_normalise(y, g, idx):
    """Z-score the targets inside each source, using statistics of ``idx`` only.

    Within a single test source the transform is an affine map, so within-source
    SRCC is unchanged by it; what changes is the weight each source carries in
    the fit. ``idx`` contains training images only, so no test label leaks in.
    """
    out = y.astype(np.float32).copy()
    for s in np.unique(g[idx]):
        m = idx[g[idx] == s]
        sd = float(y[m].std()) + 1e-3
        out[m] = (y[m] - float(y[m].mean())) / sd
    return out


def fit_mlp(X, y, seed):
    net, mu, sd = train_head(X, y, seed=seed)
    return net, mu, sd


def fit_ridge(X, y, alpha=1000.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=alpha).fit(sc.transform(X), y)
    return m, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="3k", choices=["3k", "2023"])
    ap.add_argument("--model", default="mlp", choices=["mlp", "ridge"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0,
                    help="seed for choosing the 20%% test subset of each generator")
    ap.add_argument("--match_size", action="store_true",
                    help="subsample the seen training set to the unseen size")
    ap.add_argument("--normalize", default="none", choices=["none", "source"],
                    help="source: fit per-source z-scored targets, so that no "
                         "source dominates through its label range")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.dataset == "3k":
        X, rows = load_3k(cfg.CLIP_3K)
        y = np.array([r["mos"] for r in rows], dtype=np.float32)
        g = np.array([r["generator"] for r in rows])
        gens = list(GENERATORS)
    else:
        X = np.load(cfg.CLIP_2023)
        meta = list(csv.DictReader(open(os.path.join(cfg.DATA_2023, "data.csv"),
                                        encoding="utf-8")))
        y = np.array([float(r["mos"]) for r in meta], dtype=np.float32)
        g = np.array([r["generator"] for r in meta])
        gens = sorted(set(g))

    out = {}
    print(f"{'generator':11s} {'n_test':>7s} {'unseen':>7s} {'seen':>7s} {'diff':>7s}")
    for gen in gens:
        idx_g = np.where(g == gen)[0]
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(idx_g))
        cut = int(len(idx_g) * (1 - args.test_ratio))
        rest_g, test_idx = idx_g[perm[:cut]], idx_g[perm[cut:]]
        other = np.where(g != gen)[0]

        seen_scores, unseen_scores = [], []
        for seed in seeds:
            tr_unseen = other
            tr_seen = np.concatenate([other, rest_g])
            if args.match_size:
                rs = np.random.RandomState(seed)
                tr_seen = rs.choice(tr_seen, size=len(tr_unseen), replace=False)

            y_un = (source_normalise(y, g, tr_unseen)[tr_unseen]
                    if args.normalize == "source" else y[tr_unseen])
            y_se = (source_normalise(y, g, tr_seen)[tr_seen]
                    if args.normalize == "source" else y[tr_seen])

            if args.model == "mlp":
                net, mu, sd = fit_mlp(X[tr_unseen], y_un, seed)
                p_un = predict(net, mu, sd, X[test_idx])
                net, mu, sd = fit_mlp(X[tr_seen], y_se, seed)
                p_se = predict(net, mu, sd, X[test_idx])
            else:
                m, sc = fit_ridge(X[tr_unseen], y_un)
                p_un = m.predict(sc.transform(X[test_idx]))
                m, sc = fit_ridge(X[tr_seen], y_se)
                p_se = m.predict(sc.transform(X[test_idx]))

            unseen_scores.append(stats.spearmanr(p_un, y[test_idx]).statistic)
            seen_scores.append(stats.spearmanr(p_se, y[test_idx]).statistic)

        un, se = float(np.mean(unseen_scores)), float(np.mean(seen_scores))
        out[gen] = {"n_test": int(len(test_idx)), "unseen": un, "seen": se,
                    "diff": se - un,
                    "unseen_per_seed": [float(v) for v in unseen_scores],
                    "seen_per_seed": [float(v) for v in seen_scores]}
        print(f"{gen:11s} {len(test_idx):7d} {un:7.3f} {se:7.3f} {se - un:+7.3f}")

    diffs = [v["diff"] for v in out.values()]
    out["_mean"] = {"unseen": float(np.mean([v["unseen"] for v in out.values()])),
                    "seen": float(np.mean([v["seen"] for v in out.values()])),
                    "diff": float(np.mean(diffs))}
    print(f"\nmean over generators: unseen {out['_mean']['unseen']:.3f}  "
          f"seen {out['_mean']['seen']:.3f}  diff {out['_mean']['diff']:+.3f}")

    name = args.out or (f"paired_{args.dataset}_{args.model}_split{args.split_seed}"
                        f"{'_matched' if args.match_size else ''}"
                        f"{'_srcnorm' if args.normalize == 'source' else ''}.json")
    json.dump(out, open(os.path.join(cfg.RUNS, name), "w", encoding="utf-8"), indent=2)
    print("wrote", name)


if __name__ == "__main__":
    main()
