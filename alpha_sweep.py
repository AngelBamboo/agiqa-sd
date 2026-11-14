"""Sensitivity of the ridge readout to its regularisation strength.

For every candidate penalty this fits the same readout under three settings and
reports the three metrics used in the paper:

  random   mean over three random 80/20 splits (overall SRCC)
  within   equal-weight mean over the six generators, on those same splits
  logo     mean over the six leave-one-generator-out folds

The sweep is a sensitivity analysis. The value the paper claims is the one
chosen by nested validation in nested_ridge.py, which does not see the test
source; the sweep only shows how flat the optimum is.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import agiqa.config as cfg
from nested_ridge import ALPHAS, fit_predict, load

# the sweep extends one step past the selection grid, to show the degradation
SWEEP_ALPHAS = ALPHAS + [100000]

GEN = ["sd1.5", "midjourney", "AttnGAN", "glide", "xl2.2", "DALLE2"]


def within_mean(pred, gt, gens):
    vals = [stats.spearmanr(pred[np.array(gens) == g], gt[np.array(gens) == g]).statistic
            for g in sorted(set(gens)) if (np.array(gens) == g).sum() > 30]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="3k", choices=["3k", "2023"])
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    X, y, g, prompts = load(args.dataset)
    holdouts = sorted(set(g))
    out = {}

    for a in SWEEP_ALPHAS:
        rnd, wth = [], []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(y))
            cut = int(len(y) * 0.8)
            tr, te = np.sort(idx[:cut]), np.sort(idx[cut:])
            p = fit_predict(X[tr], y[tr], X[te], a)
            rnd.append(stats.spearmanr(p, y[te]).statistic)
            wth.append(within_mean(p, y[te], [g[i] for i in te]))
        logo = []
        for h in holdouts:
            tr, te = np.where(g != h)[0], np.where(g == h)[0]
            p = fit_predict(X[tr], y[tr], X[te], a)
            logo.append(stats.spearmanr(p, y[te]).statistic)
        out[str(a)] = {"random": float(np.mean(rnd)), "random_std": float(np.std(rnd, ddof=1)),
                       "within": float(np.mean(wth)), "logo": float(np.mean(logo))}
        print(f"alpha {a:7d}  random {out[str(a)]['random']:.3f}  "
              f"within {out[str(a)]['within']:.3f}  logo {out[str(a)]['logo']:.3f}", flush=True)

    name = "alpha_sweep.json" if args.dataset == "3k" else "alpha_sweep_2023.json"
    json.dump(out, open(os.path.join(cfg.RUNS, name), "w", encoding="utf-8"), indent=2)
    print("wrote", name)


if __name__ == "__main__":
    main()
