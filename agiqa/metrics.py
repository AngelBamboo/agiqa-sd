"""Correlation metrics used by the AGIQA literature (SRCC / PLCC / KRCC)."""
import numpy as np
from scipy import stats


def _safe(fn, pred, gt):
    if len(pred) < 3 or np.std(pred) < 1e-9:
        return float("nan")
    r = fn(pred, gt)
    return float(r.statistic if hasattr(r, "statistic") else r[0])


def srcc(pred, gt):
    return _safe(stats.spearmanr, pred, gt)


def plcc(pred, gt):
    return _safe(stats.pearsonr, pred, gt)


def krcc(pred, gt):
    return _safe(stats.kendalltau, pred, gt)


def all_metrics(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    return {"srcc": srcc(pred, gt), "plcc": plcc(pred, gt), "krcc": krcc(pred, gt)}
