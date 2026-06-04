#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

_GRID = 101
_SHADE_ALPHA = 0.22


def _dedup(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, y
    order = np.argsort(x, kind="mergesort")
    x_s, y_s = x[order], y[order]
    ux, uy = [], []
    for xi, yi in zip(x_s, y_s):
        if ux and np.isclose(xi, ux[-1]):
            uy[-1] = max(uy[-1], yi)
        else:
            ux.append(float(xi))
            uy.append(float(yi))
    return np.asarray(ux), np.asarray(uy)


def _roc_on_grid(y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr, tpr = _dedup(fpr, tpr)
    tpr_i = np.interp(grid, fpr, tpr, left=0.0, right=tpr[-1] if fpr.size else 0.0)
    return tpr_i, float(auc(fpr, tpr))


def _pr_on_grid(y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(average_precision_score(y_true, y_score))
    order = np.argsort(recall, kind="mergesort")
    r, p = _dedup(recall[order], precision[order])
    prec_i = np.interp(grid, r, p, left=p[0] if r.size else 1.0, right=0.0)
    return prec_i, pr_auc


def _load(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(run_dir / "predict" / "results" / "test_predictions.tsv", sep="\t")
    return df["Label"].astype(int).to_numpy(), df["Prob"].astype(float).to_numpy()


def plot(run_dirs: List[Path], label: str, out_dir: Path, dpi: int) -> None:
    fpr_grid = np.linspace(0.0, 1.0, _GRID)
    recall_grid = np.linspace(0.0, 1.0, _GRID)

    tpr_runs, prec_runs, roc_aucs, pr_aucs = [], [], [], []
    for d in run_dirs:
        y_true, y_score = _load(d)
        tpr_i, roc_a = _roc_on_grid(y_true, y_score, fpr_grid)
        prec_i, pr_a = _pr_on_grid(y_true, y_score, recall_grid)
        tpr_runs.append(tpr_i)
        prec_runs.append(prec_i)
        roc_aucs.append(roc_a)
        pr_aucs.append(pr_a)

    tpr_mean = np.mean(tpr_runs, axis=0)
    prec_mean = np.mean(prec_runs, axis=0)
    roc_mean, pr_mean = float(np.mean(roc_aucs)), float(np.mean(pr_aucs))

    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(12, 5))
    color = "C0"

    ax_roc.fill_between(fpr_grid, np.min(tpr_runs, 0), np.max(tpr_runs, 0), color=color, alpha=_SHADE_ALPHA, linewidth=0)
    ax_prc.fill_between(recall_grid, np.min(prec_runs, 0), np.max(prec_runs, 0), color=color, alpha=_SHADE_ALPHA, linewidth=0)
    ax_roc.plot(fpr_grid, tpr_mean, color=color, lw=2, label=f"{label} ({roc_mean:.2f})")
    ax_prc.plot(recall_grid, prec_mean, color=color, lw=2, label=f"{label} ({pr_mean:.2f})")

    ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="1 - Specificity", ylabel="Sensitivity", title="AUROC")
    ax_prc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="AUPRC")
    ax_roc.legend(loc="lower right")
    ax_prc.legend(loc="lower left")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_prc_curves.png", dpi=dpi)
    fig.savefig(out_dir / "roc_prc_curves.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs=3, type=Path, help="三次运行的目录")
    p.add_argument("--label", default="model")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    plot(list(args.run_dirs), args.label, args.out_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
