#!/usr/bin/env python3
"""Render result charts: pruning tradeoff curves + differential-score heatmap.

Usage: chart.py --model qwen-0.8b
Writes results/<model>/charts/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasonprune.config import RESULTS_DIR

# Reference categorical palette (light mode), fixed slot order.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

STRATEGY_LABELS = {
    "diff": "differential (guarded)",
    "diff_noguard": "differential (no guard)",
    "know": "knowledge-importance",
    "lowmag": "low overall importance",
    "random": "random",
}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load_runs(model_key: str):
    d = RESULTS_DIR / model_key
    baseline = json.loads((d / "baseline.json").read_text())
    runs = [json.loads(l) for l in (d / "sweep.jsonl").read_text().splitlines()]
    return baseline, runs


def chart_tradeoff(model_key: str, baseline: dict, runs: list, out: Path):
    strategies = [s for s in STRATEGY_LABELS if any(
        r["config"]["strategy"] == s for r in runs)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE)
    axis_specs = [
        ("knowledge_acc", "Closed-book knowledge accuracy"),
        ("reasoning_acc", "In-context reasoning accuracy"),
    ]
    for ax, (key, title) in zip(axes[:2], axis_specs):
        style_ax(ax)
        base = baseline[key]
        ax.axhline(base, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate("unpruned", xy=(0.02, base), xycoords=("axes fraction", "data"),
                    fontsize=8, color=INK2, va="bottom")
        for i, s in enumerate(strategies):
            pts = sorted([(r["config"]["frac"], r[key]) for r in runs
                          if r["config"]["strategy"] == s])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o",
                    markersize=5, label=STRATEGY_LABELS[s])
        ax.set_xlabel("fraction of MLP channels pruned", color=INK2, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")

    ax = axes[2]
    style_ax(ax)
    ax.plot([baseline["knowledge_acc"]], [baseline["reasoning_acc"]],
            marker="*", markersize=14, color=INK, linestyle="none")
    ax.annotate("unpruned", (baseline["knowledge_acc"], baseline["reasoning_acc"]),
                textcoords="offset points", xytext=(6, -4), fontsize=8, color=INK2)
    for i, s in enumerate(strategies):
        pts = sorted([(r["config"]["frac"], r["knowledge_acc"], r["reasoning_acc"])
                      for r in runs if r["config"]["strategy"] == s])
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=SERIES[i],
                linewidth=2, marker="o", markersize=5, alpha=0.9)
    ax.set_xlabel("knowledge accuracy (want: low)", color=INK2, fontsize=9)
    ax.set_ylabel("reasoning accuracy (want: high)", color=INK2, fontsize=9)
    ax.set_title("Tradeoff path (up-left is the goal)", color=INK,
                 fontsize=10.5, loc="left")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Differential pruning — {model_key}", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out / "tradeoff.png", dpi=180, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def chart_heatmap(model_key: str, out: Path):
    scores_path = RESULTS_DIR / model_key / "scores.npz"
    if not scores_path.exists():
        return
    z = np.load(scores_path)
    i_know, i_reason = z["i_know"], z["i_reason"]
    eps = np.quantile(i_reason, 0.10, axis=1, keepdims=True) + 1e-8
    d = i_know / (i_reason + eps)
    # Per-layer channel distribution of the differential score, sorted per row.
    d_sorted = np.sort(d, axis=1)[:, ::-1]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(False)
    im = ax.imshow(np.log10(d_sorted + 1e-6), aspect="auto",
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       "seq_blue", ["#cde2fb", "#3987e5", "#0d366b"]),
                   interpolation="nearest")
    ax.set_xlabel("channels (sorted by score, per layer)", color=INK2, fontsize=9)
    ax.set_ylabel("layer", color=INK2, fontsize=9)
    ax.set_title(f"log10 knowledge/reasoning importance ratio — {model_key}",
                 color=INK, fontsize=10.5, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "diff_heatmap.png", dpi=180, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def chart_layer_profile(model_key: str, out: Path):
    scores_path = RESULTS_DIR / model_key / "scores.npz"
    if not scores_path.exists():
        return
    z = np.load(scores_path)
    i_know, i_reason = z["i_know"], z["i_reason"]
    layers = np.arange(i_know.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style_ax(ax)
    for arr, color, label in ((i_know, SERIES[0], "knowledge importance"),
                              (i_reason, SERIES[1], "reasoning importance")):
        med = np.median(arr, axis=1)
        hi = np.percentile(arr, 90, axis=1)
        ax.plot(layers, med / med.max(), color=color, linewidth=2, label=label)
        ax.plot(layers, hi / hi.max(), color=color, linewidth=1.2,
                linestyle=(0, (3, 2)), alpha=0.7)
    ax.set_xlabel("layer", color=INK2, fontsize=9)
    ax.set_ylabel("normalized importance", color=INK2, fontsize=9)
    ax.set_title("Where each capability lives (median solid, p90 dashed) — "
                 f"{model_key}", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "layer_profile.png", dpi=180, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen-0.8b")
    args = p.parse_args()
    baseline, runs = load_runs(args.model)
    out = RESULTS_DIR / args.model / "charts"
    out.mkdir(parents=True, exist_ok=True)
    chart_tradeoff(args.model, baseline, runs, out)
    chart_heatmap(args.model, out)
    chart_layer_profile(args.model, out)
    print(f"charts -> {out}")


if __name__ == "__main__":
    main()
