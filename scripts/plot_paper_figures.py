#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "GB-Persona",
    "K-means",
    "Behavior K-medoids",
    "Spectral Clustering",
    "Density Peaks",
    "DBSCAN",
    "Demographic Stratified",
    "Weighted Random",
]

SHORT_LABELS = {
    "GB-Persona": "GB-Persona",
    "K-means": "K-means",
    "Behavior K-medoids": "K-medoids",
    "Spectral Clustering": "Spectral",
    "Density Peaks": "Density Peaks",
    "DBSCAN": "DBSCAN",
    "Demographic Stratified": "Demographic",
    "Weighted Random": "Weighted Random",
}

COLORS = {
    "GB-Persona": "#D55E00",
    "K-means": "#0072B2",
    "Behavior K-medoids": "#5F4690",
    "Spectral Clustering": "#56B4E9",
    "Density Peaks": "#CC79A7",
    "DBSCAN": "#E69F00",
    "Demographic Stratified": "#009E73",
    "Weighted Random": "#7A7A7A",
}

MARKERS = {
    "GB-Persona": "o",
    "K-means": "s",
    "Behavior K-medoids": "D",
    "Spectral Clustering": "^",
    "Density Peaks": "P",
    "DBSCAN": "X",
    "Demographic Stratified": "v",
    "Weighted Random": "h",
}

LINESTYLES = {
    "GB-Persona": "-",
    "K-means": "--",
    "Behavior K-medoids": "-.",
    "Spectral Clustering": ":",
    "Density Peaks": (0, (5, 2)),
    "DBSCAN": (0, (3, 1, 1, 1)),
    "Demographic Stratified": (0, (7, 2)),
    "Weighted Random": (0, (1, 1)),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def make_budget_curve(data: pd.DataFrame, pool_size: int, output_path: Path) -> None:
    budgets = np.sort(data["requested_budget"].unique()).astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    for method in METHOD_ORDER:
        values = (
            data[data["method"] == method]
            .set_index("requested_budget")
            .loc[budgets, "mean_test_normalized_w1"]
            .to_numpy()
        )
        is_gb = method == "GB-Persona"
        ax.plot(
            budgets,
            values,
            label=method,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            linewidth=2.6 if is_gb else 1.65,
            markersize=6.2 if is_gb else 5.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=5 if is_gb else 3,
        )
    maximum = float(data["mean_test_normalized_w1"].max())
    width = float(np.ptp(budgets))
    ax.set_xlim(budgets.min() - 0.04 * width, budgets.max() + 0.04 * width)
    ax.set_ylim(0, np.ceil((maximum + 0.004) * 100) / 100)
    ax.set_xticks(budgets)
    ax.set_xlabel("Requested Persona Budget, B")
    ax.set_ylabel("Mean Test Normalized W1")
    ax.set_title(f"{pool_size}-Persona Budget-Error Curves", pad=8, fontweight="semibold")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.6,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.74)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def make_auec_bars(data: pd.DataFrame, pool_size: int, output_path: Path) -> None:
    auec = data.groupby("method", sort=False)["normalized_auec"].first()
    values = [float(auec[method]) for method in METHOD_ORDER]
    positions = np.arange(len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    bars = ax.bar(
        positions,
        values,
        width=0.68,
        color=[COLORS[method] for method in METHOD_ORDER],
        linewidth=[1.2 if method == "GB-Persona" else 0 for method in METHOD_ORDER],
        edgecolor=[
            "#222222" if method == "GB-Persona" else COLORS[method]
            for method in METHOD_ORDER
        ],
        zorder=3,
    )
    ax.set_ylim(0, max(values) * 1.20)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [SHORT_LABELS[method] for method in METHOD_ORDER],
        rotation=26,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Normalized AUEC")
    ax.set_title(f"{pool_size}-Persona Area Under the Error Curve", pad=8, fontweight="semibold")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=7.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.265, top=0.90)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the four figures used in the paper.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduced_results/figures")
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(root / "results/paper_results.csv")
    configure_style()
    for pool_size, curve_number, auec_number in ((128, 1, 2), (64, 3, 4)):
        subset = results[results["pool_size"] == pool_size]
        make_budget_curve(
            subset,
            pool_size,
            output_dir / f"figure_{curve_number}_{pool_size}_persona_budget_curve.pdf",
        )
        make_auec_bars(
            subset,
            pool_size,
            output_dir / f"figure_{auec_number}_{pool_size}_persona_auec.pdf",
        )
    print(f"Wrote four paper figures to {output_dir}")


if __name__ == "__main__":
    main()

