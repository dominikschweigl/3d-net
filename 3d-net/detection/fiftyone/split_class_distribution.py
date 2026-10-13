"""Plot per-class object counts across train/val/test splits for FiftyOne datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import fiftyone as fo

DEFAULT_DATASETS = ["3d_net_rgbt"]
DEFAULT_LABEL_FIELD = "ground_truth"
SPLITS = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#4e79a7", "val": "#59a14f", "test": "#e15759"}
CLASS_COLORS: dict[str, str] = {
    "person":     "#e15759",
    "bus":        "#edc948",
    "motorcycle": "#f28e2b",
    "car":        "#4e79a7",
    "truck":      "#59a14f",
    "bicycle":    "#b07aa1",
}
_FALLBACK_COLORS = ["#76b7b2", "#ff9da7", "#9c755f", "#bab0ac"]


def collect_split_counts(
    dataset: fo.Dataset,
    label_field: str,
) -> dict[str, dict[str, int]]:
    """Return {split: {label: count}} from all samples with a split field."""
    counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in SPLITS}

    for sample in dataset.iter_samples(progress=True):
        split = sample.get_field("split")
        if split not in SPLITS:
            continue
        detections_field = sample.get_field(label_field)
        if detections_field is None:
            continue
        for det in detections_field.detections:
            counts[split][det.label] += 1

    return {s: dict(d) for s, d in counts.items()}


def plot_split_distribution(
    rows: list[tuple[str, dict[str, dict[str, int]]]],
) -> plt.Figure:
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), squeeze=False)
    fig.suptitle("Object class distribution per split", fontsize=13)

    for row_idx, (dataset_name, split_counts) in enumerate(rows):
        all_labels_unsorted = {lbl for s in SPLITS for lbl in split_counts[s]}
        # Sort by total count across all splits, descending
        all_labels = sorted(
            all_labels_unsorted,
            key=lambda lbl: sum(split_counts[s].get(lbl, 0) for s in SPLITS),
            reverse=True,
        )
        fallback_iter = iter(_FALLBACK_COLORS)
        color_map = {lbl: CLASS_COLORS.get(lbl, next(fallback_iter, "#aaaaaa")) for lbl in all_labels}

        ax = axes[row_idx][0]

        x = np.arange(len(SPLITS))
        n_classes = len(all_labels)
        width = 0.8 / n_classes
        offsets = np.linspace(-(n_classes - 1) / 2, (n_classes - 1) / 2, n_classes) * width

        totals_per_split = {s: sum(split_counts[s].values()) for s in SPLITS}

        for i, lbl in enumerate(all_labels):
            vals = [split_counts[split].get(lbl, 0) for split in SPLITS]
            bars = ax.bar(x + offsets[i], vals, width, label=lbl, color=color_map[lbl], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:,}",
                        ha="center", va="bottom", fontsize=6, color="black",
                    )

        ax.set_xticks(x)
        split_labels = [f"{s}\n(n={totals_per_split[s]:,})" for s in SPLITS]
        ax.set_xticklabels(split_labels)
        ax.set_ylabel("Object count")
        ax.legend(fontsize=8, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        total_all = sum(totals_per_split.values())
        ax.set_title(f"{dataset_name}  ({total_all:,} detections total)", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-class object counts across train/val/test splits.",
    )
    parser.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS,
                        help="One or more FiftyOne dataset names (default: %(default)s)")
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD,
                        help="Detections field name (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows: list[tuple[str, dict]] = []
    for name in args.dataset:
        if not fo.dataset_exists(name):
            logging.error("Dataset '%s' does not exist — skipping.", name)
            continue

        dataset = fo.load_dataset(name)
        logging.info("Loaded dataset '%s' (%d samples)", name, len(dataset))

        split_counts = collect_split_counts(dataset, args.label_field)
        total = sum(sum(d.values()) for d in split_counts.values())
        logging.info("Collected %d detections", total)

        if total == 0:
            logging.warning("No detections found in '%s' field '%s' — skipping.", name, args.label_field)
            continue

        rows.append((name, split_counts))

    if not rows:
        logging.error("No data to plot.")
        sys.exit(1)

    fig = plot_split_distribution(rows)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        logging.info("Figure saved to '%s'", args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
