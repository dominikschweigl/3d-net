"""Plot the distribution of bounding box widths, heights, and areas for a FiftyOne dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import fiftyone as fo

DEFAULT_DATASETS = ["3d_net_rgbt","m3fd_rgbt", "rtdod_rgbt", "visdrone_rgbt"]
DEFAULT_LABEL_FIELD = "ground_truth"
CLASS_COLORS: dict[str, str] = {
    "person":     "#e15759",  # red
    "bus":    "#edc948",  # yellow
    "motorcycle": "#f28e2b",  # orange
    "car":        "#4e79a7",  # blue
    "truck":      "#59a14f",  # green
    "bicycle":        "#b07aa1",  # purple
}
_FALLBACK_COLORS = ["#76b7b2", "#ff9da7", "#9c755f", "#bab0ac"]


def collect_detections(
    dataset: fo.Dataset,
    label_field: str,
) -> dict[str, list[tuple[float, float, float]]]:
    """Return {label: [(w, h, area), ...]} from all samples."""
    by_label: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for sample in dataset.iter_samples(progress=True):
        detections_field = sample.get_field(label_field)
        if detections_field is None:
            continue
        for det in detections_field.detections:
            _, _, w, h = det.bounding_box
            by_label[det.label].append((w, h, w * h))

    return dict(by_label)


def _add_stats_lines(ax: plt.Axes, values: np.ndarray) -> None:
    median = float(np.median(values))
    mean = float(np.mean(values))
    ax.axvline(median, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label=f"median {median:.3f}")
    ax.axvline(mean, color="black", linestyle=":", linewidth=1.2, alpha=0.8, label=f"mean {mean:.3f}")


def _plot_row(
    axes: np.ndarray,
    by_label: dict[str, list[tuple[float, float, float]]],
    *,
    by_class: bool,
    bins: int,
    dataset_name: str = "",
) -> None:
    all_entries = [(w, h, a) for entries in by_label.values() for w, h, a in entries]
    total = len(all_entries)

    all_w = np.array([e[0] for e in all_entries])
    all_h = np.array([e[1] for e in all_entries])
    all_a = np.array([e[2] for e in all_entries])

    panels = [
        (axes[0], all_w, "Width (normalized)",  np.linspace(0, 1, bins + 1), False, 0),
        (axes[1], all_h, "Height (normalized)", np.linspace(0, 1, bins + 1), False, 1),
        (axes[2], all_a, "Area (normalized)",   None,                         True,  2),
    ]

    labels_sorted = sorted(by_label.keys())
    fallback_iter = iter(_FALLBACK_COLORS)
    color_map = {lbl: CLASS_COLORS.get(lbl, next(fallback_iter, "#aaaaaa")) for lbl in labels_sorted}

    for ax, all_vals, xlabel, fixed_bins, log_x, idx in panels:
        if log_x:
            nonzero = all_vals[all_vals > 0]
            bin_edges = np.logspace(np.log10(nonzero.min()), np.log10(nonzero.max()), bins + 1)
        else:
            bin_edges = fixed_bins

        if by_class:
            for lbl in labels_sorted:
                vals = np.array([e[idx] for e in by_label[lbl]])
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=bin_edges, alpha=0.5, label=lbl, color=color_map[lbl],
                        weights=np.ones(len(vals)) / len(vals) * 100)
        else:
            ax.hist(all_vals, bins=bin_edges, alpha=0.8, color="#4e79a7",
                    weights=np.ones(len(all_vals)) / total * 100)
            _add_stats_lines(ax, all_vals)

        if log_x:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("% of detections")

        if by_class:
            ax.legend(fontsize=8, ncol=2)
        else:
            ax.legend(fontsize=8)

    ax2d = axes[3]
    if by_class:
        scatter_order = sorted(labels_sorted, key=lambda lbl: len(by_label[lbl]), reverse=True)
        if dataset_name.startswith("visdrone") and "truck" in scatter_order and "car" in scatter_order:
            ti, ci = scatter_order.index("truck"), scatter_order.index("car")
            scatter_order[ti], scatter_order[ci] = scatter_order[ci], scatter_order[ti]
        for lbl in scatter_order:
            entries = by_label[lbl]
            if not entries:
                continue
            wx = np.array([e[0] for e in entries])
            hy = np.array([e[1] for e in entries])
            ax2d.scatter(wx, hy, s=2, alpha=0.3, label=lbl, color=color_map[lbl], rasterized=True)
        ax2d.legend(fontsize=8, ncol=2, markerscale=4)
    else:
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under("white")
        _, _, _, im = ax2d.hist2d(all_w, all_h, bins=40, cmap=cmap, vmin=1)
        ax2d.get_figure().colorbar(im, ax=ax2d, label="Count")

    lim = max(float(np.percentile(all_w, 99.9)), float(np.percentile(all_h, 99.9)))
    ax2d.set_xlim(0, lim)
    ax2d.set_ylim(0, lim)
    ax2d.set_xlabel("Width (normalized)")
    ax2d.set_ylabel("Height (normalized)")
    ax2d.set_aspect("equal")


def plot_distributions(
    rows: list[tuple[str, dict[str, list[tuple[float, float, float]]]]],
    *,
    by_class: bool,
    bins: int,
    split: str | None,
) -> plt.Figure:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n), squeeze=False)

    title = "Bounding box size distributions"
    if split:
        title += f"  (split={split})"
    fig.suptitle(title, fontsize=13)

    for row_idx, (dataset_name, by_label) in enumerate(rows):
        total = sum(len(v) for v in by_label.values())
        row_axes = axes[row_idx]
        row_axes[0].set_ylabel(f"{total:,} detections\n% of detections")
        _plot_row(row_axes, by_label, by_class=by_class, bins=bins, dataset_name=dataset_name)
        for ax in row_axes[1:3]:
            ax.set_ylabel("")

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=8.0)

    for row_idx, (dataset_name, _) in enumerate(rows):
        left_pos = axes[row_idx][0].get_position()
        right_pos = axes[row_idx][-1].get_position()
        x_center = (left_pos.x0 + right_pos.x1) / 2
        fig.text(x_center, left_pos.y1 + 0.005, dataset_name,
                 ha="center", va="bottom", fontsize=13)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot bounding box size distributions for one or more FiftyOne datasets.",
    )
    parser.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS,
                        help="One or more FiftyOne dataset names (default: %(default)s)")
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD,
                        help="Detections field name (default: %(default)s)")
    parser.add_argument("--split", default=None, help="Filter to a specific split (train/val/test)")
    parser.add_argument("--by-class", action="store_true", help="Color histograms by class label")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default: %(default)s)")
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

        if args.split:
            dataset = dataset.match({"split": args.split})
            logging.info("Filtered to split='%s' (%d samples)", args.split, len(dataset))

        by_label = collect_detections(dataset, args.label_field)
        total = sum(len(v) for v in by_label.values())
        logging.info("Collected %d detections across %d classes", total, len(by_label))

        if total == 0:
            logging.warning("No detections found in '%s' field '%s' — skipping.", name, args.label_field)
            continue

        rows.append((name, by_label))

    if not rows:
        logging.error("No data to plot.")
        sys.exit(1)

    fig = plot_distributions(rows, by_class=args.by_class, bins=args.bins, split=args.split)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        logging.info("Figure saved to '%s'", args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
