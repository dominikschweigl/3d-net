import pathlib
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FP_DIR = pathlib.Path(__file__).parent
FP_MD = FP_DIR / "fusion_method_performance.md"

DATASETS = [
    ("M3FD", "M3FD", "m3fd"),
    ("RTDOD", "RTDOD", None),
    ("Visdrone Dronevehicle", "Visdrone", None),
    ("3D-Net", "3D-Net", "3d_net_rgbt"),
    ("3D-Net Data Pool", "3D-Net Pool", None),
]

FP_METHODS = [
    ("RGB", ["RGB"]),
    ("Thermal", ["Thermal"]),
    ("Early", ["Early Fusion"]),
    ("Mid", ["Mid Fusion"]),
    ("MidP3", ["Mid P3 Fusion"]),
    ("MidCMA3", ["Mid CMA 3 Fusion", "Mid CMA3 Fusion"]),
    ("MidCFT3", ["Mid CFT 3 Fusion", "Mid CFT3 Fusion"]),
    ("Late", ["Late Fusion"]),
    ("Score", ["Score Fusion"]),
]

QUANT_RESULTS: dict[tuple[str, str], tuple[float, float]] = {
    ("M3FD", "RGB"):     (0.6368, 0.4325),
    ("M3FD", "Thermal"): (0.7363, 0.4839),
    ("M3FD", "Early"):   (0.6544, 0.4539),
    ("M3FD", "Mid"):     (0.6776, 0.4643),
    ("M3FD", "MidP3"):   (0.6869, 0.4837),
    ("M3FD", "Late"):    (0.6894, 0.4685),

    ("RTDOD", "RGB"):     (0.4142, 0.2739),
    ("RTDOD", "Thermal"): (0.5911, 0.2857),
    ("RTDOD", "Early"):   (0.7501, 0.5064),
    ("RTDOD", "Mid"):     (0.7812, 0.5310),
    ("RTDOD", "MidP3"):   (0.7781, 0.5255),
    ("RTDOD", "Late"):    (0.7832, 0.5283),

    ("Visdrone Dronevehicle", "RGB"):     (0.7353, 0.4826),
    ("Visdrone Dronevehicle", "Thermal"): (0.8823, 0.6674),
    ("Visdrone Dronevehicle", "Early"):   (0.8859, 0.6750),
    ("Visdrone Dronevehicle", "Mid"):     (0.8940, 0.6806),
    ("Visdrone Dronevehicle", "MidP3"):   (0.9065, 0.6961),
    ("Visdrone Dronevehicle", "Late"):    (0.8921, 0.6790),

    ("3D-Net", "RGB"):     (0.9261, 0.7277),
    ("3D-Net", "Thermal"): (0.4894, 0.3058),
    ("3D-Net", "Early"):   (0.9036, 0.7712),
    ("3D-Net", "Mid"):     (0.9651, 0.8311),
    ("3D-Net", "MidP3"):   (0.9534, 0.8271),
    ("3D-Net", "Late"):    (0.9347, 0.8052),

    ("3D-Net Data Pool", "RGB"):     (0.6749, 0.4477),
    ("3D-Net Data Pool", "Thermal"): (0.5040, 0.3179),
    ("3D-Net Data Pool", "Early"):   (0.7365, 0.5177),
    ("3D-Net Data Pool", "Mid"):     (0.7752, 0.5557),
    ("3D-Net Data Pool", "MidP3"):   (0.7726, 0.5592),
    ("3D-Net Data Pool", "Late"):    (0.7819, 0.5622),
}

METHOD_COLORS = {
    "RGB": "#4C78A8",
    "Thermal": "#F58518",
    "Early": "#54A24B",
    "Mid": "#E45756",
    "MidP3": "#72B7B2",
    "MidCMA3": "#B279A2",
    "MidCFT3": "#9D755D",
    "Late": "#EECA3B",
    "Score": "#BAB0AC",
}


def darken(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_fp_metrics() -> dict:
    results: dict[tuple[str, str], tuple[float, float]] = {}
    md_to_short: dict[str, str] = {}
    for short, headers in FP_METHODS:
        for h in headers:
            md_to_short[h.lower()] = short

    dataset_headers = {ds[0].lower(): ds[0] for ds in DATASETS}

    current_dataset: str | None = None
    current_method: str | None = None

    for raw in FP_MD.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            header = line[3:].strip()
            current_dataset = dataset_headers.get(header.lower())
            current_method = None
            continue
        if line.startswith("### "):
            header = line[4:].strip()
            current_method = md_to_short.get(header.lower())
            continue
        if current_dataset and current_method:
            stripped = line.strip()
            if stripped.startswith("all "):
                parts = re.split(r"\s+", stripped)
                try:
                    mAP50 = float(parts[-2])
                    mAP5095 = float(parts[-1])
                except ValueError:
                    continue
                results[(current_dataset, current_method)] = (mAP50, mAP5095)
                current_method = None
    return results


def load_quantized(ds_md: str, method_short: str) -> tuple[float, float] | None:
    return QUANT_RESULTS.get((ds_md, method_short))


def draw_bar(ax, x, height, color, edge, hatch=None, alpha=1.0, width=0.1, zorder=3):
    ax.bar(
        x,
        height,
        width,
        color=color,
        edgecolor=edge,
        linewidth=0.8,
        hatch=hatch,
        alpha=alpha,
        zorder=zorder,
    )


def make_figure_1(fp_results: dict, out_path: pathlib.Path) -> None:
    methods = [m[0] for m in FP_METHODS]
    bar_w = 0.04
    inner_gap = 0.0
    method_gap = 0.01
    slot = 2 * bar_w + inner_gap  # one method = 2 adjacent bars (mAP50, mAP50-95)
    group_w = len(methods) * slot + (len(methods) - 1) * method_gap

    x = np.arange(len(DATASETS))
    fig, ax = plt.subplots(figsize=(18, 6), dpi=150)

    for i, (ds_md, _, _) in enumerate(DATASETS):
        for j, method in enumerate(methods):
            color = METHOD_COLORS[method]
            edge = darken(color)
            slot_start = x[i] - group_w / 2 + j * (slot + method_gap)
            x50 = slot_start + bar_w / 2
            x5095 = slot_start + bar_w + inner_gap + bar_w / 2
            vals = fp_results.get((ds_md, method))
            if vals is None:
                continue
            mAP50, mAP5095 = vals
            draw_bar(ax, x50, mAP50, color, edge, hatch=None, width=bar_w)
            draw_bar(ax, x5095, mAP5095, color, edge, hatch="////", width=bar_w)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5, zorder=0)

    for i in range(len(DATASETS) - 1):
        ax.axvline(x=i + 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in DATASETS])
    ax.set_ylabel("mAP")
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylim(0, 1.05)
    ax.set_title("Full-precision YOLO11 fusion methods — mAP per dataset")

    method_handles = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor=darken(METHOD_COLORS[m]), label=m)
        for m in methods
    ]
    metric_handles = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="black", label="mAP@50"),
        mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="////", label="mAP@50-95"),
    ]

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.18)
    fig.legend(
        handles=method_handles,
        title="Fusion method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(methods),
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )
    ax.legend(handles=metric_handles, loc="upper right", frameon=True, fontsize=9)

    plt.savefig(out_path)
    print(f"Saved {out_path}")


def make_figure_2(fp_results: dict, out_path: pathlib.Path) -> None:
    methods = ["RGB", "Thermal", "Early", "Mid", "MidP3", "Late"]
    bar_w = 0.033
    pair_gap = 0.00  # small gap between FP-pair and Q-pair within a method
    method_gap = 0.02  # large gap between fusion methods
    slot = 4 * bar_w + pair_gap
    group_w = len(methods) * slot + (len(methods) - 1) * method_gap

    x = np.arange(len(DATASETS))
    fig, ax = plt.subplots(figsize=(20, 6.5), dpi=150)
    q_alpha = 0.45

    for i, (ds_md, _, _) in enumerate(DATASETS):
        for j, method in enumerate(methods):
            color = METHOD_COLORS[method]
            edge = darken(color)
            slot_start = x[i] - group_w / 2 + j * (slot + method_gap)
            positions = [
                slot_start + bar_w / 2,
                slot_start + bar_w * 1.5,
                slot_start + bar_w * 2.5 + pair_gap,
                slot_start + bar_w * 3.5 + pair_gap,
            ]

            fp_vals = fp_results.get((ds_md, method))
            q_vals = load_quantized(ds_md, method)

            if fp_vals is not None:
                draw_bar(ax, positions[0], fp_vals[0], color, edge, hatch=None, width=bar_w)
                draw_bar(ax, positions[1], fp_vals[1], color, edge, hatch="////", width=bar_w)
            if q_vals is not None:
                draw_bar(ax, positions[2], q_vals[0], color, edge, hatch=None, alpha=q_alpha, width=bar_w)
                draw_bar(ax, positions[3], q_vals[1], color, edge, hatch="////", alpha=q_alpha, width=bar_w)
            else:
                draw_bar(ax, positions[2], 1.0, "lightgrey", "grey", hatch="xx", alpha=0.35, width=bar_w)
                draw_bar(ax, positions[3], 1.0, "lightgrey", "grey", hatch="xx", alpha=0.35, width=bar_w)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5, zorder=0)

    for i in range(len(DATASETS) - 1):
        ax.axvline(x=i + 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in DATASETS])
    ax.set_ylabel("mAP")
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylim(0, 1.08)
    ax.set_title("Full-precision vs quantized YOLO11 — mAP per dataset")

    method_handles = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor=darken(METHOD_COLORS[m]), label=m)
        for m in methods
    ]
    encoding_handles = [
        mpatches.Patch(facecolor="#888888", edgecolor="black", label="FP  mAP@50"),
        mpatches.Patch(facecolor="#888888", edgecolor="black", hatch="////", label="FP  mAP@50-95"),
        mpatches.Patch(facecolor="#888888", edgecolor="black", alpha=q_alpha, label="Q  mAP@50"),
        mpatches.Patch(facecolor="#888888", edgecolor="black", hatch="////", alpha=q_alpha, label="Q  mAP@50-95"),
    ]
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.18)
    fig.legend(
        handles=method_handles,
        title="Fusion method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(methods),
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )
    ax.legend(handles=encoding_handles, loc="upper right", frameon=True, fontsize=8, ncol=1)

    plt.savefig(out_path)
    print(f"Saved {out_path}")


def print_comparison(fp_results: dict) -> None:
    methods = ["RGB", "Thermal", "Early", "Mid", "MidP3", "Late"]
    col = {"ds": 22, "m": 9, "v": 10, "d": 8}
    sep = "-" * (col["ds"] + col["m"] + col["v"] * 4 + col["d"] * 2 + 2)

    def cell(v: float | None, w: int) -> str:
        return (f"{v:.4f}" if v is not None else "  --  ").rjust(w)

    def delta(fp: float | None, q: float | None, w: int) -> str:
        if fp is None or q is None:
            return "".rjust(w)
        d = q - fp
        s = f"{d:+.4f}"
        return s.rjust(w)

    hdr = (
        f"{'Dataset':<{col['ds']}}{'Method':<{col['m']}}"
        f"{'FP@50':>{col['v']}}{'Q@50':>{col['v']}}{'Δ@50':>{col['d']}}"
        f"{'FP@50-95':>{col['v']}}{'Q@50-95':>{col['v']}}{'Δ@50-95':>{col['d']}}"
    )
    print(f"\n{'Full-precision vs Quantized — metric comparison':^{len(sep)}}")
    print(sep)
    print(hdr)
    print(sep)

    for ds_md, ds_label, _ in DATASETS:
        first = True
        for method in methods:
            fp = fp_results.get((ds_md, method))
            q = QUANT_RESULTS.get((ds_md, method))
            ds_col = ds_label if first else ""
            first = False
            print(
                f"{ds_col:<{col['ds']}}{method:<{col['m']}}"
                f"{cell(fp[0] if fp else None, col['v'])}"
                f"{cell(q[0] if q else None, col['v'])}"
                f"{delta(fp[0] if fp else None, q[0] if q else None, col['d'])}"
                f"{cell(fp[1] if fp else None, col['v'])}"
                f"{cell(q[1] if q else None, col['v'])}"
                f"{delta(fp[1] if fp else None, q[1] if q else None, col['d'])}"
            )
        print(sep)


def main() -> None:
    fp_results = parse_fp_metrics()
    print_comparison(fp_results)
    make_figure_1(fp_results, FP_DIR / "fusion_performance_full_precision.png")
    make_figure_2(fp_results, FP_DIR / "fusion_performance_full_vs_quantized.png")


if __name__ == "__main__":
    main()
