import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

FP_DIR = pathlib.Path(__file__).parent
FP_MD = FP_DIR / "fusion_method_performance.md"

FP_METHODS = [
    ("RGB", ["RGB"]),
    ("Thermal", ["Thermal"]),
    ("Early", ["Early Fusion"]),
    ("Mid", ["Mid Fusion"]),
    ("MidP3", ["Mid P3 Fusion"]),
    ("MidCMA3", ["Mid CMA 3 fusion", "Mid CMA3 Fusion"]),
    ("MidCFT3", ["Mid CFT 3 Fusion", "Mid CFT3 Fusion"]),
    ("Late", ["Late Fusion"]),
    ("Score", ["Score Fusion"]),
]

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

SUMMARY_RE = re.compile(
    r"([\d,]+) parameters.*?(?:([\d.]+) GFLOPs)?$"
)


def darken(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_model_complexity() -> dict[str, tuple[float, float | None]]:
    md_to_short: dict[str, str] = {}
    for short, headers in FP_METHODS:
        for h in headers:
            md_to_short[h.lower()] = short

    params_accum: dict[str, list[float]] = {m[0]: [] for m in FP_METHODS}
    gflops_accum: dict[str, list[float]] = {m[0]: [] for m in FP_METHODS}

    current_method: str | None = None
    for raw in FP_MD.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            current_method = None
            continue
        if line.startswith("### "):
            header = line[4:].strip()
            current_method = md_to_short.get(header.lower())
            continue
        if current_method and "parameters" in line:
            m = SUMMARY_RE.search(line)
            if m:
                params = float(m.group(1).replace(",", ""))
                params_accum[current_method].append(params)
                if m.group(2):
                    gflops_accum[current_method].append(float(m.group(2)))
                current_method = None

    result: dict[str, tuple[float, float | None]] = {}
    for short, _ in FP_METHODS:
        p_list = params_accum[short]
        g_list = gflops_accum[short]
        if p_list:
            params_mean = np.mean(p_list)
            gflops_mean = np.mean(g_list) if g_list else None
            result[short] = (params_mean, gflops_mean)
    return result


def make_figure(complexity: dict[str, tuple[float, float | None]], out_path: pathlib.Path) -> None:
    methods = [m[0] for m in FP_METHODS]
    colors = [METHOD_COLORS[m] for m in methods]
    edges = [darken(METHOD_COLORS[m]) for m in methods]

    params = [complexity[m][0] / 1e6 for m in methods]
    gflops_vals = [complexity[m][1] for m in methods]

    x = np.arange(len(methods))
    bar_w = 0.35

    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    fig.suptitle("YOLO11 fusion model complexity", fontsize=13)

    # --- Parameters ---
    bars_p = ax_p.bar(
        x,
        params,
        bar_w * 2,
        color=colors,
        edgecolor=edges,
        linewidth=0.8,
        zorder=3,
    )
    ax_p.set_axisbelow(True)
    ax_p.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5, zorder=0)
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(methods, rotation=30, ha="right")
    ax_p.set_ylabel("Parameters (M)")
    ax_p.set_title("Model parameters")

    for bar, val in zip(bars_p, params):
        ax_p.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}M",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    # --- GFLOPs (log scale) ---
    bars_g = ax_g.bar(
        x,
        gflops_vals,
        bar_w * 2,
        color=colors,
        edgecolor=edges,
        linewidth=0.8,
        zorder=3,
    )

    ax_g.set_yscale("log")
    ax_g.set_axisbelow(True)
    ax_g.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5, zorder=0)
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(methods, rotation=30, ha="right")
    ax_g.set_ylabel("GFLOPs (log scale)")
    ax_g.set_title("Computational cost (GFLOPs)")

    for bar, val in zip(bars_g, gflops_vals):
        ax_g.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ymax = max(gflops_vals)
    ax_g.set_ylim(top=ymax * 1.5)

    fig.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def main() -> None:
    complexity = parse_model_complexity()
    make_figure(complexity, FP_DIR / "model_complexity.png")


if __name__ == "__main__":
    main()
