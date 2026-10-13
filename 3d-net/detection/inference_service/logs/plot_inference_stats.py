import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).parent
LOGS_DIR = SCRIPT_DIR
FUSIONS = ["Single", "Early", "MidP3", "Mid", "Late"]

COLOR_MS = "#739EE8"
COLOR_FPS = "#E47F7B"
EDGE_MS = "#1059D6"
EDGE_FPS = "#D01F19"
BAR_WIDTH = 0.24
BAR_GAP = 0.06


def col_stats(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "median": series.median(),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75),
        "min": series.min(),
        "max": series.max(),
    }


def load_stats(fusion: str) -> tuple[dict, dict]:
    df = pd.read_csv(LOGS_DIR / f"{fusion}_Warm" / "inference_stats.csv")
    df.columns = [c.strip() for c in df.columns]
    return col_stats(df["infer_ms"]), col_stats(df["fps"])


def draw_candle(ax, center_x: float, stats: dict, candle_width: float | None = None, zorder: int = 4):
    cw = candle_width or BAR_WIDTH * 0.55
    hw = cw / 2
    cap_hw = hw * 0.9
    ax.plot([center_x, center_x], [stats["min"], stats["max"]],
            color="black", linewidth=0.7, zorder=zorder)
    ax.plot([center_x - cap_hw, center_x + cap_hw], [stats["min"], stats["min"]],
            color="black", linewidth=0.7, zorder=zorder)
    ax.plot([center_x - cap_hw, center_x + cap_hw], [stats["max"], stats["max"]],
            color="black", linewidth=0.7, zorder=zorder)
    ax.add_patch(plt.Rectangle(
        (center_x - hw, stats["q25"]), cw, stats["q75"] - stats["q25"],
        facecolor="none", edgecolor="black", linewidth=0.7, zorder=zorder,
    ))
    ax.plot([center_x - hw, center_x + hw], [stats["median"], stats["median"]],
            color="black", linewidth=0.7, zorder=zorder + 1)


ms_stats, fps_stats = zip(*[load_stats(f) for f in FUSIONS])

x = np.arange(len(FUSIONS))
offset = BAR_WIDTH / 2 + BAR_GAP / 2

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
ax2 = ax1.twinx()

ms_means = [s["mean"] for s in ms_stats]
fps_means = [s["mean"] for s in fps_stats]

ax1.bar(x - offset, ms_means, BAR_WIDTH, color=COLOR_MS, edgecolor=EDGE_MS, linewidth=0.8, zorder=3)
ax2.bar(x + offset, fps_means, BAR_WIDTH, color=COLOR_FPS, edgecolor=EDGE_FPS, linewidth=0.8, zorder=3)

for i in range(len(FUSIONS)):
    draw_candle(ax1, x[i] - offset, ms_stats[i])
    draw_candle(ax2, x[i] + offset, fps_stats[i])

ax1.set_axisbelow(True)
ax1.yaxis.grid(True, linestyle="--", color="grey", alpha=0.6, zorder=0)

for i in range(len(FUSIONS) - 1):
    ax1.axvline(x=i + 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

ax1.set_xticks(x)
ax1.set_xticklabels(FUSIONS)
ax1.set_ylabel("Inference time (ms)")
ax2.set_ylabel("FPS")
ax1.set_xlim(-0.6, len(FUSIONS) - 0.4)
ax1.set_ylim(0, max(s["max"] for s in ms_stats) * 1.12)
ax2.set_ylim(0, max(s["max"] for s in fps_stats) * 1.12)

legend_handles = [
    mpatches.Patch(color=COLOR_MS, label="Inference time (ms)"),
    mpatches.Patch(color=COLOR_FPS, label="FPS"),
]
ax1.legend(handles=legend_handles, frameon=False)

plt.tight_layout()
out = SCRIPT_DIR / "inference_stats.png"
plt.savefig(out)
print(f"Saved {out}")
