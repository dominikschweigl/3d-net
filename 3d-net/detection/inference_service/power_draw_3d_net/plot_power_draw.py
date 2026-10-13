import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_DIR = pathlib.Path(__file__).parent
FUSIONS = ["Single", "Early", "MidP3", "Mid", "Late"]

COLOR_WARM = "#739EE8"
COLOR_20FPS = "#E47F7B"
EDGE_WARM = "#1059D6"
EDGE_20FPS = "#D01F19"
BAR_WIDTH = 0.24
BAR_GAP = 0.06


def power_stats(csv_path: pathlib.Path) -> dict:
    df = pd.read_csv(csv_path, skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    s = df["Power (W)"]
    return {
        "mean": s.mean(),
        "median": s.median(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "min": s.min(),
        "max": s.max(),
    }


def draw_candle(ax, center_x, stats, candle_width=None, zorder=4):
    cw = candle_width or BAR_WIDTH * 0.55
    hw = cw / 2
    cap_hw = hw * 0.9
    # wick: min to max
    ax.plot([center_x, center_x], [stats["min"], stats["max"]],
            color="black", linewidth=0.7, zorder=zorder)
    # min cap
    ax.plot([center_x - cap_hw, center_x + cap_hw], [stats["min"], stats["min"]],
            color="black", linewidth=0.7, zorder=zorder)
    # max cap
    ax.plot([center_x - cap_hw, center_x + cap_hw], [stats["max"], stats["max"]],
            color="black", linewidth=0.7, zorder=zorder)
    # box: Q25 to Q75, transparent fill
    ax.add_patch(plt.Rectangle(
        (center_x - hw, stats["q25"]), cw, stats["q75"] - stats["q25"],
        facecolor="none", edgecolor="black", linewidth=0.7, zorder=zorder,
    ))
    # median line
    ax.plot([center_x - hw, center_x + hw], [stats["median"], stats["median"]],
            color="black", linewidth=0.7, zorder=zorder + 1)


warm_stats = [power_stats(DATA_DIR / f"{f}_Warm.CSV") for f in FUSIONS]
fps20_stats = [power_stats(DATA_DIR / f"{f}_20fps.CSV") for f in FUSIONS]

warm_means = [s["mean"] for s in warm_stats]
fps20_means = [s["mean"] for s in fps20_stats]

x = np.arange(len(FUSIONS))

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

offset = BAR_WIDTH / 2 + BAR_GAP / 2
ax.bar(x - offset, warm_means, BAR_WIDTH, color=COLOR_WARM, edgecolor=EDGE_WARM, linewidth=0.8, zorder=3)
ax.bar(x + offset, fps20_means, BAR_WIDTH, color=COLOR_20FPS, edgecolor=EDGE_20FPS, linewidth=0.8, zorder=3)

for i, (ws, fs) in enumerate(zip(warm_stats, fps20_stats)):
    draw_candle(ax, x[i] - offset, ws)
    draw_candle(ax, x[i] + offset, fs)

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.6, zorder=0)

# dashed vertical separators between groups
for i in range(len(FUSIONS) - 1):
    ax.axvline(x=i + 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(FUSIONS)
ax.set_ylabel("Power (W)")
ax.set_xlim(-0.6, len(FUSIONS) - 0.4)
ax.set_ylim(0, max(s["max"] for s in warm_stats + fps20_stats) * 1.12)


legend_handles = [
    mpatches.Patch(color=COLOR_WARM, label="fps uncapped"),
    mpatches.Patch(color=COLOR_20FPS, label="20 fps capped"),
]
ax.legend(handles=legend_handles, frameon=False)

plt.tight_layout()
out = DATA_DIR / "power_draw.png"
plt.savefig(out)
print(f"Saved {out}")
