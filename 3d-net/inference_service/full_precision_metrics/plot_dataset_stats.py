"""
Dataset statistics plots for the bachelor thesis.
Run this script to regenerate all figures in images/.

To update values: edit DATASETS below.
To add a dataset: add a new entry to DATASETS and COLORS.
"""

import plotly.graph_objects as go

# ── Data ──────────────────────────────────────────────────────────────────────
# Annotation instance counts per unified class.
# Use 0 for classes absent in a dataset.
# Order must match CLASSES.

CLASSES = ["Car", "Bus", "Truck", "Motorcycle", "Bicycle", "Person"]

DATASETS = {
    "DroneVehicle": [440_163,   16_535, 43119,  0,      0,      0],  # car includes van; truck includes freight car
    "RTDOD":        [78_063,    2_336,  7_580,  26_378, 2_693,  57_515],
    "M3FD":         [18_296,    700,    1_008,  521,    0,      11_477],
    "3D-Net":       [  4502,     101,    2872,    20,      205,      1360],
}

COLORS = {
    "DroneVehicle": "#4C78A8",  # vega blue
    "RTDOD":        "#F58518",  # vega orange
    "M3FD":         "#54A24B",  # vega green
    "3D-Net":       "#E45756",  # vega red
}

FONT    = "Helvetica Neue, Arial, sans-serif"
GRAY_HI = "#111827"
GRAY_LO = "#6B7280"

# Approximate inner plot height in pixels — used to decide which labels fit inside a segment.
# Adjust if the figure height changes significantly.
PLOT_H_PX    = 400
MIN_LABEL_PX = 14  # min segment height in px to show a label inside


def darken(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Plot: class distribution ───────────────────────────────────────────────────
def plot_class_distribution():
    totals = [
        sum(DATASETS[ds][i] for ds in DATASETS)
        for i in range(len(CLASSES))
    ]
    y_max = max(totals) * 1.14
    px_per_unit = PLOT_H_PX / y_max

    fig = go.Figure()

    for ds_name, counts in DATASETS.items():
        color = COLORS[ds_name]
        text = [
            f"{c:,}" if c > 0 and c * px_per_unit >= MIN_LABEL_PX else ""
            for c in counts
        ]
        fig.add_trace(go.Bar(
            name=ds_name,
            x=CLASSES,
            y=counts,
            text=text,
            textposition="inside",
            insidetextanchor="middle",
            marker_color=color,
            marker_line_color=darken(color),
            marker_line_width=0.8,
            textfont=dict(color="white", size=10, family=FONT),
        ))

    # Build annotations: total above every bar, plus a color-coded breakdown
    # of any segments too small to label inside.
    annotations = []
    for i, cls in enumerate(CLASSES):
        total = totals[i]

        # Collect hidden labels, top segment first (reversed dataset order).
        hidden = []
        for ds_name, counts in reversed(list(DATASETS.items())):
            count = counts[i]
            if count > 0 and count * px_per_unit < MIN_LABEL_PX:
                c = COLORS[ds_name]
                hidden.append(f"<span style='color:{c}'>{count:,}</span>")

        if hidden:
            text = f"<b>{total:,}</b><br>" + "<br>".join(hidden)
            font_size = 9.5
        else:
            text = f"<b>{total:,}</b>"
            font_size = 11

        annotations.append(dict(
            x=cls, y=total,
            text=text,
            showarrow=False,
            yanchor="bottom",
            yshift=6,
            font=dict(size=font_size, color=GRAY_HI, family=FONT),
            align="center",
        ))

    fig.update_layout(
        barmode="stack",
        bargap=0.25,
        template="simple_white",
        font=dict(family=FONT, size=12, color=GRAY_HI),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            title=dict(
                text="Number of Instances",
                font=dict(size=12, color=GRAY_LO),
                standoff=12,
            ),
            tickformat=",",
            tickfont=dict(size=11, color=GRAY_LO),
            showgrid=True,
            gridcolor="rgba(128,128,128,0.5)",
            griddash="dash",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            range=[0, y_max],
        ),
        xaxis=dict(
            title=None,
            tickfont=dict(size=13, color=GRAY_HI),
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticklen=0,
        ),
        legend=dict(
            title=dict(text="Dataset", font=dict(size=12, color=GRAY_HI)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e0e0e0",
            borderwidth=1,
            font=dict(size=12),
            x=0.99, xanchor="right",
            y=0.99, yanchor="top",
            traceorder="normal",
        ),
        uniformtext=dict(mode="hide", minsize=9),
        annotations=annotations,
        width=780,
        height=500,
        margin=dict(l=70, r=30, t=40, b=55),
    )

    fig.write_image("images/dataset_class_distribution.pdf")
    fig.write_image("images/dataset_class_distribution.png", scale=2)
    print("Saved: images/dataset_class_distribution.{pdf,png}")


# ── Split statistics ──────────────────────────────────────────────────────────
# Images and annotations per dataset per split.
# RGB counts are used where RGB and thermal differ.
# Update "Ours" once self-collection is complete.

SPLITS = ["Train", "Val", "Test", "Total"]

SPLIT_IMAGES = {
    #                   Train    Val    Test   Total
    "DroneVehicle":  [17_951,  1_469,  8_980, 28_400],
    # "FLIR ADAS v2":  [10_318,  1_085,  3_749, 15_152],
    "RTDOD":         [11_003,  1_221,  3_968, 16_192],
    "M3FD":          [ 2_905,    628,    667,  4_200],
    "3D-Net":        [     8485,      1889,      1889,      12263],  # placeholder
}

SPLIT_ANNOTATIONS = {
    #                   Train      Val     Test    Total
    "DroneVehicle":  [286_050,  22_463, 143_330, 451_843],
    # "FLIR ADAS v2":  [169_174,  16_909,  84_786, 270_869],
    "RTDOD":         [108_285,  14_160,  56_576, 179_021],
    "M3FD":          [ 24_052,   4_858,   5_497,  34_407],
    "3D-Net":        [      6389,       1403,       1268,       9060],  # placeholder
}

SPLIT_COLORS = {
    "Train": "#4C78A8",   # vega blue
    "Val":   "#54A24B",   # vega green
    "Test":  "#EECA3B",   # vega yellow
    "Total": "#BAB0AC",   # vega grey
}


# ── Plot: split statistics ─────────────────────────────────────────────────────
def plot_split_stats():
    from plotly.subplots import make_subplots

    datasets = list(SPLIT_IMAGES.keys())
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Images per Split", "Annotations per Split"),
    )

    shown_in_legend = set()

    for row, data_dict in enumerate([SPLIT_IMAGES, SPLIT_ANNOTATIONS], start=1):
        for split in SPLITS:
            color = SPLIT_COLORS[split]
            values = [data_dict[ds][SPLITS.index(split)] for ds in datasets]
            show_legend = split not in shown_in_legend
            if show_legend:
                shown_in_legend.add(split)
            fig.add_trace(
                go.Bar(
                    name=split,
                    x=datasets,
                    y=values,
                    marker_color=color,
                    marker_line_color=darken(color),
                    marker_line_width=0.8,
                    text=[f"{v:,}" if v > 0 else "" for v in values],
                    textposition="outside",
                    textfont=dict(size=9, color=GRAY_HI, family=FONT),
                    legendgroup=split,
                    showlegend=show_legend,
                ),
                row=row, col=1,
            )

    axis_style = dict(
        tickformat=",",
        tickfont=dict(size=11, color=GRAY_LO),
        title_font=dict(size=12, color=GRAY_LO),
        showgrid=True,
        gridcolor="rgba(128,128,128,0.5)",
        griddash="dash",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )

    fig.update_layout(
        barmode="group",
        bargap=0.20,
        bargroupgap=0.06,
        template="simple_white",
        font=dict(family=FONT, size=12, color=GRAY_HI),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title=dict(text="Split", font=dict(size=12, color=GRAY_HI)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e0e0e0",
            borderwidth=1,
            font=dict(size=12),
            orientation="v",
            x=0.99, xanchor="right",
            y=0.89, yanchor="top",
            traceorder="normal",
        ),
        width=900,
        height=680,
        margin=dict(l=70, r=30, t=50, b=55),
    )

    fig.update_yaxes(**axis_style, title_text="Images", row=1, col=1)
    fig.update_yaxes(**axis_style, title_text="Annotations", row=2, col=1)
    fig.update_xaxes(
        tickfont=dict(size=13, color=GRAY_HI),
        showline=True, linecolor="black", linewidth=1, mirror=True,
        ticklen=0,
        row=2, col=1,
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color=GRAY_HI, family=FONT)

    fig.write_image("images/dataset_split_stats.pdf")
    fig.write_image("images/dataset_split_stats.png", scale=2)
    print("Saved: images/dataset_split_stats.{pdf,png}")


# ── Combined plot ─────────────────────────────────────────────────────────────
def plot_combined():
    from plotly.subplots import make_subplots

    # Left col: images (row 1) + annotations (row 2); right col: class dist (rows 1+2)
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        column_widths=[0.52, 0.48],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        specs=[
            [{"type": "bar"}, {"type": "bar", "rowspan": 2}],
            [{"type": "bar"}, None],
        ],
        subplot_titles=(
            "Images per Split", "Class Distribution",
            "Annotations per Split", "",
        ),
    )

    axis_style = dict(
        tickformat=",",
        tickfont=dict(size=10, color=GRAY_LO),
        title_font=dict(size=11, color=GRAY_LO),
        showgrid=True,
        gridcolor="rgba(128,128,128,0.5)",
        griddash="dash",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )

    # ── Left: split stats ──────────────────────────────────────────────────────
    datasets = list(SPLIT_IMAGES.keys())
    shown_in_legend = set()

    img_max  = max(v for ds in SPLIT_IMAGES.values()      for v in ds)
    ann_max  = max(v for ds in SPLIT_ANNOTATIONS.values() for v in ds)

    for row, data_dict in enumerate([SPLIT_IMAGES, SPLIT_ANNOTATIONS], start=1):
        for split in SPLITS:
            color = SPLIT_COLORS[split]
            values = [data_dict[ds][SPLITS.index(split)] for ds in datasets]
            show_legend = split not in shown_in_legend
            if show_legend:
                shown_in_legend.add(split)
            fig.add_trace(
                go.Bar(
                    name=split,
                    x=datasets,
                    y=values,
                    marker_color=color,
                    marker_line_color=darken(color),
                    marker_line_width=0.8,
                    text=[f"{v:,}" if v > 0 else "" for v in values],
                    textposition="outside",
                    textfont=dict(size=8, color=GRAY_HI, family=FONT),
                    legendgroup=split,
                    legend="legend2",
                    showlegend=show_legend,
                ),
                row=row, col=1,
            )

    fig.update_yaxes(**axis_style, title_text="Images",
                     range=[0, img_max * 1.22], row=1, col=1)
    fig.update_yaxes(**axis_style, title_text="Annotations",
                     range=[0, ann_max * 1.22], row=2, col=1)
    fig.update_xaxes(
        tickfont=dict(size=11, color=GRAY_HI), showline=True,
        linecolor="black", linewidth=1, mirror=True, ticklen=0, row=2, col=1,
    )
    fig.update_xaxes(
        tickfont=dict(size=11, color=GRAY_HI), showline=True,
        linecolor="black", linewidth=1, mirror=True, ticklen=0, row=1, col=1,
    )

    # ── Right: class distribution ──────────────────────────────────────────────
    totals = [
        sum(DATASETS[ds][i] for ds in DATASETS)
        for i in range(len(CLASSES))
    ]
    y_max_cls = max(totals) * 1.18
    # Right subplot spans full height: 640px - top/bottom margins - title row ≈ 490px
    px_per_unit = 490 / y_max_cls
    min_label_px_cls = 10  # lower threshold so Bus/Truck DroneVehicle segments get labelled

    for ds_name, counts in DATASETS.items():
        color = COLORS[ds_name]
        text = [
            f"{c:,}" if c > 0 and c * px_per_unit >= min_label_px_cls else ""
            for c in counts
        ]
        fig.add_trace(
            go.Bar(
                name=ds_name,
                x=CLASSES,
                y=counts,
                text=text,
                textposition="inside",
                insidetextanchor="middle",
                marker_color=color,
                marker_line_color=darken(color),
                marker_line_width=0.8,
                textfont=dict(color="white", size=9, family=FONT),
                legendgroup=ds_name,
                legend="legend",
                showlegend=True,
            ),
            row=1, col=2,
        )

    # Totals above class bars
    class_annotations = []
    for i, cls in enumerate(CLASSES):
        total = totals[i]
        hidden = []
        for ds_name, counts in reversed(list(DATASETS.items())):
            count = counts[i]
            if count > 0 and count * px_per_unit < min_label_px_cls:
                c = COLORS[ds_name]
                hidden.append(f"<span style='color:{c}'>{count:,}</span>")
        if hidden:
            label = f"<b>{total:,}</b><br>" + "<br>".join(hidden)
            fsize = 8.5
        else:
            label = f"<b>{total:,}</b>"
            fsize = 10
        class_annotations.append(dict(
            x=cls, y=total,
            xref="x2", yref="y2",
            text=label,
            showarrow=False,
            yanchor="bottom",
            yshift=5,
            font=dict(size=fsize, color=GRAY_HI, family=FONT),
            align="center",
        ))

    fig.update_yaxes(
        **axis_style,
        title_text="Instances",
        range=[0, y_max_cls],
        row=1, col=2,
    )
    fig.update_xaxes(
        tickfont=dict(size=11, color=GRAY_HI), showline=True,
        linecolor="black", linewidth=1, mirror=True, ticklen=0, row=1, col=2,
    )

    legend_style = dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#e0e0e0",
        borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        barmode="stack",
        bargap=0.22,
        bargroupgap=0.05,
        template="simple_white",
        font=dict(family=FONT, size=12, color=GRAY_HI),
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Dataset legend — top-right of class distribution panel
        legend=dict(**legend_style,
            title=dict(text="Dataset", font=dict(size=11, color=GRAY_HI)),
            x=1.0, xanchor="right",
            y=1.0, yanchor="top",
        ),
        # Split legend — inside left panel, top-right of images subplot
        legend2=dict(**legend_style,
            title=dict(text="Split", font=dict(size=11, color=GRAY_HI)),
            x=0.49, xanchor="right",
            y=0.91, yanchor="top",
        ),
        width=1280,
        height=640,
        margin=dict(l=65, r=30, t=55, b=55),
    )

    # Add class distribution annotations AFTER update_layout so they are
    # appended rather than merged into the subplot title annotation slots.
    for ann in class_annotations:
        fig.add_annotation(**ann)

    # Override barmode to "group" only for left subplots — not possible directly,
    # so set the stacked mode globally and patch the left traces to use offsetgroup.
    for trace in fig.data:
        if trace.xaxis == "x" or trace.xaxis == "x3":
            trace.update(offsetgroup=trace.name)

    for ann in fig.layout.annotations:
        if ann.text in ("Images per Split", "Annotations per Split", "Class Distribution", ""):
            ann.font = dict(size=13, color=GRAY_HI, family=FONT)

    fig.write_image("images/dataset_combined.pdf")
    fig.write_image("images/dataset_combined.png", scale=2)
    print("Saved: images/dataset_combined.{pdf,png}")


if __name__ == "__main__":
    plot_class_distribution()
    plot_split_stats()
    plot_combined()
