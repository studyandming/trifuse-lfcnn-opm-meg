#!/usr/bin/env python3
"""
Generate a compact explanatory figure clarifying the difference between
triaxial OPM signal channels and static geometry metadata.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from release_utils import FIGURES_DIR

OUT_DIRS = [FIGURES_DIR]

COLORS = {
    "neutral": "#4C566A",
    "blue": "#2E86AB",
    "green": "#2A9D8F",
    "orange": "#E07A5F",
    "gold": "#E9C46A",
    "gray_fill": "#F4F6F8",
    "light_blue": "#EAF4FB",
    "light_green": "#EAF8F5",
    "light_orange": "#FCF0EC",
}


def save_all(fig, name):
    for out_dir in OUT_DIRS:
        os.makedirs(str(out_dir), exist_ok=True)
        fig.savefig(
            str(Path(out_dir) / f"{name}.png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        fig.savefig(
            str(Path(out_dir) / f"{name}.pdf"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def add_round_box(ax, x, y, w, h, title, body, edge, face, title_fs=8.6, body_fs=6.7):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.08 * w,
        y + h - 0.16,
        title,
        ha="left",
        va="top",
        fontsize=title_fs,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.08 * w,
        y + h - 0.38,
        body,
        ha="left",
        va="top",
        fontsize=body_fs,
        color=COLORS["neutral"],
        linespacing=1.3,
    )


def add_arrow(ax, start, end, color, text=None, text_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if text and text_xy:
        ax.text(
            text_xy[0],
            text_xy[1],
            text,
            fontsize=7.1,
            color=color,
            fontweight="bold",
            ha="center",
            va="center",
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    ax.text(
        5.3,
        4.63,
        "One triaxial OPM sensor: signal channels versus geometry metadata",
        ha="center",
        va="top",
        fontsize=9.7,
        fontweight="bold",
        color=COLORS["neutral"],
    )

    # Left: intuitive sensor view on the scalp.
    scalp = Circle((2.15, 2.25), 1.0, facecolor="#FFF7E8", edgecolor="#D7B98E", linewidth=1.2)
    ax.add_patch(scalp)
    sensor = Rectangle((1.92, 2.15), 0.46, 0.34, facecolor=COLORS["blue"], edgecolor=COLORS["blue"], linewidth=1.1)
    ax.add_patch(sensor)
    ax.text(2.15, 2.32, "OPM", ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")

    add_arrow(ax, (2.15, 2.32), (3.05, 2.95), COLORS["blue"], r"$s_X(t)$", (3.35, 3.08))
    add_arrow(ax, (2.15, 2.32), (1.15, 3.12), COLORS["green"], r"$s_Y(t)$", (0.92, 3.25))
    add_arrow(ax, (2.15, 2.32), (2.15, 3.72), COLORS["orange"], r"$s_Z(t)$", (2.15, 3.96))
    ax.text(
        2.15,
        1.0,
        "Three time-series channels are recorded\nfrom one physical triaxial sensor.",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["neutral"],
    )

    # Middle: channels.tsv interpretation.
    add_round_box(
        ax,
        3.65,
        1.1,
        3.2,
        2.55,
        "What `channels.tsv` stores",
        "For each axis row, the file contains:\n"
        "1. Sensor position  p = (Px, Py, Pz)\n"
        "2. Axis orientation  o = (Ox, Oy, Oz)\n\n"
        "These are static geometry metadata,\nnot extra neural channels.",
        COLORS["orange"],
        COLORS["light_orange"],
    )

    ax.text(5.25, 0.58, "Same sensor, three axis rows:\n`LR[X]`, `LR[Y]`, `LR[Z]`", ha="center", va="center",
            fontsize=6.7, color=COLORS["neutral"])

    # Right: table-like breakdown.
    add_round_box(
        ax,
        7.25,
        2.2,
        2.85,
        1.45,
        "Per-axis row",
        "`LR [X]`: position + X-axis direction\n"
        "`LR [Y]`: same position + Y direction\n"
        "`LR [Z]`: same position + Z direction",
        COLORS["green"],
        COLORS["light_green"],
        title_fs=8.2,
        body_fs=6.35,
    )

    add_round_box(
        ax,
        7.25,
        0.72,
        2.85,
        1.08,
        "Key distinction",
        "Signal channels: 3 measured magnetic-field\ncomponents.\n"
        "Geometry metadata: where the sensor sits\nand how each axis points.",
        COLORS["blue"],
        COLORS["light_blue"],
        title_fs=8.2,
        body_fs=6.15,
    )

    # Arrows linking panels.
    ax.add_patch(FancyArrowPatch((3.0, 2.25), (3.65, 2.25), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.2, color="#6B7280"))
    ax.add_patch(FancyArrowPatch((6.85, 2.92), (7.25, 2.92), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.2, color="#6B7280"))
    ax.add_patch(FancyArrowPatch((6.85, 1.45), (7.25, 1.25), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.2, color="#6B7280"))

    save_all(fig, "sensor_geometry_control")


if __name__ == "__main__":
    main()
