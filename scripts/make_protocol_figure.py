#!/usr/bin/env python3
"""
Generate a compact dataset and evaluation protocol figure for the IJCB paper.
Outputs both PNG and PDF to the project-level and LaTeX figure directories.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from release_utils import FIGURES_DIR

OUT_DIRS = [FIGURES_DIR]

COLORS = {
    "neutral": "#4C566A",
    "blue": "#2E86AB",
    "green": "#2A9D8F",
    "orange": "#E07A5F",
    "gold": "#E9C46A",
    "purple": "#7B6DCC",
    "gray_fill": "#F4F6F8",
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


def add_box(ax, x, y, w, h, title, lines, edgecolor, facecolor, title_fs=8.3, body_fs=6.6):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.08 * w,
        y + h - 0.17,
        title,
        ha="left",
        va="top",
        fontsize=title_fs,
        fontweight="bold",
        color=edgecolor,
    )
    ax.text(
        x + 0.08 * w,
        y + h - 0.38,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=body_fs,
        color=COLORS["neutral"],
        linespacing=1.3,
    )


def add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="#6B7280",
            shrinkA=2,
            shrinkB=2,
        )
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

    fig, ax = plt.subplots(figsize=(5.5, 3.45))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.25)
    ax.axis("off")

    add_box(
        ax,
        0.3,
        2.55,
        2.45,
        1.22,
        "Dataset",
        [
            "10 subjects, 2 runs each",
            "Naturalistic movie viewing",
            "49 tri-axial OPM sensors",
            "600 s per run",
        ],
        COLORS["blue"],
        "#EAF4FB",
    )

    add_box(
        ax,
        3.2,
        2.55,
        2.25,
        1.22,
        "Preprocessing",
        [
            "Band-pass filter",
            "Resample to 200 Hz",
            "Artifact rejection",
            "5 s non-overlap windows",
        ],
        COLORS["green"],
        "#EAF8F5",
    )

    add_box(
        ax,
        5.95,
        2.55,
        3.95,
        1.22,
        "Feature / model branch",
        [
            "LogVar / PSD directly",
            "or z-score -> tri-branch LF-Net",
            "Window-level scores / labels",
        ],
        COLORS["orange"],
        "#FCF0EC",
    )

    add_arrow(ax, 2.75, 3.14, 3.2, 3.14)
    add_arrow(ax, 5.45, 3.14, 5.95, 3.14)

    add_box(
        ax,
        0.45,
        0.34,
        2.75,
        1.18,
        "Closed-set identification",
        [
            "run-1 train -> run-2 test",
            "Swap directions",
            "Mean subject-ID accuracy",
        ],
        COLORS["purple"],
        "#F2EEFD",
        title_fs=7.7,
        body_fs=6.2,
    )

    add_box(
        ax,
        3.65,
        0.34,
        2.7,
        1.18,
        "Verification / CMC",
        [
            "run-1 subject templates",
            "run-2 probe windows",
            "Cosine -> ROC / DET / CMC",
        ],
        COLORS["gold"],
        "#FCF6E6",
        title_fs=7.7,
        body_fs=6.2,
    )

    add_box(
        ax,
        6.8,
        0.34,
        2.7,
        1.18,
        "Open-set protocol",
        [
            "1-3 subjects as unknown",
            "Threshold from run-1 gallery",
            "20 random splits",
            "Known acc. + reject rate",
        ],
        COLORS["neutral"],
        COLORS["gray_fill"],
        title_fs=7.7,
        body_fs=6.2,
    )

    add_arrow(ax, 6.75, 2.55, 1.85, 1.54)
    add_arrow(ax, 7.3, 2.55, 5.0, 1.54)
    add_arrow(ax, 8.0, 2.55, 8.1, 1.54)

    ax.text(
        5.1,
        4.08,
        "OPM-MEG dataset and cross-run biometric evaluation protocol",
        ha="center",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["neutral"],
    )

    save_all(fig, "protocol")


if __name__ == "__main__":
    main()
