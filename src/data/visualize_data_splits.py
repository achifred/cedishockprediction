"""
Visualizes the chronological train / validation / test splits used for the
GHS/USD shock prediction models, plus a schematic of how the LSTM's 3D
input sequences (lookback window -> forecast horizon) are constructed.

Deliberately excludes the walk-forward CV folds -- those were used for
experimentation only and are not part of the reported methodology.

Output: data_splits_and_sequence.png
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch
import pandas as pd

from src.config.config import VISUALIZE_DATA_SPLITS_PLOT_FILE
from src.utils.utils import get_path

# ---------------------------------------------------------------
# Configuration -- edit here if the split dates or LSTM parameters change
# ---------------------------------------------------------------
SPLITS = [
    {"name": "Train",      "start": "2010-01-01", "end": "2019-12-31", "color": "#4C72B0"},
    {"name": "Validation", "start": "2020-01-01", "end": "2023-12-31", "color": "#DD8452"},
    {"name": "Test",       "start": "2024-01-01", "end": "2025-12-31", "color": "#55A868"},
]

LOOKBACK_WEEKS = 8
FORECAST_HORIZON_MIN_WEEKS = 2
FORECAST_HORIZON_MAX_WEEKS = 4




def draw_split_timeline(ax):
    bar_height = 0.6
    y_center = 0

    for split in SPLITS:
        start = pd.Timestamp(split["start"])
        end = pd.Timestamp(split["end"])
        width_days = (end - start).days

        ax.add_patch(Rectangle(
            (mdates.date2num(start), y_center - bar_height / 2),
            width_days, bar_height,
            facecolor=split["color"], edgecolor="black", linewidth=1.0, alpha=0.9,
        ))

        midpoint = start + (end - start) / 2
        ax.text(
            mdates.date2num(midpoint), y_center, split["name"],
            ha="center", va="center", fontsize=12, fontweight="bold", color="white",
        )

        # start/end date labels below the bar -- staggered vertically so the
        # end-of-one-split / start-of-next-split labels (one day apart) don't
        # collide horizontally
        label_specs = [(start, "left", -0.42), (end, "right", -0.62)]
        for date_point, label_align, y_offset in label_specs:
            ax.text(
                mdates.date2num(date_point), y_center + y_offset,
                date_point.strftime("%Y-%m-%d"),
                ha=label_align, va="top", fontsize=8.5, color="#333333",
            )

    overall_start = pd.Timestamp(SPLITS[0]["start"])
    overall_end = pd.Timestamp(SPLITS[-1]["end"])
    ax.set_xlim(
        mdates.date2num(overall_start) - 60,
        mdates.date2num(overall_end) + 60,
    )
    ax.set_ylim(-1.6, 1.0)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Chronological Train / Validation / Test Split (Weekly Data, 2010\u20132025)",
                 fontsize=13, fontweight="bold", pad=14)

    # boundary markers between splits
    for split in SPLITS[1:]:
        boundary = mdates.date2num(pd.Timestamp(split["start"]))
        ax.axvline(boundary, color="#777777", linestyle="--", linewidth=0.8, ymin=0.08, ymax=0.92)


def draw_sequence_schematic(ax):
    """
    Generic schematic (relative week indices, not tied to a specific
    calendar date) showing: a LOOKBACK_WEEKS-wide input window ending at
    a reference week t, a gap representing the forecast horizon start,
    and the FORECAST_HORIZON_MIN..MAX_WEEKS window in which the shock
    target is evaluated.
    """
    week_width = 1.0
    bar_y = 0
    bar_height = 0.5

    lookback_start = -LOOKBACK_WEEKS
    lookback_end = 0  # reference week t (inclusive)

    # lookback window (input sequence)
    ax.add_patch(Rectangle(
        (lookback_start, bar_y - bar_height / 2), LOOKBACK_WEEKS, bar_height,
        facecolor="#4C72B0", edgecolor="black", linewidth=1.0, alpha=0.85,
    ))
    ax.text(
        (lookback_start + lookback_end) / 2, bar_y, f"Lookback window\n({LOOKBACK_WEEKS} weeks of features)",
        ha="center", va="center", fontsize=9.5, color="white", fontweight="bold",
    )

    # reference week marker (t)
    ax.axvline(lookback_end, color="black", linestyle="-", linewidth=1.2, ymin=0.15, ymax=0.85)
    ax.text(lookback_end, bar_y - bar_height / 2 - 0.35, "week t\n(prediction made here)",
            ha="center", va="top", fontsize=8.5)

    # forecast horizon window (target evaluation)
    horizon_start = FORECAST_HORIZON_MIN_WEEKS
    horizon_width = FORECAST_HORIZON_MAX_WEEKS - FORECAST_HORIZON_MIN_WEEKS + 1
    ax.add_patch(Rectangle(
        (horizon_start, bar_y - bar_height / 2), horizon_width, bar_height,
        facecolor="#C44E52", edgecolor="black", linewidth=1.0, alpha=0.85, hatch="///",
    ))
    ax.text(
        horizon_start + horizon_width / 2, bar_y,
        f"Forecast horizon\n(weeks t+{FORECAST_HORIZON_MIN_WEEKS} to t+{FORECAST_HORIZON_MAX_WEEKS})",
        ha="center", va="center", fontsize=9.5, color="white", fontweight="bold",
    )

    # arrow from reference week to the forecast horizon
    arrow = FancyArrowPatch(
        (lookback_end + 0.15, bar_y), (horizon_start - 0.15, bar_y),
        arrowstyle="-|>", mutation_scale=14, color="#333333", linewidth=1.3,
        connectionstyle="arc3,rad=-0.3",
    )
    ax.add_patch(arrow)
    ax.text((lookback_end + horizon_start) / 2, bar_y + 0.55, "shock target\nevaluated over\nthis window",
            ha="center", va="bottom", fontsize=8.5, style="italic", color="#444444")

    ax.set_xlim(lookback_start - 1.5, horizon_start + horizon_width + 1.5)
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title("Input Sequence Construction (3D Sequence \u2192 Target Window)",
                 fontsize=12, fontweight="bold", pad=12)


def visualize_data_splits():
    fig, (top_axis, bottom_axis) = plt.subplots(
        2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios": [1.1, 1]}
    )
    fig.suptitle(
        "Figure: Dataset Splitting Strategy and Sequence Construction",
        fontsize=14, fontweight="bold", y=0.98,
    )

    draw_split_timeline(top_axis)
    draw_sequence_schematic(bottom_axis)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(get_path(VISUALIZE_DATA_SPLITS_PLOT_FILE), dpi=300, bbox_inches="tight")
    print(f"Saved: {VISUALIZE_DATA_SPLITS_PLOT_FILE}")

