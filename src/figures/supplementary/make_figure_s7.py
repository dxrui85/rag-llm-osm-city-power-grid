from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
FIG = ROOT / "figure_assets"
QA = ROOT / "qa"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S7"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
FIG = OUTPUT_ROOT / "Figure_S7"
QA = OUTPUT_ROOT / "Figure_S7_qa"

BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
SLATE = "#8796A5"
DARK_GREY = "#5B6573"
GRID = "#D9DEE5"
TEXT = "#20242A"

CLASSES = ["ACLineSegment", "ConnectivityNode", "EnergySource"]
CLASS_LABELS = ["Power lines", "Connection points", "Power generation facilities"]
OUTCOMES = ["Correct", "Wrong class", "Insufficient", "INVALID", "MISSING"]
OUTCOME_LABELS = ["Correct", "Wrong class", "Insufficient\nevidence", "Invalid\noutput", "Missing\nresponse"]
DISPOSITIONS = ["Retained correct", "Rescued to correct", "Degraded from correct", "Unresolved / other"]
DISPOSITION_LABELS = {
    "Retained correct": "Retained correct",
    "Rescued to correct": "Corrected by H2",
    "Degraded from correct": "Degraded from correct",
    "Unresolved / other": "Unresolved or other",
}
DISPOSITION_COLORS = {
    "Retained correct": BLUE,
    "Rescued to correct": LIGHT_BLUE,
    "Degraded from correct": ORANGE,
    "Unresolved / other": SLATE,
}
ORIGIN_COLORS = {
    "Insufficient": LIGHT_BLUE,
    "Wrong class": ORANGE,
    "INVALID": SLATE,
    "MISSING": DARK_GREY,
}
ORIGIN_LABELS = {
    "Insufficient": "Insufficient evidence",
    "Wrong class": "Wrong class",
    "INVALID": "Invalid output",
    "MISSING": "Missing response",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 11.3,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.edgecolor": "#7B8491",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.105, 1.07, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def grid_axis(ax: plt.Axes, axis: str) -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def make_figure() -> Path:
    configure()
    transitions = pd.read_csv(DATA / "paired_outcome_transition_matrix.csv")
    dispositions = pd.read_csv(DATA / "paired_disposition_by_gold_class.csv")
    rescued = pd.read_csv(DATA / "rescued_pair_origins_by_gold_class.csv")
    balance = pd.read_csv(DATA / "rescue_degradation_balance_by_gold_class.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    fig.subplots_adjust(left=0.095, right=0.97, top=0.90, bottom=0.14, wspace=0.34, hspace=0.47)
    fig.suptitle("Paired prediction transitions from H1 LLM to H2 RAG-LLM", fontsize=15.5, fontweight="bold")

    # a | Transition matrix
    ax = axes[0, 0]
    matrix = (
        transitions.pivot(index="h1_outcome", columns="h2_outcome", values="row_percent")
        .reindex(index=OUTCOMES, columns=OUTCOMES)
        .to_numpy()
    )
    heatmap_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "main_blue", ["#F7FBFF", "#C6E4F3", LIGHT_BLUE, BLUE, "#004B78"]
    )
    image = ax.imshow(matrix, cmap=heatmap_cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(5), OUTCOME_LABELS)
    ax.set_yticks(range(5), OUTCOME_LABELS)
    ax.set_xlabel("H2 RAG-LLM outcome")
    ax.set_ylabel("H1 LLM outcome")
    ax.set_title("Within-pair outcome transition matrix", loc="left")
    panel_label(ax, "a")
    for i in range(5):
        for j in range(5):
            value = matrix[i, j]
            if value > 0:
                label = "<0.1" if value < 0.1 else f"{value:.1f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold" if value > 20 else "normal",
                    color="white" if value > 55 else TEXT,
                )
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.18, aspect=28)
    cbar.set_label("Share within each H1 LLM outcome (%)")
    cbar.outline.set_linewidth(0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # b | Paired dispositions by facility class
    ax = axes[0, 1]
    x = np.arange(3)
    bottom = np.zeros(3)
    for state in DISPOSITIONS:
        values = (
            dispositions[dispositions["paired_disposition"] == state]
            .set_index("gold_class")
            .reindex(CLASSES)["percent"]
            .to_numpy()
        )
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.48,
            label=DISPOSITION_LABELS[state],
            color=DISPOSITION_COLORS[state],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        bottom += values
    ax.set_xticks(x, CLASS_LABELS)
    ax.set_ylim(60, 101.5)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_ylabel("Paired positions (%)")
    ax.set_title("Paired outcomes by facility class", loc="left")
    panel_label(ax, "b")
    grid_axis(ax, "y")
    for i, facility_class in enumerate(CLASSES):
        retained = float(
            dispositions[
                (dispositions["gold_class"] == facility_class)
                & (dispositions["paired_disposition"] == "Retained correct")
            ]["percent"].iloc[0]
        )
        corrected = float(
            dispositions[
                (dispositions["gold_class"] == facility_class)
                & (dispositions["paired_disposition"] == "Rescued to correct")
            ]["percent"].iloc[0]
        )
        if retained > 60:
            visible_midpoint = 60 + (retained - 60) / 2
            ax.text(i, visible_midpoint, f"{retained:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if corrected > 6:
            ax.text(i, retained + corrected / 2, f"+{corrected:.1f}%", ha="center", va="center", color=TEXT, fontweight="bold")
    # Make the non-zero baseline explicit for the truncated panel-b axis.
    break_style_b = dict(transform=ax.transAxes, color="#58616D", clip_on=False, linewidth=1.0)
    ax.plot((-0.012, 0.012), (-0.012, 0.012), **break_style_b)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)

    # c | Origin of corrected positions
    ax = axes[1, 0]
    origins = ["Insufficient", "Wrong class", "INVALID", "MISSING"]
    x = np.arange(3)
    bottom = np.zeros(3)
    for origin in origins:
        values = (
            rescued[rescued["h1_outcome"] == origin]
            .set_index("gold_class")
            .reindex(CLASSES)["percent_of_rescues"]
            .to_numpy()
        )
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.48,
            label=ORIGIN_LABELS[origin],
            color=ORIGIN_COLORS[origin],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        bottom += values
    ax.set_xticks(x, CLASS_LABELS)
    ax.set_ylim(90, 101.5)
    ax.set_yticks([90, 92, 94, 96, 98, 100])
    ax.set_ylabel("Share of corrected pairs (%)")
    ax.set_title("H1 LLM origins of positions corrected by H2", loc="left")
    panel_label(ax, "c")
    grid_axis(ax, "y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)
    for i, facility_class in enumerate(CLASSES):
        n = int(rescued[rescued["gold_class"] == facility_class]["rescued_class_denominator"].iloc[0])
        if n > 0:
            ax.text(i, 100.65, f"n={n:,}", ha="center", va="center", fontsize=8.2, color=TEXT, fontweight="bold")
        else:
            ax.text(i, 95.5, "No corrected\npairs", ha="center", va="center", fontsize=9, color=DARK_GREY)
    # Make the non-zero baseline explicit for the truncated panel-c axis.
    break_style = dict(transform=ax.transAxes, color="#58616D", clip_on=False, linewidth=1.0)
    ax.plot((-0.012, 0.012), (-0.012, 0.012), **break_style)

    # d | Correction-degradation balance
    ax = axes[1, 1]
    balanced = balance.set_index("gold_class").reindex(CLASSES)
    y = np.arange(3)
    ax.barh(
        y,
        balanced["rescued_percent"],
        color=BLUE,
        height=0.34,
        label="Corrected by H2",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.barh(
        y,
        -balanced["degraded_percent"],
        color=ORANGE,
        height=0.34,
        label="Degraded from correct",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    negative_limit = 2.5
    positive_limit = 40.0
    ax.set_xlim(-negative_limit, positive_limit)
    ax.set_xticks([-2, 0, 10, 20, 30, 40])
    ax.axvline(0, color="#3F4650", linewidth=0.8)
    ax.set_yticks(y, CLASS_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Share of class-paired positions (%)")
    ax.set_title("Correction-degradation balance", loc="left")
    panel_label(ax, "d")
    grid_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)
    for i, facility_class in enumerate(CLASSES):
        correction = float(balanced.loc[facility_class, "rescued_percent"])
        degradation = float(balanced.loc[facility_class, "degraded_percent"])
        net = float(balanced.loc[facility_class, "net_correct_gain_percentage_points"])
        if correction >= 3:
            ax.text(correction - 0.6, i, f"+{correction:.1f}", va="center", ha="right", color="white", fontweight="bold")
        else:
            ax.text(correction + 0.35, i, f"+{correction:.1f}", va="center", ha="left")
        if degradation > 0:
            ax.text(-degradation - 0.12, i, f"−{degradation:.1f}", va="center", ha="right")
        net_x = correction + 0.8 if correction > 0 else 1.2
        ax.text(net_x, i - 0.18, f"net {net:+.1f} pp", va="center", ha="left", fontweight="bold", fontsize=8.5)

    output = FIG / "Figure_S7_paired_prediction_transitions_revised"
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    with Image.open(output.with_suffix(".png")) as image_file:
        rgb = image_file.convert("RGB")
        rgb.save(output.with_suffix(".tiff"), format="TIFF", compression="raw", dpi=(300, 300))
        ImageOps.grayscale(rgb).save(QA / "Figure_S7_grayscale_preview.png")
    return output


def write_audit(output: Path) -> None:
    source_files = sorted(DATA.glob("*.csv"))
    output_files = [output.with_suffix(ext) for ext in [".png", ".pdf", ".svg", ".tiff"]]
    audit = {
        "figure": "Figure S7",
        "revision_date": "2026-11-26",
        "core_conclusion": "Paired H1-to-H2 transitions are dominated by retained or corrected predictions, with class-specific differences.",
        "archetype": "quantitative grid",
        "changes": [
            "Aligned all four panels with the main-figure blue, light-blue, orange, and slate palette.",
            "Removed every hatch and texture pattern from panels b, c, and d.",
            "Used solid fills with white boundaries for stacked bars.",
            "Removed both bottom figure notes.",
            "Replaced reader-visible internal all-caps failure labels with normal descriptive wording.",
            "Retained all frozen values without recomputation.",
            "Narrowed the panel-b and panel-c bars to match the visual bar width used in the surrounding supplementary figures.",
            "Truncated the panel-b y-axis to 60-101.5% so that near-complete stacked totals remain distinguishable.",
            "Truncated the panel-c y-axis to 90-101.5% so that small outcome shares near 100% remain visually distinguishable.",
            "Reduced the unused negative range in panel d to -2.5% while retaining the positive range to 40%, and repositioned value labels.",
        ],
        "input_sha256": {path.name: sha256(path) for path in source_files},
        "output_sha256": {path.name: sha256(path) for path in output_files},
    }
    (QA / "Figure_S7_revision_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "README.md").write_text(
        "# Figure S7 revision\n\n"
        "This standalone package truncates the panel-b y-axis and removes the unused negative range in panel d. "
        "The frozen source tables are unchanged. The revised figure has not yet been integrated "
        "into the Supplementary Information Word file.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    write_audit(figure)
    print(figure.with_suffix(".png"))
