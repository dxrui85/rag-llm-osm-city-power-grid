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
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S6"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
FIG = OUTPUT_ROOT / "Figure_S6"
QA = OUTPUT_ROOT / "Figure_S6_qa"

BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
PALE_ORANGE = "#E69F00"
SLATE_BLUE = "#8796A5"
DARK_GREY = "#5B6573"
GRID = "#D9DEE5"
TEXT = "#20242A"

OUTCOME_COLORS = {
    "Correct": BLUE,
    "Wrong class": ORANGE,
    "Insufficient": LIGHT_BLUE,
    "INVALID": SLATE_BLUE,
    "MISSING": DARK_GREY,
}
OUTCOME_LABELS = {
    "Correct": "Correct",
    "Wrong class": "Wrong class",
    "Insufficient": "Insufficient evidence",
    "INVALID": "Invalid output",
    "MISSING": "Missing response",
}
OUTCOMES = list(OUTCOME_COLORS)
RULE_STATES = ["MATCHED", "RULE_NONE"]
METHODS = ["H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]


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
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.1,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#7B8491",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#8A939F")
    ax.spines["bottom"].set_color("#8A939F")


def outcome_values(counts: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows = [
        ("MATCHED", "H1_no_retrieval_LLM"),
        ("MATCHED", "H2_KB_RAG_LLM"),
        ("RULE_NONE", "H1_no_retrieval_LLM"),
        ("RULE_NONE", "H2_KB_RAG_LLM"),
    ]
    matrix = []
    for state, method in rows:
        row = []
        for outcome in OUTCOMES:
            selected = counts[
                (counts["rule_match_state"] == state)
                & (counts["method"] == method)
                & (counts["outcome"] == outcome)
            ]
            row.append(float(selected["percent"].iloc[0]))
        matrix.append(row)
    labels = [
        "Applicable rule\nH1 LLM",
        "Applicable rule\nH2 RAG-LLM",
        "No applicable rule\nH1 LLM",
        "No applicable rule\nH2 RAG-LLM",
    ]
    return np.asarray(matrix), labels


def draw_stacked(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    zoom: bool = False,
) -> None:
    x = np.arange(matrix.shape[0])
    bottoms = np.zeros(matrix.shape[0])
    for j, outcome in enumerate(OUTCOMES):
        values = matrix[:, j]
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.7,
            width=0.66,
            label=OUTCOME_LABELS[outcome],
            zorder=3,
        )
        bottoms += values
    ax.set_xticks(x)
    if zoom:
        ax.set_xticklabels(["Rule\nH1", "Rule\nH2", "No rule\nH1", "No rule\nH2"], fontsize=7.4)
        ax.set_ylim(98.4, 100.05)
        ax.set_yticks([98.5, 99.0, 99.5, 100.0])
        ax.set_yticklabels(["98.5", "99.0", "99.5", "100"])
        ax.set_title("Upper range enlarged", fontsize=8.5, loc="center", pad=4)
        ax.set_ylabel("Prediction positions (%)", labelpad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#737C88")
            spine.set_visible(True)
    else:
        ax.set_xticklabels(["H1\nLLM", "H2\nRAG-LLM", "H1\nLLM", "H2\nRAG-LLM"], fontsize=7.9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Prediction positions (%)")
        ax.set_title("Prediction outcomes by rule availability", loc="left")
        ax.text(
            0.5,
            -0.145,
            "Applicable rule",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            2.5,
            -0.145,
            "No applicable rule",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            clip_on=False,
        )
    style_axis(ax, "y")


def make_figure() -> Path:
    configure()
    coverage = pd.read_csv(DATA / "evidence_channel_coverage.csv")
    rule_comp = pd.read_csv(DATA / "rule_supported_class_composition.csv")
    counts = pd.read_csv(DATA / "prediction_outcomes_by_rule_applicability.csv")
    changes = pd.read_csv(DATA / "h2_minus_h1_pathway_changes_by_rule_applicability.csv")

    fig = plt.figure(figsize=(13.8, 9.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.97,
        top=0.91,
        bottom=0.12,
        wspace=0.30,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    panel_c_grid = grid[1, 0].subgridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.18)
    ax_c = fig.add_subplot(panel_c_grid[0, 0])
    ax_c_zoom = fig.add_subplot(panel_c_grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Integrated knowledge evidence and prediction pathways",
        fontsize=15,
        fontweight="bold",
        y=0.972,
    )

    cov = coverage.set_index("evidence_channel").loc[
        ["EX retrieved examples", "Shared TOPO context", "RULE evidence"]
    ]
    y = np.arange(3)
    bars = ax_a.barh(
        y,
        cov["coverage_percent"],
        color=[BLUE, LIGHT_BLUE, ORANGE],
        height=0.58,
        zorder=3,
    )
    ax_a.set_yticks(
        y,
        [
            "Retrieved labelled examples\n(H2 RAG-LLM only)",
            "Shared topology context\n(H1 LLM and H2 RAG-LLM)",
            "Retrieved power system rules\n(H2 RAG-LLM only)",
        ],
    )
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 108)
    ax_a.set_xlabel("Objects with available evidence (%)")
    ax_a.set_title("Evidence availability across study objects", loc="left")
    panel_label(ax_a, "a")
    style_axis(ax_a, "x")
    for bar, value in zip(bars, cov["coverage_percent"]):
        ax_a.text(
            min(float(value) + 1.3, 101.5),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    comp = rule_comp.set_index("rule_supported_class").loc[
        ["ACLineSegment", "ConnectivityNode", "EnergySource"]
    ]
    yy = np.arange(3)
    bars = ax_b.barh(yy, comp["object_count"], color=[BLUE, LIGHT_BLUE, ORANGE], height=0.58, zorder=3)
    ax_b.set_yticks(yy, ["Power lines", "Connection points", "Power generation facilities"])
    ax_b.invert_yaxis()
    ax_b.set_xscale("log")
    ax_b.set_xlim(1, 1800)
    ax_b.set_xlabel("Objects with applicable retrieved rules (log scale)")
    ax_b.set_title("Rule support is sparse and class specific", loc="left")
    panel_label(ax_b, "b")
    style_axis(ax_b, "x")
    for bar, count, share in zip(bars, comp["object_count"], comp["share_of_rule_matched_percent"]):
        ax_b.text(
            float(count) * 1.16,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count):,} ({float(share):.1f}%)",
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold",
        )

    matrix, labels = outcome_values(counts)
    draw_stacked(ax_c, matrix, labels, zoom=False)
    draw_stacked(ax_c_zoom, matrix, labels, zoom=True)
    panel_label(ax_c, "c")
    handles, legend_labels = ax_c.get_legend_handles_labels()
    ax_c.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.30),
        columnspacing=1.1,
        handlelength=1.3,
    )
    mat = (
        changes.pivot(index="rule_match_state", columns="outcome", values="h2_minus_h1_percentage_points")
        .reindex(index=RULE_STATES, columns=["Correct", "Insufficient"])
    )
    vals = mat.to_numpy()
    cmap = mpl.colors.LinearSegmentedColormap.from_list("orange_white_blue", [ORANGE, "#F7F7F7", BLUE])
    image = ax_d.imshow(vals, cmap=cmap, vmin=-40, vmax=40, aspect="auto")
    ax_d.set_xticks([0, 1], ["Correct", "Insufficient evidence"])
    ax_d.set_yticks([0, 1], ["Applicable rule", "No applicable rule"])
    ax_d.set_title("H2 RAG-LLM minus H1 LLM: outcome shift", loc="left")
    panel_label(ax_d, "d")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            value = vals[i, j]
            ax_d.text(
                j,
                i,
                f"{value:+.1f} pp",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if abs(value) > 24 else TEXT,
            )
    cbar = fig.colorbar(image, ax=ax_d, orientation="horizontal", fraction=0.055, pad=0.20, aspect=28)
    cbar.set_label("Percentage point difference")
    cbar.outline.set_linewidth(0.6)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    output = FIG / "Figure_S6_knowledge_evidence_mechanism_revised"
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    with Image.open(output.with_suffix(".png")) as image_file:
        rgb = image_file.convert("RGB")
        rgb.save(output.with_suffix(".tiff"), format="TIFF", compression="raw", dpi=(300, 300))
        ImageOps.grayscale(rgb).save(QA / "Figure_S6_grayscale_preview.png")

    return output


def write_audit(output: Path) -> None:
    inputs = sorted(DATA.glob("*.csv"))
    outputs = [output.with_suffix(ext) for ext in [".png", ".pdf", ".svg", ".tiff"]]
    audit = {
        "figure": "Figure S6",
        "revision_date": "2026-11-26",
        "changes": [
            "Aligned the palette with the main figures using blue, light blue, and orange families.",
            "Removed all hatch patterns from panel c and used solid fills.",
            "Added an enlarged 98.4-100% view beside panel c.",
            "Used the same Prediction positions (%) y-axis title in the main and enlarged panel-c axes.",
            "Removed the arrow between the main and enlarged panel-c axes.",
            "Changed invalid output from pink to a restrained slate-blue fill.",
            "Removed the two bottom figure notes.",
            "Retained the frozen numeric data without recomputation.",
        ],
        "input_sha256": {path.name: sha256(path) for path in inputs},
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    (QA / "Figure_S6_revision_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "README.md").write_text(
        "# Figure S6 revision\n\n"
        "This folder contains a standalone revision of Figure S6. The source CSV files are copied "
        "from the frozen Figure S20 evidence-mechanism package. The figure has not yet been "
        "integrated into the Supplementary Information Word file.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    figure_output = make_figure()
    write_audit(figure_output)
    print(figure_output.with_suffix(".png"))
