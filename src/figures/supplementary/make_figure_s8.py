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
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S8"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
FIG = OUTPUT_ROOT / "Figure_S8"
QA = OUTPUT_ROOT / "Figure_S8_qa"

METRICS = [
    "effective_assembly_rate",
    "endpoint_valid_connection_rate",
    "dangling_line_rate",
    "missing_gold_line_rate",
    "isolated_node_rate",
    "largest_connected_component_retention_rate",
]

LABELS = {
    "effective_assembly_rate": "Effective\nassembly",
    "endpoint_valid_connection_rate": "Valid endpoint\nconnection",
    "dangling_line_rate": "Fewer dangling\nlines",
    "missing_gold_line_rate": "Fewer missing\nlines",
    "isolated_node_rate": "Fewer isolated\nnodes",
    "largest_connected_component_retention_rate": "LCC\nretention",
}

BLUE = "#3B73B9"
ORANGE = "#C96F35"
GREY = "#A8ADB3"
TEXT = "#22262D"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(DATA / "direction_aligned_h2_advantage_by_unit.csv")
    summary = pd.read_csv(DATA / "graph_metric_advantage_summary.csv")
    if len(paired) != 90:
        raise RuntimeError(f"Expected 90 paired metric rows, found {len(paired)}")
    if set(paired["metric"]) != set(METRICS):
        raise RuntimeError("Metric domain differs from the frozen six-metric contract")
    if len(summary) != 6:
        raise RuntimeError(f"Expected six metric summaries, found {len(summary)}")
    if not (summary["unit_denominator"] == 15).all():
        raise RuntimeError("Direction-consistency denominator differs from 15")
    return paired, summary


def make_figure(paired: pd.DataFrame, summary: pd.DataFrame) -> Path:
    configure_plotting()
    fig = plt.figure(figsize=(14.2, 8.8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.75, 1.0],
        height_ratios=[1, 1],
        left=0.08,
        right=0.97,
        top=0.89,
        bottom=0.22,
        wspace=0.30,
        hspace=0.38,
    )
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_range = fig.add_subplot(gs[0, 1])
    ax_cons = fig.add_subplot(gs[1, 1])
    fig.suptitle("Graph-assembly trade-offs across repeats and spatial folds", fontsize=16, fontweight="bold")

    unit_order = [f"R{repeat_id}–F{fold}" for repeat_id in [1, 2, 3] for fold in [0, 1, 2, 3, 4]]
    matrix = (
        paired.pivot(index="unit_label", columns="metric", values="h2_favorable_advantage_pp")
        .reindex(index=unit_order, columns=METRICS)
        .to_numpy()
    )
    if np.isnan(matrix).any():
        raise RuntimeError("Incomplete repeat-by-fold heatmap matrix")

    limit = 80
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "adverse_neutral_favorable", [ORANGE, "#F7F7F7", BLUE]
    )
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    ax_heat.set_xticks(range(len(METRICS)), [LABELS[metric] for metric in METRICS], fontsize=8.5)
    ax_heat.set_yticks(range(len(unit_order)), unit_order, fontsize=8.5)
    ax_heat.set_xlabel("Graph metric (direction aligned so positive favors H2)")
    ax_heat.set_ylabel("Random repeat × outer spatial fold")
    ax_heat.set_title("Direction-aligned H2 advantage", loc="left", fontsize=12.5, fontweight="bold")
    panel_label(ax_heat, "a")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax_heat.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if abs(value) > 42 else TEXT,
                fontweight="bold" if abs(value) > 25 else "normal",
            )
    colorbar = fig.colorbar(image, ax=ax_heat, orientation="horizontal", fraction=0.045, pad=0.12, aspect=35)
    colorbar.set_label("Favorable H2 difference (percentage points)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    for boundary in [4.5, 9.5]:
        ax_heat.axhline(boundary, color="white", linewidth=2.0)

    ordered = summary.set_index("metric").reindex(METRICS)
    y = np.arange(len(METRICS))
    for index, metric in enumerate(METRICS):
        low = float(ordered.loc[metric, "minimum_advantage_pp"])
        high = float(ordered.loc[metric, "maximum_advantage_pp"])
        mean = float(ordered.loc[metric, "mean_advantage_pp"])
        ax_range.plot([low, high], [index, index], color="#7A7A7A", linewidth=2)
        ax_range.scatter(
            [mean],
            [index],
            s=60,
            color=BLUE if mean >= 0 else ORANGE,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax_range.text(high + 1.5, index, f"mean {mean:+.1f}", va="center", fontsize=8.2)
    ax_range.axvline(0, color="#333333", linewidth=0.8)
    ax_range.set_yticks(y, [LABELS[metric].replace("\n", " ") for metric in METRICS], fontsize=8.5)
    ax_range.invert_yaxis()
    ax_range.set_xlim(-35, 90)
    ax_range.set_xlabel("Favorable H2 difference (pp)")
    ax_range.set_title("Mean and range across 15 units", loc="left", fontsize=12, fontweight="bold")
    panel_label(ax_range, "b")
    ax_range.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    ax_range.spines[["top", "right", "left"]].set_visible(False)
    ax_range.tick_params(axis="y", length=0)

    favorable = ordered["favorable_unit_count"].to_numpy()
    adverse = ordered["adverse_unit_count"].to_numpy()
    unchanged = ordered["zero_unit_count"].to_numpy()
    ax_cons.barh(y, favorable, color=BLUE, label="Favors H2", height=0.58, edgecolor="white", linewidth=0.7)
    ax_cons.barh(
        y,
        adverse,
        left=favorable,
        color=ORANGE,
        label="Favors H1",
        height=0.58,
        edgecolor="white",
        linewidth=0.7,
    )
    ax_cons.barh(
        y,
        unchanged,
        left=favorable + adverse,
        color=GREY,
        label="No change",
        height=0.58,
        edgecolor="white",
        linewidth=0.7,
    )
    ax_cons.set_yticks(y, [LABELS[metric].replace("\n", " ") for metric in METRICS], fontsize=8.5)
    ax_cons.invert_yaxis()
    ax_cons.set_xlim(0, 15)
    ax_cons.set_xticks([0, 3, 6, 9, 12, 15])
    ax_cons.set_xlabel("Repeat × fold units (n=15)")
    ax_cons.set_title("Direction consistency", loc="left", fontsize=12, fontweight="bold")
    panel_label(ax_cons, "c")
    for index, (good, bad, same) in enumerate(zip(favorable, adverse, unchanged)):
        if good:
            ax_cons.text(good / 2, index, str(int(good)), ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if bad:
            ax_cons.text(good + bad / 2, index, str(int(bad)), ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if same:
            ax_cons.text(good + bad + same / 2, index, str(int(same)), ha="center", va="center", color=TEXT, fontsize=8.5, fontweight="bold")
    ax_cons.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax_cons.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    ax_cons.spines[["top", "right", "left"]].set_visible(False)
    ax_cons.tick_params(axis="y", length=0)

    fig.text(
        0.5,
        0.060,
        "Positive values favor H2: higher assembly, endpoint and LCC rates; lower dangling-line, missing-line and isolated-node rates.",
        ha="center",
        fontsize=9,
        color="#3F3F3F",
    )
    fig.text(
        0.5,
        0.032,
        "Units are descriptive repeat × fold views, not independent cities or time points. No inferential test was performed.",
        ha="center",
        fontsize=9,
        color="#3F3F3F",
    )

    output = FIG / "Figure_S8_graph_assembly_tradeoff_stability_revised"
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    with Image.open(output.with_suffix(".png")) as image_file:
        rgb = image_file.convert("RGB")
        rgb.save(output.with_suffix(".tiff"), format="TIFF", compression="raw", dpi=(300, 300))
        ImageOps.grayscale(rgb).save(QA / "Figure_S8_grayscale_preview.png")
    return output


def write_audit(output: Path) -> None:
    source_files = sorted(DATA.glob("*.csv"))
    output_files = [output.with_suffix(extension) for extension in [".png", ".pdf", ".svg", ".tiff"]]
    audit = {
        "figure": "Figure S8",
        "revision_date": "2026-11-26",
        "core_conclusion": "Graph assembly gains are stable across repeated spatial folds, while dangling-line behavior remains a visible trade-off.",
        "archetype": "asymmetric mixed-modality quantitative figure",
        "change_scope": [
            "Removed hatch and dot patterns from panel c.",
            "Used solid blue, orange, and grey fills for the three direction-consistency states.",
            "Retained all frozen values and left panels a and b unchanged in meaning and layout.",
        ],
        "input_sha256": {path.name: sha256(path) for path in source_files},
        "output_sha256": {path.name: sha256(path) for path in output_files},
    }
    (QA / "Figure_S8_revision_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "README.md").write_text(
        "# Figure S8 revision\n\n"
        "Panel c now uses solid colour fills without hatch or dot patterns. Frozen values are unchanged. "
        "This figure has not yet been integrated into the Supplementary Information Word file.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    paired_data, summary_data = load_data()
    figure_path = make_figure(paired_data, summary_data)
    write_audit(figure_path)
    print(figure_path.with_suffix(".png"))
