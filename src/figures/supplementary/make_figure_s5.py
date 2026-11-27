from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "figure_assets"
QA = ROOT / "qa"

COUNTS_FILE = DATA / "class_outcome_counts_and_percentages.csv"
DELTA_FILE = DATA / "h2_minus_h1_class_outcome_changes.csv"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S5"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
OUT = OUTPUT_ROOT / "Figure_S5"
QA = OUTPUT_ROOT / "Figure_S5_qa"
COUNTS_FILE = DATA / "class_outcome_counts_and_percentages.csv"
DELTA_FILE = DATA / "h2_minus_h1_class_outcome_changes.csv"

EXPECTED_HASHES = {
    COUNTS_FILE: "A358E5F143D3744240A4809FB1D9A98C1ADD228D9B6C740296E59D754CC3791F",
    DELTA_FILE: "E1F28525C0C26B0780AE7E08FD1366070F457871DBE42B19EF3E79CA9AC00982",
}

METHOD_ORDER = ["H0_no_completion", "H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]
CLASS_ORDER = ["ACLineSegment", "ConnectivityNode", "EnergySource"]
CLASS_LABELS = ["Line\n(n=1,339)", "Node\n(n=17,289)", "Source\n(n=104)"]
OUTCOME_ORDER = [
    "ACLineSegment",
    "ConnectivityNode",
    "EnergySource",
    "Insufficient evidence",
    "INVALID",
    "MISSING",
    "No completion",
]
OUTCOME_LABELS = [
    "Predicted\nLine",
    "Predicted\nNode",
    "Predicted\nSource",
    "Insufficient\nevidence",
    "Invalid\noutput",
    "Missing\nresponse",
    "No\ncompletion",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {path.name}: {actual} != {expected}")
    counts = pd.read_csv(COUNTS_FILE)
    delta = pd.read_csv(DELTA_FILE)
    if len(counts) != len(METHOD_ORDER) * len(CLASS_ORDER) * len(OUTCOME_ORDER):
        raise RuntimeError("Unexpected count-table dimensions")
    if len(delta) != len(CLASS_ORDER) * len(OUTCOME_ORDER):
        raise RuntimeError("Unexpected delta-table dimensions")
    for method in METHOD_ORDER:
        totals = (
            counts[counts.method == method]
            .groupby("gold_class")["row_percent"]
            .sum()
            .reindex(CLASS_ORDER)
        )
        if not np.allclose(totals.to_numpy(), 100.0):
            raise RuntimeError(f"Row percentages do not sum to 100 for {method}")
    return counts, delta


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def matrix_for(counts: pd.DataFrame, method: str) -> np.ndarray:
    return (
        counts[counts.method == method]
        .pivot(index="gold_class", columns="outcome", values="row_percent")
        .reindex(index=CLASS_ORDER, columns=OUTCOME_ORDER)
        .to_numpy(dtype=float)
    )


def annotate(ax, matrix: np.ndarray, difference: bool = False) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if difference:
                if abs(value) < 0.05:
                    label = ""
                elif abs(value) < 0.1:
                    label = "+<0.1" if value > 0 else "−<0.1"
                else:
                    label = f"{value:+.1f}"
                color = "white" if abs(value) >= 24 else "#202020"
            else:
                if value == 0:
                    label = ""
                elif value < 0.1:
                    label = "<0.1"
                else:
                    label = f"{value:.1f}"
                color = "white" if value >= 55 else "#202020"
            if label:
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=color,
                    fontweight="bold" if abs(value) >= 20 else "normal",
                )


def style_axis(ax, title: str, panel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=8)
    ax.text(-0.085, 1.08, panel, transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(OUTCOME_LABELS)), OUTCOME_LABELS, fontsize=8)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=4)
    ax.set_yticks(np.arange(len(CLASS_LABELS)), CLASS_LABELS, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel("Prediction or response state", fontsize=9, labelpad=8)
    ax.set_ylabel("Reference class (unique objects)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def outline_correct_cells(ax) -> None:
    for index in range(3):
        ax.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#111111",
                linewidth=1.35,
            )
        )


def make_figure(counts: pd.DataFrame, delta: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.45), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.855, bottom=0.075, wspace=0.19, hspace=0.48)
    fig.suptitle(
        "Class-specific prediction outcomes across completion methods",
        fontsize=16,
        fontweight="bold",
    )

    sequential_cmap = mpl.colormaps["Blues"]
    sequential_norm = Normalize(vmin=0.0, vmax=100.0)
    difference_cmap = LinearSegmentedColormap.from_list(
        "orange_white_blue", ["#C06C3B", "#F7F7F7", "#3F6FB3"]
    )
    difference_norm = Normalize(vmin=-40.0, vmax=40.0)

    panels = [
        (axes[0, 0], "H0_no_completion", "H0 no completion", "a"),
        (axes[0, 1], "H1_no_retrieval_LLM", "H1 LLM", "b"),
        (axes[1, 0], "H2_KB_RAG_LLM", "H2 RAG-LLM", "c"),
    ]
    for ax, method, title, panel in panels:
        matrix = matrix_for(counts, method)
        image = ax.imshow(
            matrix,
            cmap=sequential_cmap,
            norm=sequential_norm,
            aspect="auto",
            interpolation="nearest",
        )
        annotate(ax, matrix)
        style_axis(ax, title, panel)
        if method != "H0_no_completion":
            outline_correct_cells(ax)
        colorbar = fig.colorbar(
            image,
            ax=ax,
            orientation="horizontal",
            fraction=0.05,
            pad=0.18,
            aspect=32,
        )
        colorbar.set_label("Within-reference-class share (%)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    delta_matrix = (
        delta.pivot(index="gold_class", columns="outcome", values="h2_minus_h1_pp")
        .reindex(index=CLASS_ORDER, columns=OUTCOME_ORDER)
        .to_numpy(dtype=float)
    )
    ax = axes[1, 1]
    image = ax.imshow(
        delta_matrix,
        cmap=difference_cmap,
        norm=difference_norm,
        aspect="auto",
        interpolation="nearest",
    )
    annotate(ax, delta_matrix, difference=True)
    style_axis(ax, "H2 minus H1", "d")
    outline_correct_cells(ax)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.18,
        aspect=32,
    )
    colorbar.set_label("Change in within-reference-class share (percentage points)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return fig


def export(fig: plt.Figure) -> dict[str, dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Figure_S5_class_specific_prediction_outcomes"
    outputs: dict[str, dict[str, object]] = {}
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 300},
        ".tiff": {"dpi": 300},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white", **kwargs)
        outputs[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    plt.close(fig)
    return outputs


def write_qa(outputs: dict[str, dict[str, object]]) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    png = OUT / "Figure_S5_class_specific_prediction_outcomes.png"
    Image.open(png).convert("L").save(
        QA / "Figure_S5_class_specific_prediction_outcomes_grayscale.png",
        dpi=(300, 300),
    )
    svg = (OUT / "Figure_S5_class_specific_prediction_outcomes.svg").read_text(encoding="utf-8")
    forbidden_reader_terms = [
        "Gold class",
        "Gold-class",
        ">INVALID<",
        ">MISSING<",
        "Black outlines mark",
        "Three random repeats pooled",
    ]
    report = {
        "source_hash_gate": True,
        "source_data_unchanged": True,
        "new_title": "Class-specific prediction outcomes across completion methods",
        "reader_y_axis": "Reference class (unique objects)",
        "bottom_figure_notes_removed": True,
        "forbidden_reader_terms_absent": all(term not in svg for term in forbidden_reader_terms),
        "editable_svg_text": "<text" in svg,
        "exported_files": outputs,
    }
    report["passed"] = all(
        [
            report["source_hash_gate"],
            report["source_data_unchanged"],
            report["bottom_figure_notes_removed"],
            report["forbidden_reader_terms_absent"],
            report["editable_svg_text"],
        ]
    )
    (QA / "figure_s5_terminology_revision_qa.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure()
    counts, delta = verify_and_load()
    outputs = export(make_figure(counts, delta))
    write_qa(outputs)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
