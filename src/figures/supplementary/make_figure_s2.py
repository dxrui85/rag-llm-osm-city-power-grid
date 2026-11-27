from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "figure_assets"
QA = ROOT / "qa"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S2"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
OUT = OUTPUT_ROOT / "Figure_S2"
QA = OUTPUT_ROOT / "Figure_S2_qa"

EXPECTED_HASHES = {
    "class_level_metrics_by_repeat.csv": "DEEB3517AE1E26A4DFDCA572B655BAAC1B23B49E4DF736A1A9140486E2D886E0",
    "class_level_scale_paired_differences.csv": "B6E297C2A7D0A5B8950CD87AC414B9BAE5B86F244BBC35F58BFBDE27695C9FDE",
    "class_support.csv": "06A21E2E52D2009CDFFBE7D2532BB8CC3CB870809EC5BA95BBE41DA458D11153",
}

CLASSES = ["ACLineSegment", "ConnectivityNode", "EnergySource"]
CLASS_LABEL = {
    "ACLineSegment": "AC line segment",
    "ConnectivityNode": "Connectivity node",
    "EnergySource": "Energy source",
}
METHODS = ["H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]
METHOD_LABEL = {
    "H1_no_retrieval_LLM": "H1 LLM",
    "H2_KB_RAG_LLM": "H2 RAG-LLM",
}
METHOD_COLOR = {
    "H1_no_retrieval_LLM": "#0072B2",
    "H2_KB_RAG_LLM": "#D55E00",
}
METHOD_MARKER = {
    "H1_no_retrieval_LLM": "o",
    "H2_KB_RAG_LLM": "D",
}
SCALE_MARKER = {0.125: "o", 0.10: "s"}
SCALE_LABEL = {0.125: "0.125° main", 0.10: "0.10° sensitivity"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, expected in EXPECTED_HASHES.items():
        path = DATA / name
        if not path.exists():
            raise RuntimeError(f"Missing frozen source data: {path}")
        if sha256(path) != expected:
            raise RuntimeError(f"Source data hash drift: {path}")

    detail = pd.read_csv(DATA / "class_level_metrics_by_repeat.csv")
    paired = pd.read_csv(DATA / "class_level_scale_paired_differences.csv")
    support = pd.read_csv(DATA / "class_support.csv")

    if len(detail) != 36 or len(paired) != 18 or len(support) != 3:
        raise RuntimeError("Unexpected source-data row count")
    if set(detail["class_name"]) != set(CLASSES):
        raise RuntimeError("Class domain drift")
    if set(detail["condition"]) != set(METHODS):
        raise RuntimeError("Method domain drift")
    if set(detail["repeat_id"]) != {1, 2, 3}:
        raise RuntimeError("Repeat domain drift")
    if set(np.round(detail["scale_degree"], 3)) != {0.10, 0.125}:
        raise RuntimeError("Scale domain drift")
    return detail, support, paired


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )


def support_panel(ax, support: pd.DataFrame) -> None:
    ordered = support.set_index("class_name").loc[CLASSES].reset_index()
    y = np.arange(len(ordered))
    bars = ax.barh(
        y,
        ordered["support"],
        color=["#7AA6C2", "#5B8DB8", "#A6BDD0"],
        height=0.62,
    )
    ax.set_yticks(y, [CLASS_LABEL[value] for value in ordered["class_name"]])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Number of evaluated objects (log scale)")
    ax.set_title("Reference class distribution", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    for bar, value in zip(bars, ordered["support"]):
        ax.text(
            value * 1.12,
            bar.get_y() + bar.get_height() / 2,
            f"n = {value:,}",
            va="center",
            fontsize=6.5,
        )
    ax.text(
        0.02,
        0.04,
        "H0 no completion: F1 = 0 for all classes",
        transform=ax.transAxes,
        fontsize=6.4,
        color="#555555",
    )
    panel_label(ax, "a")


def method_panel(ax, detail: pd.DataFrame, method: str, panel: str) -> None:
    subset = detail[detail["condition"] == method]
    centers = np.arange(len(CLASSES))
    for index, class_name in enumerate(CLASSES):
        class_rows = subset[subset["class_name"] == class_name]
        for repeat in sorted(class_rows["repeat_id"].unique()):
            pair = class_rows[class_rows["repeat_id"] == repeat].set_index("scale_degree")
            xs = [index - 0.11, index + 0.11]
            ys = [float(pair.loc[0.125, "f1"]), float(pair.loc[0.10, "f1"])]
            ax.plot(xs, ys, color=METHOD_COLOR[method], alpha=0.36, linewidth=0.8)
            ax.scatter(
                xs[0],
                ys[0],
                marker=SCALE_MARKER[0.125],
                color=METHOD_COLOR[method],
                s=24,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            ax.scatter(
                xs[1],
                ys[1],
                marker=SCALE_MARKER[0.10],
                color="white",
                edgecolor=METHOD_COLOR[method],
                s=24,
                linewidth=1.0,
                zorder=3,
            )
        means = class_rows.groupby("scale_degree")["f1"].mean()
        ax.plot(
            [index - 0.11, index + 0.11],
            [means.loc[0.125], means.loc[0.10]],
            color=METHOD_COLOR[method],
            linewidth=2.1,
            zorder=4,
        )
    ax.set_xticks(centers, ["AC line", "Connectivity\nnode", "Energy\nsource"])
    if method == "H2_KB_RAG_LLM":
        ax.set_ylim(0.94, 1.002)
        ax.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
        ax.set_ylabel("Class F1 (0.94–1.00)")
    else:
        ax.set_ylim(0.0, 1.035)
        ax.set_ylabel("Class F1")
    ax.set_title(f"{METHOD_LABEL[method]} class-wise recovery", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor="white",
            label=SCALE_LABEL[0.125],
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor=METHOD_COLOR[method],
            label=SCALE_LABEL[0.10],
            markersize=5,
        ),
    ]
    legend_location = "lower left" if method == "H2_KB_RAG_LLM" else "lower right"
    ax.legend(handles=handles, loc=legend_location, fontsize=6.2)
    panel_label(ax, panel)


def delta_panel(ax, paired: pd.DataFrame) -> None:
    y = np.arange(len(CLASSES))
    offsets = {"H1_no_retrieval_LLM": -0.12, "H2_KB_RAG_LLM": 0.12}
    for method in METHODS:
        subset = paired[paired["condition"] == method]
        for index, class_name in enumerate(CLASSES):
            values = (
                subset[subset["class_name"] == class_name]["delta_010_minus_0125"].to_numpy()
                * 100
            )
            yy = np.full(len(values), index + offsets[method])
            ax.scatter(
                values,
                yy,
                color=METHOD_COLOR[method],
                marker=METHOD_MARKER[method],
                s=22,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.4,
            )
            mean = float(values.mean())
            ax.scatter(
                [mean],
                [index + offsets[method]],
                color=METHOD_COLOR[method],
                marker=METHOD_MARKER[method],
                s=48,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=4,
            )
            ax.text(
                mean + (0.18 if mean >= 0 else -0.18),
                index + offsets[method],
                f"{mean:+.2f}",
                ha="left" if mean >= 0 else "right",
                va="center",
                fontsize=6.1,
                color=METHOD_COLOR[method],
            )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, [CLASS_LABEL[value] for value in CLASSES])
    ax.invert_yaxis()
    ax.set_xlim(-5.2, 2.3)
    ax.set_xlabel("0.10° - 0.125° class F1 (percentage points)")
    ax.set_title("Paired scale differences", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKER[method],
            color="none",
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor="white",
            label=METHOD_LABEL[method],
            markersize=5,
        )
        for method in METHODS
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2)
    panel_label(ax, "d")


def draw(detail: pd.DataFrame, support: pd.DataFrame, paired: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    support_panel(fig.add_subplot(grid[0, 0]), support)
    method_panel(fig.add_subplot(grid[0, 1]), detail, "H1_no_retrieval_LLM", "b")
    method_panel(fig.add_subplot(grid[1, 0]), detail, "H2_KB_RAG_LLM", "c")
    delta_panel(fig.add_subplot(grid[1, 1]), paired)
    return fig


def export(fig: plt.Figure) -> dict[str, dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    stem = OUT / "Figure_S2_classwise_scale_sensitivity_zoomed"
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white", **kwargs)
        outputs[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    plt.close(fig)
    return outputs


def write_qa(outputs: dict[str, dict[str, object]]) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    png = OUT / "Figure_S2_classwise_scale_sensitivity_zoomed.png"
    Image.open(png).convert("L").save(
        QA / "Figure_S2_classwise_scale_sensitivity_zoomed_grayscale.png",
        dpi=(300, 300),
    )
    svg = (OUT / "Figure_S2_classwise_scale_sensitivity_zoomed.svg").read_text(
        encoding="utf-8"
    )
    report = {
        "source_hash_gate": True,
        "source_data_unchanged": True,
        "panel_a_title": "Reference class distribution",
        "panel_a_x_label": "Number of evaluated objects (log scale)",
        "panel_c_y_axis_expanded": True,
        "panel_c_y_axis_range": [0.94, 1.002],
        "bottom_figure_note_removed": True,
        "editable_svg_text": "<text" in svg,
        "exported_files": outputs,
    }
    report["passed"] = all(
        [
            report["source_hash_gate"],
            report["source_data_unchanged"],
            report["panel_c_y_axis_expanded"],
            report["bottom_figure_note_removed"],
            report["editable_svg_text"],
        ]
    )
    (QA / "figure_s2_revision_qa.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure()
    detail, support, paired = verify_and_load()
    outputs = export(draw(detail, support, paired))
    write_qa(outputs)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
