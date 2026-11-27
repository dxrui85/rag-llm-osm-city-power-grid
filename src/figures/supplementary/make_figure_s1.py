from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "figure_assets"
QA = ROOT / "qa"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S1"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
OUT = OUTPUT_ROOT / "Figure_S1"
QA = OUTPUT_ROOT / "Figure_S1_qa"

METHOD_ORDER = ["H0_no_completion", "H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]
METHOD_LABEL = {
    "H0_no_completion": "H0 no completion",
    "H1_no_retrieval_LLM": "H1 LLM",
    "H2_KB_RAG_LLM": "H2 RAG-LLM",
}
METHOD_SHORT = {
    "H0_no_completion": "H0",
    "H1_no_retrieval_LLM": "H1",
    "H2_KB_RAG_LLM": "H2",
}

# Harmonised with the blue, coral and neutral palette used in the main-text figures.
METHOD_COLOR = {
    "H0_no_completion": "#8C8C8C",
    "H1_no_retrieval_LLM": "#4E79A7",
    "H2_KB_RAG_LLM": "#E76F51",
}
METHOD_MARKER = {
    "H0_no_completion": "s",
    "H1_no_retrieval_LLM": "o",
    "H2_KB_RAG_LLM": "D",
}

STATE_ORDER = [
    "valid_prediction",
    "valid_insufficient_evidence",
    "INVALID",
    "missing",
    "H0_unresolved",
]
STATE_LABEL = {
    "valid_prediction": "Prediction returned",
    "valid_insufficient_evidence": "Insufficient evidence",
    "INVALID": "Invalid response",
    "missing": "Missing response",
    "H0_unresolved": "No completion",
}
STATE_COLOR = {
    "valid_prediction": "#4E79A7",
    "valid_insufficient_evidence": "#F4A261",
    "INVALID": "#C96C75",
    "missing": "#7F8C8D",
    "H0_unresolved": "#B7B7B7",
}

GRAPH_METRIC_LABEL = {
    "effective_assembly_rate": "Effective assembly",
    "endpoint_valid_connection_rate": "Valid endpoints",
    "missing_gold_line_rate": "Missing lines",
    "dangling_line_rate": "Dangling lines",
    "isolated_node_rate": "Isolated nodes",
    "largest_connected_component_retention_rate": "Largest component",
}


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
            "axes.titlesize": 8.7,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    object_data = pd.read_csv(DATA / "object_scale_metrics.csv")
    failure = pd.read_csv(DATA / "failure_state_summary.csv")
    graph = pd.read_csv(DATA / "graph_scale_delta_summary.csv")
    object_data["parameter_value"] = object_data["parameter_value"].astype(float)
    object_data["value"] = object_data["value"].astype(float)
    object_data["repeat_id"] = object_data["case_id"].map(
        lambda value: int(re.search(r"repeat=(\d+)", str(value)).group(1))
    )
    failure["scale_degree"] = failure["scale_degree"].astype(float)
    failure["mean_rate"] = failure["mean_rate"].astype(float)
    graph["mean_delta_010_minus_0125"] = graph["mean_delta_010_minus_0125"].astype(float)
    return object_data, failure, graph


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.115,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )


def style_axis(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#E6E9EC", linewidth=0.55, zorder=0)
    ax.tick_params(length=2.5, width=0.65)


def draw_object_panel(ax, object_data: pd.DataFrame) -> None:
    data = object_data[object_data["variable"] == "macro_f1"].copy()
    x_map = {0.125: 0.0, 0.10: 1.0}
    offsets = {
        "H0_no_completion": -0.18,
        "H1_no_retrieval_LLM": 0.0,
        "H2_KB_RAG_LLM": 0.18,
    }
    for method in METHOD_ORDER:
        subset = data[data["condition"] == method]
        for repeat_id in [1, 2, 3]:
            values = []
            xs = []
            for scale in [0.125, 0.10]:
                row = subset[
                    np.isclose(subset["parameter_value"], scale)
                    & (subset["repeat_id"] == repeat_id)
                ]
                if len(row) != 1:
                    raise RuntimeError(f"Expected one row for {method}, repeat {repeat_id}, scale {scale}")
                xs.append(x_map[scale] + offsets[method])
                values.append(float(row.iloc[0]["value"]))
            ax.plot(xs, values, color=METHOD_COLOR[method], alpha=0.34, linewidth=0.85, zorder=2)
            ax.scatter(
                xs,
                values,
                color=METHOD_COLOR[method],
                marker=METHOD_MARKER[method],
                s=26,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        means = []
        mean_x = []
        for scale in [0.125, 0.10]:
            rows = subset[np.isclose(subset["parameter_value"], scale)]
            means.append(float(rows["value"].mean()))
            mean_x.append(x_map[scale] + offsets[method])
        ax.plot(mean_x, means, color=METHOD_COLOR[method], linewidth=2.1, zorder=4)

    ax.set_xticks([0, 1], ["0.125° primary", "0.10° sensitivity"])
    ax.set_ylim(-0.035, 1.035)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Object-level category recovery", loc="left", fontweight="bold")
    style_axis(ax, "y")
    panel_label(ax, "a")


def draw_failure_panel(ax, failure: pd.DataFrame) -> None:
    rows: list[tuple[float, str]] = []
    labels: list[str] = []
    y_positions: list[float] = []
    cursor = 0.0
    for method in METHOD_ORDER:
        for scale in [0.125, 0.10]:
            rows.append((scale, method))
            labels.append(f"{METHOD_SHORT[method]}  {scale:.3f}°" if scale == 0.125 else f"{METHOD_SHORT[method]}  {scale:.2f}°")
            y_positions.append(cursor)
            cursor += 0.70
        cursor += 0.24

    y = np.array(y_positions)
    left = np.zeros(len(rows))
    for state in STATE_ORDER:
        values = []
        for scale, method in rows:
            q = failure[
                np.isclose(failure["scale_degree"], scale)
                & (failure["condition"] == method)
                & (failure["state"] == state)
            ]
            values.append(float(q.iloc[0]["mean_rate"]) if len(q) else 0.0)
        ax.barh(
            y,
            values,
            left=left,
            height=0.54,
            color=STATE_COLOR[state],
            edgecolor="white",
            linewidth=0.55,
            label=STATE_LABEL[state],
        )
        left += np.array(values)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of fixed denominator")
    ax.set_title("Response-state composition", loc="left", fontweight="bold")
    style_axis(ax, "x")
    state_handles = [
        Patch(facecolor=STATE_COLOR[state], edgecolor="none", label=STATE_LABEL[state])
        for state in STATE_ORDER
    ]
    ax.legend(
        handles=state_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.21),
        ncol=3,
        fontsize=5.9,
        handlelength=1.35,
        columnspacing=0.9,
        borderaxespad=0,
    )
    panel_label(ax, "b")


def draw_graph_panel(ax, graph: pd.DataFrame) -> None:
    metrics = [
        "effective_assembly_rate",
        "endpoint_valid_connection_rate",
        "missing_gold_line_rate",
        "dangling_line_rate",
        "isolated_node_rate",
        "largest_connected_component_retention_rate",
    ]
    y = np.arange(len(metrics))
    values_all: list[float] = []
    for index, method in enumerate(["H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]):
        subset = graph[graph["condition"] == method].set_index("metric")
        values = np.array([float(subset.loc[m, "mean_delta_010_minus_0125"]) * 100 for m in metrics])
        values_all.extend(values.tolist())
        yy = y + (-0.14 if index == 0 else 0.14)
        ax.scatter(
            values,
            yy,
            s=31,
            marker=METHOD_MARKER[method],
            color=METHOD_COLOR[method],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for value, ypos in zip(values, yy):
            ax.plot([0, value], [ypos, ypos], color=METHOD_COLOR[method], linewidth=1.3, alpha=0.72, zorder=2)
            ax.text(
                value + (0.72 if value >= 0 else -0.72),
                ypos,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=6.0,
                color=METHOD_COLOR[method],
                clip_on=False,
            )
    ax.axvline(0, color="#333333", linewidth=0.75)
    ax.set_yticks(y, [GRAPH_METRIC_LABEL[m] for m in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("0.10° − 0.125° (percentage points)")
    ax.set_xlim(min(-2.5, min(values_all) - 1.8), max(12.0, max(values_all) + 2.0))
    ax.set_title("Graph-assembly sensitivity", loc="left", fontweight="bold")
    style_axis(ax, "x")
    panel_label(ax, "c")


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#6F7F8C",
    fontsize: float = 6.3,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#24323D",
        fontweight=weight,
        linespacing=1.16,
        zorder=3,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], style: str = "-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=0.9,
            color="#667785",
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def draw_mechanism_panel(ax) -> None:
    ax.set_axis_off()
    ax.set_title("Evidence, prediction and assembly", loc="left", fontweight="bold", pad=4)
    panel_label(ax, "d")

    # Upper workflow: every arrow joins the midpoint of a box edge and stays clear of text.
    add_box(ax, 0.02, 0.70, 0.25, 0.17, "Visible OSM fields\nTopology context", "#E8EFF6", fontsize=5.7)
    add_box(ax, 0.02, 0.44, 0.25, 0.18, "Labelled examples\nPower system rules", "#FBE3D9", fontsize=5.7)
    add_box(ax, 0.37, 0.74, 0.25, 0.15, "H1 LLM\nDirect prediction", "#DDEAF4", fontsize=5.9, weight="bold")
    add_box(ax, 0.37, 0.49, 0.25, 0.19, "H2 RAG-LLM\nRetrieval and\nprediction", "#F9D8CF", fontsize=5.8, weight="bold")
    add_box(ax, 0.73, 0.57, 0.25, 0.24, "Structural role\nmapping and\ngraph assembly", "#E5EFEA", fontsize=5.7, weight="bold")

    add_arrow(ax, (0.27, 0.785), (0.37, 0.815))
    add_arrow(ax, (0.27, 0.735), (0.37, 0.61))
    add_arrow(ax, (0.27, 0.53), (0.37, 0.575))
    add_arrow(ax, (0.62, 0.815), (0.73, 0.735))
    add_arrow(ax, (0.62, 0.585), (0.73, 0.645))

    # Lower row: the two prespecified scales feed one descriptive comparison.
    add_box(ax, 0.03, 0.19, 0.34, 0.12, "0.125° primary scale", "#F1F3F5", fontsize=6.0)
    add_box(ax, 0.63, 0.19, 0.34, 0.12, "0.10° sensitivity scale", "#F1F3F5", fontsize=6.0)
    add_arrow(ax, (0.37, 0.25), (0.63, 0.25), style="<->")
    ax.text(
        0.50,
        0.355,
        "Descriptive scale comparison",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.6,
        color="#52606A",
    )
    add_box(
        ax,
        0.17,
        0.012,
        0.66,
        0.115,
        "Stable H2 object recovery\nChanges varied across\ngraph assembly metrics",
        "#FFF0CC",
        fontsize=5.7,
        weight="bold",
    )
    add_arrow(ax, (0.50, 0.19), (0.50, 0.127))


def export(fig) -> dict[str, dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        path = OUT / f"Figure_S1_spatial_scale_sensitivity_revised{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white", **kwargs)
        outputs[path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(fig)
    return outputs


def make_figure(object_data: pd.DataFrame, failure: pd.DataFrame, graph: pd.DataFrame):
    fig = plt.figure(figsize=(183 / 25.4, 174 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.00, 1.12],
        height_ratios=[1.00, 1.02],
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.905,
        hspace=0.62,
        wspace=0.52,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_object_panel(ax_a, object_data)
    draw_failure_panel(ax_b, failure)
    draw_graph_panel(ax_c, graph)
    draw_mechanism_panel(ax_d)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            linewidth=1.8,
            markersize=4.8,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=3,
        fontsize=6.8,
        title="Methods",
        title_fontsize=6.8,
        handlelength=1.8,
        columnspacing=1.4,
    )
    return fig


def write_qa(outputs: dict[str, dict[str, object]]) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    png = OUT / "Figure_S1_spatial_scale_sensitivity_revised.png"
    image = Image.open(png)
    image.convert("L").save(QA / "Figure_S1_grayscale_preview.png", dpi=(300, 300))
    record = {
        "figure": "Figure S1",
        "changes": [
            "Panel b now uses solid colour fills without hatching.",
            "Panels a and c use a shared legend outside the plotting areas.",
            "Panel d boxes and arrows were repositioned to avoid overlap.",
            "The bottom footnote was removed.",
            "The palette was harmonised with the main-text blue, coral and neutral colours.",
        ],
        "source_rows": {
            "object_scale_metrics": len(pd.read_csv(DATA / "object_scale_metrics.csv")),
            "failure_state_summary": len(pd.read_csv(DATA / "failure_state_summary.csv")),
            "graph_scale_delta_summary": len(pd.read_csv(DATA / "graph_scale_delta_summary.csv")),
        },
        "png_pixels": {"width": image.width, "height": image.height},
        "outputs": outputs,
        "status": "generated_pending_visual_review",
    }
    (QA / "Figure_S1_generation_qa.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure()
    object_data, failure, graph = load_data()
    fig = make_figure(object_data, failure, graph)
    outputs = export(fig)
    write_qa(outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
