from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter

# Portable release overrides. The historical definitions above are retained as provenance.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
OUT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT", REPOSITORY_ROOT / "outputs" / "Figure_3"))
SOURCE = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_3" / "graph_assembly_results.csv"

COLORS = {
    "h1": "#4C78A8",
    "h2": "#E76F51",
    "ink": "#263238",
    "muted": "#66727A",
    "grid": "#E5E9EC",
    "detail_bg": "#FAFBFC",
    "detail_edge": "#A9B4BD",
}

TEXT = {
    "en": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "higher": "Higher values are favourable",
        "lower": "Lower values are favourable",
        "detail": "Detail view",
        "rate": "Rate",
        "high_labels": ["Effective assembly", "Valid endpoints", "Largest component\nretained"],
        "low_labels": ["Missing lines", "Isolated nodes", "Dangling lines"],
        "box_note": "Boxes show interquartile ranges; centre lines show medians; whiskers show min–max across 15 repeat–fold graph units.",
    },
    "zh": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "higher": "数值越高越优",
        "lower": "数值越低越优",
        "detail": "局部放大",
        "rate": "比例",
        "high_labels": ["有效装配", "端点有效连接", "最大连通分量\n保留率"],
        "low_labels": ["缺失线路", "孤立节点", "悬空线路"],
        "box_note": "箱体表示四分位区间，中线表示中位数，须线表示15个重复—折图单元的最小值至最大值。",
    },
}

HIGH_METRICS = [
    "effective_assembly_rate",
    "endpoint_valid_connection_rate",
    "largest_connected_component_retention_rate",
]
LOW_METRICS = ["missing_gold_line_rate", "isolated_node_rate", "dangling_line_rate"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def set_style(lang: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial" if lang == "en" else "Microsoft YaHei",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def condition_key(value: str) -> str:
    return "h1" if str(value).startswith("H1") else "h2"


def draw_box(ax: plt.Axes, values: np.ndarray, position: float, color: str, width: float = 0.22) -> None:
    bp = ax.boxplot(
        [np.asarray(values, dtype=float)],
        positions=[position],
        widths=width,
        vert=False,
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        zorder=3,
        boxprops={"facecolor": mpl.colors.to_rgba(color, 0.28), "edgecolor": color, "linewidth": 1.05},
        whiskerprops={"color": color, "linewidth": 1.05},
        capprops={"color": color, "linewidth": 1.05},
        medianprops={"color": color, "linewidth": 1.8},
    )
    for patch in bp["boxes"]:
        patch.set_joinstyle("round")


def values_for(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    h1 = df[(df["metric"] == metric) & (df["cond"] == "h1")]["value"].astype(float).to_numpy()
    h2 = df[(df["metric"] == metric) & (df["cond"] == "h2")]["value"].astype(float).to_numpy()
    return h1, h2


def style_main_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=6)


def draw_main_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics: list[str],
    labels: list[str],
    detail_ranges: list[tuple[float, float]],
    title: str,
    letter: str,
    xlabel: str,
) -> None:
    y = np.arange(3)[::-1]
    for yi, metric, (xmin, xmax) in zip(y, metrics, detail_ranges):
        h1, h2 = values_for(df, metric)
        draw_box(ax, h1, yi + 0.16, COLORS["h1"])
        draw_box(ax, h2, yi - 0.16, COLORS["h2"])
        ax.add_patch(
            Rectangle(
                (xmin, yi - 0.38),
                xmax - xmin,
                0.76,
                facecolor=mpl.colors.to_rgba(COLORS["h2"], 0.045),
                edgecolor=mpl.colors.to_rgba(COLORS["h2"], 0.55),
                linewidth=0.7,
                linestyle=(0, (2, 2)),
                zorder=0,
            )
        )
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.48, 2.48)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=7.8, fontweight="bold", pad=9, color=COLORS["ink"])
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    style_main_axis(ax)


def draw_detail_axis(
    ax: plt.Axes,
    h1: np.ndarray,
    h2: np.ndarray,
    xlim: tuple[float, float],
    ticks: list[float],
    show_h1: bool,
    title: str | None = None,
) -> None:
    if show_h1:
        draw_box(ax, h1, 0.14, COLORS["h1"], width=0.20)
        draw_box(ax, h2, -0.14, COLORS["h2"], width=0.20)
    else:
        draw_box(ax, h2, 0.0, COLORS["h2"], width=0.25)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(-0.44, 0.44)
    ax.set_yticks([])
    ax.set_facecolor(COLORS["detail_bg"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["detail_edge"])
        spine.set_linewidth(0.75)
    if title:
        ax.set_title(title, fontsize=7.0, color=COLORS["muted"], fontweight="bold", pad=5)


def build_figure(lang: str) -> None:
    set_style(lang)
    t = TEXT[lang]
    df = pd.read_csv(SOURCE)
    df = df[df["condition"].isin(["H1_no_retrieval_LLM", "H2_KB_RAG_LLM"])].copy()
    df["cond"] = df["condition"].map(condition_key)

    fig = plt.figure(figsize=(7.20, 4.05))
    outer = fig.add_gridspec(1, 2, left=0.145, right=0.985, bottom=0.16, top=0.79, wspace=0.42)
    left = outer[0, 0].subgridspec(3, 2, width_ratios=[3.05, 1.20], hspace=0.60, wspace=0.14)
    right = outer[0, 1].subgridspec(3, 2, width_ratios=[3.05, 1.20], hspace=0.60, wspace=0.14)

    ax_a = fig.add_subplot(left[:, 0])
    ax_b = fig.add_subplot(right[:, 0])
    a_detail = [fig.add_subplot(left[i, 1]) for i in range(3)]
    b_detail = [fig.add_subplot(right[i, 1]) for i in range(3)]

    a_ranges = [(0.965, 1.001), (0.965, 1.001), (0.87, 1.001)]
    b_ranges = [(-0.001, 0.035), (0.90, 1.001), (0.35, 0.72)]
    draw_main_panel(ax_a, df, HIGH_METRICS, t["high_labels"], a_ranges, t["higher"], "a", t["rate"])
    draw_main_panel(ax_b, df, LOW_METRICS, t["low_labels"], b_ranges, t["lower"], "b", t["rate"])

    a_ticks = [[0.97, 0.985, 1.00], [0.97, 0.985, 1.00], [0.88, 0.94, 1.00]]
    b_ticks = [[0.00, 0.015, 0.03], [0.91, 0.95, 0.99], [0.38, 0.55, 0.72]]
    for idx, (metric, xlim, ticks) in enumerate(zip(HIGH_METRICS, a_ranges, a_ticks)):
        h1, h2 = values_for(df, metric)
        draw_detail_axis(a_detail[idx], h1, h2, xlim, ticks, show_h1=False, title=t["detail"] if idx == 0 else None)
    for idx, (metric, xlim, ticks) in enumerate(zip(LOW_METRICS, b_ranges, b_ticks)):
        h1, h2 = values_for(df, metric)
        draw_detail_axis(b_detail[idx], h1, h2, xlim, ticks, show_h1=idx in {1, 2}, title=t["detail"] if idx == 0 else None)

    handles = [
        Patch(facecolor=mpl.colors.to_rgba(COLORS["h1"], 0.28), edgecolor=COLORS["h1"], label=t["methods"][0]),
        Patch(facecolor=mpl.colors.to_rgba(COLORS["h2"], 0.28), edgecolor=COLORS["h2"], label=t["methods"][1]),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False, columnspacing=1.8)
    fig.text(0.5, 0.89, t["box_note"], ha="center", va="center", fontsize=6.2, color=COLORS["muted"])

    target = OUT / "figures" / ("English" if lang == "en" else "Chinese")
    target.mkdir(parents=True, exist_ok=True)
    stem = "Figure3_Graph_Assembly_Boxplots_Zoom_Round10"
    fig.savefig(target / f"{stem}.svg")
    fig.savefig(target / f"{stem}.pdf")
    fig.savefig(target / f"{stem}.png", dpi=450)
    fig.savefig(target / f"{stem}.tiff", dpi=600)
    plt.close(fig)


def write_audit() -> None:
    df = pd.read_csv(SOURCE)
    selected = df[df["condition"].isin(["H1_no_retrieval_LLM", "H2_KB_RAG_LLM"])]
    audit = {
        "figure": "Figure3_Graph_Assembly_Boxplots_Zoom_Round10",
        "source_sha256": sha256(SOURCE),
        "source_rows": int(len(df)),
        "selected_rows": int(len(selected)),
        "selected_graph_units_per_metric_and_method": 15,
        "changes": [
            "Replaced repeat-fold point clouds and mean markers with grouped horizontal box plots.",
            "Retained full 0-1 axes for direct H1/H2 comparison.",
            "Added metric-specific detail axes for compressed distributions.",
            "Used the same blue and coral method palette as revised Figure 2.",
            "No frozen values, denominators or statistical definitions were changed.",
        ],
        "inferential_statistics_added": False,
    }
    qa = OUT / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "Figure3_Round10_data_and_design_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_figure("en")
    build_figure("zh")
    write_audit()
