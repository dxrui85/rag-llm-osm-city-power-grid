from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Portable release overrides. The historical definitions above are retained as provenance.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
OUT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT", REPOSITORY_ROOT / "outputs" / "Figure_2"))
SOURCE = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_2" / "object_level_results.csv"

COLORS = {
    "h1": "#4C78A8",
    "h2": "#E76F51",
    "returned": "#4C78A8",
    "insufficient": "#F2A07B",
    "invalid": "#A9B4BD",
    "ink": "#263238",
    "muted": "#66727A",
    "grid": "#E5E9EC",
    "zoom_bg": "#FFF7F3",
}

TEXT = {
    "en": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "metrics": ["Accuracy", "Macro F1", "Prediction\ncoverage"],
        "classes": ["Power lines", "Network connection\npoints", "Power generation\nfacilities"],
        "states": ["Prediction returned", "Insufficient evidence", "Invalid or missing"],
        "score": "Score",
        "f1": "F1 score",
        "share": "Share of prediction positions (%)",
        "zoom": "H2 detail",
        "high_zoom": "High-score detail",
        "box_note": "Boxes show interquartile ranges; centre lines show medians; whiskers show min–max across three repeats.",
    },
    "zh": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "metrics": ["准确率", "宏平均 F1", "预测覆盖率"],
        "classes": ["电力线路", "网络连接点", "发电设施"],
        "states": ["返回预测", "证据不足", "无效或缺失"],
        "score": "指标值",
        "f1": "F1 值",
        "share": "占全部预测位置的比例（%）",
        "zoom": "H2 局部放大",
        "high_zoom": "高分区间放大",
        "box_note": "箱体表示四分位区间，中线表示中位数，须线表示三次重复的最小值至最大值。",
    },
}


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
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
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


def clean_axis(ax: plt.Axes, hide_left: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )


def draw_box(ax: plt.Axes, values: np.ndarray, position: float, color: str, width: float = 0.23) -> None:
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
        boxprops={"facecolor": mpl.colors.to_rgba(color, 0.28), "edgecolor": color, "linewidth": 1.1},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
        medianprops={"color": color, "linewidth": 2.0},
    )
    for patch in bp["boxes"]:
        patch.set_joinstyle("round")


def full_and_zoom_panel(
    ax: plt.Axes,
    ax_zoom: plt.Axes,
    groups: list[tuple[str, np.ndarray, np.ndarray]],
    labels: list[str],
    xlabel: str,
    zoom_xlim: tuple[float, float],
    zoom_ticks: list[float],
    zoom_title: str,
    zoom_h1_indices: set[int] | None = None,
) -> None:
    zoom_h1_indices = zoom_h1_indices or set()
    base_positions = np.arange(len(groups))[::-1] * 1.18
    offset = 0.18
    for idx, (base, (_, h1, h2)) in enumerate(zip(base_positions, groups)):
        draw_box(ax, h1, base + offset, COLORS["h1"])
        draw_box(ax, h2, base - offset, COLORS["h2"])
        if idx in zoom_h1_indices:
            draw_box(ax_zoom, h1, base + 0.13, COLORS["h1"], width=0.21)
            draw_box(ax_zoom, h2, base - 0.13, COLORS["h2"], width=0.21)
        else:
            draw_box(ax_zoom, h2, base, COLORS["h2"], width=0.28)

    ymin, ymax = base_positions.min() - 0.48, base_positions.max() + 0.48
    ax.set_yticks(base_positions, labels)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.15, 1.02)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(xlabel)
    ax.axvspan(zoom_xlim[0], 1.02, color=COLORS["h2"], alpha=0.055, zorder=0)
    ax.axvline(zoom_xlim[0], color=COLORS["h2"], alpha=0.55, linewidth=0.75, linestyle=(0, (2, 2)))
    clean_axis(ax)

    ax_zoom.set_facecolor(COLORS["zoom_bg"])
    ax_zoom.set_ylim(ymin, ymax)
    ax_zoom.set_xlim(*zoom_xlim)
    ax_zoom.set_xticks(zoom_ticks)
    ax_zoom.set_yticks(base_positions, [""] * len(base_positions))
    ax_zoom.tick_params(axis="y", length=0)
    ax_zoom.grid(axis="x", color="#EEDFD9", linewidth=0.65, zorder=0)
    ax_zoom.set_axisbelow(True)
    for spine in ax_zoom.spines.values():
        spine.set_visible(True)
        spine.set_color(mpl.colors.to_rgba(COLORS["h2"], 0.7))
        spine.set_linewidth(0.8)
    ax_zoom.set_title(zoom_title, color=COLORS["h2"], fontsize=7.0, fontweight="bold", pad=4)


def build_figure(lang: str) -> None:
    set_style(lang)
    t = TEXT[lang]
    df = pd.read_csv(SOURCE)
    df["cond"] = df["condition"].map(condition_key)

    fig = plt.figure(figsize=(7.20, 4.48))
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.10, 0.82],
        width_ratios=[1.0, 1.08],
        left=0.135,
        right=0.985,
        bottom=0.115,
        top=0.81,
        hspace=0.66,
        wspace=0.42,
    )
    gs_a = outer[0, 0].subgridspec(1, 2, width_ratios=[2.75, 1.10], wspace=0.12)
    gs_b = outer[0, 1].subgridspec(1, 2, width_ratios=[2.75, 1.10], wspace=0.12)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_a_zoom = fig.add_subplot(gs_a[0, 1])
    ax_b = fig.add_subplot(gs_b[0, 0])
    ax_b_zoom = fig.add_subplot(gs_b[0, 1])
    ax_c = fig.add_subplot(outer[1, :])

    a_df = df[df["panel"] == "a"].copy()
    a_groups = []
    for column in ["accuracy", "macro_f1", "prediction_coverage"]:
        h1 = a_df[a_df["cond"] == "h1"][column].astype(float).to_numpy()
        h2 = a_df[a_df["cond"] == "h2"][column].astype(float).to_numpy()
        a_groups.append((column, h1, h2))
    full_and_zoom_panel(ax_a, ax_a_zoom, a_groups, t["metrics"], t["score"], (0.97, 1.001), [0.97, 0.98, 0.99, 1.00], t["zoom"])
    panel_label(ax_a, "a")

    b_df = df[df["panel"] == "b"].copy()
    b_groups = []
    for class_code in ["ACLineSegment", "ConnectivityNode", "EnergySource"]:
        h1 = b_df[(b_df["class"] == class_code) & (b_df["cond"] == "h1")]["f1"].astype(float).to_numpy()
        h2 = b_df[(b_df["class"] == class_code) & (b_df["cond"] == "h2")]["f1"].astype(float).to_numpy()
        b_groups.append((class_code, h1, h2))
    full_and_zoom_panel(
        ax_b,
        ax_b_zoom,
        b_groups,
        t["classes"],
        t["f1"],
        (0.94, 1.001),
        [0.94, 0.96, 0.98, 1.00],
        t["high_zoom"],
        zoom_h1_indices={2},
    )
    panel_label(ax_b, "b")

    handles = [
        Patch(facecolor=mpl.colors.to_rgba(COLORS["h1"], 0.28), edgecolor=COLORS["h1"], label=t["methods"][0]),
        Patch(facecolor=mpl.colors.to_rgba(COLORS["h2"], 0.28), edgecolor=COLORS["h2"], label=t["methods"][1]),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False, columnspacing=1.8)
    fig.text(0.5, 0.905, t["box_note"], ha="center", va="center", fontsize=6.25, color=COLORS["muted"])

    c_df = df[df["panel"] == "c"].copy()
    state_group = {"valid_prediction": 0, "valid_insufficient_evidence": 1, "INVALID": 2, "missing": 2}
    state_colors = [COLORS["returned"], COLORS["insufficient"], COLORS["invalid"]]
    for row_y, cond in [(1, "h1"), (0, "h2")]:
        values = np.zeros(3)
        for _, row in c_df[c_df["cond"] == cond].iterrows():
            values[state_group[str(row["state"])]] += float(row["proportion"]) * 100
        left = 0.0
        for idx, value in enumerate(values):
            ax_c.barh(row_y, value, left=left, height=0.38, color=state_colors[idx], edgecolor="white", linewidth=0.7)
            if value >= 4:
                ax_c.text(left + value / 2, row_y, f"{value:.1f}%", ha="center", va="center", fontsize=6.5,
                          color="white" if idx == 0 else COLORS["ink"], fontweight="bold" if idx == 0 else "normal")
            left += value
    ax_c.set_yticks([1, 0], t["methods"])
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel(t["share"])
    clean_axis(ax_c)
    ax_c.text(
        -0.055,
        1.07,
        "c",
        transform=ax_c.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )
    ax_c.legend(
        handles=[Patch(color=color, label=label) for color, label in zip(state_colors, t["states"])],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        columnspacing=1.3,
    )

    target = OUT / "figures" / ("English" if lang == "en" else "Chinese")
    target.mkdir(parents=True, exist_ok=True)
    stem = "Figure2_Object_Level_Recovery_Zoom_Round9"
    fig.savefig(target / f"{stem}.svg")
    fig.savefig(target / f"{stem}.pdf")
    fig.savefig(target / f"{stem}.png", dpi=450)
    fig.savefig(target / f"{stem}.tiff", dpi=600)
    plt.close(fig)


def write_audit() -> None:
    source_df = pd.read_csv(SOURCE)
    audit = {
        "figure": "Figure2_Object_Level_Recovery_Zoom_Round9",
        "source_sha256": sha256(SOURCE),
        "source_rows": int(len(source_df)),
        "panel_counts": {str(k): int(v) for k, v in source_df.groupby("panel").size().items()},
        "changes": [
            "Panels a and b retain a common full scale for direct H1/H2 comparison.",
            "Detail axes enlarge narrow high-score ranges without changing any values.",
            "Panel b's high-score detail axis additionally shows the compressed H1 power-generation-facilities box plot.",
            "Panel c now uses the same blue, coral and blue-grey visual family as panels a and b.",
            "All frozen data and denominator-preserving response-state composition are unchanged.",
        ],
        "zoom_ranges": {"panel_a": [0.97, 1.001], "panel_b": [0.94, 1.001]},
        "inferential_statistics_added": False,
    }
    qa = OUT / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "Figure2_Round9_data_and_design_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_figure("en")
    build_figure("zh")
    write_audit()
