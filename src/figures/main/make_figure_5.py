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

# Portable release overrides. The historical definitions above are retained as provenance.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
OUT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT", REPOSITORY_ROOT / "outputs" / "Figure_5"))
SOURCE_DIR = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_5"
PRED = SOURCE_DIR / "prediction_outcomes_by_rule_applicability.csv"
EVIDENCE = SOURCE_DIR / "evidence_channel_coverage.csv"
PATHS = SOURCE_DIR / "paired_disposition_by_gold_class.csv"

COLORS = {
    "h1": "#4C78A8",
    "h2": "#E76F51",
    "retained": "#4C78A8",
    "rescued": "#E76F51",
    "degraded": "#C97A87",
    "unresolved": "#A9B4BD",
    "ink": "#263238",
    "muted": "#66727A",
    "grid": "#E5E9EC",
}

TEXT = {
    "en": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "panel_a_title": "Correct predictions by available evidence",
        "panel_b_title": "Prediction transitions from H1 to H2",
        "groups": ["Labelled examples\n+ matched rule\n(n = 3,495)", "Labelled examples only\n(no rule match)\n(n = 52,701)"],
        "ylabel": "Correct predictions (%)",
        "xlabel_a": "Knowledge evidence available to H2",
        "classes": ["Power lines", "Network connection\npoints", "Power generation\nfacilities"],
        "paths": ["Retained\ncorrect", "Corrected\nby H2", "Degraded", "Unresolved\nor other"],
        "xlabel_b": "Share within each facility class (%)",
    },
    "zh": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "panel_a_title": "按可用证据分层的正确预测率",
        "panel_b_title": "从H1到H2的预测结果转移",
        "groups": ["已标注样例\n+匹配规则\n（n = 3,495）", "仅已标注样例\n（无规则匹配）\n（n = 52,701）"],
        "ylabel": "正确预测率（%）",
        "xlabel_a": "H2可用的知识证据",
        "classes": ["电力线路", "网络连接点", "发电设施"],
        "paths": ["保持正确", "由H2\n纠正", "由正确\n变差", "未解决\n或其他"],
        "xlabel_b": "各设施类别内的比例（%）",
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
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=COLORS["ink"])


def blend_with_white(hex_color: str, strength: float) -> tuple[float, float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    mixed = np.ones(3) * (1 - strength) + rgb * strength
    return (*mixed, 1.0)


def build_figure(lang: str) -> None:
    set_style(lang)
    t = TEXT[lang]
    pred = pd.read_csv(PRED)
    paths = pd.read_csv(PATHS)

    fig = plt.figure(figsize=(7.20, 3.38))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.93, 1.30], left=0.105, right=0.985, bottom=0.22, top=0.78, wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    states = ["MATCHED", "RULE_NONE"]
    methods = ["H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]
    # Pull the two evidence strata inward while retaining room for their
    # multi-line labels. A fixed x range prevents autoscaling from undoing
    # the intended visual compression.
    xs = [0.12, 0.88]
    for x, state in zip(xs, states):
        values = []
        for method in methods:
            row = pred[(pred["rule_match_state"] == state) & (pred["method"] == method) & (pred["outcome"] == "Correct")]
            values.append(float(row["percent"].iloc[0]))
        ax_a.plot([x - 0.085, x + 0.085], values, color="#CDD3D7", linewidth=2.6, zorder=1)
        ax_a.scatter(x - 0.085, values[0], s=50, color=COLORS["h1"], edgecolor="white", linewidth=0.7, zorder=3)
        ax_a.scatter(x + 0.085, values[1], s=50, color=COLORS["h2"], edgecolor="white", linewidth=0.7, zorder=3)
        ax_a.text(x - 0.085, values[0] + 1.15, f"{values[0]:.1f}", ha="center", color=COLORS["h1"], fontsize=6.4)
        ax_a.text(x + 0.085, values[1] + 1.15, f"{values[1]:.1f}", ha="center", color=COLORS["h2"], fontsize=6.4)
    ax_a.set_xticks(xs, t["groups"])
    ax_a.set_xlim(-0.12, 1.12)
    ax_a.set_ylim(55, 102)
    ax_a.set_yticks([60, 70, 80, 90, 100])
    ax_a.set_ylabel(t["ylabel"])
    ax_a.set_xlabel(t["xlabel_a"], labelpad=8)
    ax_a.set_title(t["panel_a_title"], loc="left", fontweight="bold", pad=10, color=COLORS["ink"])
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax_a.set_axisbelow(True)
    ax_a.spines[["top", "right"]].set_visible(False)
    panel_label(ax_a, "a")

    class_codes = ["ACLineSegment", "ConnectivityNode", "EnergySource"]
    dispositions = ["Retained correct", "Rescued to correct", "Degraded from correct", "Unresolved / other"]
    base_colors = [COLORS["retained"], COLORS["rescued"], COLORS["degraded"], COLORS["unresolved"]]
    mat = np.zeros((3, 4))
    for i, class_code in enumerate(class_codes):
        for j, disposition in enumerate(dispositions):
            row = paths[(paths["gold_class"] == class_code) & (paths["paired_disposition"] == disposition)]
            mat[i, j] = float(row["percent"].iloc[0])
            strength = 0.16 + 0.76 * (mat[i, j] / 100.0)
            ax_b.add_patch(
                mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=blend_with_white(base_colors[j], strength), edgecolor="white", linewidth=1.0)
            )
            text_color = "white" if strength > 0.57 else COLORS["ink"]
            ax_b.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6.7, color=text_color, fontweight="normal")
    ax_b.set_xlim(-0.5, 3.5)
    ax_b.set_ylim(2.5, -0.5)
    ax_b.set_yticks(range(3), t["classes"])
    ax_b.set_xticks(range(4), t["paths"])
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.set_xlabel(t["xlabel_b"], labelpad=7)
    ax_b.set_title(t["panel_b_title"], loc="left", fontweight="bold", pad=10, color=COLORS["ink"])
    panel_label(ax_b, "b")

    handles = [
        Line2D([0], [0], marker="o", markersize=6, color="none", markerfacecolor=COLORS["h1"], markeredgecolor="white", label=t["methods"][0]),
        Line2D([0], [0], marker="o", markersize=6, color="none", markerfacecolor=COLORS["h2"], markeredgecolor="white", label=t["methods"][1]),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, columnspacing=1.8)

    target = OUT / "figures" / ("English" if lang == "en" else "Chinese")
    target.mkdir(parents=True, exist_ok=True)
    stem = "Figure5_Retrieval_Mechanism_Round14"
    fig.savefig(target / f"{stem}.svg")
    fig.savefig(target / f"{stem}.pdf")
    fig.savefig(target / f"{stem}.png", dpi=450)
    fig.savefig(target / f"{stem}.tiff", dpi=600)
    plt.close(fig)


def write_audit() -> None:
    ev = pd.read_csv(EVIDENCE)
    example_coverage = float(ev.loc[ev["evidence_channel"] == "EX retrieved examples", "coverage_percent"].iloc[0])
    rule_coverage = float(ev.loc[ev["evidence_channel"] == "RULE evidence", "coverage_percent"].iloc[0])
    audit = {
        "figure": "Figure5_Retrieval_Mechanism_Round14",
        "source_hashes": {p.name: sha256(p) for p in [PRED, EVIDENCE, PATHS]},
        "example_retrieval_coverage_percent": example_coverage,
        "rule_evidence_coverage_percent": rule_coverage,
        "label_decision": {
            "MATCHED": "Labelled examples + matched rule",
            "RULE_NONE": "Labelled examples only (no rule match)",
            "rationale": "Labelled examples were available for all H2 positions; the grouping variable distinguishes whether a matching rule was also available.",
        },
        "changes": [
            "Removed the former evidence-availability panel.",
            "Renumbered the former panels b and c as panels a and b.",
            "Replaced ambiguous rule-retrieval labels with combined example/rule evidence labels.",
            "Defined the strata by evidence available to H2; H1 itself uses no retrieved evidence.",
            "Recoloured both panels using the same blue, coral and blue-grey family as Figures 1-3.",
            "Standardized all numerical labels in panel b to regular font weight.",
            "Reduced the horizontal separation between the two evidence strata in panel a while preserving label readability.",
        ],
        "inferential_statistics_added": False,
    }
    qa = OUT / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "Figure5_Round14_data_and_design_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_figure("en")
    build_figure("zh")
    write_audit()
