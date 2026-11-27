from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "figure_assets"
QA = ROOT / "qa"

RATES_FILE = DATA / "spatial_block_comparison.csv"
BOUNDARY_FILE = DATA / "410100.geojson"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
DATA = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S4"
OUTPUT_ROOT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
OUT = OUTPUT_ROOT / "Figure_S4"
QA = OUTPUT_ROOT / "Figure_S4_qa"
RATES_FILE = DATA / "spatial_block_comparison.csv"
BOUNDARY_FILE = Path(os.environ.get("ZHENGZHOU_BOUNDARY_GEOJSON", DATA_ROOT / "third_party" / "zhengzhou_administrative_boundary.geojson"))

EXPECTED_HASHES = {
    RATES_FILE: "130C44CCBDEA36A13E381FABD1ADFA02ACEFFCF3E9C2FD4E730AB0BC504C5888",
    BOUNDARY_FILE: "239797C921E3B1EB34357513AC3BA31AAAF9A7BFBADD9570DDD957659ABCEA12",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_and_load() -> pd.DataFrame:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {path.name}: {actual} != {expected}")
    rates = pd.read_csv(RATES_FILE)
    if len(rates) != 63:
        raise RuntimeError(f"Expected 63 occupied blocks, found {len(rates)}")
    return rates


def polygon_rings(geometry: dict):
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    for polygon in polygons:
        for ring in polygon:
            yield np.asarray(ring, dtype=float)


def load_boundary() -> list[np.ndarray]:
    obj = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
    rings: list[np.ndarray] = []
    for feature in obj["features"]:
        rings.extend(polygon_rings(feature["geometry"]))
    return rings


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_cells(ax, rates: pd.DataFrame, column: str, cmap, norm) -> None:
    for row in rates.itertuples(index=False):
        value = getattr(row, column)
        ax.add_patch(
            Rectangle(
                (row.lon_min, row.lat_min),
                0.125,
                0.125,
                facecolor=cmap(norm(value)),
                edgecolor="white",
                linewidth=0.35,
                zorder=1,
            )
        )
        if row.unique_objects < 30:
            ax.add_patch(
                Rectangle(
                    (row.lon_min, row.lat_min),
                    0.125,
                    0.125,
                    facecolor="none",
                    edgecolor="#202020",
                    linewidth=0.35,
                    hatch="///",
                    zorder=2,
                )
            )


def decorate_map(ax, rings: list[np.ndarray], label: str, title: str) -> None:
    for ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#4A4A4A", linewidth=0.65, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.text(-0.09, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.grid(False)
    ax.tick_params(labelsize=8)


def make_figure(rates: pd.DataFrame, rings: list[np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.5), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.11, wspace=0.16, hspace=0.50)
    fig.suptitle(
        "Spatial distribution of category recovery across Zhengzhou",
        fontsize=16,
        fontweight="bold",
    )

    rings_xy = np.vstack(rings)
    xmin, ymin = rings_xy.min(axis=0)
    xmax, ymax = rings_xy.max(axis=0)

    score_cmap = mpl.colormaps["YlGnBu"]
    score_norm = Normalize(vmin=0.0, vmax=1.0)

    # Panel b: narrow high-value intervals receive visibly different colours.
    h2_bounds = np.array([0.66, 0.90, 0.94, 0.97, 0.98, 0.99, 0.995, 1.00001])
    h2_colors = [
        "#EDF7FB",
        "#CBE7F1",
        "#8FD0DF",
        "#49AFC6",
        "#2385B6",
        "#2459A1",
        "#192F6A",
    ]
    h2_cmap = ListedColormap(h2_colors, name="h2_high_contrast")
    h2_norm = BoundaryNorm(h2_bounds, h2_cmap.N, clip=True)

    # Panel c: resolution is concentrated around the observed 30-40 pp range.
    gain_bounds = np.array([0.0, 25.0, 30.0, 33.0, 36.0, 40.0, 60.0, 80.0, 100.0001])
    gain_colors = [
        "#FFF4CC",
        "#FDDC92",
        "#FDB66E",
        "#F47C5E",
        "#DF4B5E",
        "#B8326A",
        "#7A2876",
        "#3E1B63",
    ]
    gain_cmap = ListedColormap(gain_colors, name="gain_high_contrast")
    gain_norm = BoundaryNorm(gain_bounds, gain_cmap.N, clip=True)

    support_cmap = mpl.colormaps["viridis"]
    support_norm = LogNorm(
        vmin=max(1, int(rates["unique_objects"].min())),
        vmax=int(rates["unique_objects"].max()),
    )

    specs = [
        (axes[0, 0], "h1_correct_share", score_cmap, score_norm, "a", "H1 LLM: correct share"),
        (axes[0, 1], "h2_correct_share", h2_cmap, h2_norm, "b", "H2 RAG-LLM: correct share"),
        (axes[1, 0], "h2_minus_h1_pp", gain_cmap, gain_norm, "c", "H2 minus H1 (percentage points)"),
        (axes[1, 1], "unique_objects", support_cmap, support_norm, "d", "Unique objects per occupied block"),
    ]

    for ax, column, cmap, norm, label, title in specs:
        draw_cells(ax, rates, column, cmap, norm)
        decorate_map(ax, rings, label, title)
        ax.set_xlim(xmin - 0.06, xmax + 0.06)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        cbar = fig.colorbar(
            scalar,
            ax=ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.12,
            aspect=30,
            boundaries=(h2_bounds if column == "h2_correct_share" else gain_bounds if column == "h2_minus_h1_pp" else None),
        )
        cbar.ax.tick_params(labelsize=8)
        if column == "h1_correct_share":
            cbar.set_label("Correct share of fixed denominator", fontsize=9)
        elif column == "h2_correct_share":
            cbar.set_ticks([0.66, 0.90, 0.94, 0.97, 0.98, 0.99, 1.00])
            cbar.set_ticklabels(["0.66", "0.90", "0.94", "0.97", "0.98", "0.99", "1.00"])
            cbar.set_label("Correct share (expanded high-value scale)", fontsize=9)
        elif column == "h2_minus_h1_pp":
            cbar.set_ticks([0, 25, 30, 33, 36, 40, 60, 80, 100])
            cbar.set_label("Percentage-point gain (expanded central range)", fontsize=9)
        else:
            cbar.set_label("Unique object count (log scale)", fontsize=9)

    axes[0, 0].text(
        0.02,
        0.025,
        "Hatched cells: <30 unique objects",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color="#202020",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0),
        zorder=5,
    )
    axes[1, 1].annotate(
        "N",
        xy=(0.93, 0.94),
        xytext=(0.93, 0.84),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#202020", linewidth=1.2),
    )

    positive = int((rates["h2_minus_h1_pp"] > 1e-12).sum())
    tied = int((rates["h2_minus_h1_pp"].abs() <= 1e-12).sum())
    fig.text(
        0.5,
        0.012,
        f"0.125° occupied blocks; three repeats pooled descriptively. H2 exceeded H1 in {positive}/63 blocks ({tied} tie). "
        "Insufficient-evidence, invalid and missing positions remain errors in the fixed denominator.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    return fig


def export(fig: plt.Figure) -> dict[str, dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Figure_S4_spatial_category_recovery_high_contrast"
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


def write_qa(rates: pd.DataFrame, outputs: dict[str, dict[str, object]]) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    png = OUT / "Figure_S4_spatial_category_recovery_high_contrast.png"
    Image.open(png).convert("L").save(
        QA / "Figure_S4_spatial_category_recovery_high_contrast_grayscale.png",
        dpi=(300, 300),
    )
    svg = (OUT / "Figure_S4_spatial_category_recovery_high_contrast.svg").read_text(encoding="utf-8")
    report = {
        "source_hash_gate": True,
        "source_data_unchanged": True,
        "occupied_blocks": int(len(rates)),
        "panel_b_value_range": [float(rates.h2_correct_share.min()), float(rates.h2_correct_share.max())],
        "panel_b_discrete_boundaries": [0.66, 0.90, 0.94, 0.97, 0.98, 0.99, 0.995, 1.00001],
        "panel_c_value_range": [float(rates.h2_minus_h1_pp.min()), float(rates.h2_minus_h1_pp.max())],
        "panel_c_discrete_boundaries": [0, 25, 30, 33, 36, 40, 60, 80, 100.0001],
        "editable_svg_text": "<text" in svg,
        "exported_files": outputs,
    }
    report["passed"] = all(
        [
            report["source_hash_gate"],
            report["source_data_unchanged"],
            report["occupied_blocks"] == 63,
            report["editable_svg_text"],
        ]
    )
    (QA / "figure_s4_colour_revision_qa.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure()
    rates = verify_and_load()
    outputs = export(make_figure(rates, load_boundary()))
    write_qa(rates, outputs)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
