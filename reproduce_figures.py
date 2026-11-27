"""Regenerate all figures supported by the public data package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    ("Figure 2", "src/figures/main/make_figure_2.py", "Figure_2", "single"),
    ("Figure 3", "src/figures/main/make_figure_3.py", "Figure_3", "single"),
    ("Figure 5", "src/figures/main/make_figure_5.py", "Figure_5", "single"),
    ("Figure S1", "src/figures/supplementary/make_figure_s1.py", "Figure_S1", "root"),
    ("Figure S2", "src/figures/supplementary/make_figure_s2.py", "Figure_S2", "root"),
    ("Figure S3", "src/figures/supplementary/make_figure_s3.py", "Figure_S3", "single"),
    ("Figure S5", "src/figures/supplementary/make_figure_s5.py", "Figure_S5", "root"),
    ("Figure S6", "src/figures/supplementary/make_figure_s6.py", "Figure_S6", "root"),
    ("Figure S7", "src/figures/supplementary/make_figure_s7.py", "Figure_S7", "root"),
    ("Figure S8", "src/figures/supplementary/make_figure_s8.py", "Figure_S8", "root"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate public figures without API or network access")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    report: dict[str, object] = {
        "api_calls": 0,
        "network_calls": 0,
        "generated": [],
        "not_regenerated": [
            {"figure": "Figure 1", "reason": "editable PowerPoint source is distributed; rendering depends on desktop presentation software"},
            {"figure": "Figure 4", "reason": "administrative boundary is excluded pending redistribution-rights confirmation"},
            {"figure": "Figure S4", "reason": "administrative boundary is excluded pending redistribution-rights confirmation"},
        ],
    }
    for label, relative, folder, output_mode in SCRIPTS:
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "RAG_OSM_DATA_ROOT": str(data_root)})
        if output_mode == "single":
            env["RAG_OSM_FIGURE_OUTPUT"] = str(output / folder)
        else:
            env["RAG_OSM_FIGURE_OUTPUT_ROOT"] = str(output)
        process = subprocess.run(
            [sys.executable, str(ROOT / relative)], cwd=ROOT, env=env,
            text=True, capture_output=True, check=False,
        )
        record = {
            "figure": label,
            "script": relative,
            "returncode": process.returncode,
            "stdout": process.stdout[-2000:],
            "stderr": process.stderr[-2000:],
        }
        report["generated"].append(record)
        if process.returncode != 0:
            (output / "offline_figure_reproduction_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise SystemExit(f"Figure reproduction failed: {label}")
    report_path = output / "offline_figure_reproduction_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "generated": len(SCRIPTS), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
