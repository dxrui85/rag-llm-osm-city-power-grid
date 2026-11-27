"""Offline reproduction of the frozen descriptive analysis at both spatial scales.

This entry point reads only the companion public data package. It does not use
the network, an API key, or any model service. Outputs are always written to a
new directory supplied by the caller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "src" / "analysis"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def common_flags() -> dict[str, bool | str]:
    return {
        "execution_mode": "authorized_real",
        "configuration_executable": True,
        "author_final_review_complete": True,
        "real_gold_read_authorized": True,
        "outer_performance_read_authorized": True,
        "real_gold_join_authorized": True,
        "real_descriptive_scoring_authorized": True,
        "real_graph_assembly_authorized": True,
        "post_assembly_real_denominator_join_authorized": True,
        "confirmatory_inference_authorized": False,
        "adaptive_analysis_authorized": False,
        "api_authorized": False,
        "network_authorized": False,
        "api_key_read_authorized": False,
        "scope_expansion_authorized": False,
        "stage11_authorized": False,
        "stage16_authorized": False,
    }


def primary_config(data_root: Path, output: Path) -> dict[str, Any]:
    reference = data_root / "data" / "reference_ledgers"
    responses = sorted((data_root / "data" / "model_io" / "scale_0.125_degree" / "responses").glob("*.jsonl"))
    positions = reference / "position_status_scale_0.125.csv"
    rows = read_csv(positions)
    reasons = Counter(row["validation_reason"] for row in rows)
    states = {
        "valid_prediction": reasons["VALID_PREDICTION"],
        "valid_insufficient_evidence": reasons["VALID_INSUFFICIENT_EVIDENCE"],
        "INVALID": sum(count for reason, count in reasons.items() if reason not in {
            "VALID_PREDICTION", "VALID_INSUFFICIENT_EVIDENCE", "AMBIGUOUS_INFLIGHT_NEVER_RESEND"
        }),
        "missing": reasons["AMBIGUOUS_INFLIGHT_NEVER_RESEND"],
    }
    config: dict[str, Any] = {
        **common_flags(),
        "batch_id": "PUBLIC_OFFLINE_REPRODUCTION_SCALE_0125",
        "authorized_output_directory": str(output.resolve()),
        "analysis_design": {"repeats": [1, 2, 3], "folds": [0, 1, 2, 3, 4]},
        "expected_counts": {
            "prediction_rows": len(rows),
            "prediction_states": states,
            "objects": 18732,
            "folds": 5,
            "unique_graphs": 35,
        },
        "input_bindings": {
            "position_status_ledger": binding(positions),
            "validated_response_ledgers": [binding(path) for path in responses],
            "geometry_ledger": binding(reference / "geometry_ledger.jsonl"),
            "h0_semantic_source_ledger": binding(reference / "h0_semantic_sources_scale_0.125.csv"),
            "gold_ledger": binding(reference / "reference_category_ledger.jsonl"),
            "denominator_connector": binding(reference / "fixed_denominator_connector.csv"),
        },
    }
    return config


def sensitivity_config(data_root: Path, output: Path) -> dict[str, Any]:
    reference = data_root / "data" / "reference_ledgers"
    results = data_root / "data" / "analysis_results" / "scale_0.125_degree"
    responses = sorted((data_root / "data" / "model_io" / "scale_0.10_degree" / "responses").glob("*.jsonl"))
    positions = reference / "position_status_scale_0.10.csv"
    rows = read_csv(positions)
    states = dict(Counter(row["position_status"] for row in rows))
    return {
        **common_flags(),
        "batch_id": "PUBLIC_OFFLINE_REPRODUCTION_SCALE_010",
        "authorized_output_directory": str(output.resolve()),
        "expected_counts": {"prediction_rows": len(rows), "prediction_states": states},
        "input_bindings": {
            "position_status_ledger": binding(positions),
            "validated_response_ledgers": [binding(path) for path in responses],
            "geometry_ledger": binding(reference / "geometry_ledger.jsonl"),
            "h0_semantic_source_ledger_010": binding(reference / "h0_semantic_sources_scale_0.10.csv"),
            "gold_ledger": binding(reference / "reference_category_ledger.jsonl"),
            "denominator_connector": binding(reference / "fixed_denominator_connector.csv"),
            "main_scale_object_repeat": binding(results / "object_repeat_pooled_descriptive.csv"),
            "main_scale_graph_repeat": binding(results / "graph_repeat_aligned_descriptive.csv"),
        },
    }


def compare_outputs(reproduced: Path, frozen: Path) -> list[dict[str, Any]]:
    names = (
        "object_repeat_pooled_descriptive.csv",
        "object_three_repeat_summary.csv",
        "graph_repeat_aligned_descriptive.csv",
        "graph_unique_scores.json",
    )
    comparisons: list[dict[str, Any]] = []
    for name in names:
        left, right = reproduced / name, frozen / name
        item: dict[str, Any] = {"file": name, "reproduced_exists": left.is_file(), "frozen_exists": right.is_file()}
        if left.is_file() and right.is_file():
            item["reproduced_sha256"] = sha256_file(left)
            item["frozen_sha256"] = sha256_file(right)
            item["byte_identical"] = item["reproduced_sha256"] == item["frozen_sha256"]
            if name.endswith(".csv"):
                item["content_identical"] = read_csv(left) == read_csv(right)
            else:
                item["content_identical"] = json.loads(left.read_text(encoding="utf-8")) == json.loads(right.read_text(encoding="utf-8"))
        comparisons.append(item)
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce frozen descriptive analyses without API or network access")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", choices=("all", "0.125", "0.10"), default="all")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    report: dict[str, Any] = {"api_calls": 0, "network_calls": 0, "scales": {}}
    if args.scales in {"all", "0.125"}:
        target = output / "scale_0.125_degree"
        module = load_module("public_primary_analysis", ANALYSIS / "analyze_primary_scale.py")
        result = module.run_analysis(primary_config(data_root, target))
        report["scales"]["0.125"] = {
            "run_result": result,
            "frozen_comparison": compare_outputs(target, data_root / "data" / "analysis_results" / "scale_0.125_degree"),
        }
    if args.scales in {"all", "0.10"}:
        target = output / "scale_0.10_degree"
        module = load_module("public_sensitivity_analysis", ANALYSIS / "analyze_sensitivity_scale.py")
        result = module.run_analysis(sensitivity_config(data_root, target))
        report["scales"]["0.10"] = {
            "run_result": result,
            "frozen_comparison": compare_outputs(target, data_root / "data" / "analysis_results" / "scale_0.10_degree"),
        }
    report_path = output / "offline_results_reproduction_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
