from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("H1_no_retrieval_LLM", "H2_KB_RAG_LLM")
CONDITIONS = ("H0_no_completion", *METHODS)
TARGET_CLASSES = ("ACLineSegment", "ConnectivityNode", "EnergySource")
BASE_EXECUTOR = Path(__file__).with_name("role_assembly_and_scoring.py")


class ContractViolation(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ContractViolation(f"invalid JSONL:{path.name}:{line_no}") from exc
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ContractViolation("cannot write empty normalized prediction ledger")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_base():
    spec = importlib.util.spec_from_file_location("outer_analysis_base_20261111", BASE_EXECUTOR)
    if spec is None or spec.loader is None:
        raise ContractViolation("cannot load frozen base executor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.METHODS = METHODS
    module.CONDITIONS = CONDITIONS
    return module


def validate_activation(config: dict[str, Any]) -> None:
    required_true = (
        "configuration_executable", "author_final_review_complete", "real_gold_read_authorized",
        "outer_performance_read_authorized", "real_gold_join_authorized",
        "real_descriptive_scoring_authorized", "real_graph_assembly_authorized",
        "post_assembly_real_denominator_join_authorized",
    )
    prohibited = (
        "confirmatory_inference_authorized", "adaptive_analysis_authorized", "api_authorized",
        "network_authorized", "api_key_read_authorized", "scope_expansion_authorized",
        "stage11_authorized",
    )
    if config.get("execution_mode") == "synthetic_regression":
        if config.get("synthetic_fixture_authorized") is not True:
            raise ContractViolation("synthetic fixture authorization absent")
        root = Path(config["synthetic_fixture_root"]).resolve()
        paths = [Path(v["path"]).resolve() for k, v in config["input_bindings"].items() if k != "validated_response_ledgers"]
        paths.extend(Path(v["path"]).resolve() for v in config["input_bindings"]["validated_response_ledgers"])
        if any(root != p and root not in p.parents for p in paths):
            raise ContractViolation("synthetic input escaped fixture root")
        return
    if config.get("execution_mode") != "authorized_real":
        raise ContractViolation("real executor remains non-executable")
    drift = [key for key in required_true if config.get(key) is not True]
    drift += [key for key in prohibited if config.get(key) is not False]
    if drift:
        raise ContractViolation("activation flag drift:" + ",".join(drift))


def verify_bindings(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    bindings = config["input_bindings"]
    for name, binding in bindings.items():
        if name == "validated_response_ledgers":
            continue
        path = Path(binding["path"])
        if not path.is_file():
            raise ContractViolation(f"missing bound input:{name}")
        digest = sha256_file(path)
        if digest != binding["sha256"].upper():
            raise ContractViolation(f"hash drift:{name}")
        observed[name] = digest
    for index, binding in enumerate(bindings["validated_response_ledgers"], 1):
        path = Path(binding["path"])
        if not path.is_file():
            raise ContractViolation(f"missing validated response ledger:{index}")
        digest = sha256_file(path)
        if digest != binding["sha256"].upper():
            raise ContractViolation(f"hash drift:validated_response_ledger_{index}")
        observed[f"validated_response_ledger_{index}"] = digest
    return observed


def _validation_to_prediction(response: dict[str, Any]) -> tuple[str, str]:
    validation = response.get("validation") or {}
    state = validation.get("state")
    parsed = validation.get("parsed_content") or {}
    if state == "INVALID":
        return "INVALID", ""
    if state != "VALID":
        raise ContractViolation("validated response state outside VALID/INVALID")
    status = parsed.get("status")
    if status == "prediction":
        predicted = parsed.get("class")
        if predicted not in TARGET_CLASSES:
            raise ContractViolation("predicted class outside frozen ontology")
        return "valid_prediction", predicted
    if status == "insufficient_evidence":
        return "valid_insufficient_evidence", ""
    raise ContractViolation("VALID response has unsupported parsed status")


def materialize_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
    status_rows = read_csv(Path(config["input_bindings"]["position_status_ledger"]["path"]))
    response_by_sequence: dict[int, dict[str, Any]] = {}
    for binding in config["input_bindings"]["validated_response_ledgers"]:
        for response in read_jsonl(Path(binding["path"])):
            sequence = int(response["global_sequence"])
            if sequence in response_by_sequence:
                raise ContractViolation("duplicate validated response position")
            response_by_sequence[sequence] = response

    if len(status_rows) != int(config["expected_counts"]["prediction_rows"]):
        raise ContractViolation("position-status row count drift")
    seen_identity: set[tuple[int, int, int, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    expected_sequences = set()
    for row in status_rows:
        sequence = int(row["global_sequence"])
        expected_sequences.add(sequence)
        method = row["method"]
        if method not in METHODS:
            raise ContractViolation("prediction method drift")
        identity = (int(row["repeat_id"]), int(row["seed"]), int(row["outer_fold"]), method, row["object_id"])
        if identity in seen_identity:
            raise ContractViolation("duplicate prediction identity")
        seen_identity.add(identity)
        position_state = row["position_state"]
        response = response_by_sequence.get(sequence)
        if position_state == "MISSING_QUARANTINED":
            if response is not None:
                raise ContractViolation("quarantined missing position has a response")
            prediction_state, predicted_class = "missing", ""
        elif position_state in {"VALID", "INVALID"}:
            if response is None:
                raise ContractViolation("received position lacks validated response")
            for key, expected in (
                ("method", method), ("object_id", row["object_id"]),
                ("repeat_id", int(row["repeat_id"])), ("outer_fold", int(row["outer_fold"])),
            ):
                if response.get(key) != expected:
                    raise ContractViolation(f"response/status identity drift:{key}")
            prediction_state, predicted_class = _validation_to_prediction(response)
            if position_state == "VALID" and prediction_state == "INVALID":
                raise ContractViolation("status ledger says VALID but response validates INVALID")
            if position_state == "INVALID" and prediction_state != "INVALID":
                raise ContractViolation("status ledger says INVALID but response is not INVALID")
        else:
            raise ContractViolation("position state outside frozen contract")
        normalized.append({
            "global_sequence": sequence,
            "pair_id": row["pair_id"],
            "repeat_id": int(row["repeat_id"]),
            "seed": int(row["seed"]),
            "outer_test_fold": int(row["outer_fold"]),
            "method": method,
            "object_id": row["object_id"],
            "partition_id": row.get("source_run", ""),
            "technical_status": position_state,
            "prediction_state": prediction_state,
            "predicted_class": predicted_class,
            "metric_token": predicted_class or f"__{prediction_state.upper()}__",
            "gold_class": "NOT_READ",
            "correctness": "NOT_COMPUTED",
        })
    if set(response_by_sequence) - expected_sequences:
        raise ContractViolation("validated response outside frozen position domain")
    counts: dict[str, int] = {}
    for row in normalized:
        counts[row["prediction_state"]] = counts.get(row["prediction_state"], 0) + 1
    expected_states = config["expected_counts"].get("prediction_states")
    if expected_states is not None and counts != expected_states:
        raise ContractViolation(f"prediction-state count drift:{counts}")
    return sorted(normalized, key=lambda r: int(r["global_sequence"]))


def load_geometry_and_h0(config: dict[str, Any], base):
    geometry_rows = read_jsonl(Path(config["input_bindings"]["geometry_ledger"]["path"]))
    h0_rows = read_csv(Path(config["input_bindings"]["h0_semantic_source_ledger"]["path"]))
    geometry = {row["object_id"]: row for row in geometry_rows}
    if len(geometry) != len(geometry_rows) or len(geometry) != int(config["expected_counts"]["objects"]):
        raise ContractViolation("geometry object-domain drift")
    if any("gold_class" in row or "candidate_class" in row for row in geometry_rows):
        raise ContractViolation("gold field in geometry ledger")
    h0: dict[tuple[int, str], dict[str, str]] = {}
    for row in h0_rows:
        key = (int(row["outer_test_fold"]), row["object_id"])
        if key in h0:
            raise ContractViolation("duplicate H0 semantic-source identity")
        if base._truth(row.get("masked_gold_accessed_by_H0_branch")):
            raise ContractViolation("H0 masked branch reports gold access")
        h0[key] = row
    expected_h0 = int(config["expected_counts"]["objects"]) * int(config["expected_counts"]["folds"])
    if len(h0) != expected_h0:
        raise ContractViolation("H0 semantic-source row count drift")
    return geometry, h0


def descriptive_outputs_safe(joined: list[dict[str, Any]], output_dir: Path, base) -> None:
    """Preserve the frozen aggregation order while allowing explicit NA in tiny synthetic folds."""
    base.write_csv(output_dir / "immutable_gold_join_ledger.csv", joined)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(int(row["repeat_id"]), int(row["outer_test_fold"]), row["condition"])].append(row)
    diagnostics = []
    for key, rows in sorted(groups.items()):
        score = base.score_rows(rows)
        diagnostics.append({"repeat_id": key[0], "outer_test_fold": key[1], "condition": key[2],
                            "n": score["n"], "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                            "prediction_coverage": score["prediction_coverage"],
                            "state_counts_json": json.dumps(score["state_counts"], sort_keys=True),
                            "per_class_json": json.dumps(score["per_class"], sort_keys=True)})
    base.write_csv(output_dir / "object_fold_diagnostics.csv", diagnostics)
    pooled = []
    for repeat in sorted({int(r["repeat_id"]) for r in joined}):
        for condition in CONDITIONS:
            rows = [r for r in joined if int(r["repeat_id"]) == repeat and r["condition"] == condition]
            score = base.score_rows(rows)
            pooled.append({"repeat_id": repeat, "condition": condition, "n": score["n"],
                           "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                           "prediction_coverage": score["prediction_coverage"],
                           "state_counts_json": json.dumps(score["state_counts"], sort_keys=True),
                           "per_class_json": json.dumps(score["per_class"], sort_keys=True)})
    base.write_csv(output_dir / "object_repeat_pooled_descriptive.csv", pooled)
    summaries = []
    for condition in CONDITIONS:
        rows = [r for r in pooled if r["condition"] == condition]
        for metric in ("accuracy", "macro_f1", "prediction_coverage"):
            values = [float(r[metric]) for r in rows if r[metric] is not None]
            mean = sum(values) / len(values) if values else None
            sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else (0.0 if len(values) == 1 else None)
            summaries.append({"condition": condition, "metric": metric, "repeats": len(values),
                              "mean": mean, "sample_sd": sd,
                              "minimum": min(values) if values else None,
                              "maximum": max(values) if values else None})
    base.write_csv(output_dir / "object_three_repeat_summary.csv", summaries)


def run_analysis(config: dict[str, Any], activation_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_activation(config)
    observed = verify_bindings(config)
    output_dir = Path(config["authorized_output_directory"])
    if output_dir.exists():
        raise ContractViolation("authorized output directory already exists")
    base = load_base()
    predictions = materialize_predictions(config)
    geometry, h0 = load_geometry_and_h0(config, base)
    repeats = [int(v) for v in config["analysis_design"]["repeats"]]
    folds = [int(v) for v in config["analysis_design"]["folds"]]

    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "activation_diff_audit.json", activation_audit or {"mode": config["execution_mode"]})
    atomic_csv(output_dir / "normalized_prediction_ledger_before_gold.csv", predictions)
    graphs, _ = base.build_graphs(predictions, geometry, h0, output_dir, repeats, folds)
    if len(graphs) != int(config["expected_counts"]["unique_graphs"]):
        raise ContractViolation("unique graph count drift")
    marker = output_dir / "graph_hash_freeze_complete.json"
    if not marker.is_file():
        raise ContractViolation("graph hash freeze marker absent")

    # Gold-derived inputs remain unopened until all graph files, hashes and the marker are durable.
    gold_rows = read_jsonl(Path(config["input_bindings"]["gold_ledger"]["path"]))
    denominator_rows = read_csv(Path(config["input_bindings"]["denominator_connector"]["path"]))
    joined = base.join_gold(predictions, gold_rows, geometry, repeats)
    descriptive_outputs_safe(joined, output_dir, base)
    base.graph_outputs(graphs, denominator_rows, repeats, folds, output_dir)
    result = {
        "status": "COMPLETED_DESCRIPTIVE_ONLY",
        "batch_id": config["batch_id"],
        "prediction_rows": len(predictions),
        "three_condition_rows": len(joined),
        "unique_graphs": len(graphs),
        "repeat_aligned_graph_units": len(repeats) * len(folds) * 3,
        "gold_opened_after_graph_hash_freeze": True,
        "confirmatory_inference_performed": False,
        "api_calls": 0,
        "network_calls": 0,
        "stage11_entered": False,
        "input_hashes": observed,
        "graph_hash_freeze_index_sha256": sha256_file(output_dir / "graph_hash_freeze_index.csv"),
    }
    atomic_json(output_dir / "analysis_completion_audit.json", result)
    manifest = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        manifest.append({"path": str(path.relative_to(output_dir)).replace("\\", "/"),
                         "sha256": sha256_file(path), "bytes": path.stat().st_size})
    atomic_json(output_dir / "output_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-executable candidate for H0/H1/H2-KB outer descriptive analysis")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_analysis(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
