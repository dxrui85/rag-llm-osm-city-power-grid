from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_CLASSES = ("ACLineSegment", "ConnectivityNode", "EnergySource")
METHODS = ("H1_no_retrieval_LLM", "H2_KB_RAG_LLM")
CONDITIONS = ("H0", *METHODS)
BASE_EXECUTOR = Path(__file__).with_name("role_assembly_and_scoring.py")
BASE_EXECUTOR_SHA256 = "3D71614C072C47B6057663E67F6C1E8362BA292E0FEB40FAF7D67362BDBC1B59"


class ContractViolation(RuntimeError):
    pass


def load_base_executor():
    if not BASE_EXECUTOR.is_file() or sha256_file(BASE_EXECUTOR) != BASE_EXECUTOR_SHA256:
        raise ContractViolation("base ROLE-ASSEMBLY executor hash drift")
    spec = importlib.util.spec_from_file_location("scale010_role_assembly_base", BASE_EXECUTOR)
    if spec is None or spec.loader is None:
        raise ContractViolation("cannot load frozen ROLE-ASSEMBLY base")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.METHODS = METHODS
    module.CONDITIONS = ("H0_no_completion", *METHODS)
    return module


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ContractViolation("refuse empty output table")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_config(config: dict[str, Any]) -> None:
    mode = config.get("execution_mode")
    if mode == "synthetic_regression":
        if config.get("synthetic_fixture_authorized") is not True:
            raise ContractViolation("synthetic fixture authorization absent")
        root = Path(config["synthetic_fixture_root"]).resolve()
        for binding in config["input_bindings"].values():
            items = binding if isinstance(binding, list) else [binding]
            for item in items:
                p = Path(item["path"]).resolve()
                if p != root and root not in p.parents:
                    raise ContractViolation("synthetic input escaped fixture root")
        return
    if mode != "authorized_real":
        raise ContractViolation("candidate remains non-executable")
    required_true = (
        "configuration_executable", "author_final_review_complete", "real_gold_read_authorized",
        "outer_performance_read_authorized", "real_gold_join_authorized",
        "real_descriptive_scoring_authorized", "real_graph_assembly_authorized",
        "post_assembly_real_denominator_join_authorized",
    )
    must_false = (
        "confirmatory_inference_authorized", "adaptive_analysis_authorized", "api_authorized",
        "network_authorized", "api_key_read_authorized", "scope_expansion_authorized",
        "stage16_authorized",
    )
    drift = [k for k in required_true if config.get(k) is not True]
    drift += [k for k in must_false if config.get(k) is not False]
    if drift:
        raise ContractViolation("activation flag drift:" + ",".join(drift))


def verify_bindings(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, binding in config["input_bindings"].items():
        items = binding if isinstance(binding, list) else [binding]
        for index, item in enumerate(items, 1):
            path = Path(item["path"])
            if not path.is_file():
                raise ContractViolation(f"missing bound input:{name}")
            digest = sha256_file(path)
            if digest != item["sha256"].upper():
                raise ContractViolation(f"hash drift:{name}")
            observed[f"{name}_{index}" if len(items) > 1 else name] = digest
    return observed


def materialize_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
    positions = read_csv(Path(config["input_bindings"]["position_status_ledger"]["path"]))
    responses: dict[int, dict[str, Any]] = {}
    for binding in config["input_bindings"]["validated_response_ledgers"]:
        for row in read_jsonl(Path(binding["path"])):
            seq = int(row["global_sequence"])
            if seq in responses:
                raise ContractViolation("duplicate validated response position")
            responses[seq] = row
    if len(positions) != int(config["expected_counts"]["prediction_rows"]):
        raise ContractViolation("prediction position row-count drift")

    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    sequences: set[int] = set()
    for row in positions:
        seq = int(row["global_sequence"])
        sequences.add(seq)
        method = row["method"]
        if method not in METHODS:
            raise ContractViolation("method drift")
        identity = (int(row["repeat_id"]), int(row["seed"]), int(row["outer_fold"]), method, row["object_id"])
        if identity in identities:
            raise ContractViolation("duplicate prediction identity")
        identities.add(identity)
        state = row.get("position_status") or row.get("position_state")
        response = responses.get(seq)
        if state == "MISSING_QUARANTINED":
            if response is not None:
                raise ContractViolation("quarantined missing position has response")
            prediction_state, predicted_class = "MISSING_QUARANTINED", ""
        else:
            if response is None:
                raise ContractViolation("received position lacks validated response")
            if (response.get("object_id") != row["object_id"] or response.get("method") != method
                    or int(response.get("outer_fold")) != int(row["outer_fold"])
                    or int(response.get("repeat_id")) != int(row["repeat_id"])):
                raise ContractViolation("response identity drift")
            validation = response.get("validation") or {}
            parsed = validation.get("parsed_content") or {}
            if validation.get("state") == "INVALID":
                prediction_state, predicted_class = "INVALID", ""
            elif validation.get("state") == "VALID" and parsed.get("status") == "insufficient_evidence":
                prediction_state, predicted_class = "VALID_INSUFFICIENT_EVIDENCE", ""
            elif validation.get("state") == "VALID" and parsed.get("status") == "prediction":
                prediction_state, predicted_class = "VALID_PREDICTION", parsed.get("class", "")
            else:
                raise ContractViolation("response validation schema outside frozen contract")
            allowed = {"VALID_PREDICTION", "VALID_INSUFFICIENT_EVIDENCE", "INVALID"}
            if prediction_state not in allowed:
                raise ContractViolation("response state outside frozen contract")
            if state != prediction_state:
                raise ContractViolation("position and response state mismatch")
            if prediction_state == "VALID_PREDICTION" and predicted_class not in TARGET_CLASSES:
                raise ContractViolation("predicted class outside frozen ontology")
            if prediction_state != "VALID_PREDICTION" and predicted_class:
                raise ContractViolation("failed state carries predicted class")
        normalized_state = {
            "VALID_PREDICTION": "valid_prediction",
            "VALID_INSUFFICIENT_EVIDENCE": "valid_insufficient_evidence",
            "INVALID": "INVALID",
            "MISSING_QUARANTINED": "missing",
        }[prediction_state]
        normalized.append({
            "global_sequence": seq,
            "repeat_id": int(row["repeat_id"]),
            "seed": int(row["seed"]),
            "outer_test_fold": int(row["outer_fold"]),
            "pair_id": row["pair_id"],
            "method": method,
            "object_id": row["object_id"],
            "technical_status": prediction_state,
            "prediction_state": normalized_state,
            "predicted_class": predicted_class,
            "metric_token": predicted_class if normalized_state == "valid_prediction" else f"__{normalized_state.upper()}__",
            "gold_class": "NOT_READ",
            "correctness": "NOT_COMPUTED",
        })
    if set(responses) - sequences:
        raise ContractViolation("response outside frozen position domain")
    expected_states = config["expected_counts"].get("prediction_states")
    observed_states = dict(Counter(r["technical_status"] for r in normalized))
    if expected_states is not None and observed_states != expected_states:
        raise ContractViolation(f"prediction state count drift:{observed_states}")
    return sorted(normalized, key=lambda r: r["global_sequence"])


def build_graphs_before_gold(predictions: list[dict[str, Any]], geometry: list[dict[str, Any]], output: Path, events: list[str]) -> list[dict[str, Any]]:
    geometry_by_id = {r["object_id"]: r for r in geometry}
    if len(geometry_by_id) != len(geometry):
        raise ContractViolation("duplicate geometry object_id")
    if any("gold_class" in r for r in geometry):
        raise ContractViolation("gold field present in pre-gold geometry")
    repeats = sorted({r["repeat_id"] for r in predictions})
    folds = sorted({r["outer_fold_010"] for r in predictions})
    graphs: list[dict[str, Any]] = []
    graph_dir = output / "graphs_before_gold"
    graph_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        keys = [(None, fold, "H0")]
        keys.extend((repeat, fold, condition) for repeat in repeats for condition in METHODS)
        for repeat, fold_id, condition in keys:
            if condition == "H0":
                selected: list[dict[str, Any]] = []
            else:
                selected = [r for r in predictions if r["repeat_id"] == repeat and r["outer_fold_010"] == fold_id and r["method"] == condition and r["prediction_state"] == "VALID_PREDICTION"]
            vertices = sorted(r["object_id"] for r in selected if r["predicted_class"] in {"ConnectivityNode", "EnergySource"})
            edges = []
            for row in selected:
                if row["predicted_class"] != "ACLineSegment":
                    continue
                geom = geometry_by_id.get(row["object_id"])
                if geom is None:
                    raise ContractViolation("prediction object absent from geometry")
                endpoints = geom.get("endpoint_object_ids", [])
                edges.append({"object_id": row["object_id"], "endpoints": endpoints[:2]})
            payload = {"repeat_id": repeat, "outer_fold_010": fold_id, "condition": condition, "vertices": vertices, "edges": edges}
            filename = f"graph_{condition}_fold{fold_id}_repeat{repeat if repeat is not None else 'H0'}.json"
            path = graph_dir / filename
            write_json(path, payload)
            graphs.append({"repeat_id": repeat, "outer_fold_010": fold_id, "condition": condition, "path": filename, "sha256": sha256_file(path)})
    write_csv(output / "graph_hash_freeze_index.csv", graphs)
    events.append("GRAPH_HASH_FREEZE_COMPLETE")
    write_json(output / "graph_hash_freeze_complete.json", {"status": "COMPLETE_BEFORE_GOLD_READ", "gold_rows_opened": 0, "graph_count": len(graphs)})
    return graphs


def join_gold_by_object_id(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold: dict[str, str] = {}
    for row in gold_rows:
        object_id = row["object_id"]
        if object_id in gold:
            raise ContractViolation("gold object_id non-unique")
        if row["gold_class"] not in TARGET_CLASSES:
            raise ContractViolation("gold class outside frozen ontology")
        gold[object_id] = row["gold_class"]
    joined: list[dict[str, Any]] = []
    for row in predictions:
        if row["object_id"] not in gold:
            raise ContractViolation("zero gold match")
        out = dict(row)
        out["gold_class"] = gold[row["object_id"]]
        out["correctness"] = bool(row["prediction_state"] == "VALID_PREDICTION" and row["predicted_class"] == out["gold_class"])
        joined.append(out)
    if len(joined) != len(predictions):
        raise ContractViolation("join row-count drift")
    return joined


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for cls in TARGET_CLASSES:
        tp = sum(r["prediction_state"] == "VALID_PREDICTION" and r["predicted_class"] == cls and r["gold_class"] == cls for r in rows)
        fp = sum(r["prediction_state"] == "VALID_PREDICTION" and r["predicted_class"] == cls and r["gold_class"] != cls for r in rows)
        fn = sum(r["gold_class"] == cls and not (r["prediction_state"] == "VALID_PREDICTION" and r["predicted_class"] == cls) for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    correct = sum(bool(r["correctness"]) for r in rows)
    predicted = sum(r["prediction_state"] == "VALID_PREDICTION" for r in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "prediction_coverage": predicted / n if n else 0.0,
        "state_counts": dict(Counter(r["prediction_state"] for r in rows)),
        "per_class": per_class,
    }


def descriptive_outputs(joined: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    write_csv(output / "immutable_gold_join_ledger.csv", joined)
    fold_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        fold_groups[(row["repeat_id"], row["outer_fold_010"], row["method"])].append(row)
    diagnostics = []
    for (repeat, fold, method), rows in sorted(fold_groups.items()):
        score = score_rows(rows)
        diagnostics.append({"repeat_id": repeat, "outer_fold_010": fold, "condition": method, "n": score["n"], "macro_f1": score["macro_f1"], "accuracy": score["accuracy"], "prediction_coverage": score["prediction_coverage"], "state_counts_json": json.dumps(score["state_counts"], sort_keys=True)})
    write_csv(output / "object_fold010_diagnostics.csv", diagnostics)
    pooled: list[dict[str, Any]] = []
    repeats = sorted({r["repeat_id"] for r in joined})
    for repeat in repeats:
        for method in METHODS:
            rows = [r for r in joined if r["repeat_id"] == repeat and r["method"] == method]
            score = score_rows(rows)
            pooled.append({"repeat_id": repeat, "seed": rows[0]["seed"], "condition": method, "n": score["n"], "macro_f1": score["macro_f1"], "accuracy": score["accuracy"], "prediction_coverage": score["prediction_coverage"], "state_counts_json": json.dumps(score["state_counts"], sort_keys=True)})
    write_csv(output / "object_repeat_pooled_descriptive_scale010.csv", pooled)
    summaries = []
    for method in METHODS:
        method_rows = [r for r in pooled if r["condition"] == method]
        for metric in ("macro_f1", "accuracy", "prediction_coverage"):
            values = [float(r[metric]) for r in method_rows]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0
            summaries.append({"condition": method, "metric": metric, "repeats": len(values), "mean": mean, "sample_sd": sd, "minimum": min(values), "maximum": max(values)})
    write_csv(output / "object_three_repeat_summary_scale010.csv", summaries)
    return pooled


def compare_scales(pooled010: list[dict[str, Any]], main0125: list[dict[str, str]], output: Path) -> list[dict[str, Any]]:
    required = {"seed", "condition", "metric", "value"}
    if not main0125 or not required.issubset(main0125[0]):
        raise ContractViolation("main-scale summary schema drift")
    lookup: dict[tuple[int, str, str], float] = {}
    for row in main0125:
        if "outer_fold" in row and row["outer_fold"] not in {"", "POOLED", "ALL"}:
            raise ContractViolation("fold-level cross-scale pairing forbidden")
        key = (int(row["seed"]), row["condition"], row["metric"])
        if key in lookup:
            raise ContractViolation("duplicate main-scale pooled identity")
        lookup[key] = float(row["value"])
    rows: list[dict[str, Any]] = []
    for current in pooled010:
        for metric in ("macro_f1", "accuracy", "prediction_coverage"):
            key = (int(current["seed"]), current["condition"], metric)
            if key not in lookup:
                raise ContractViolation("missing same-seed main-scale pooled result")
            v010 = float(current[metric])
            v0125 = lookup[key]
            rows.append({"seed": current["seed"], "condition": current["condition"], "metric": metric, "scale010": v010, "scale0125": v0125, "delta_010_minus_0125": v010 - v0125, "comparison_unit": "same_seed_five_fold_pooled"})
    write_csv(output / "scale010_vs_scale0125_descriptive.csv", rows)
    return rows


def join_gold_and_add_h0(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold: dict[str, str] = {}
    for row in gold_rows:
        object_id = row["object_id"]
        if object_id in gold:
            raise ContractViolation("gold object_id non-unique")
        if row["gold_class"] not in TARGET_CLASSES:
            raise ContractViolation("gold class outside frozen ontology")
        gold[object_id] = row["gold_class"]
    joined: list[dict[str, Any]] = []
    h1_rows: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for row in predictions:
        if row["object_id"] not in gold:
            raise ContractViolation("zero gold match")
        out = dict(row)
        out["condition"] = row["method"]
        out["gold_class"] = gold[row["object_id"]]
        out["correct"] = bool(row["prediction_state"] == "valid_prediction" and row["predicted_class"] == out["gold_class"])
        out["gold_join_count"] = 1
        joined.append(out)
        if row["method"] == METHODS[0]:
            h1_rows[(row["repeat_id"], row["seed"], row["outer_test_fold"], row["object_id"])] = row
    for (repeat, seed, fold, object_id), source in sorted(h1_rows.items()):
        joined.append({
            "global_sequence": "", "pair_id": source["pair_id"], "repeat_id": repeat,
            "seed": seed, "outer_test_fold": fold, "method": "H0_no_completion",
            "condition": "H0_no_completion", "object_id": object_id,
            "technical_status": "H0_UNRESOLVED", "prediction_state": "H0_unresolved",
            "predicted_class": "", "metric_token": "__NO_COMPLETION__",
            "gold_class": gold[object_id], "correct": False, "gold_join_count": 1,
        })
    if len(joined) != len(predictions) + len(h1_rows):
        raise ContractViolation("joined row count drift")
    return joined


def reindex_denominators_to_scale010(denominator_rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> list[dict[str, str]]:
    fold_by_object: dict[str, int] = {}
    for row in predictions:
        object_id, fold = row["object_id"], int(row["outer_test_fold"])
        if object_id in fold_by_object and fold_by_object[object_id] != fold:
            raise ContractViolation("scale010 object fold drift")
        fold_by_object[object_id] = fold
    by_object: dict[str, dict[str, str]] = {}
    for row in denominator_rows:
        object_id = row["object_id"]
        if object_id in by_object:
            raise ContractViolation("denominator object_id non-unique")
        by_object[object_id] = dict(row)
    if set(by_object) != set(fold_by_object):
        raise ContractViolation("denominator object domain drift")
    result = []
    for object_id in sorted(by_object):
        row = dict(by_object[object_id])
        row["outer_test_fold"] = str(fold_by_object[object_id])
        result.append(row)
    return result


def compare_object_scales(output: Path, main_path: Path) -> list[dict[str, Any]]:
    scale010 = read_csv(output / "object_repeat_pooled_descriptive.csv")
    main0125 = read_csv(main_path)
    lookup = {(int(r["repeat_id"]), r["condition"]): r for r in main0125}
    if len(lookup) != len(main0125):
        raise ContractViolation("duplicate main-scale object pooled identity")
    rows = []
    for current in scale010:
        if current["condition"] == "H0_no_completion":
            continue
        key = (int(current["repeat_id"]), current["condition"])
        if key not in lookup:
            raise ContractViolation("missing main-scale object pooled identity")
        prior = lookup[key]
        for metric in ("macro_f1", "accuracy", "prediction_coverage"):
            v010, v0125 = float(current[metric]), float(prior[metric])
            rows.append({"repeat_id": current["repeat_id"], "condition": current["condition"], "metric": metric,
                         "scale010": v010, "scale0125": v0125, "delta_010_minus_0125": v010 - v0125,
                         "comparison_unit": "same_repeat_five_fold_pooled"})
    write_csv(output / "object_scale010_vs_scale0125_descriptive.csv", rows)
    return rows


GRAPH_METRIC_KEYS = {
    "effective_assembly_rate": ("effective_assembly", "D_object"),
    "endpoint_valid_connection_rate": ("valid_endpoint_slots", "D_endpoint"),
    "dangling_line_rate": ("dangling_gold_lines", "D_gold_line"),
    "missing_gold_line_rate": ("missing_gold_lines", "D_gold_line"),
    "isolated_node_rate": ("isolated_or_missing_gold_vertices", "D_gold_vertex"),
    "largest_connected_component_retention_rate": ("reference_LCC_sources_retained_in_best_component", "D_reference_LCC"),
}


def pool_graph_rows(rows: list[dict[str, str]]) -> dict[tuple[int, str, str], float]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["repeat_id"]), row["condition"])].append(row)
    pooled: dict[tuple[int, str, str], float] = {}
    for (repeat, condition), group in grouped.items():
        for metric, (num_key, den_key) in GRAPH_METRIC_KEYS.items():
            numerator = sum(int(json.loads(r["numerators_json"])[num_key]) for r in group)
            denominator = sum(int(json.loads(r["denominators_json"])[den_key]) for r in group)
            pooled[(repeat, condition, metric)] = numerator / denominator if denominator else 0.0
    return pooled


def compare_graph_scales(output: Path, main_path: Path) -> list[dict[str, Any]]:
    current = pool_graph_rows(read_csv(output / "graph_repeat_aligned_descriptive.csv"))
    main = pool_graph_rows(read_csv(main_path))
    rows = []
    for key, value010 in sorted(current.items()):
        repeat, condition, metric = key
        if key not in main:
            raise ContractViolation("missing main-scale graph pooled identity")
        value0125 = main[key]
        rows.append({"repeat_id": repeat, "condition": condition, "metric": metric,
                     "scale010": value010, "scale0125": value0125,
                     "delta_010_minus_0125": value010 - value0125,
                     "comparison_unit": "same_repeat_five_fold_pooled_numerator_denominator"})
    write_csv(output / "graph_scale010_vs_scale0125_descriptive.csv", rows)
    return rows


def run_analysis(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    observed_hashes = verify_bindings(config)
    output = Path(config["authorized_output_directory"])
    if output.exists():
        raise ContractViolation("authorized output directory already exists")
    predictions = materialize_predictions(config)
    base = load_base_executor()
    geometry_rows = read_jsonl(Path(config["input_bindings"]["geometry_ledger"]["path"]))
    if any("gold_class" in row or "candidate_class" in row for row in geometry_rows):
        raise ContractViolation("gold field present in pre-gold geometry")
    fold_by_object = {row["object_id"]: int(row["outer_test_fold"]) for row in predictions if row["method"] == METHODS[0]}
    geometry = {}
    for source in geometry_rows:
        if source["object_id"] not in fold_by_object:
            raise ContractViolation("geometry object absent from scale010 prediction domain")
        row = dict(source)
        row["fold_id"] = fold_by_object[source["object_id"]]
        geometry[row["object_id"]] = row
    h0_rows = read_csv(Path(config["input_bindings"]["h0_semantic_source_ledger_010"]["path"]))
    h0 = {(int(row["outer_test_fold"]), row["object_id"]): row for row in h0_rows}
    expected_folds = {int(row["outer_test_fold"]) for row in predictions}
    h0_objects = {row["object_id"] for row in h0_rows}
    if (
        len(geometry) != len(geometry_rows)
        or h0_objects != set(geometry)
        or len(h0) != len(geometry) * len(expected_folds)
        or {fold for fold, _ in h0} != expected_folds
    ):
        raise ContractViolation("geometry or H0 object domain drift")
    if any(str(row.get("masked_gold_accessed_by_H0_branch", "")).lower() != "false" for row in h0_rows):
        raise ContractViolation("H0 source reports gold access")
    output.mkdir(parents=True)
    write_csv(output / "normalized_prediction_ledger_before_gold.csv", predictions)
    repeats = sorted({int(r["repeat_id"]) for r in predictions})
    folds = sorted({int(r["outer_test_fold"]) for r in predictions})
    graphs, _ = base.build_graphs(predictions, geometry, h0, output, repeats, folds)
    marker = output / "graph_hash_freeze_complete.json"
    if not marker.is_file():
        raise ContractViolation("graph freeze marker absent")
    events = ["GRAPH_HASH_FREEZE_COMPLETE", "GOLD_READ"]
    gold = read_jsonl(Path(config["input_bindings"]["gold_ledger"]["path"]))
    joined = join_gold_and_add_h0(predictions, gold)
    base.descriptive_outputs(joined, output)
    denominator = read_csv(Path(config["input_bindings"]["denominator_connector"]["path"]))
    denominator010 = reindex_denominators_to_scale010(denominator, predictions)
    base.graph_outputs(graphs, denominator010, repeats, folds, output)
    object_comparison = compare_object_scales(output, Path(config["input_bindings"]["main_scale_object_repeat"]["path"]))
    graph_comparison = compare_graph_scales(output, Path(config["input_bindings"]["main_scale_graph_repeat"]["path"]))
    result = {
        "status": "COMPLETED_SYNTHETIC_REGRESSION" if config["execution_mode"] == "synthetic_regression" else "COMPLETED_DESCRIPTIVE_ONLY",
        "prediction_rows": len(predictions),
        "joined_rows": len(joined),
        "unique_graphs": len(graphs),
        "scale_comparison_rows": len(object_comparison) + len(graph_comparison),
        "event_order": events,
        "gold_join_key": ["object_id"],
        "cross_scale_unit": "same_repeat_five_fold_pooled",
        "confirmatory_inference_performed": False,
        "api_calls": 0,
        "network_calls": 0,
        "stage16_entered": False,
        "input_hashes": observed_hashes,
    }
    write_json(output / "analysis_completion_audit.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit("This frozen candidate has no direct CLI activation path; use a separately reviewed wrapper.")
