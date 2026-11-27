from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


TARGET_CLASSES = ("ACLineSegment", "ConnectivityNode", "EnergySource")
METHODS = ("H1_no_retrieval_LLM", "H2_RAG_LLM")
CONDITIONS = ("H0_no_completion", *METHODS)
FAILURE_STATES = {"valid_insufficient_evidence", "INVALID", "missing", "H0_unresolved", "uncompleted"}
ASSEMBLER_VERSION = "ROLE-ASSEMBLY-v0.2-canonical-shared-anchor"


class ContractViolation(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ContractViolation(f"cannot infer fields for empty CSV: {path.name}")
    fields = fields or list(rows[0])
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ContractViolation(f"invalid JSONL {path.name}:{line_no}") from exc
    return rows


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def verify_input_bindings(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, binding in config["input_bindings"].items():
        path = Path(binding["path"])
        if not path.is_file():
            raise ContractViolation(f"missing bound input:{name}")
        digest = sha256_file(path)
        if digest != binding["sha256"].upper():
            raise ContractViolation(f"hash drift:{name}")
        observed[name] = digest
    return observed


def validate_activation(config: dict[str, Any]) -> None:
    if config.get("execution_mode") == "synthetic_regression":
        if config.get("synthetic_fixture_authorized") is not True:
            raise ContractViolation("synthetic fixture authorization absent")
        fixture_root = Path(config["synthetic_fixture_root"]).resolve()
        for binding in config["input_bindings"].values():
            if fixture_root not in Path(binding["path"]).resolve().parents:
                raise ContractViolation("synthetic input escaped fixture root")
        return
    if config.get("execution_mode") != "authorized_real":
        raise ContractViolation("real executor remains non-executable")
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
    drift = [key for key in required_true if config.get(key) is not True]
    drift += [key for key in prohibited if config.get(key) is not False]
    if drift:
        raise ContractViolation("activation flag drift:" + ",".join(drift))


def _prediction_identity(row: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (int(row["repeat_id"]), int(row["seed"]), int(row["outer_test_fold"]), row["method"], row["object_id"])


def load_preassembly(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[int, str], dict[str, str]]]:
    predictions = read_csv(Path(config["input_bindings"]["prediction_ledger"]["path"]))
    geometry_rows = read_jsonl(Path(config["input_bindings"]["geometry_ledger"]["path"]))
    h0_rows = read_csv(Path(config["input_bindings"]["h0_semantic_source_ledger"]["path"]))

    identities = [_prediction_identity(row) for row in predictions]
    if len(identities) != len(set(identities)):
        raise ContractViolation("duplicate prediction identity")
    if len(predictions) != int(config["expected_counts"]["prediction_rows"]):
        raise ContractViolation("prediction row count drift")
    for row in predictions:
        if row["method"] not in METHODS or row["prediction_state"] not in {
            "valid_prediction", "valid_insufficient_evidence", "INVALID", "missing"
        }:
            raise ContractViolation("prediction method/state drift")
        if row.get("gold_class") != "NOT_READ" or row.get("correctness") != "NOT_COMPUTED":
            raise ContractViolation("preassembly prediction ledger contains opened gold/performance")

    geometry = {row["object_id"]: row for row in geometry_rows}
    if len(geometry) != len(geometry_rows) or len(geometry) != int(config["expected_counts"]["objects"]):
        raise ContractViolation("geometry object-domain drift")
    if any("gold_class" in row or "candidate_class" in row for row in geometry_rows):
        raise ContractViolation("gold field in geometry ledger")

    h0: dict[tuple[int, str], dict[str, str]] = {}
    for row in h0_rows:
        key = (int(row["outer_test_fold"]), row["object_id"])
        if key in h0:
            raise ContractViolation("duplicate H0 semantic source identity")
        if _truth(row.get("masked_gold_accessed_by_H0_branch")):
            raise ContractViolation("H0 masked branch reports gold access")
        h0[key] = row
    if len(h0) != int(config["expected_counts"]["objects"]) * int(config["expected_counts"]["folds"]):
        raise ContractViolation("H0 semantic-source row count drift")
    return predictions, geometry, h0


def map_structural_role(record: dict[str, Any]) -> dict[str, Any]:
    role = {
        "object_id": record["object_id"], "condition": record["condition"],
        "prediction_state": record["prediction_state"], "predicted_class": record["predicted_class"],
        "structural_role": "unresolved_semantic_object", "assembly_eligibility": False,
        "assembly_block_reason": None, "frozen_geometry_ref": record["frozen_geometry_ref"],
        "TOPO_id": record["TOPO_id"], "geometry": record["geometry"],
        "source_granularity": record["source_granularity"], "boundary_flag": record["boundary_flag"],
    }
    if record["prediction_state"] != "valid_prediction":
        role["assembly_block_reason"] = f"prediction_state:{record['prediction_state']}"
        return role
    cls, geom = record["predicted_class"], record["geometry"]
    if cls == "ACLineSegment":
        role["structural_role"] = "edge_candidate"
        nodes = geom["osm_node_ids"]
        if geom["object_type"] != "way":
            role["assembly_block_reason"] = "line_requires_way"
        elif geom["is_closed"] or geom["geometry_type"] == "closed_way":
            role["assembly_block_reason"] = "closed_line_not_ordinary_endpoint_edge"
        elif len(nodes) < 2 or nodes[0] == nodes[-1]:
            role["assembly_block_reason"] = "line_requires_two_distinct_endpoint_nodes"
        else:
            role["assembly_eligibility"] = True
    elif cls == "ConnectivityNode":
        role["structural_role"] = "connectivity_vertex_candidate"
        if geom["object_type"] == "node" and geom["geometry_type"] == "point" and not geom["is_closed"]:
            role["assembly_eligibility"] = True
        else:
            role["assembly_block_reason"] = "connectivity_vertex_requires_node_point"
    elif cls == "EnergySource":
        role["structural_role"] = "source_vertex_candidate"
        node_ok = geom["object_type"] == "node" and geom["geometry_type"] == "point"
        way_ok = geom["object_type"] == "way" and geom.get("representative_point_id") is not None
        if node_ok or way_ok:
            role["assembly_eligibility"] = True
        else:
            role["assembly_block_reason"] = "energy_source_requires_point_or_representative_point"
    else:
        role["assembly_block_reason"] = "class_outside_frozen_ontology"
    return role


def assemble_graph(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: row["object_id"])
    if len({row["object_id"] for row in ordered}) != len(ordered):
        raise ContractViolation("duplicate object in graph")
    if len({row["condition"] for row in ordered}) != 1:
        raise ContractViolation("one condition required per graph")
    roles = [map_structural_role(row) for row in ordered]
    by_id = {row["object_id"]: row for row in roles}
    vertices: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for role in roles:
        if role["assembly_eligibility"] and role["structural_role"] in {
            "connectivity_vertex_candidate", "source_vertex_candidate"
        }:
            vid = f"V:{role['object_id']}"
            vertices[vid] = {"vertex_id": vid, "source_object_ids": [role["object_id"]],
                             "role": role["structural_role"], "geometry_ref": role["frozen_geometry_ref"],
                             "boundary_flag": role["boundary_flag"], "source_granularity": role["source_granularity"]}
    for role in roles:
        if not role["assembly_eligibility"] or role["structural_role"] != "edge_candidate":
            continue
        geom = role["geometry"]
        linked = list(geom.get("endpoint_object_ids", [])) + [None, None]
        endpoints, bases = [], []
        for node_id, linked_id in zip((geom["osm_node_ids"][0], geom["osm_node_ids"][-1]), linked[:2]):
            linked_role = by_id.get(linked_id) if linked_id else None
            if linked_role and linked_role["assembly_eligibility"] and linked_role["structural_role"] in {
                "connectivity_vertex_candidate", "source_vertex_candidate"
            }:
                vid, basis = f"V:{linked_id}", "explicit_endpoint_object"
            else:
                vid, basis = f"ANCHOR:OSM_NODE:{node_id}", "shared_osm_node_anchor"
                vertices.setdefault(vid, {"vertex_id": vid, "source_object_ids": [],
                                          "role": "deterministic_shared_endpoint_anchor",
                                          "geometry_ref": f"OSM_NODE:{node_id}",
                                          "boundary_flag": role["boundary_flag"], "source_granularity": None})
                vertices[vid]["source_object_ids"] = sorted(set(vertices[vid]["source_object_ids"] + [role["object_id"]]))
            endpoints.append(vid); bases.append(basis)
        edges.append({"edge_id": f"E:{role['object_id']}", "source_object_id": role["object_id"],
                      "u": endpoints[0], "v": endpoints[1], "connection_basis": bases,
                      "boundary_flag": role["boundary_flag"]})
    graph = {"graph_contract_version": ASSEMBLER_VERSION, "condition": ordered[0]["condition"],
             "roles": roles, "vertices": sorted(vertices.values(), key=lambda x: x["vertex_id"]),
             "edges": sorted(edges, key=lambda x: x["source_object_id"]),
             "audit": {"input_objects": len(ordered), "eligible_objects": sum(r["assembly_eligibility"] for r in roles),
                       "gold_labels_read_by_assembler": 0, "distance_edges": 0,
                       "canonical_anchor_rule": "ANCHOR:OSM_NODE:<node_id>"}}
    graph["graph_sha256"] = sha256_json(graph)
    return graph


def _role_input(oid: str, condition: str, state: str, predicted: str | None, geom_row: dict[str, Any]) -> dict[str, Any]:
    geom = geom_row["geometry"]
    granularity = None
    if predicted == "EnergySource":
        granularity = "generator_unit" if geom["object_type"] == "node" else "plant_site"
    return {"object_id": oid, "condition": condition, "prediction_state": state,
            "predicted_class": predicted, "frozen_geometry_ref": geom_row["frozen_geometry_ref"],
            "TOPO_id": geom_row["TOPO_id"], "geometry": geom, "source_granularity": granularity,
            "boundary_flag": geom_row["boundary_flag"]}


def build_graphs(predictions: list[dict[str, Any]], geometry: dict[str, dict[str, Any]],
                 h0: dict[tuple[int, str], dict[str, str]], output_dir: Path,
                 repeats: list[int], folds: list[int]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pred_index = {(int(r["repeat_id"]), int(r["outer_test_fold"]), r["method"], r["object_id"]): r for r in predictions}
    graphs: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    for fold in folds:
        records = []
        for oid, geom in geometry.items():
            source = h0[(fold, oid)]
            predicted = source["assigned_class"] or None
            state = "valid_prediction" if predicted else "uncompleted"
            records.append(_role_input(oid, "H0_no_completion", state, predicted, geom))
        unit = f"H0|F{fold}"
        graphs[unit] = assemble_graph(records)
    for repeat in repeats:
        for fold in folds:
            for method in METHODS:
                records = []
                for oid, geom in geometry.items():
                    source = h0[(fold, oid)]
                    if int(geom["fold_id"]) == fold:
                        row = pred_index.get((repeat, fold, method, oid))
                        if row is None:
                            raise ContractViolation(f"missing prediction identity:{repeat}:{fold}:{method}:{oid}")
                        state = row["prediction_state"]
                        predicted = row["predicted_class"] or None if state == "valid_prediction" else None
                    else:
                        predicted = source["assigned_class"] or None
                        state = "valid_prediction" if predicted else "uncompleted"
                    records.append(_role_input(oid, method, state, predicted, geom))
                unit = f"R{repeat}|F{fold}|{method}"
                graphs[unit] = assemble_graph(records)
    graph_dir = output_dir / "graphs_frozen_before_gold"
    graph_dir.mkdir(parents=True)
    for unit, graph in sorted(graphs.items()):
        safe = unit.replace("|", "__")
        write_json(graph_dir / f"{safe}.json", graph)
        index_rows.append({"analysis_unit_id": unit, "condition": graph["condition"],
                           "graph_sha256": graph["graph_sha256"], "gold_read_before_hash": "false",
                           "vertices": len(graph["vertices"]), "edges": len(graph["edges"])})
    write_csv(output_dir / "graph_hash_freeze_index.csv", index_rows)
    write_json(output_dir / "graph_hash_freeze_complete.json", {
        "status": "COMPLETE_BEFORE_GOLD_READ", "unique_graphs": len(graphs),
        "H0_unique_graphs": len(folds), "H1_H2_graphs": len(repeats) * len(folds) * 2,
        "gold_rows_opened": 0, "index_sha256": sha256_file(output_dir / "graph_hash_freeze_index.csv")})
    return graphs, index_rows


def join_gold(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]],
              geometry: dict[str, dict[str, Any]], repeats: list[int]) -> list[dict[str, Any]]:
    gold: dict[tuple[str, int], str] = {}
    for row in gold_rows:
        oid = row["object_id"]
        if oid not in geometry:
            raise ContractViolation("gold object outside geometry domain")
        key = (oid, int(geometry[oid]["fold_id"]))
        if key in gold:
            raise ContractViolation("multiple gold matches")
        if row["gold_class"] not in TARGET_CLASSES:
            raise ContractViolation("gold class outside ontology")
        gold[key] = row["gold_class"]
    joined: list[dict[str, Any]] = []
    seen = set()
    for row in predictions:
        identity = _prediction_identity(row)
        if identity in seen:
            raise ContractViolation("duplicate prediction identity")
        seen.add(identity)
        key = (row["object_id"], int(row["outer_test_fold"]))
        if key not in gold:
            raise ContractViolation("zero gold match")
        item = dict(row); item["condition"] = row["method"]; item["gold_class"] = gold[key]
        item["correct"] = row["prediction_state"] == "valid_prediction" and row["predicted_class"] == gold[key]
        item["gold_join_count"] = 1
        joined.append(item)
    pairs = {(int(r["repeat_id"]), int(r["seed"]), int(r["outer_test_fold"]), r["object_id"]): r for r in predictions if r["method"] == METHODS[0]}
    for (repeat, seed, fold, oid), _ in pairs.items():
        joined.append({"global_sequence": "", "pair_id": f"R{repeat}|F{fold}|{oid}", "repeat_id": repeat,
                       "seed": seed, "outer_test_fold": fold, "method": "H0_no_completion",
                       "condition": "H0_no_completion", "object_id": oid, "partition_id": "H0_REUSED",
                       "technical_status": "H0_UNRESOLVED", "prediction_state": "H0_unresolved",
                       "predicted_class": "", "metric_token": "__NO_COMPLETION__",
                       "gold_class": gold[(oid, fold)], "correct": False, "gold_join_count": 1})
    expected = len(predictions) + len(pairs)
    if len(joined) != expected:
        raise ContractViolation("joined row count drift")
    return joined


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {cls: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cls in TARGET_CLASSES}
    states = Counter(); correct = 0; confusion = Counter()
    for row in rows:
        gold = row["gold_class"]
        state = row["prediction_state"]
        predicted = row["predicted_class"] if state == "valid_prediction" else None
        pred_token = predicted or f"__{state}__"
        confusion[(gold, pred_token)] += 1
        states[state] += 1
        correct += int(predicted == gold and predicted in TARGET_CLASSES)
        counts[gold]["support"] += 1
        for cls in TARGET_CLASSES:
            counts[cls]["tp"] += int(predicted == cls and gold == cls)
            counts[cls]["fp"] += int(predicted == cls and gold != cls)
            counts[cls]["fn"] += int(predicted != cls and gold == cls)
    per_class = {}
    f1s = []
    for cls, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["support"] else None
        f1 = None if recall is None else (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if f1 is not None: f1s.append(f1)
        per_class[cls] = {**c, "precision": precision, "recall": recall, "f1": f1}
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None,
            "macro_f1": sum(f1s) / len(f1s) if len(f1s) == len(TARGET_CLASSES) else None,
            "prediction_coverage": sum(s == "valid_prediction" for s in (r["prediction_state"] for r in rows)) / len(rows) if rows else None,
            "state_counts": dict(sorted(states.items())), "per_class": per_class,
            "confusion": [{"gold_class": g, "prediction_token": p, "count": n} for (g, p), n in sorted(confusion.items())]}


def descriptive_outputs(joined: list[dict[str, Any]], output_dir: Path) -> None:
    write_csv(output_dir / "immutable_gold_join_ledger.csv", joined)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(int(row["repeat_id"]), int(row["outer_test_fold"]), row["condition"])].append(row)
    diagnostics = []
    for key, rows in sorted(groups.items()):
        score = score_rows(rows)
        diagnostics.append({"repeat_id": key[0], "outer_test_fold": key[1], "condition": key[2],
                            "n": score["n"], "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                            "prediction_coverage": score["prediction_coverage"],
                            "state_counts_json": json.dumps(score["state_counts"], sort_keys=True),
                            "per_class_json": json.dumps(score["per_class"], sort_keys=True)})
    write_csv(output_dir / "object_fold_diagnostics.csv", diagnostics)
    pooled = []
    for repeat in sorted({int(r["repeat_id"]) for r in joined}):
        for condition in CONDITIONS:
            rows = [r for r in joined if int(r["repeat_id"]) == repeat and r["condition"] == condition]
            score = score_rows(rows)
            pooled.append({"repeat_id": repeat, "condition": condition, "n": score["n"],
                           "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                           "prediction_coverage": score["prediction_coverage"],
                           "state_counts_json": json.dumps(score["state_counts"], sort_keys=True),
                           "per_class_json": json.dumps(score["per_class"], sort_keys=True)})
    write_csv(output_dir / "object_repeat_pooled_descriptive.csv", pooled)
    summaries = []
    for condition in CONDITIONS:
        rows = [r for r in pooled if r["condition"] == condition]
        for metric in ("accuracy", "macro_f1", "prediction_coverage"):
            values = [float(r[metric]) for r in rows if r[metric] is not None]
            summaries.append({"condition": condition, "metric": metric, "repeats": len(values),
                              "mean": sum(values) / len(values),
                              "sample_sd": math.sqrt(sum((x - sum(values)/len(values))**2 for x in values)/(len(values)-1)) if len(values)>1 else 0.0,
                              "minimum": min(values), "maximum": max(values)})
    write_csv(output_dir / "object_three_repeat_summary.csv", summaries)


def connected_components(graph: dict[str, Any]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for vertex in graph["vertices"]: adjacency[vertex["vertex_id"]]
    for edge in graph["edges"]:
        adjacency[edge["u"]].add(edge["v"]); adjacency[edge["v"]].add(edge["u"])
    unseen = set(adjacency); result = []
    while unseen:
        queue = deque([min(unseen)]); component = set()
        while queue:
            node = queue.popleft()
            if node in component: continue
            component.add(node); queue.extend(sorted(adjacency[node] - component))
        unseen -= component; result.append(component)
    return result


def ratio(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def score_graph(graph: dict[str, Any], denominator_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not denominator_rows or any(not _truth(r["request_access_forbidden"]) for r in denominator_rows):
        raise ContractViolation("invalid post-assembly denominator connector")
    planned = {r["object_id"] for r in denominator_rows}
    gold_lines = {r["object_id"] for r in denominator_rows if _truth(r["in_D_gold_line"])}
    gold_vertices = {r["object_id"] for r in denominator_rows if _truth(r["in_D_gold_vertex"])}
    reference = {r["object_id"] for r in denominator_rows if _truth(r["in_reference_LCC_source_set"])}
    roles = {r["object_id"]: r for r in graph["roles"]}
    edges = {e["source_object_id"]: e for e in graph["edges"]}
    vertices_by_source = {}
    for v in graph["vertices"]:
        if not v["role"].startswith("deterministic_"):
            for oid in v["source_object_ids"]: vertices_by_source[oid] = v["vertex_id"]
    degree = Counter()
    for e in graph["edges"]: degree[e["u"]] += 1; degree[e["v"]] += 1
    eligible = {oid for oid in planned if oid in roles and roles[oid]["assembly_eligibility"]}
    assembled_lines = gold_lines & set(edges)
    valid_slots = sum(int(edges[o]["u"] in degree) + int(edges[o]["v"] in degree) for o in assembled_lines)
    dangling = sum(degree[edges[o]["u"]] == 1 or degree[edges[o]["v"]] == 1 for o in assembled_lines)
    isolated = sum(vertices_by_source.get(o) is None or degree[vertices_by_source[o]] == 0 for o in gold_vertices)
    vertex_sources = {v["vertex_id"]: set(v["source_object_ids"]) for v in graph["vertices"]}
    overlaps = []
    for component in connected_components(graph):
        sources = set().union(*(vertex_sources.get(v, set()) for v in component))
        sources |= {e["source_object_id"] for e in graph["edges"] if e["u"] in component and e["v"] in component}
        overlaps.append(len(sources & reference))
    num = {"effective_assembly": len(eligible), "valid_endpoint_slots": valid_slots,
           "dangling_gold_lines": dangling, "missing_gold_lines": len(gold_lines - assembled_lines),
           "isolated_or_missing_gold_vertices": isolated,
           "reference_LCC_sources_retained_in_best_component": max(overlaps, default=0)}
    den = {"D_object": len(planned), "D_gold_line": len(gold_lines), "D_gold_vertex": len(gold_vertices),
           "D_endpoint": 2 * len(gold_lines), "D_reference_LCC": len(reference)}
    return {"numerators": num, "denominators": den,
            "metrics": {"effective_assembly_rate": ratio(num["effective_assembly"], den["D_object"]),
                        "endpoint_valid_connection_rate": ratio(num["valid_endpoint_slots"], den["D_endpoint"]),
                        "dangling_line_rate": ratio(num["dangling_gold_lines"], den["D_gold_line"]),
                        "missing_gold_line_rate": ratio(num["missing_gold_lines"], den["D_gold_line"]),
                        "isolated_node_rate": ratio(num["isolated_or_missing_gold_vertices"], den["D_gold_vertex"]),
                        "largest_connected_component_retention_rate": ratio(num["reference_LCC_sources_retained_in_best_component"], den["D_reference_LCC"])}}


def graph_outputs(graphs: dict[str, dict[str, Any]], denominator_rows: list[dict[str, str]],
                  repeats: list[int], folds: list[int], output_dir: Path) -> None:
    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in denominator_rows: by_fold[int(row["outer_test_fold"])].append(row)
    unique_scores = {}
    for unit, graph in graphs.items():
        fold = int(unit.split("|F")[1].split("|")[0])
        unique_scores[unit] = {"graph_sha256": graph["graph_sha256"], **score_graph(graph, by_fold[fold])}
    write_json(output_dir / "graph_unique_scores.json", unique_scores)
    aligned = []
    for repeat in repeats:
        for fold in folds:
            for condition in CONDITIONS:
                unit = f"H0|F{fold}" if condition == "H0_no_completion" else f"R{repeat}|F{fold}|{condition}"
                score = unique_scores[unit]
                aligned.append({"repeat_id": repeat, "outer_test_fold": fold, "condition": condition,
                                "unique_graph_unit": unit, "independent_stochastic_evidence": condition != "H0_no_completion",
                                "graph_sha256": score["graph_sha256"],
                                "numerators_json": json.dumps(score["numerators"], sort_keys=True),
                                "denominators_json": json.dumps(score["denominators"], sort_keys=True),
                                **score["metrics"]})
    write_csv(output_dir / "graph_repeat_aligned_descriptive.csv", aligned)


def run_analysis(config: dict[str, Any], activation_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_activation(config)
    observed_hashes = verify_input_bindings(config)
    output_dir = Path(config["authorized_output_directory"])
    if output_dir.exists():
        raise ContractViolation("authorized output directory already exists")
    predictions, geometry, h0 = load_preassembly(config)
    repeats = [int(x) for x in config["analysis_design"]["repeats"]]
    folds = [int(x) for x in config["analysis_design"]["folds"]]
    output_dir.mkdir(parents=True)
    write_json(output_dir / "activation_diff_audit.json", activation_audit or {"mode": config["execution_mode"]})
    graphs, graph_index = build_graphs(predictions, geometry, h0, output_dir, repeats, folds)
    if len(graphs) != int(config["expected_counts"]["unique_graphs"]):
        raise ContractViolation("unique graph count drift")
    # Gold-derived inputs are deliberately unopened until every graph hash is durable.
    gold_rows = read_jsonl(Path(config["input_bindings"]["gold_ledger"]["path"]))
    denominator_rows = read_csv(Path(config["input_bindings"]["denominator_connector"]["path"]))
    joined = join_gold(predictions, gold_rows, geometry, repeats)
    descriptive_outputs(joined, output_dir)
    graph_outputs(graphs, denominator_rows, repeats, folds, output_dir)
    result = {"status": "COMPLETED_DESCRIPTIVE_ONLY", "batch_id": config["batch_id"],
              "prediction_rows": len(predictions), "three_condition_rows": len(joined),
              "unique_graphs": len(graphs), "repeat_aligned_graph_units": len(repeats) * len(folds) * 3,
              "gold_opened_after_graph_hash_freeze": True, "confirmatory_inference_performed": False,
              "api_calls": 0, "network_calls": 0, "stage11_entered": False,
              "input_hashes": observed_hashes,
              "graph_hash_freeze_index_sha256": sha256_file(output_dir / "graph_hash_freeze_index.csv")}
    write_json(output_dir / "analysis_completion_audit.json", result)
    manifest = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        manifest.append({"path": str(path.relative_to(output_dir)).replace("\\", "/"),
                         "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(output_dir / "output_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash-gated local outer-gold descriptive/graph analysis candidate")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_analysis(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
