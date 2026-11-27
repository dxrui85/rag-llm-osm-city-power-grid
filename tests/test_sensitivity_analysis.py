from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTOR = HERE.parent / "src" / "analysis" / "analyze_sensitivity_scale.py"


def load_executor():
    spec = importlib.util.spec_from_file_location("scale010_executor", EXECUTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ex = load_executor()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.positions = self.root / "positions.csv"
        self.responses = self.root / "responses.jsonl"
        self.geometry = self.root / "geometry.jsonl"
        self.h0 = self.root / "h0_scale010.csv"
        self.gold = self.root / "gold.jsonl"
        self.denominator = self.root / "denominator.csv"
        self.main_object = self.root / "main0125_object.csv"
        self.main_graph = self.root / "main0125_graph.csv"
        self.output = self.root / "output"
        self.position_rows = []
        self.response_rows = []
        seq = 0
        objects = [("way:1", "ACLineSegment", 0), ("node:2", "ConnectivityNode", 0), ("node:3", "EnergySource", 1)]
        for repeat, seed in ((1, 101), (2, 102)):
            for object_id, gold_class, fold in objects:
                for method in ("H1_no_retrieval_LLM", "H2_KB_RAG_LLM"):
                    seq += 1
                    state = "VALID_PREDICTION"
                    predicted = gold_class if method == "H2_KB_RAG_LLM" else ("ConnectivityNode" if gold_class == "ACLineSegment" else gold_class)
                    if seq == 5:
                        state, predicted = "VALID_INSUFFICIENT_EVIDENCE", ""
                    elif seq == 8:
                        state, predicted = "INVALID", ""
                    elif seq == 11:
                        state, predicted = "MISSING_QUARANTINED", ""
                    pair_id = f"R{repeat}-F{fold}-{object_id}"
                    self.position_rows.append({"global_sequence": seq, "repeat_id": repeat, "seed": seed, "outer_fold": fold, "pair_id": pair_id, "method": method, "object_id": object_id, "position_status": state})
                    if state != "MISSING_QUARANTINED":
                        if state == "INVALID":
                            validation = {"state": "INVALID", "reason": "SYNTHETIC_SCHEMA", "parsed_content": None}
                        elif state == "VALID_INSUFFICIENT_EVIDENCE":
                            validation = {"state": "VALID", "reason": state, "parsed_content": {"status": "insufficient_evidence", "class": None, "method": method}}
                        else:
                            validation = {"state": "VALID", "reason": state, "parsed_content": {"status": "prediction", "class": predicted, "method": method}}
                        self.response_rows.append({"global_sequence": seq, "method": method, "object_id": object_id, "outer_fold": fold, "repeat_id": repeat, "validation": validation})
        geometry = [
            {"object_id": "way:1", "fold_id": 4, "frozen_geometry_ref": "G1", "TOPO_id": "T1", "boundary_flag": "core", "geometry": {"object_type": "way", "geometry_type": "open_way", "is_closed": False, "osm_node_ids": [10, 20], "endpoint_object_ids": ["node:2", "node:3"]}},
            {"object_id": "node:2", "fold_id": 4, "frozen_geometry_ref": "G2", "TOPO_id": "T2", "boundary_flag": "core", "geometry": {"object_type": "node", "geometry_type": "point", "is_closed": False, "osm_node_ids": [10], "endpoint_object_ids": [], "representative_point_id": "OSM_NODE:10"}},
            {"object_id": "node:3", "fold_id": 4, "frozen_geometry_ref": "G3", "TOPO_id": "T3", "boundary_flag": "core", "geometry": {"object_type": "node", "geometry_type": "point", "is_closed": False, "osm_node_ids": [20], "endpoint_object_ids": [], "representative_point_id": "OSM_NODE:20"}},
        ]
        h0_rows = []
        for outer_fold in (0, 1):
            for object_id, gold_class, object_fold in objects:
                assigned = "" if outer_fold == object_fold else gold_class
                h0_rows.append({"outer_test_fold": outer_fold, "object_id": object_id, "object_fold": object_fold, "semantic_source": "SYNTHETIC", "assigned_class": assigned, "masked_gold_accessed_by_H0_branch": "false", "role_input_sha256": "SYNTHETIC"})
        gold = [{"object_id": oid, "gold_class": cls, "outer_fold_0125": 4} for oid, cls, _ in objects]
        denominator = []
        for object_id, gold_class, _ in objects:
            denominator.append({"object_id": object_id, "outer_test_fold": 4, "in_D_object": "true", "in_D_gold_line": str(gold_class == "ACLineSegment").lower(), "in_D_gold_vertex": str(gold_class != "ACLineSegment").lower(), "closed_gold_line_excluded_from_ordinary_edge_denominator": "false", "in_reference_LCC_source_set": "true", "use": "evaluation_only", "request_access_forbidden": "true"})
        main_object = []
        for repeat in (1, 2):
            for condition in ("H0_no_completion", "H1_no_retrieval_LLM", "H2_KB_RAG_LLM"):
                main_object.append({"repeat_id": repeat, "condition": condition, "n": 3, "accuracy": 0.5, "macro_f1": 0.5, "prediction_coverage": 0.5, "state_counts_json": "{}", "per_class_json": "{}"})
        main_graph = []
        nums = {"effective_assembly": 1, "valid_endpoint_slots": 1, "dangling_gold_lines": 0, "missing_gold_lines": 0, "isolated_or_missing_gold_vertices": 0, "reference_LCC_sources_retained_in_best_component": 1}
        dens = {"D_object": 3, "D_gold_line": 1, "D_gold_vertex": 2, "D_endpoint": 2, "D_reference_LCC": 3}
        for repeat in (1, 2):
            for fold in (0, 1):
                for condition in ("H0_no_completion", "H1_no_retrieval_LLM", "H2_KB_RAG_LLM"):
                    main_graph.append({"repeat_id": repeat, "outer_test_fold": fold, "condition": condition, "unique_graph_unit": "S", "independent_stochastic_evidence": "false", "graph_sha256": "S", "numerators_json": json.dumps(nums), "denominators_json": json.dumps(dens), "effective_assembly_rate": 1/3, "endpoint_valid_connection_rate": 0.5, "dangling_line_rate": 0, "missing_gold_line_rate": 0, "isolated_node_rate": 0, "largest_connected_component_retention_rate": 1/3})
        write_csv(self.positions, self.position_rows)
        write_jsonl(self.responses, self.response_rows)
        write_jsonl(self.geometry, geometry)
        write_csv(self.h0, h0_rows)
        write_jsonl(self.gold, gold)
        write_csv(self.denominator, denominator)
        write_csv(self.main_object, main_object)
        write_csv(self.main_graph, main_graph)

    def binding(self, path: Path) -> dict:
        return {"path": str(path), "sha256": ex.sha256_file(path)}

    def config(self) -> dict:
        return {
            "execution_mode": "synthetic_regression",
            "synthetic_fixture_authorized": True,
            "synthetic_fixture_root": str(self.root),
            "authorized_output_directory": str(self.output),
            "expected_counts": {
                "prediction_rows": len(self.position_rows),
                "prediction_states": dict(Counter(r["position_status"] for r in self.position_rows)),
            },
            "input_bindings": {
                "position_status_ledger": self.binding(self.positions),
                "validated_response_ledgers": [self.binding(self.responses)],
                "geometry_ledger": self.binding(self.geometry),
                "h0_semantic_source_ledger_010": self.binding(self.h0),
                "gold_ledger": self.binding(self.gold),
                "denominator_connector": self.binding(self.denominator),
                "main_scale_object_repeat": self.binding(self.main_object),
                "main_scale_graph_repeat": self.binding(self.main_graph),
            },
        }

    def refresh_positions(self):
        write_csv(self.positions, self.position_rows)

    def refresh_responses(self):
        write_jsonl(self.responses, self.response_rows)

    def close(self):
        self.tmp.cleanup()


class Tests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()

    def tearDown(self):
        self.fx.close()

    def violation(self, fn, text):
        with self.assertRaises(Exception) as cm:
            fn()
        self.assertIn(text, str(cm.exception))

    def test_01_candidate_real_execution_blocked(self):
        cfg = self.fx.config(); cfg["execution_mode"] = "candidate_requires_activation"
        self.violation(lambda: ex.run_analysis(cfg), "non-executable")

    def test_02_synthetic_fixture_escape_blocked(self):
        cfg = self.fx.config(); cfg["input_bindings"]["gold_ledger"] = {"path": str(EXECUTOR), "sha256": ex.sha256_file(EXECUTOR)}
        self.violation(lambda: ex.run_analysis(cfg), "escaped fixture root")

    def test_03_hash_drift_blocked(self):
        cfg = self.fx.config(); self.fx.positions.write_text("drift", encoding="utf-8")
        self.violation(lambda: ex.run_analysis(cfg), "hash drift")

    def test_04_output_overwrite_blocked(self):
        cfg = self.fx.config(); self.fx.output.mkdir()
        self.violation(lambda: ex.run_analysis(cfg), "already exists")

    def test_05_duplicate_prediction_identity_blocked(self):
        self.fx.position_rows[1]["method"] = self.fx.position_rows[0]["method"]
        self.fx.refresh_positions(); cfg = self.fx.config()
        self.violation(lambda: ex.materialize_predictions(cfg), "duplicate prediction identity")

    def test_06_duplicate_response_blocked(self):
        self.fx.response_rows.append(dict(self.fx.response_rows[0])); self.fx.refresh_responses(); cfg = self.fx.config()
        self.violation(lambda: ex.materialize_predictions(cfg), "duplicate validated response")

    def test_07_quarantined_response_blocked(self):
        row = next(r for r in self.fx.position_rows if r["position_status"] == "MISSING_QUARANTINED")
        self.fx.response_rows.append({"global_sequence": row["global_sequence"], "method": row["method"], "object_id": row["object_id"], "outer_fold": row["outer_fold"], "repeat_id": row["repeat_id"], "validation": {"state": "VALID", "reason": "VALID_PREDICTION", "parsed_content": {"status": "prediction", "class": "EnergySource", "method": row["method"]}}})
        self.fx.refresh_responses(); cfg = self.fx.config()
        self.violation(lambda: ex.materialize_predictions(cfg), "missing position has response")

    def test_08_invalid_and_missing_preserved(self):
        rows = ex.materialize_predictions(self.fx.config()); c = Counter(r["technical_status"] for r in rows)
        self.assertEqual(c["INVALID"], 1); self.assertEqual(c["MISSING_QUARANTINED"], 1)

    def test_09_gold_closed_before_join(self):
        rows = ex.materialize_predictions(self.fx.config())
        self.assertTrue(all(r["gold_class"] == "NOT_READ" and r["correctness"] == "NOT_COMPUTED" for r in rows))

    def test_10_gold_object_id_unique_required(self):
        rows = ex.read_jsonl(self.fx.gold); rows.append(dict(rows[0])); write_jsonl(self.fx.gold, rows)
        cfg = self.fx.config()
        self.violation(lambda: ex.run_analysis(cfg), "gold object_id non-unique")

    def test_11_gold_join_ignores_main_scale_fold(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["gold_join_key"], ["object_id"])
        self.assertEqual(result["joined_rows"], len(self.fx.position_rows) + len(self.fx.position_rows) // 2)

    def test_12_graphs_frozen_before_gold_read(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["event_order"], ["GRAPH_HASH_FREEZE_COMPLETE", "GOLD_READ"])
        marker = json.loads((self.fx.output / "graph_hash_freeze_complete.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["gold_rows_opened"], 0)

    def test_13_h0_reused_across_repeats(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["unique_graphs"], 10)
        rows = ex.read_csv(self.fx.output / "graph_hash_freeze_index.csv")
        self.assertEqual(sum(r["condition"] == "H0_no_completion" for r in rows), 2)

    def test_14_fixed_denominator_includes_failures(self):
        ex.run_analysis(self.fx.config())
        joined = ex.read_csv(self.fx.output / "immutable_gold_join_ledger.csv")
        self.assertEqual(len(joined), len(self.fx.position_rows) + len(self.fx.position_rows) // 2)
        self.assertEqual(sum(r["prediction_state"] in {"INVALID", "missing", "valid_insufficient_evidence"} for r in joined), 3)

    def test_15_repeat_pooled_not_fold_paired(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["cross_scale_unit"], "same_repeat_five_fold_pooled")
        rows = ex.read_csv(self.fx.output / "object_scale010_vs_scale0125_descriptive.csv")
        self.assertTrue(all(r["comparison_unit"] == "same_repeat_five_fold_pooled" for r in rows))

    def test_16_duplicate_main_scale_pooled_identity_blocked(self):
        rows = ex.read_csv(self.fx.main_object); rows.append(dict(rows[0])); write_csv(self.fx.main_object, rows)
        cfg = self.fx.config()
        self.violation(lambda: ex.run_analysis(cfg), "duplicate main-scale object pooled identity")

    def test_17_no_inference_api_network_or_stage16(self):
        result = ex.run_analysis(self.fx.config())
        self.assertFalse(result["confirmatory_inference_performed"]); self.assertFalse(result["stage16_entered"])
        self.assertEqual(result["api_calls"], 0); self.assertEqual(result["network_calls"], 0)

    def test_18_scale_comparison_cardinality(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["scale_comparison_rows"], 48)

    def test_19_geometry_gold_field_blocked(self):
        rows = ex.read_jsonl(self.fx.geometry); rows[0]["gold_class"] = "ACLineSegment"; write_jsonl(self.fx.geometry, rows)
        cfg = self.fx.config()
        self.violation(lambda: ex.run_analysis(cfg), "gold field present")

    def test_20_wrong_prediction_class_blocked(self):
        row = next(r for r in self.fx.response_rows if (r.get("validation") or {}).get("reason") == "VALID_PREDICTION")
        row["validation"]["parsed_content"]["class"] = "Transformer"; self.fx.refresh_responses(); cfg = self.fx.config()
        self.violation(lambda: ex.materialize_predictions(cfg), "outside frozen ontology")


if __name__ == "__main__":
    unittest.main(verbosity=2)
