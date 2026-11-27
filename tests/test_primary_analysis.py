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
SOURCE = HERE.parent / "src" / "analysis"
EXECUTOR = SOURCE / "analyze_primary_scale.py"
WRAPPER = SOURCE / "activate_analysis_once.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ex = load(EXECUTOR, "h1_h2kb_executor_under_test")
aw = load(WRAPPER, "h1_h2kb_wrapper_under_test")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.status = self.root / "status.csv"
        self.responses = self.root / "validated.jsonl"
        self.geometry = self.root / "geometry.jsonl"
        self.h0 = self.root / "h0.csv"
        self.gold = self.root / "gold.jsonl"
        self.denominator = self.root / "denominator.csv"
        self.output = self.root / "result"
        self.status_rows = [
            {"global_sequence": 1, "repeat_id": 1, "seed": 101, "outer_fold": 0, "pair_id": "P1", "pair_member_order": 1, "method": "H1_no_retrieval_LLM", "object_id": "way:1", "position_state": "VALID", "validation_reason": "VALID_PREDICTION", "source_run": "S"},
            {"global_sequence": 2, "repeat_id": 1, "seed": 101, "outer_fold": 0, "pair_id": "P1", "pair_member_order": 2, "method": "H2_KB_RAG_LLM", "object_id": "way:1", "position_state": "VALID", "validation_reason": "VALID_INSUFFICIENT_EVIDENCE", "source_run": "S"},
            {"global_sequence": 3, "repeat_id": 1, "seed": 101, "outer_fold": 0, "pair_id": "P2", "pair_member_order": 1, "method": "H1_no_retrieval_LLM", "object_id": "node:2", "position_state": "INVALID", "validation_reason": "SCHEMA", "source_run": "S"},
            {"global_sequence": 4, "repeat_id": 1, "seed": 101, "outer_fold": 0, "pair_id": "P2", "pair_member_order": 2, "method": "H2_KB_RAG_LLM", "object_id": "node:2", "position_state": "MISSING_QUARANTINED", "validation_reason": "AMBIGUOUS_IN_FLIGHT", "source_run": "Q"},
        ]
        self.response_rows = [
            {"global_sequence": 1, "method": "H1_no_retrieval_LLM", "object_id": "way:1", "outer_fold": 0, "repeat_id": 1,
             "validation": {"state": "VALID", "reason": "VALID_PREDICTION", "parsed_content": {"status": "prediction", "class": "ACLineSegment", "method": "H1_no_retrieval_LLM"}}},
            {"global_sequence": 2, "method": "H2_KB_RAG_LLM", "object_id": "way:1", "outer_fold": 0, "repeat_id": 1,
             "validation": {"state": "VALID", "reason": "VALID_INSUFFICIENT_EVIDENCE", "parsed_content": {"status": "insufficient_evidence", "method": "H2_KB_RAG_LLM"}}},
            {"global_sequence": 3, "method": "H1_no_retrieval_LLM", "object_id": "node:2", "outer_fold": 0, "repeat_id": 1,
             "validation": {"state": "INVALID", "reason": "SCHEMA", "parsed_content": None}},
        ]
        geometry_rows = [
            {"object_id": "way:1", "fold_id": 0, "frozen_geometry_ref": "G:way:1", "TOPO_id": "T1", "boundary_flag": "core",
             "geometry": {"object_type": "way", "geometry_type": "open_way", "is_closed": False, "osm_node_ids": [11, 12], "endpoint_object_ids": ["node:2", None]}},
            {"object_id": "node:2", "fold_id": 0, "frozen_geometry_ref": "G:node:2", "TOPO_id": "T2", "boundary_flag": "core",
             "geometry": {"object_type": "node", "geometry_type": "point", "is_closed": False, "osm_node_ids": [11], "endpoint_object_ids": []}},
        ]
        h0_rows = [
            {"outer_test_fold": 0, "object_id": "way:1", "assigned_class": "", "masked_gold_accessed_by_H0_branch": "false"},
            {"outer_test_fold": 0, "object_id": "node:2", "assigned_class": "", "masked_gold_accessed_by_H0_branch": "false"},
        ]
        gold_rows = [
            {"object_id": "way:1", "gold_class": "ACLineSegment"},
            {"object_id": "node:2", "gold_class": "ConnectivityNode"},
        ]
        denominator_rows = [
            {"outer_test_fold": 0, "object_id": "way:1", "request_access_forbidden": "true", "in_D_gold_line": "true", "in_D_gold_vertex": "false", "in_reference_LCC_source_set": "true"},
            {"outer_test_fold": 0, "object_id": "node:2", "request_access_forbidden": "true", "in_D_gold_line": "false", "in_D_gold_vertex": "true", "in_reference_LCC_source_set": "true"},
        ]
        write_csv(self.status, self.status_rows)
        write_jsonl(self.responses, self.response_rows)
        write_jsonl(self.geometry, geometry_rows)
        write_csv(self.h0, h0_rows)
        write_jsonl(self.gold, gold_rows)
        write_csv(self.denominator, denominator_rows)

    def binding(self, path: Path) -> dict:
        return {"path": str(path), "sha256": ex.sha256_file(path)}

    def config(self) -> dict:
        false_flags = {key: False for key in aw.ALLOWED_TRUE + aw.MUST_REMAIN_FALSE}
        return {
            "batch_id": "SYNTH-H1-H2KB",
            "execution_mode": "synthetic_regression",
            "synthetic_fixture_authorized": True,
            "synthetic_fixture_root": str(self.root),
            **false_flags,
            "authorized_output_directory": str(self.output),
            "expected_counts": {
                "prediction_rows": 4, "objects": 2, "folds": 1, "method_pairs": 2,
                "three_condition_rows": 6, "unique_graphs": 3, "repeat_aligned_graph_units": 3,
                "prediction_states": {"valid_prediction": 1, "valid_insufficient_evidence": 1, "INVALID": 1, "missing": 1},
            },
            "analysis_design": {"repeats": [1], "folds": [0], "conditions": ["H0_no_completion", "H1_no_retrieval_LLM", "H2_KB_RAG_LLM"]},
            "input_bindings": {
                "position_status_ledger": self.binding(self.status),
                "validated_response_ledgers": [self.binding(self.responses)],
                "geometry_ledger": self.binding(self.geometry),
                "h0_semantic_source_ledger": self.binding(self.h0),
                "gold_ledger": self.binding(self.gold),
                "denominator_connector": self.binding(self.denominator),
            },
        }

    def refresh_status(self):
        write_csv(self.status, self.status_rows)

    def refresh_responses(self):
        write_jsonl(self.responses, self.response_rows)

    def close(self):
        self.tmp.cleanup()


class TestH1H2KBExecutor(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()

    def tearDown(self):
        self.fx.close()

    def assert_violation(self, fn, fragment: str):
        with self.assertRaises(Exception) as cm:
            fn()
        self.assertIn(fragment, str(cm.exception))

    def test_01_nonactivated_real_mode_blocked(self):
        cfg = self.fx.config(); cfg["execution_mode"] = "candidate_requires_activation"; cfg["synthetic_fixture_authorized"] = False
        self.assert_violation(lambda: ex.run_analysis(cfg), "non-executable")

    def test_02_input_hash_drift_blocked(self):
        cfg = self.fx.config(); self.fx.status.write_text("drift", encoding="utf-8")
        self.assert_violation(lambda: ex.run_analysis(cfg), "hash drift")

    def test_03_output_directory_overwrite_blocked(self):
        cfg = self.fx.config(); self.fx.output.mkdir()
        self.assert_violation(lambda: ex.run_analysis(cfg), "already exists")

    def test_04_duplicate_prediction_identity_blocked(self):
        self.fx.status_rows[1]["method"] = "H1_no_retrieval_LLM"
        self.fx.refresh_status(); cfg = self.fx.config()
        self.assert_violation(lambda: ex.materialize_predictions(cfg), "duplicate prediction identity")

    def test_05_duplicate_validated_response_blocked(self):
        self.fx.response_rows.append(dict(self.fx.response_rows[0])); self.fx.refresh_responses(); cfg = self.fx.config()
        self.assert_violation(lambda: ex.materialize_predictions(cfg), "duplicate validated response")

    def test_06_missing_position_with_response_blocked(self):
        extra = {"global_sequence": 4, "method": "H2_KB_RAG_LLM", "object_id": "node:2", "outer_fold": 0, "repeat_id": 1,
                 "validation": {"state": "VALID", "parsed_content": {"status": "prediction", "class": "ConnectivityNode"}}}
        self.fx.response_rows.append(extra); self.fx.refresh_responses(); cfg = self.fx.config()
        self.assert_violation(lambda: ex.materialize_predictions(cfg), "missing position has a response")

    def test_07_received_position_without_response_blocked(self):
        self.fx.response_rows = self.fx.response_rows[1:]; self.fx.refresh_responses(); cfg = self.fx.config()
        self.assert_violation(lambda: ex.materialize_predictions(cfg), "lacks validated response")

    def test_08_method_drift_blocked(self):
        self.fx.status_rows[0]["method"] = "H2_RAG_LLM"; self.fx.refresh_status(); cfg = self.fx.config()
        self.assert_violation(lambda: ex.materialize_predictions(cfg), "method drift")

    def test_09_status_response_identity_drift_blocked(self):
        self.fx.response_rows[0]["object_id"] = "node:2"; self.fx.refresh_responses(); cfg = self.fx.config()
        self.assert_violation(lambda: ex.materialize_predictions(cfg), "identity drift")

    def test_10_prediction_state_counts_frozen(self):
        rows = ex.materialize_predictions(self.fx.config())
        self.assertEqual(Counter(r["prediction_state"] for r in rows), Counter({"valid_prediction": 1, "valid_insufficient_evidence": 1, "INVALID": 1, "missing": 1}))

    def test_11_gold_markers_remain_closed_in_normalized_ledger(self):
        rows = ex.materialize_predictions(self.fx.config())
        self.assertTrue(all(r["gold_class"] == "NOT_READ" and r["correctness"] == "NOT_COMPUTED" for r in rows))

    def test_12_h2kb_method_preserved(self):
        rows = ex.materialize_predictions(self.fx.config())
        self.assertEqual(sum(r["method"] == "H2_KB_RAG_LLM" for r in rows), 2)

    def test_13_h0_gold_leakage_blocked(self):
        rows = ex.read_csv(self.fx.h0); rows[0]["masked_gold_accessed_by_H0_branch"] = "true"; write_csv(self.fx.h0, rows)
        cfg = self.fx.config(); base = ex.load_base()
        self.assert_violation(lambda: ex.load_geometry_and_h0(cfg, base), "reports gold access")

    def test_14_geometry_gold_field_blocked(self):
        rows = ex.read_jsonl(self.fx.geometry); rows[0]["gold_class"] = "ACLineSegment"; write_jsonl(self.fx.geometry, rows)
        cfg = self.fx.config(); base = ex.load_base()
        self.assert_violation(lambda: ex.load_geometry_and_h0(cfg, base), "gold field")

    def test_15_synthetic_end_to_end_cardinality_and_order(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["prediction_rows"], 4)
        self.assertEqual(result["three_condition_rows"], 6)
        self.assertEqual(result["unique_graphs"], 3)
        marker = json.loads((self.fx.output / "graph_hash_freeze_complete.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["status"], "COMPLETE_BEFORE_GOLD_READ")
        self.assertEqual(marker["gold_rows_opened"], 0)

    def test_16_failure_states_persist_to_pre_gold_ledger(self):
        ex.run_analysis(self.fx.config())
        rows = ex.read_csv(self.fx.output / "normalized_prediction_ledger_before_gold.csv")
        states = Counter(r["prediction_state"] for r in rows)
        self.assertEqual(states["INVALID"], 1); self.assertEqual(states["missing"], 1)

    def test_17_no_api_network_inference_or_stage11(self):
        result = ex.run_analysis(self.fx.config())
        self.assertEqual(result["api_calls"], 0); self.assertEqual(result["network_calls"], 0)
        self.assertFalse(result["confirmatory_inference_performed"]); self.assertFalse(result["stage11_entered"])

    def test_18_activation_diff_exact(self):
        source = {"batch_id": "B", "execution_mode": "candidate_requires_activation", "authorized_output_directory": "X",
                  **{key: False for key in aw.ALLOWED_TRUE + aw.MUST_REMAIN_FALSE}}
        activated, audit = aw.build_activated_config(source, "B")
        self.assertEqual(set(audit["changed_fields"]), {"execution_mode", *aw.ALLOWED_TRUE})
        self.assertTrue(all(activated[k] for k in aw.ALLOWED_TRUE))
        self.assertTrue(all(activated[k] is False for k in aw.MUST_REMAIN_FALSE))

    def test_19_activation_rejects_preenabled_source(self):
        source = {"batch_id": "B", "execution_mode": "candidate_requires_activation", "authorized_output_directory": "X",
                  **{key: False for key in aw.ALLOWED_TRUE + aw.MUST_REMAIN_FALSE}}
        source["outer_performance_read_authorized"] = True
        self.assert_violation(lambda: aw.build_activated_config(source, "B"), "not all false")

    def test_20_synthetic_fixture_escape_blocked(self):
        cfg = self.fx.config(); cfg["input_bindings"]["gold_ledger"] = {"path": str(EXECUTOR), "sha256": ex.sha256_file(EXECUTOR)}
        self.assert_violation(lambda: ex.run_analysis(cfg), "escaped fixture root")


if __name__ == "__main__":
    unittest.main(verbosity=2)
