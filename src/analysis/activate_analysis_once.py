from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ALLOWED_TRUE = (
    "configuration_executable", "author_final_review_complete", "real_gold_read_authorized",
    "outer_performance_read_authorized", "real_gold_join_authorized",
    "real_descriptive_scoring_authorized", "real_graph_assembly_authorized",
    "post_assembly_real_denominator_join_authorized",
)
MUST_REMAIN_FALSE = (
    "confirmatory_inference_authorized", "adaptive_analysis_authorized", "api_authorized",
    "network_authorized", "api_key_read_authorized", "scope_expansion_authorized",
    "stage11_authorized",
)


class ActivationViolation(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("h1_h2kb_outer_analysis_candidate", path)
    if spec is None or spec.loader is None:
        raise ActivationViolation("cannot load bound executor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_activated_config(source: dict[str, Any], expected_batch: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if source.get("batch_id") != expected_batch:
        raise ActivationViolation("batch id mismatch")
    if source.get("execution_mode") != "candidate_requires_activation":
        raise ActivationViolation("source config mode drift")
    if any(source.get(key) is not False for key in ALLOWED_TRUE + MUST_REMAIN_FALSE):
        raise ActivationViolation("source authorization flags are not all false")
    activated = json.loads(json.dumps(source))
    activated["execution_mode"] = "authorized_real"
    for key in ALLOWED_TRUE:
        activated[key] = True
    changed = {key for key in activated if activated.get(key) != source.get(key)}
    allowed = {"execution_mode", *ALLOWED_TRUE}
    if changed != allowed:
        raise ActivationViolation(f"activation diff drift:{sorted(changed ^ allowed)}")
    if any(activated[key] is not False for key in MUST_REMAIN_FALSE):
        raise ActivationViolation("prohibited flag changed")
    audit = {
        "activation_policy": "ONE-TIME-H1-H2KB-OUTER-DESCRIPTIVE-ANALYSIS-v1",
        "batch_id": expected_batch,
        "changed_fields": {
            "execution_mode": ["candidate_requires_activation", "authorized_real"],
            **{key: [False, True] for key in ALLOWED_TRUE},
        },
        "unchanged_prohibited_fields": {key: activated[key] for key in MUST_REMAIN_FALSE},
        "authorized_output_directory": activated["authorized_output_directory"],
    }
    return activated, audit


def activate_once(source_path: Path, executor_path: Path, expected_source_sha256: str,
                  expected_executor_sha256: str, expected_batch: str) -> dict[str, Any]:
    if sha256_file(source_path) != expected_source_sha256.upper():
        raise ActivationViolation("source config hash drift")
    if sha256_file(executor_path) != expected_executor_sha256.upper():
        raise ActivationViolation("executor hash drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    activated, audit = build_activated_config(source, expected_batch)
    audit["source_config_sha256"] = expected_source_sha256.upper()
    audit["executor_sha256"] = expected_executor_sha256.upper()
    return load_module(executor_path).run_analysis(activated, audit)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time activation wrapper for frozen H1/H2-KB local analysis")
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-executor-sha256", required=True)
    parser.add_argument("--expected-batch", required=True)
    args = parser.parse_args()
    result = activate_once(args.source_config, args.executor, args.expected_source_sha256,
                           args.expected_executor_sha256, args.expected_batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
