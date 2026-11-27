"""Single offline entry point for tests, descriptive results and figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
    process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout[-12000:],
        "stderr": process.stderr[-12000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete offline reproducibility workflow")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-results", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    records: dict[str, object] = {"api_calls": 0, "network_calls": 0}
    tests = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], env)
    records["synthetic_tests"] = tests
    if tests["returncode"] != 0:
        (output / "offline_reproduction_report.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        return 1
    if not args.skip_results:
        results = run([
            sys.executable, str(ROOT / "reproduce_results.py"),
            "--data-root", str(args.data_root.resolve()),
            "--output-dir", str(output / "results"),
        ], env)
        records["results"] = results
        if results["returncode"] != 0:
            (output / "offline_reproduction_report.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            return 1
    if not args.skip_figures:
        figures = run([
            sys.executable, str(ROOT / "reproduce_figures.py"),
            "--data-root", str(args.data_root.resolve()),
            "--output-dir", str(output / "figures"),
        ], env)
        records["figures"] = figures
        if figures["returncode"] != 0:
            (output / "offline_reproduction_report.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            return 1
    records["status"] = "PASS"
    report = output / "offline_reproduction_report.json"
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
