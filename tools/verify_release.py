from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a release package against a SHA-256 manifest.")
    parser.add_argument("package_root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = row["relative_path"]
            expected = row["sha256"].upper()
            target = args.package_root / relative
            if not target.is_file():
                failures.append(f"MISSING {relative}")
                continue
            observed = sha256(target)
            if observed != expected:
                failures.append(f"HASH_MISMATCH {relative}: {observed} != {expected}")

    if failures:
        print("\n".join(failures))
        return 1
    print("All manifest entries verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

