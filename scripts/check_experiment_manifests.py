#!/usr/bin/env python3
"""Validate reproducibility manifests without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "experiments" / "manifests"
EXPERIMENT_ID = re.compile(r"^EXP-\d{8}-\d{3}(?:-[a-z0-9-]+)?$")
SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
DECISIONS = {"planned", "accept", "reject", "investigate"}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_manifest(document: Any, source: str = "manifest") -> list[str]:
    """Return human-readable violations for one experiment document."""

    if not isinstance(document, dict):
        return [f"{source}: root must be a JSON object"]

    errors: list[str] = []
    required_strings = (
        "experiment_id",
        "task_id",
        "hypothesis",
        "upstream_commit",
        "code_commit",
        "model",
        "model_revision",
        "eval_data_version",
        "normalizer_version",
        "command",
        "hardware",
        "decision",
    )
    for field in required_strings:
        if not _non_empty_string(document.get(field)):
            errors.append(f"{source}: {field} must be a non-empty string")

    experiment_id = document.get("experiment_id")
    if _non_empty_string(experiment_id) and not EXPERIMENT_ID.fullmatch(experiment_id):
        errors.append(f"{source}: experiment_id has an invalid format")

    for field in ("config_sha256", "data_sha256"):
        value = document.get(field)
        if not _non_empty_string(value) or not SHA256.fullmatch(value):
            errors.append(f"{source}: {field} must contain a SHA-256 digest")

    seed = document.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append(f"{source}: seed must be an integer")

    decision = document.get("decision")
    if _non_empty_string(decision) and decision not in DECISIONS:
        errors.append(f"{source}: decision must be one of {sorted(DECISIONS)}")

    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{source}: metrics must be a JSON object")
    else:
        for field in ("content_cer", "rtf_p50", "peak_rss_mb"):
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{source}: metrics.{field} must be numeric")

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        _non_empty_string(item) for item in artifacts
    ):
        errors.append(f"{source}: artifacts must be a list of non-empty strings")
    return errors


def validate_directory(directory: Path) -> list[str]:
    if not directory.is_dir():
        return [f"{directory}: manifest directory does not exist"]
    errors: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        errors.extend(validate_manifest(document, str(path)))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_MANIFEST_DIR)
    args = parser.parse_args()
    errors = validate_directory(args.directory)
    if errors:
        print("\n".join(errors))
        return 1
    count = len(list(args.directory.glob("*.json")))
    print(f"Experiment manifest check passed ({count} manifest(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
