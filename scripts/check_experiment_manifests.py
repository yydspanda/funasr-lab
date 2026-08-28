#!/usr/bin/env python3
"""Validate experiment provenance manifests without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "experiments" / "manifests"
DEFAULT_ROADMAP_PATH = REPO_ROOT / ".notes" / "asr" / "delivery-roadmap.md"
DEFAULT_CODE_REF = "refs/remotes/origin/develop"
EXPERIMENT_ID = re.compile(r"^EXP-\d{8}-\d{3}(?:-[a-z0-9-]+)?$")
TASK_ID = re.compile(r"^[A-Z][A-Z0-9]*-(?:\d{2}|SYNC)$")
ROADMAP_TASK_ROW = re.compile(
    r"^\|\s*`(?P<task_id>[A-Z][A-Z0-9]*-(?:\d{2}|SYNC))`\s*\|",
    re.MULTILINE,
)
ROADMAP_BASELINE_COMMIT = re.compile(
    r"^- \*\*Baseline Commit:\*\* `(?P<commit>[0-9a-f]{40})`$",
    re.MULTILINE,
)
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^sha256:([0-9a-f]{64})$")
MODEL_ROLE = re.compile(r"^[a-z][a-z0-9_-]*$")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DECISIONS = {"planned", "accept", "reject", "investigate"}
ARTIFACT_KINDS = {"report", "prediction", "log", "checkpoint", "other"}
FLOATING_REVISIONS = {"head", "latest", "main", "master", "trunk"}
PLACEHOLDERS = {
    "...",
    "changeme",
    "example",
    "n/a",
    "na",
    "none",
    "null",
    "placeholder",
    "pinned-revision",
    "replace-me",
    "tbd",
    "test",
    "test-host",
    "todo",
    "unknown",
}
EMPTY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb924"
    "27ae41e4649b934ca495991b7852b855"
)
TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_id",
    "task_id",
    "hypothesis",
    "upstream_commit",
    "code_commit",
    "models",
    "config_sha256",
    "data_sha256",
    "eval_data_version",
    "normalizer_version",
    "hardware",
    "seed",
    "command",
    "metrics",
    "artifacts",
    "decision",
}
REQUIRED_METRICS = {
    "content_cer",
    "substitutions",
    "deletions",
    "insertions",
    "reference_units",
    "utterance_count",
    "failed_count",
    "rtf_p50",
    "rtf_p95",
    "peak_rss_mb",
}
NULLABLE_METRICS = {
    "mer",
    "first_partial_ms",
    "finalization_ms",
    "partial_churn",
}
NON_NEGATIVE_METRICS = REQUIRED_METRICS | NULLABLE_METRICS
COUNT_METRICS = {
    "substitutions",
    "deletions",
    "insertions",
    "reference_units",
    "utterance_count",
    "failed_count",
}


class DuplicateJsonKeyError(ValueError):
    """Raised when provenance JSON contains an ambiguous duplicate object key."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKeyError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in PLACEHOLDERS:
        return True
    if re.fullmatch(r"<[^<>]+>", stripped):
        return True
    if re.fullmatch(r"[<{\[].*(?:todo|tbd|replace|placeholder|value).*[>}\]]", lowered):
        return True
    return bool(re.fullmatch(r"(?:todo|tbd|replace|placeholder)[: _-].*", lowered))


def _is_periodic(value: str) -> bool:
    """Identify visually plausible but mechanically repeated fake object IDs."""

    for width in (1, 2, 4, 8, 10, 16, 20, 32):
        if len(value) % width == 0 and value == value[:width] * (len(value) // width):
            return True
    return False


def _invalid_hex_identifier(value: str) -> bool:
    return len(set(value)) < 6 or _is_periodic(value)


def _validate_string(
    document: dict[str, Any], field: str, source: str, errors: list[str]
) -> None:
    value = document.get(field)
    if not _non_empty_string(value):
        errors.append(f"{source}: {field} must be a non-empty string")
    elif _is_placeholder(value):
        errors.append(f"{source}: {field} must not be a placeholder")


def _validate_commit(value: Any, field: str, source: str, errors: list[str]) -> None:
    if not _non_empty_string(value) or not GIT_COMMIT.fullmatch(value):
        errors.append(
            f"{source}: {field} must be a lowercase, full 40-character Git commit"
        )
    elif _invalid_hex_identifier(value):
        errors.append(f"{source}: {field} looks like a placeholder Git commit")


def _validate_sha256(value: Any, field: str, source: str, errors: list[str]) -> None:
    match = SHA256.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        errors.append(
            f"{source}: {field} must use the format sha256:<64 lowercase hex chars>"
        )
        return
    digest = match.group(1)
    if digest == EMPTY_SHA256 or _invalid_hex_identifier(digest):
        errors.append(f"{source}: {field} looks like a placeholder or empty digest")


def _validate_models(value: Any, source: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{source}: models must be a non-empty list")
        return

    roles: set[str] = set()
    expected_fields = {"role", "identifier", "revision", "sha256"}
    for index, component in enumerate(value):
        prefix = f"{source}: models[{index}]"
        if not isinstance(component, dict):
            errors.append(f"{prefix} must be a JSON object")
            continue

        missing = expected_fields - component.keys()
        unknown = component.keys() - expected_fields
        for field in sorted(missing):
            errors.append(f"{prefix}.{field} is required")
        if unknown:
            errors.append(f"{prefix} has unknown fields: {', '.join(sorted(unknown))}")

        role = component.get("role")
        if not _non_empty_string(role) or not MODEL_ROLE.fullmatch(role):
            errors.append(f"{prefix}.role has an invalid format")
        elif role in roles:
            errors.append(f"{prefix}.role duplicates model role {role!r}")
        else:
            roles.add(role)

        identifier = component.get("identifier")
        if not _non_empty_string(identifier) or _is_placeholder(identifier):
            errors.append(f"{prefix}.identifier must identify the resolved model")

        revision = component.get("revision")
        if not _non_empty_string(revision) or _is_placeholder(revision):
            errors.append(
                f"{prefix}.revision must be a pinned, non-placeholder revision"
            )
        else:
            stripped_revision = revision.strip()
            if revision != stripped_revision:
                errors.append(
                    f"{prefix}.revision must not contain surrounding whitespace"
                )
            if stripped_revision.lower().rsplit("/", 1)[-1] in FLOATING_REVISIONS:
                errors.append(f"{prefix}.revision must not use a floating revision")

        _validate_sha256(component.get("sha256"), f"models[{index}].sha256", source, errors)


def _validate_hardware(value: Any, source: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{source}: hardware must be a JSON object")
        return

    required = {
        "host_id",
        "os",
        "cpu_model",
        "logical_cpu_count",
        "memory_bytes",
        "device",
    }
    allowed = required | {"accelerator"}
    for field in sorted(required - value.keys()):
        errors.append(f"{source}: hardware.{field} is required")
    unknown = value.keys() - allowed
    if unknown:
        errors.append(
            f"{source}: hardware has unknown fields: {', '.join(sorted(unknown))}"
        )

    for field in ("host_id", "os", "cpu_model", "device"):
        field_value = value.get(field)
        if not _non_empty_string(field_value) or _is_placeholder(field_value):
            errors.append(
                f"{source}: hardware.{field} must be a concrete non-placeholder string"
            )

    accelerator = value.get("accelerator")
    if accelerator is not None and (
        not _non_empty_string(accelerator) or _is_placeholder(accelerator)
    ):
        errors.append(f"{source}: hardware.accelerator must be null or a concrete string")

    for field in ("logical_cpu_count", "memory_bytes"):
        field_value = value.get(field)
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value <= 0
        ):
            errors.append(f"{source}: hardware.{field} must be a positive integer")


def _validate_command(value: Any, source: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{source}: command must be a JSON object")
        return

    expected = {"working_directory", "argv", "environment"}
    for field in sorted(expected - value.keys()):
        errors.append(f"{source}: command.{field} is required")
    unknown = value.keys() - expected
    if unknown:
        errors.append(f"{source}: command has unknown fields: {', '.join(sorted(unknown))}")

    working_directory = value.get("working_directory")
    if not _non_empty_string(working_directory) or _is_placeholder(working_directory):
        errors.append(f"{source}: command.working_directory must be concrete")

    argv = value.get("argv")
    if not isinstance(argv, list) or len(argv) < 2:
        errors.append(f"{source}: command.argv must contain the complete argument vector")
    else:
        for index, argument in enumerate(argv):
            if not _non_empty_string(argument) or _is_placeholder(argument):
                errors.append(
                    f"{source}: command.argv[{index}] must be a complete, "
                    "non-placeholder argument"
                )

    environment = value.get("environment")
    if not isinstance(environment, dict):
        errors.append(f"{source}: command.environment must be a JSON object")
    else:
        for name, field_value in environment.items():
            if not isinstance(name, str) or not ENVIRONMENT_NAME.fullmatch(name):
                errors.append(f"{source}: command.environment has invalid name {name!r}")
            if not isinstance(field_value, str):
                errors.append(f"{source}: command.environment.{name} must be a string")


def _validate_metrics(value: Any, source: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{source}: metrics must be a JSON object")
        return

    for field in sorted(REQUIRED_METRICS - value.keys()):
        errors.append(f"{source}: metrics.{field} is required")

    for field, metric in value.items():
        if metric is None and field in NULLABLE_METRICS:
            continue
        if not _finite_number(metric):
            errors.append(f"{source}: metrics.{field} must be a finite number")
            continue
        if field in NON_NEGATIVE_METRICS and metric < 0:
            errors.append(f"{source}: metrics.{field} must be non-negative")

    for field in sorted(COUNT_METRICS):
        metric = value.get(field)
        if not isinstance(metric, int) or isinstance(metric, bool):
            errors.append(f"{source}: metrics.{field} must be an integer")

    for field in ("reference_units", "utterance_count"):
        metric = value.get(field)
        if isinstance(metric, int) and not isinstance(metric, bool) and metric <= 0:
            errors.append(f"{source}: metrics.{field} must be greater than zero")

    failed_count = value.get("failed_count")
    utterance_count = value.get("utterance_count")
    if (
        isinstance(failed_count, int)
        and not isinstance(failed_count, bool)
        and isinstance(utterance_count, int)
        and not isinstance(utterance_count, bool)
        and failed_count > utterance_count
    ):
        errors.append(
            f"{source}: metrics.failed_count cannot exceed metrics.utterance_count"
        )

    for field in ("rtf_p50", "peak_rss_mb"):
        metric = value.get(field)
        if _finite_number(metric) and metric <= 0:
            errors.append(f"{source}: metrics.{field} must be greater than zero")

    p50 = value.get("rtf_p50")
    p95 = value.get("rtf_p95")
    if all(_finite_number(metric) for metric in (p50, p95)) and p95 < p50:
        errors.append(f"{source}: metrics.rtf_p95 must be greater than or equal to rtf_p50")

    component_values = [
        value.get("substitutions"),
        value.get("deletions"),
        value.get("insertions"),
    ]
    reference_units = value.get("reference_units")
    content_cer = value.get("content_cer")
    if (
        all(
            isinstance(metric, int) and not isinstance(metric, bool)
            for metric in component_values
        )
        and isinstance(reference_units, int)
        and not isinstance(reference_units, bool)
        and reference_units > 0
        and _finite_number(content_cer)
    ):
        calculated_cer = sum(component_values) / reference_units
        if not math.isclose(content_cer, calculated_cer, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(
                f"{source}: metrics.content_cer must equal "
                "(substitutions + deletions + insertions) / reference_units"
            )


def _validate_artifacts(value: Any, source: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{source}: artifacts must be a list")
        return

    expected = {"kind", "path", "sha256"}
    for index, artifact in enumerate(value):
        prefix = f"{source}: artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be a JSON object")
            continue
        for field in sorted(expected - artifact.keys()):
            errors.append(f"{prefix}.{field} is required")
        unknown = artifact.keys() - expected
        if unknown:
            errors.append(f"{prefix} has unknown fields: {', '.join(sorted(unknown))}")
        kind = artifact.get("kind")
        if not _non_empty_string(kind) or kind not in ARTIFACT_KINDS:
            errors.append(
                f"{prefix}.kind must be one of {sorted(ARTIFACT_KINDS)}"
            )
        path = artifact.get("path")
        if not _non_empty_string(path) or _is_placeholder(path):
            errors.append(f"{prefix}.path must be a concrete path")
        _validate_sha256(
            artifact.get("sha256"), f"artifacts[{index}].sha256", source, errors
        )


def validate_manifest(document: Any, source: str = "manifest") -> list[str]:
    """Return human-readable violations for one experiment document."""

    if not isinstance(document, dict):
        return [f"{source}: root must be a JSON object"]

    errors: list[str] = []
    for field in sorted(TOP_LEVEL_FIELDS - document.keys()):
        errors.append(f"{source}: {field} is required")
    unknown = document.keys() - TOP_LEVEL_FIELDS
    if unknown:
        errors.append(f"{source}: unknown fields: {', '.join(sorted(unknown))}")

    if document.get("schema_version") != 1 or isinstance(
        document.get("schema_version"), bool
    ):
        errors.append(f"{source}: schema_version must be integer 1")

    experiment_id = document.get("experiment_id")
    if not _non_empty_string(experiment_id) or not EXPERIMENT_ID.fullmatch(
        experiment_id
    ):
        errors.append(f"{source}: experiment_id has an invalid format")

    task_id = document.get("task_id")
    if not _non_empty_string(task_id) or not TASK_ID.fullmatch(task_id):
        errors.append(f"{source}: task_id has an invalid format")

    _validate_string(document, "hypothesis", source, errors)
    hypothesis = document.get("hypothesis")
    if _non_empty_string(hypothesis) and len(hypothesis.strip()) < 12:
        errors.append(f"{source}: hypothesis must be at least 12 characters")

    _validate_commit(document.get("upstream_commit"), "upstream_commit", source, errors)
    _validate_commit(document.get("code_commit"), "code_commit", source, errors)
    _validate_models(document.get("models"), source, errors)
    _validate_sha256(document.get("config_sha256"), "config_sha256", source, errors)
    _validate_sha256(document.get("data_sha256"), "data_sha256", source, errors)

    for field in ("eval_data_version", "normalizer_version"):
        _validate_string(document, field, source, errors)

    _validate_hardware(document.get("hardware"), source, errors)

    seed = document.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append(f"{source}: seed must be a non-negative integer")

    _validate_command(document.get("command"), source, errors)
    _validate_artifacts(document.get("artifacts"), source, errors)

    decision = document.get("decision")
    if not _non_empty_string(decision) or decision not in DECISIONS:
        errors.append(f"{source}: decision must be one of {sorted(DECISIONS)}")
        _validate_metrics(document.get("metrics"), source, errors)
    elif decision == "planned":
        if document.get("metrics") is not None:
            errors.append(
                f"{source}: planned experiment metrics must be null until execution"
            )
        artifacts = document.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            errors.append(
                f"{source}: planned experiment artifacts must be empty until execution"
            )
    else:
        _validate_metrics(document.get("metrics"), source, errors)
        artifacts = document.get("artifacts")
        has_report = isinstance(artifacts, list) and any(
            isinstance(artifact, dict) and artifact.get("kind") == "report"
            for artifact in artifacts
        )
        if not has_report:
            errors.append(
                f"{source}: executed experiment decision {decision!r} requires at least "
                "one hashed artifact with kind 'report'"
            )
    return errors


def _read_roadmap_control(
    roadmap_path: Path,
) -> tuple[set[str], str | None, list[str]]:
    try:
        roadmap = roadmap_path.read_text(encoding="utf-8")
    except OSError as exc:
        return set(), None, [f"{roadmap_path}: cannot read roadmap control record: {exc}"]

    task_ids = {
        match.group("task_id") for match in ROADMAP_TASK_ROW.finditer(roadmap)
    }
    errors: list[str] = []
    if not task_ids:
        errors.append(f"{roadmap_path}: no roadmap task rows found")
    baseline_matches = list(ROADMAP_BASELINE_COMMIT.finditer(roadmap))
    if len(baseline_matches) != 1:
        errors.append(
            f"{roadmap_path}: expected exactly one full Baseline Commit field, "
            f"found {len(baseline_matches)}"
        )
        baseline_commit = None
    else:
        baseline_commit = baseline_matches[0].group("commit")
    return task_ids, baseline_commit, errors


def _git_commit_exists(
    repo_root: Path,
    commit: str,
    *,
    git_executable: str = "git",
    git_environment: Mapping[str, str] | None = None,
) -> bool:
    completed = subprocess.run(
        [git_executable, "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment,
    )
    return completed.returncode == 0


def _git_is_ancestor(
    repo_root: Path,
    ancestor: str,
    descendant: str,
    *,
    git_executable: str = "git",
    git_environment: Mapping[str, str] | None = None,
) -> bool:
    completed = subprocess.run(
        [git_executable, "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment,
    )
    return completed.returncode == 0


def validate_directory(
    directory: Path,
    roadmap_path: Path = DEFAULT_ROADMAP_PATH,
    *,
    repo_root: Path = REPO_ROOT,
    verify_git: bool = True,
    code_ref: str = DEFAULT_CODE_REF,
    git_executable: str = "git",
    git_environment: Mapping[str, str] | None = None,
) -> list[str]:
    if not directory.is_dir():
        return [f"{directory}: manifest directory does not exist"]
    tracked_manifest_directory = (
        directory.resolve()
        == (repo_root.resolve() / "experiments/manifests").resolve()
    )
    task_ids, baseline_commit, errors = _read_roadmap_control(roadmap_path)
    manifest_paths = sorted(directory.glob("*.json"))
    code_ref_exists = True
    if verify_git and manifest_paths and not _git_commit_exists(
        repo_root,
        code_ref,
        git_executable=git_executable,
        git_environment=git_environment,
    ):
        code_ref_exists = False
        errors.append(
            f"{code_ref}: target code ref does not resolve; fetch the target branch "
            "before validating manifests"
        )
    experiment_sources: dict[str, Path] = {}
    for path in manifest_paths:
        try:
            document = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (OSError, json.JSONDecodeError, DuplicateJsonKeyError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        errors.extend(validate_manifest(document, str(path)))
        if isinstance(document, dict):
            if (
                tracked_manifest_directory
                and document.get("task_id") == "EVAL-01"
                and (
                    document.get("decision") != "planned"
                    or document.get("metrics") is not None
                    or document.get("artifacts") != []
                )
            ):
                errors.append(
                    f"{path}: tracked EVAL-01 manifests must remain planned with "
                    "null metrics and no artifacts; result-bearing copies are private"
                )
            experiment_id = document.get("experiment_id")
            if _non_empty_string(experiment_id) and EXPERIMENT_ID.fullmatch(
                experiment_id
            ):
                expected_name = f"{experiment_id}.json"
                if path.name != expected_name:
                    errors.append(
                        f"{path}: filename must be {expected_name!r} to match experiment_id"
                    )
                previous = experiment_sources.get(experiment_id)
                if previous is not None:
                    errors.append(
                        f"{path}: duplicate experiment_id {experiment_id!r}; "
                        f"already recorded by {previous}"
                    )
                else:
                    experiment_sources[experiment_id] = path
            task_id = document.get("task_id")
            if _non_empty_string(task_id) and task_id not in task_ids:
                errors.append(
                    f"{path}: task_id {task_id!r} is not registered in {roadmap_path}"
                )
            if verify_git and baseline_commit is not None:
                upstream_commit = document.get("upstream_commit")
                code_commit = document.get("code_commit")
                if _non_empty_string(upstream_commit) and GIT_COMMIT.fullmatch(
                    upstream_commit
                ):
                    if not _git_commit_exists(
                        repo_root,
                        upstream_commit,
                        git_executable=git_executable,
                        git_environment=git_environment,
                    ):
                        errors.append(
                            f"{path}: upstream_commit does not resolve to a Git commit"
                        )
                    elif not _git_is_ancestor(
                        repo_root,
                        upstream_commit,
                        baseline_commit,
                        git_executable=git_executable,
                        git_environment=git_environment,
                    ):
                        errors.append(
                            f"{path}: upstream_commit is not in the accepted upstream "
                            "baseline history"
                        )
                if _non_empty_string(code_commit) and GIT_COMMIT.fullmatch(code_commit):
                    if not _git_commit_exists(
                        repo_root,
                        code_commit,
                        git_executable=git_executable,
                        git_environment=git_environment,
                    ):
                        errors.append(
                            f"{path}: code_commit does not resolve to a Git commit"
                        )
                    elif not _git_is_ancestor(
                        repo_root,
                        code_commit,
                        "HEAD",
                        git_executable=git_executable,
                        git_environment=git_environment,
                    ):
                        errors.append(
                            f"{path}: code_commit is not an ancestor of the checked-out HEAD"
                        )
                    elif code_ref_exists and not _git_is_ancestor(
                        repo_root,
                        code_commit,
                        code_ref,
                        git_executable=git_executable,
                        git_environment=git_environment,
                    ):
                        errors.append(
                            f"{path}: code_commit is not reachable from target code "
                            f"ref {code_ref}; land the code before registering the run"
                        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--roadmap", type=Path, default=DEFAULT_ROADMAP_PATH, help="task registry"
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="skip resolving manifest commits and ancestry",
    )
    parser.add_argument(
        "--code-ref",
        default=DEFAULT_CODE_REF,
        help=(
            "durable target branch that must already contain code_commit "
            f"(default: {DEFAULT_CODE_REF})"
        ),
    )
    args = parser.parse_args()
    errors = validate_directory(
        args.directory,
        args.roadmap,
        verify_git=not args.no_git,
        code_ref=args.code_ref,
    )
    if errors:
        print("\n".join(errors))
        return 1
    count = len(list(args.directory.glob("*.json")))
    print(f"Experiment manifest check passed ({count} manifest(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
