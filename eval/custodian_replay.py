"""Blind-safe artifacts for an offline EVAL-01 custodian replay.

This module never constructs or imports an ASR model.  It freezes the boundary
between a reference-free sealed audio projection, one pre-registered candidate,
the decoder's exact predictions, and the restricted scoring process that owns
the sealed references.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.collection import SEALED_INPUT_PROJECTION_KIND
from eval.collection import CollectionDescriptor
from eval.collection import ValidatedCollection
from eval.collection import build_sealed_input_projection
from eval.collection import canonical_json_bytes
from eval.collection import sha256_bytes
from eval.core_report import CORE_SCHEMA_VERSION
from eval.core_report import CoreReportValidationError
from eval.core_report import IDENTITY_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import adapt_hypothesis
from eval.core_report import canonical_core_bytes
from eval.core_report import validate_core_report
from eval.execution_envelope import ExecutionEnvelopeError
from eval.execution_envelope import LoadedExecutionEnvelope
from eval.execution_envelope import canonical_execution_envelope_bytes
from eval.execution_envelope import peak_rss_mib
from eval.execution_envelope import validate_execution_envelope
from eval.execution_envelope import validate_execution_envelope_for_predictions
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import RecordIdentityError
from eval.record_identity import record_input_sha256
from eval.sealed_candidate_contract import SealedCandidateContractError
from eval.sealed_candidate_contract import validate_sealed_candidate_execution
from scripts.check_experiment_manifests import validate_directory
from scripts.check_experiment_manifests import validate_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_GIT_PATH = Path("/usr/bin/git")
CANDIDATE_LOCK_SCHEMA_VERSION = 2
CANDIDATE_LOCK_KIND = "asr-evaluation-candidate-lock"
PREDICTION_BUNDLE_SCHEMA_VERSION = 2
PREDICTION_BUNDLE_KIND = "asr-evaluation-predictions"
RECEIPT_SCHEMA_VERSION = 2
INPUT_EXPORT_RECEIPT_KIND = "asr-evaluation-input-export-receipt"
PREDICTION_FREEZE_RECEIPT_KIND = "asr-evaluation-prediction-freeze-receipt"
CUSTODIAN_SCORE_RECEIPT_KIND = "asr-evaluation-custodian-score-receipt"
SEALED_SPLIT = "sealed-blind"

MAX_CANDIDATE_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_PREDICTION_ITEMS = 1_000_000
MAX_HYPOTHESIS_CHARACTERS = 16_384
MAX_TOTAL_HYPOTHESIS_CHARACTERS = 1_000_000
MAX_RAW_PREDICTION_LINE_BYTES = 128 * 1024
MAX_SHORT_STRING_CHARACTERS = 256
MAX_EXPERIMENT_ID_CHARACTERS = 128
MAX_PREDICTION_ID_CHARACTERS = 512
MAX_REASON_CODE_CHARACTERS = 128

SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
EXPERIMENT_ID_PATTERN = re.compile(
    r"^EXP-[0-9]{8}-[0-9]{3}(?:-[a-z0-9-]+)?$"
)
REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
PYTHON_VERSION_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$"
)
PYTHON_CACHE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
UNICODE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

CANDIDATE_REGISTRATION_FIELD_NAMES = (
    "candidate_registration_commit",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
)

SCORER_SOURCE_PATHS = (
    "eval/__init__.py",
    "eval/collection.py",
    "eval/core_report.py",
    "eval/custodian_replay.py",
    "eval/execution_envelope.py",
    "eval/normalizers/__init__.py",
    "eval/normalizers/zh_content.py",
    "eval/offline_baseline.py",
    "eval/record_identity.py",
    "eval/scoring.py",
    "eval/sealed_candidate_contract.py",
    "eval/sealed_decoder.py",
    "requirements/lab-cpu.lock",
    "scripts/check_experiment_manifests.py",
    "scripts/__init__.py",
    "scripts/replay_asr_evaluation.py",
)

HYPOTHESIS_ADAPTER_VERSIONS = frozenset(
    {
        IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
        SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
    }
)

CANDIDATE_FREEZE_FIELDS = frozenset(
    {
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
    }
)

CANDIDATE_LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "access_class",
        "dataset_id",
        "revision",
        "split",
        "data_sha256",
        "input_projection_sha256",
        "hypothesis_adapter_version",
        "record_identity_version",
        "record_input_sha256",
        "decode_item_count",
        "decode_item_ids_sha256",
        "source_manifest_decision",
        "candidate_registration_commit",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
        "candidate",
        "candidate_freeze_sha256",
    }
)

PREDICTION_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "access_class",
        "dataset_id",
        "revision",
        "split",
        "input_projection_sha256",
        "candidate_lock_sha256",
        "input_export_receipt_sha256",
        "raw_predictions_sha256",
        "execution_envelope_sha256",
        "hypothesis_adapter_version",
        "item_count",
        "items_sha256",
        "items",
    }
)

PREDICTION_ITEM_FIELDS = frozenset(
    {"id", "raw_text", "status", "reason_code"}
)

TERMINAL_METRIC_FIELDS = frozenset(
    {
        "content_cer",
        "substitutions",
        "deletions",
        "insertions",
        "reference_units",
        "utterance_count",
        "failed_count",
        "excluded_count",
        "mer",
        "rtf_p50",
        "rtf_p95",
        "peak_rss_mb",
        "rtf_attempted_count",
        "retried_count",
        "model_load_seconds",
        "cold_inference_seconds",
        "cold_start_seconds",
        "warm_wall_seconds",
        "warm_audio_seconds",
    }
)

INPUT_EXPORT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "access_class",
        "experiment_id",
        "dataset_id",
        "revision",
        "split",
        "decode_item_count",
        "input_projection_sha256",
        "candidate_lock_sha256",
        "candidate_freeze_sha256",
        "candidate_registration_commit",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
    }
)

PREDICTION_FREEZE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "access_class",
        "experiment_id",
        "dataset_id",
        "revision",
        "split",
        "expected_decode_item_count",
        "prediction_item_count",
        "missing_prediction_count",
        "input_projection_sha256",
        "candidate_lock_sha256",
        "candidate_freeze_sha256",
        "candidate_registration_commit",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
        "hypothesis_adapter_version",
        "prediction_artifact_sha256",
        "prediction_items_sha256",
        "input_export_receipt_sha256",
        "raw_predictions_sha256",
        "execution_envelope_sha256",
        "runner_code_commit",
        "runner_source_sha256",
    }
)

CUSTODIAN_SCORE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "access_class",
        "experiment_id",
        "dataset_id",
        "revision",
        "evaluation_scope",
        "data_sha256",
        "input_projection_sha256",
        "record_identity_version",
        "record_input_sha256",
        "hypothesis_adapter_version",
        "prediction_input_sha256",
        "candidate_lock_sha256",
        "candidate_freeze_sha256",
        "candidate_registration_commit",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
        "prediction_artifact_sha256",
        "prediction_items_sha256",
        "input_export_receipt_sha256",
        "prediction_freeze_receipt_sha256",
        "execution_envelope_sha256",
        "runner_code_commit",
        "runner_source_sha256",
        "scorer_code_commit",
        "scorer_source_sha256",
        "scorer_runtime",
        "core_schema_version",
        "core_sha256",
        "public_release",
    }
)

SCORER_RUNTIME_FIELDS = frozenset(
    {
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "dependency_lock_sha256",
        "installed_dependencies_sha256",
        "installed_dependency_count",
        "unicode_version",
    }
)

PUBLIC_RELEASE_FIELDS = frozenset(
    {"state", "summary_sha256", "reason_code"}
)

SEALED_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "dataset_id",
        "revision",
        "split",
        "manifest_sha256",
        "manifest_record_count",
        "item_count",
        "items",
    }
)

SEALED_INPUT_ITEM_FIELDS = frozenset(
    {
        "id",
        "split",
        "audio",
        "audio_sha256",
        "duration_seconds",
        "sample_rate",
        "channels",
        "sample_width_bits",
    }
)


class CustodianReplayError(ValueError):
    """Raised when a custodian artifact or replay boundary is invalid."""


class _DuplicateJsonKeyError(ValueError):
    pass


class _InvalidJsonConstantError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedArtifact:
    """One strictly parsed canonical artifact and its exact byte identity."""

    document: dict[str, Any]
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class LoadedPredictionItems:
    """Strict raw decoder JSONL plus its exact byte identity."""

    items: tuple[dict[str, Any], ...]
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class RegisteredCandidateManifest:
    """One planned manifest proven byte-identical to a reachable Git blob."""

    document: dict[str, Any]
    payload: bytes
    sha256: str
    repository_path: str
    registration_commit: str


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> None:
    raise _InvalidJsonConstantError(f"non-standard JSON constant {value!r}")


def _strict_fields(
    document: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    missing = sorted(expected - set(document))
    unknown = sorted(set(document) - expected)
    if missing:
        raise CustodianReplayError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise CustodianReplayError(
            f"{context} has unknown field(s): {', '.join(unknown)}"
        )


def _string(
    value: Any,
    context: str,
    *,
    allow_empty: bool = False,
    maximum_characters: int | None = None,
) -> str:
    if not isinstance(value, str):
        raise CustodianReplayError(f"{context} must be a string")
    if not allow_empty and not value:
        raise CustodianReplayError(f"{context} must be a non-empty string")
    if value != value.strip() and not allow_empty:
        raise CustodianReplayError(
            f"{context} must not contain surrounding whitespace"
        )
    if maximum_characters is not None and len(value) > maximum_characters:
        raise CustodianReplayError(
            f"{context} exceeds the {maximum_characters}-character limit"
        )
    return value


def _sha256(value: Any, context: str) -> str:
    digest = _string(value, context)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise CustodianReplayError(
            f"{context} must use sha256:<64 lowercase hex characters>"
        )
    return digest


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CustodianReplayError(f"{context} must be a non-negative integer")
    return value


def _bounded_nonnegative_integer(
    value: Any,
    context: str,
    *,
    maximum: int,
) -> int:
    result = _nonnegative_integer(value, context)
    if result > maximum:
        raise CustodianReplayError(f"{context} exceeds the {maximum} limit")
    return result


def _trusted_git_context(
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, dict[str, str]]:
    """Return the fixed Git executable and configuration-free environment."""

    try:
        git_metadata = TRUSTED_GIT_PATH.lstat()
    except OSError as exc:
        raise CustodianReplayError("trusted Git executable is unavailable") from exc
    if (
        stat.S_ISLNK(git_metadata.st_mode)
        or not stat.S_ISREG(git_metadata.st_mode)
        or stat.S_IMODE(git_metadata.st_mode) & 0o022
    ):
        raise CustodianReplayError("trusted Git executable is unsafe")
    git_environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/dev/null",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    root = repository_root.resolve()
    try:
        common_directory = subprocess.run(
            [
                str(TRUSTED_GIT_PATH),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        ).stdout.strip()
        replacement_refs = subprocess.run(
            [
                str(TRUSTED_GIT_PATH),
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace/",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CustodianReplayError("cannot inspect the trusted Git object view") from exc
    common_path = Path(common_directory)
    try:
        common_metadata = common_path.lstat()
    except OSError as exc:
        raise CustodianReplayError("Git common directory is unavailable") from exc
    if (
        not common_path.is_absolute()
        or stat.S_ISLNK(common_metadata.st_mode)
        or not stat.S_ISDIR(common_metadata.st_mode)
    ):
        raise CustodianReplayError("Git common directory is unsafe")
    if replacement_refs:
        raise CustodianReplayError("Git replacement refs are forbidden")
    if os.path.lexists(common_path / "info/grafts"):
        raise CustodianReplayError("Git grafts are forbidden")
    return str(TRUSTED_GIT_PATH), git_environment


def scorer_code_identity(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    code_commit: str,
) -> tuple[str, str]:
    """Bind current scorer bytes to one explicit, immutable Git commit."""

    root = repository_root.resolve()
    git_executable, git_environment = _trusted_git_context(root)
    requested_commit = code_commit
    if GIT_COMMIT_PATTERN.fullmatch(code_commit) is None:
        raise CustodianReplayError(
            "requested scorer_code_commit must be a full Git commit"
        )
    try:
        completed = subprocess.run(
            [
                git_executable,
                "rev-parse",
                "--verify",
                f"{requested_commit}^{{commit}}",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CustodianReplayError(
            "cannot resolve the scorer Git commit"
        ) from exc
    code_commit = completed.stdout.strip()
    if not GIT_COMMIT_PATTERN.fullmatch(code_commit):
        raise CustodianReplayError("scorer Git commit is not a full commit identity")
    if code_commit != requested_commit:
        raise CustodianReplayError("requested scorer_code_commit did not resolve exactly")

    inventory: list[dict[str, str]] = []
    for relative_path in SCORER_SOURCE_PATHS:
        payload = _regular_file_bytes(
            root / relative_path,
            MAX_CANDIDATE_MANIFEST_BYTES,
            f"scorer source {relative_path}",
        )
        try:
            committed = subprocess.run(
                [git_executable, "show", f"{code_commit}:{relative_path}"],
                cwd=root,
                check=True,
                capture_output=True,
                env=git_environment,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CustodianReplayError(
                f"cannot load committed scorer source {relative_path}"
            ) from exc
        if payload != committed:
            raise CustodianReplayError(
                f"scorer source {relative_path} differs from scorer_code_commit; "
                "commit the scorer before a sealed replay"
            )
        inventory.append(
            {
                "path": relative_path,
                "sha256": sha256_bytes(payload),
            }
        )
    return code_commit, sha256_bytes(canonical_json_bytes(inventory))


def _regular_file_bytes(path: Path, maximum_bytes: int, context: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise CustodianReplayError(
            f"cannot safely read {context}: O_NOFOLLOW is unavailable"
        )
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CustodianReplayError(f"cannot read {context}: {exc.strerror}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CustodianReplayError(
                f"{context} must be a regular, non-symlink file"
            )
        if before.st_size > maximum_bytes:
            raise CustodianReplayError(
                f"{context} exceeds the {maximum_bytes}-byte safety limit"
            )

        payload = bytearray()
        while len(payload) <= maximum_bytes:
            remaining = maximum_bytes + 1 - len(payload)
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum_bytes:
            raise CustodianReplayError(
                f"{context} exceeds the {maximum_bytes}-byte safety limit"
            )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CustodianReplayError(f"{context} changed while it was read")
        path_after = path.lstat()
        if (
            not stat.S_ISREG(path_after.st_mode)
            or (path_after.st_dev, path_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise CustodianReplayError(
                f"{context} path changed while it was read"
            )
        return bytes(payload)
    except OSError as exc:
        raise CustodianReplayError(f"cannot read {context}: {exc.strerror}") from exc
    finally:
        os.close(descriptor)


def _parse_json_document(
    payload: bytes,
    context: str,
    *,
    require_canonical: bool,
) -> dict[str, Any]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise CustodianReplayError(f"{context} must not contain a UTF-8 BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CustodianReplayError(f"{context} must use valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        _InvalidJsonConstantError,
    ) as exc:
        raise CustodianReplayError(f"{context} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CustodianReplayError(f"{context} root must be a JSON object")
    if require_canonical and payload != canonical_json_bytes(value):
        raise CustodianReplayError(f"{context} must use canonical JSON bytes")
    return value


def load_planned_candidate_manifest(
    path: Path,
    registration_commit: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> RegisteredCandidateManifest:
    """Load one planned manifest from its exact, reachable registration blob."""

    root = repository_root.resolve()
    registered_directory = root / "experiments/manifests"
    supplied = path if path.is_absolute() else root / path
    if supplied.parent != registered_directory:
        raise CustodianReplayError(
            "candidate manifest must use its exact experiments/manifests path"
        )
    if registered_directory.resolve() != registered_directory:
        raise CustodianReplayError(
            "candidate manifest directory must not contain symlinks"
        )

    payload = _regular_file_bytes(
        supplied,
        MAX_CANDIDATE_MANIFEST_BYTES,
        "candidate manifest",
    )
    document = _parse_json_document(
        payload,
        "candidate manifest",
        require_canonical=False,
    )
    violations = validate_manifest(document, "candidate manifest")
    if violations:
        raise CustodianReplayError(violations[0])
    if document.get("decision") != "planned":
        raise CustodianReplayError(
            "candidate manifest must still have decision 'planned' before blind replay"
        )
    expected_name = f"{document['experiment_id']}.json"
    if supplied.name != expected_name:
        raise CustodianReplayError(
            f"candidate manifest filename must be {expected_name!r}"
        )
    repository_path = f"experiments/manifests/{expected_name}"
    if supplied != root / repository_path:
        raise CustodianReplayError(
            "candidate manifest must use its exact experiments/manifests path"
        )

    registration = _string(
        registration_commit,
        "candidate registration commit",
        maximum_characters=40,
    )
    if GIT_COMMIT_PATTERN.fullmatch(registration) is None:
        raise CustodianReplayError(
            "candidate registration commit must be a full Git commit"
        )
    git_executable, git_environment = _trusted_git_context(root)
    try:
        resolved_registration = subprocess.run(
            [
                git_executable,
                "rev-parse",
                "--verify",
                f"{registration}^{{commit}}",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CustodianReplayError(
            "candidate registration commit does not resolve"
        ) from exc
    if resolved_registration != registration:
        raise CustodianReplayError(
            "candidate registration commit did not resolve exactly"
        )

    for ancestor, descendant, message in (
        (
            document["code_commit"],
            registration,
            "candidate code_commit must be an ancestor of its registration commit",
        ),
        (
            registration,
            "HEAD",
            "candidate registration commit must be reachable from checked-out HEAD",
        ),
        (
            registration,
            "refs/remotes/origin/develop",
            "candidate registration commit must be reachable from origin/develop; "
            "fetch the durable branch before replay",
        ),
    ):
        completed = subprocess.run(
            [git_executable, "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root,
            check=False,
            capture_output=True,
            env=git_environment,
        )
        if completed.returncode != 0:
            raise CustodianReplayError(message)

    blob_spec = f"{registration}:{repository_path}"
    try:
        size_text = subprocess.run(
            [git_executable, "cat-file", "-s", blob_spec],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        ).stdout.strip()
        blob_size = int(size_text)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise CustodianReplayError(
            "candidate manifest is not tracked by its registration commit"
        ) from exc
    if blob_size > MAX_CANDIDATE_MANIFEST_BYTES:
        raise CustodianReplayError(
            "registered candidate manifest exceeds the safety limit"
        )
    try:
        registered_payload = subprocess.run(
            [git_executable, "show", blob_spec],
            cwd=root,
            check=True,
            capture_output=True,
            env=git_environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CustodianReplayError(
            "cannot load the registered candidate manifest blob"
        ) from exc
    if len(registered_payload) != blob_size or registered_payload != payload:
        raise CustodianReplayError(
            "candidate manifest bytes do not match its registration commit"
        )

    governance_violations = validate_directory(
        registered_directory,
        roadmap_path=root / ".notes/asr/delivery-roadmap.md",
        repo_root=root,
        # The exact candidate blob is proven above. Directory-wide governance
        # applies to the current checkout so later manifests do not invalidate
        # an older, still-reachable registration.
        code_ref="HEAD",
        git_executable=git_executable,
        git_environment=git_environment,
    )
    if governance_violations:
        raise CustodianReplayError(governance_violations[0])
    if _regular_file_bytes(
        supplied,
        MAX_CANDIDATE_MANIFEST_BYTES,
        "candidate manifest",
    ) != payload:
        raise CustodianReplayError(
            "candidate manifest changed during governance validation"
        )
    return RegisteredCandidateManifest(
        document=document,
        payload=payload,
        sha256=sha256_bytes(payload),
        repository_path=repository_path,
        registration_commit=registration,
    )


def load_terminal_candidate_manifest(path: Path) -> dict[str, Any]:
    """Load one canonical, result-bearing private EVAL-01 manifest."""

    payload = _regular_file_bytes(
        path,
        MAX_CANDIDATE_MANIFEST_BYTES,
        "terminal candidate manifest",
    )
    document = _parse_json_document(
        payload,
        "terminal candidate manifest",
        require_canonical=True,
    )
    violations = validate_manifest(document, str(path))
    if violations:
        raise CustodianReplayError(violations[0])
    if document.get("decision") == "planned":
        raise CustodianReplayError(
            "terminal candidate manifest must contain a result"
        )
    return document


def load_restricted_core_report(path: Path) -> dict[str, Any]:
    """Load one canonical restricted core before terminal validation."""

    payload = _regular_file_bytes(path, MAX_ARTIFACT_BYTES, "restricted core")
    document = _parse_json_document(
        payload,
        "restricted core",
        require_canonical=True,
    )
    try:
        validate_core_report(document)
    except CoreReportValidationError as exc:
        raise CustodianReplayError(str(exc)) from exc
    return document


def candidate_freeze_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project immutable pre-result facts from one valid planned manifest."""

    violations = validate_manifest(dict(manifest), "candidate manifest")
    if violations:
        raise CustodianReplayError(violations[0])
    if manifest.get("decision") != "planned":
        raise CustodianReplayError("candidate manifest decision must be planned")
    projection = _candidate_projection(manifest)
    _validate_candidate_projection(projection)
    return projection


def _candidate_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the immutable identity fields from an already validated manifest."""

    _strict_fields(
        {key: manifest[key] for key in manifest if key in CANDIDATE_FREEZE_FIELDS},
        CANDIDATE_FREEZE_FIELDS,
        "candidate freeze projection",
    )
    return {
        key: deepcopy(manifest[key])
        for key in sorted(CANDIDATE_FREEZE_FIELDS)
    }


def candidate_manifest_freeze_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash immutable candidate facts from a planned or terminal manifest."""

    violations = validate_manifest(dict(manifest), "candidate manifest")
    if violations:
        raise CustodianReplayError(violations[0])
    return candidate_freeze_sha256(_candidate_projection(manifest))


def candidate_freeze_sha256(candidate: Mapping[str, Any]) -> str:
    """Return the stable hash used after metrics/artifacts update the manifest."""

    _validate_candidate_projection(candidate)
    return sha256_bytes(canonical_json_bytes(candidate))


def validate_registered_candidate_binding(
    evidence: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> RegisteredCandidateManifest:
    """Re-prove one lock or receipt's registered candidate against Git."""

    candidate = evidence.get("candidate")
    if isinstance(candidate, Mapping):
        experiment_id = candidate.get("experiment_id")
    else:
        experiment_id = evidence.get("experiment_id")
    if not isinstance(experiment_id, str):
        raise CustodianReplayError(
            "registered candidate evidence has no experiment_id"
        )
    _validate_candidate_registration(evidence, "registered candidate", experiment_id)
    registered = load_planned_candidate_manifest(
        Path(evidence["candidate_manifest_path"]),
        evidence["candidate_registration_commit"],
        repository_root=repository_root,
    )
    if registered.sha256 != evidence["candidate_manifest_sha256"]:
        raise CustodianReplayError(
            "registered candidate manifest hash does not match evidence"
        )
    if candidate_manifest_freeze_sha256(registered.document) != evidence[
        "candidate_freeze_sha256"
    ]:
        raise CustodianReplayError(
            "registered candidate facts do not match evidence"
        )
    if isinstance(candidate, Mapping) and _candidate_projection(
        registered.document
    ) != dict(candidate):
        raise CustodianReplayError(
            "registered candidate projection does not match candidate lock"
        )
    return registered


def _validate_candidate_projection(candidate: Any) -> Mapping[str, Any]:
    if not isinstance(candidate, Mapping):
        raise CustodianReplayError("candidate must be a JSON object")
    _strict_fields(candidate, CANDIDATE_FREEZE_FIELDS, "candidate")
    planned_document = dict(candidate)
    planned_document.update(
        {"metrics": None, "artifacts": [], "decision": "planned"}
    )
    violations = validate_manifest(planned_document, "candidate")
    if violations:
        raise CustodianReplayError(violations[0])
    _validate_candidate_schema_bounds(candidate)
    return candidate


def _validate_candidate_schema_bounds(candidate: Mapping[str, Any]) -> None:
    """Mirror candidate-lock JSON Schema bounds in the semantic validator."""

    def check_text(value: str, maximum: int, context: str) -> None:
        if len(value) > maximum:
            raise CustodianReplayError(
                f"{context} exceeds the {maximum}-character limit"
            )

    if candidate["task_id"] != "EVAL-01":
        raise CustodianReplayError("candidate task_id must be EVAL-01")
    check_text(
        candidate["experiment_id"],
        MAX_EXPERIMENT_ID_CHARACTERS,
        "candidate.experiment_id",
    )
    check_text(
        candidate["hypothesis"],
        MAX_HYPOTHESIS_CHARACTERS,
        "candidate.hypothesis",
    )
    models = candidate["models"]
    if len(models) > 32:
        raise CustodianReplayError("candidate.models exceeds the 32-item limit")
    for index, component in enumerate(models):
        check_text(component["role"], 64, f"candidate.models[{index}].role")
        check_text(
            component["identifier"],
            1024,
            f"candidate.models[{index}].identifier",
        )
        check_text(
            component["revision"],
            MAX_SHORT_STRING_CHARACTERS,
            f"candidate.models[{index}].revision",
        )
    for field in ("eval_data_version", "normalizer_version"):
        check_text(
            candidate[field],
            MAX_SHORT_STRING_CHARACTERS,
            f"candidate.{field}",
        )
    if candidate["seed"] > 4_294_967_295:
        raise CustodianReplayError("candidate.seed exceeds the 4294967295 limit")

    hardware = candidate["hardware"]
    hardware_text_limits = {
        "host_id": MAX_SHORT_STRING_CHARACTERS,
        "os": 1024,
        "cpu_model": 1024,
        "device": MAX_SHORT_STRING_CHARACTERS,
    }
    for field, maximum in hardware_text_limits.items():
        check_text(hardware[field], maximum, f"candidate.hardware.{field}")
    accelerator = hardware.get("accelerator")
    if accelerator is not None:
        check_text(accelerator, 1024, "candidate.hardware.accelerator")
    if hardware["logical_cpu_count"] > 1_048_576:
        raise CustodianReplayError(
            "candidate.hardware.logical_cpu_count exceeds the 1048576 limit"
        )
    if hardware["memory_bytes"] > 1_152_921_504_606_846_976:
        raise CustodianReplayError(
            "candidate.hardware.memory_bytes exceeds the schema limit"
        )

    command = candidate["command"]
    check_text(
        command["working_directory"],
        4096,
        "candidate.command.working_directory",
    )
    argv = command["argv"]
    if len(argv) > 1024:
        raise CustodianReplayError("candidate.command.argv exceeds the 1024-item limit")
    for index, argument in enumerate(argv):
        check_text(
            argument,
            MAX_HYPOTHESIS_CHARACTERS,
            f"candidate.command.argv[{index}]",
        )
    environment = command["environment"]
    if len(environment) > 256:
        raise CustodianReplayError(
            "candidate.command.environment exceeds the 256-property limit"
        )
    for name, value in environment.items():
        check_text(name, 256, "candidate.command.environment property name")
        check_text(value, 8192, f"candidate.command.environment.{name}")


def _validate_candidate_command_adapter(
    candidate: Mapping[str, Any],
    hypothesis_adapter_version: str,
) -> None:
    adapter = _string(hypothesis_adapter_version, "hypothesis_adapter_version")
    if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CustodianReplayError("hypothesis_adapter_version is unsupported")
    try:
        validate_sealed_candidate_execution(candidate, adapter, REPOSITORY_ROOT)
    except SealedCandidateContractError as exc:
        raise CustodianReplayError(str(exc)) from exc


def validate_candidate_request(
    descriptor: CollectionDescriptor,
    candidate_manifest: Mapping[str, Any],
    *,
    hypothesis_adapter_version: str,
) -> dict[str, Any]:
    """Validate candidate facts that must pass before references are opened."""

    candidate = candidate_freeze_projection(candidate_manifest)
    if candidate["task_id"] != "EVAL-01":
        raise CustodianReplayError("candidate task_id must be EVAL-01")
    if candidate["data_sha256"] != descriptor.raw_sha256:
        raise CustodianReplayError(
            "candidate data_sha256 does not match collection descriptor"
        )
    expected_data_version = f"{descriptor.dataset_id}-{descriptor.revision}"
    if candidate["eval_data_version"] != expected_data_version:
        raise CustodianReplayError(
            "candidate eval_data_version does not match collection descriptor"
        )
    if candidate["normalizer_version"] != descriptor.normalizer_version:
        raise CustodianReplayError(
            "candidate normalizer_version does not match collection descriptor"
        )

    _validate_candidate_command_adapter(candidate, hypothesis_adapter_version)
    return candidate


def _validate_sealed_input_projection(document: Any) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise CustodianReplayError("sealed input projection must be a JSON object")
    _strict_fields(document, SEALED_INPUT_FIELDS, "sealed input projection")
    if document["schema_version"] != 2 or isinstance(
        document["schema_version"], bool
    ):
        raise CustodianReplayError("sealed input projection schema_version must be 2")
    if document["kind"] != SEALED_INPUT_PROJECTION_KIND:
        raise CustodianReplayError("sealed input projection kind is unsupported")
    if document["split"] != SEALED_SPLIT:
        raise CustodianReplayError("sealed input projection split must be sealed-blind")
    for field in ("dataset_id", "revision"):
        _string(document[field], f"sealed input projection.{field}")
    _sha256(
        document["manifest_sha256"],
        "sealed input projection.manifest_sha256",
    )
    manifest_count = _nonnegative_integer(
        document["manifest_record_count"],
        "sealed input projection.manifest_record_count",
    )
    item_count = _nonnegative_integer(
        document["item_count"], "sealed input projection.item_count"
    )
    if item_count > manifest_count:
        raise CustodianReplayError(
            "sealed input projection item_count exceeds manifest_record_count"
        )
    items = document["items"]
    if not isinstance(items, list) or len(items) != item_count:
        raise CustodianReplayError(
            "sealed input projection items must match item_count"
        )
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        context = f"sealed input projection.items[{index}]"
        if not isinstance(item, Mapping):
            raise CustodianReplayError(f"{context} must be a JSON object")
        _strict_fields(item, SEALED_INPUT_ITEM_FIELDS, context)
        utterance_id = _string(item["id"], f"{context}.id")
        if utterance_id in seen_ids:
            raise CustodianReplayError(
                f"sealed input projection has duplicate item id at index {index}"
            )
        seen_ids.add(utterance_id)
        if item["split"] != SEALED_SPLIT:
            raise CustodianReplayError(f"{context}.split must be sealed-blind")
        audio = _string(item["audio"], f"{context}.audio")
        if Path(audio).is_absolute() or ".." in Path(audio).parts:
            raise CustodianReplayError(f"{context}.audio must be a safe relative path")
        _sha256(item["audio_sha256"], f"{context}.audio_sha256")
        duration = item["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise CustodianReplayError(
                f"{context}.duration_seconds must be a positive finite number"
            )
        for field in ("sample_rate", "channels", "sample_width_bits"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CustodianReplayError(f"{context}.{field} must be positive")
    return document


def parse_sealed_input_projection(payload: bytes) -> LoadedArtifact:
    """Strictly parse canonical reference-free projection bytes."""

    if len(payload) > MAX_ARTIFACT_BYTES:
        raise CustodianReplayError(
            f"sealed input projection exceeds the {MAX_ARTIFACT_BYTES}-byte limit"
        )
    document = _parse_json_document(
        payload,
        "sealed input projection",
        require_canonical=True,
    )
    _validate_sealed_input_projection(document)
    return LoadedArtifact(document, payload, sha256_bytes(payload))


def load_sealed_input_projection(path: Path) -> LoadedArtifact:
    payload = _regular_file_bytes(
        path,
        MAX_ARTIFACT_BYTES,
        "sealed input projection",
    )
    return parse_sealed_input_projection(payload)


def _decode_item_ids(document: Mapping[str, Any]) -> list[str]:
    return [str(item["id"]) for item in document["items"]]


def _decode_item_ids_sha256(document: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_decode_item_ids(document)))


def _sealed_records(collection: ValidatedCollection) -> tuple[dict[str, Any], ...]:
    records = tuple(
        record
        for record in collection.records
        if record.get("split") == SEALED_SPLIT
    )
    if not records:
        raise CustodianReplayError("collection contains no sealed-blind records")
    return records


def build_candidate_lock(
    descriptor: CollectionDescriptor,
    collection: ValidatedCollection,
    sealed_input: LoadedArtifact,
    candidate_manifest: RegisteredCandidateManifest,
    *,
    hypothesis_adapter_version: str,
) -> dict[str, Any]:
    """Bind one pre-result candidate to one validated sealed decode input."""

    projection = _validate_sealed_input_projection(sealed_input.document)
    if sealed_input.payload != build_sealed_input_projection(collection):
        raise CustodianReplayError(
            "sealed input projection bytes do not match the validated collection"
        )
    if descriptor.raw_sha256 != collection.summary.get("data_sha256"):
        raise CustodianReplayError(
            "descriptor identity does not match validated collection summary"
        )
    if projection["dataset_id"] != descriptor.dataset_id:
        raise CustodianReplayError("sealed input dataset_id does not match descriptor")
    if projection["revision"] != descriptor.revision:
        raise CustodianReplayError("sealed input revision does not match descriptor")
    if sealed_input.sha256 != descriptor.blind["input_projection_sha256"]:
        raise CustodianReplayError(
            "sealed input hash does not match descriptor blind projection"
        )

    candidate = validate_candidate_request(
        descriptor,
        candidate_manifest.document,
        hypothesis_adapter_version=hypothesis_adapter_version,
    )
    if candidate["normalizer_version"] != collection.summary.get(
        "normalizer_version"
    ):
        raise CustodianReplayError(
            "candidate normalizer_version does not match collection"
        )
    adapter = hypothesis_adapter_version

    sealed_records = _sealed_records(collection)
    try:
        sealed_record_digest = record_input_sha256(sealed_records)
    except RecordIdentityError as exc:
        raise CustodianReplayError(str(exc)) from exc
    expected_decode_ids = [
        str(record["id"])
        for record in sealed_records
        if record.get("evaluation_status") != "excluded"
    ]
    if not expected_decode_ids:
        raise CustodianReplayError(
            "sealed-blind replay requires at least one decode-eligible record"
        )
    if _decode_item_ids(projection) != expected_decode_ids:
        raise CustodianReplayError(
            "sealed input item order does not match decode-eligible collection records"
        )

    return {
        "schema_version": CANDIDATE_LOCK_SCHEMA_VERSION,
        "kind": CANDIDATE_LOCK_KIND,
        "state": "frozen",
        "access_class": "restricted",
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "split": SEALED_SPLIT,
        "data_sha256": descriptor.raw_sha256,
        "input_projection_sha256": sealed_input.sha256,
        "hypothesis_adapter_version": adapter,
        "record_identity_version": RECORD_IDENTITY_VERSION,
        "record_input_sha256": sealed_record_digest,
        "decode_item_count": len(expected_decode_ids),
        "decode_item_ids_sha256": _decode_item_ids_sha256(projection),
        "source_manifest_decision": "planned",
        "candidate_registration_commit": candidate_manifest.registration_commit,
        "candidate_manifest_path": candidate_manifest.repository_path,
        "candidate_manifest_sha256": candidate_manifest.sha256,
        "candidate": candidate,
        "candidate_freeze_sha256": candidate_freeze_sha256(candidate),
    }


def validate_candidate_lock(document: Any) -> None:
    if not isinstance(document, Mapping):
        raise CustodianReplayError("candidate lock must be a JSON object")
    _strict_fields(document, CANDIDATE_LOCK_FIELDS, "candidate lock")
    if document["schema_version"] != CANDIDATE_LOCK_SCHEMA_VERSION or isinstance(
        document["schema_version"], bool
    ):
        raise CustodianReplayError("candidate lock schema_version is unsupported")
    if document["kind"] != CANDIDATE_LOCK_KIND:
        raise CustodianReplayError("candidate lock kind is unsupported")
    if document["state"] != "frozen":
        raise CustodianReplayError("candidate lock state must be frozen")
    if document["access_class"] != "restricted":
        raise CustodianReplayError("candidate lock access_class must be restricted")
    if document["split"] != SEALED_SPLIT:
        raise CustodianReplayError("candidate lock split must be sealed-blind")
    for field in ("dataset_id", "revision"):
        _string(
            document[field],
            f"candidate lock.{field}",
            maximum_characters=MAX_SHORT_STRING_CHARACTERS,
        )
    adapter = _string(
        document["hypothesis_adapter_version"],
        "candidate lock.hypothesis_adapter_version",
    )
    if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CustodianReplayError(
            "candidate lock hypothesis_adapter_version is unsupported"
        )
    for field in (
        "data_sha256",
        "input_projection_sha256",
        "record_input_sha256",
        "decode_item_ids_sha256",
        "candidate_manifest_sha256",
        "candidate_freeze_sha256",
    ):
        _sha256(document[field], f"candidate lock.{field}")
    if document["record_identity_version"] != RECORD_IDENTITY_VERSION:
        raise CustodianReplayError(
            "candidate lock record_identity_version is unsupported"
        )
    decode_item_count = _bounded_nonnegative_integer(
        document["decode_item_count"],
        "candidate lock.decode_item_count",
        maximum=MAX_PREDICTION_ITEMS,
    )
    if decode_item_count == 0:
        raise CustodianReplayError(
            "candidate lock must contain at least one decode-eligible item"
        )
    if document["source_manifest_decision"] != "planned":
        raise CustodianReplayError(
            "candidate lock source_manifest_decision must be planned"
        )
    candidate = _validate_candidate_projection(document["candidate"])
    if candidate["task_id"] != "EVAL-01":
        raise CustodianReplayError("candidate task_id must be EVAL-01")
    _validate_candidate_registration(
        document,
        "candidate lock",
        candidate["experiment_id"],
    )
    if candidate["data_sha256"] != document["data_sha256"]:
        raise CustodianReplayError(
            "candidate data_sha256 does not match candidate lock"
        )
    if candidate["eval_data_version"] != (
        f"{document['dataset_id']}-{document['revision']}"
    ):
        raise CustodianReplayError(
            "candidate eval_data_version does not match candidate lock"
        )
    argv = candidate["command"]["argv"]
    adapter_positions = [
        index
        for index, argument in enumerate(argv)
        if argument == "--hypothesis-adapter-version"
    ]
    if len(adapter_positions) != 1:
        raise CustodianReplayError(
            "candidate command must bind exactly one hypothesis adapter"
        )
    adapter_index = adapter_positions[0]
    if adapter_index + 1 >= len(argv) or argv[adapter_index + 1] != adapter:
        raise CustodianReplayError(
            "candidate command hypothesis adapter does not match candidate lock"
        )
    if candidate_freeze_sha256(candidate) != document["candidate_freeze_sha256"]:
        raise CustodianReplayError(
            "candidate_freeze_sha256 does not match candidate projection"
        )


def canonical_candidate_lock_bytes(document: Mapping[str, Any]) -> bytes:
    validate_candidate_lock(document)
    return canonical_json_bytes(document)


def load_candidate_lock(path: Path) -> LoadedArtifact:
    payload = _regular_file_bytes(path, MAX_ARTIFACT_BYTES, "candidate lock")
    document = _parse_json_document(
        payload,
        "candidate lock",
        require_canonical=True,
    )
    validate_candidate_lock(document)
    return LoadedArtifact(document, payload, sha256_bytes(payload))


def validate_decode_handoff(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
) -> None:
    """Bind a reference-free input to its custodian-owned candidate lock."""

    projection = _validate_sealed_input_projection(sealed_input.document)
    lock = candidate_lock.document
    validate_candidate_lock(lock)
    for field in ("dataset_id", "revision", "split"):
        if lock[field] != projection[field]:
            raise CustodianReplayError(
                f"candidate lock {field} does not match sealed input"
            )
    if lock["input_projection_sha256"] != sealed_input.sha256:
        raise CustodianReplayError(
            "candidate lock input projection does not match sealed input bytes"
        )
    projection_ids = _decode_item_ids(projection)
    if lock["decode_item_count"] != len(projection_ids):
        raise CustodianReplayError(
            "candidate lock decode_item_count does not match sealed input"
        )
    if lock["decode_item_ids_sha256"] != _decode_item_ids_sha256(projection):
        raise CustodianReplayError(
            "candidate lock decode item identity does not match sealed input"
        )


def _validate_prediction_items(
    items: Any,
    *,
    hypothesis_adapter_version: str | None = None,
) -> list[Mapping[str, Any]]:
    if not isinstance(items, list):
        raise CustodianReplayError("prediction bundle items must be an array")
    if len(items) > MAX_PREDICTION_ITEMS:
        raise CustodianReplayError("prediction bundle contains too many items")
    normalized: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    total_hypothesis_characters = 0
    for index, item in enumerate(items):
        context = f"prediction bundle.items[{index}]"
        if not isinstance(item, Mapping):
            raise CustodianReplayError(f"{context} must be a JSON object")
        _strict_fields(item, PREDICTION_ITEM_FIELDS, context)
        utterance_id = _string(
            item["id"],
            f"{context}.id",
            maximum_characters=MAX_PREDICTION_ID_CHARACTERS,
        )
        if utterance_id in seen_ids:
            raise CustodianReplayError(
                f"prediction bundle has duplicate item id at index {index}"
            )
        seen_ids.add(utterance_id)
        raw_text = _string(
            item["raw_text"],
            f"{context}.raw_text",
            allow_empty=True,
        )
        if len(raw_text) > MAX_HYPOTHESIS_CHARACTERS:
            raise CustodianReplayError(
                f"{context}.raw_text exceeds the "
                f"{MAX_HYPOTHESIS_CHARACTERS}-character limit"
            )
        total_hypothesis_characters += len(raw_text)
        if total_hypothesis_characters > MAX_TOTAL_HYPOTHESIS_CHARACTERS:
            raise CustodianReplayError(
                "prediction bundle raw_text exceeds the "
                f"{MAX_TOTAL_HYPOTHESIS_CHARACTERS}-character total limit"
            )
        status_value = item["status"]
        if not isinstance(status_value, str) or status_value not in {
            "ok",
            "empty",
            "failed",
        }:
            raise CustodianReplayError(
                f"{context}.status must be ok, empty, or failed"
            )
        reason_code = item["reason_code"]
        if status_value == "ok":
            if not raw_text:
                raise CustodianReplayError(f"{context}: ok text must be non-empty")
            if reason_code is not None:
                raise CustodianReplayError(
                    f"{context}: ok reason_code must be null"
                )
        else:
            if status_value == "failed" and raw_text:
                raise CustodianReplayError(
                    f"{context}: failed raw_text must be empty"
                )
            if (
                not isinstance(reason_code, str)
                or REASON_CODE_PATTERN.fullmatch(reason_code) is None
            ):
                raise CustodianReplayError(
                    f"{context}.reason_code must be a stable snake_case code"
                )
            if len(reason_code) > MAX_REASON_CODE_CHARACTERS:
                raise CustodianReplayError(
                    f"{context}.reason_code exceeds the "
                    f"{MAX_REASON_CODE_CHARACTERS}-character limit"
                )
        if hypothesis_adapter_version is not None:
            try:
                display_text = adapt_hypothesis(
                    raw_text,
                    hypothesis_adapter_version,
                )
            except CoreReportValidationError as exc:
                raise CustodianReplayError(str(exc)) from exc
            if status_value == "ok" and not display_text:
                raise CustodianReplayError(
                    f"{context}: ok prediction must not adapt to empty text"
                )
            if status_value == "empty" and display_text:
                raise CustodianReplayError(
                    f"{context}: empty prediction must adapt to empty text"
                )
        normalized.append(item)
    return normalized


def load_prediction_items_jsonl_artifact(path: Path) -> LoadedPredictionItems:
    """Load strict, reference-free decoder items before canonical freezing.

    Every non-empty line is one JSON object with the exact prediction-item
    fields. An empty file deliberately represents zero returned predictions so
    the scorer can account for every expected ID as ``missing_prediction``.
    """

    payload = _regular_file_bytes(
        path,
        MAX_ARTIFACT_BYTES,
        "raw prediction JSONL",
    )
    if payload.startswith(b"\xef\xbb\xbf"):
        raise CustodianReplayError(
            "raw prediction JSONL must not contain a UTF-8 BOM"
        )
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if line in {b"\n", b"\r\n"}:
            raise CustodianReplayError(
                f"raw prediction JSONL line {line_number} must not be blank"
            )
        if len(line) > MAX_RAW_PREDICTION_LINE_BYTES:
            raise CustodianReplayError(
                f"raw prediction JSONL line {line_number} exceeds the "
                f"{MAX_RAW_PREDICTION_LINE_BYTES}-byte limit"
            )
        item = _parse_json_document(
            line,
            f"raw prediction JSONL line {line_number}",
            require_canonical=False,
        )
        items.append(item)
        if len(items) > MAX_PREDICTION_ITEMS:
            raise CustodianReplayError("raw prediction JSONL contains too many items")
    _validate_prediction_items(items)
    return LoadedPredictionItems(
        items=tuple(items),
        payload=payload,
        sha256=sha256_bytes(payload),
    )


def load_prediction_items_jsonl(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only the parsed prediction items."""

    return list(load_prediction_items_jsonl_artifact(path).items)


def _validate_prediction_id_subsequence(
    prediction_ids: Sequence[str],
    decode_ids: Sequence[str],
) -> None:
    """Require prediction IDs to preserve decode order without extras."""

    decode_positions = {
        utterance_id: index for index, utterance_id in enumerate(decode_ids)
    }
    unexpected = [
        utterance_id
        for utterance_id in prediction_ids
        if utterance_id not in decode_positions
    ]
    if unexpected:
        raise CustodianReplayError(
            "prediction items contain ID(s) absent from sealed decode input"
        )
    positions = [decode_positions[utterance_id] for utterance_id in prediction_ids]
    if positions != sorted(positions):
        raise CustodianReplayError(
            "prediction items must preserve sealed decode ID order"
        )


def build_prediction_bundle(
    sealed_input: LoadedArtifact,
    candidate_lock_sha256: str,
    predictions: Sequence[Mapping[str, Any]],
    *,
    input_export_receipt_sha256: str,
    raw_predictions_sha256: str,
    execution_envelope_sha256: str,
    hypothesis_adapter_version: str,
) -> dict[str, Any]:
    """Build the exact reference-free prediction artifact expected by scoring."""

    projection = _validate_sealed_input_projection(sealed_input.document)
    lock_digest = _sha256(candidate_lock_sha256, "candidate_lock_sha256")
    input_export_receipt_digest = _sha256(
        input_export_receipt_sha256,
        "input_export_receipt_sha256",
    )
    raw_predictions_digest = _sha256(
        raw_predictions_sha256,
        "raw_predictions_sha256",
    )
    execution_envelope_digest = _sha256(
        execution_envelope_sha256,
        "execution_envelope_sha256",
    )
    adapter = _string(
        hypothesis_adapter_version,
        "hypothesis_adapter_version",
    )
    if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CustodianReplayError("hypothesis_adapter_version is unsupported")
    if isinstance(predictions, (str, bytes, bytearray)) or not isinstance(
        predictions, Sequence
    ):
        raise CustodianReplayError("predictions must be an ordered sequence")
    copied_items = [deepcopy(dict(item)) for item in predictions]
    _validate_prediction_items(
        copied_items,
        hypothesis_adapter_version=adapter,
    )
    _validate_prediction_id_subsequence(
        [str(item["id"]) for item in copied_items],
        _decode_item_ids(projection),
    )
    items_digest = sha256_bytes(canonical_json_bytes(copied_items))
    bundle = {
        "schema_version": PREDICTION_BUNDLE_SCHEMA_VERSION,
        "kind": PREDICTION_BUNDLE_KIND,
        "state": "frozen",
        "access_class": "restricted",
        "dataset_id": projection["dataset_id"],
        "revision": projection["revision"],
        "split": SEALED_SPLIT,
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": lock_digest,
        "input_export_receipt_sha256": input_export_receipt_digest,
        "raw_predictions_sha256": raw_predictions_digest,
        "execution_envelope_sha256": execution_envelope_digest,
        "hypothesis_adapter_version": adapter,
        "item_count": len(copied_items),
        "items_sha256": items_digest,
        "items": copied_items,
    }
    validate_prediction_bundle(bundle)
    return bundle


def validate_prediction_bundle(document: Any) -> None:
    if not isinstance(document, Mapping):
        raise CustodianReplayError("prediction bundle must be a JSON object")
    _strict_fields(document, PREDICTION_BUNDLE_FIELDS, "prediction bundle")
    if document["schema_version"] != PREDICTION_BUNDLE_SCHEMA_VERSION or isinstance(
        document["schema_version"], bool
    ):
        raise CustodianReplayError("prediction bundle schema_version is unsupported")
    if document["kind"] != PREDICTION_BUNDLE_KIND:
        raise CustodianReplayError("prediction bundle kind is unsupported")
    if document["state"] != "frozen":
        raise CustodianReplayError("prediction bundle state must be frozen")
    if document["access_class"] != "restricted":
        raise CustodianReplayError(
            "prediction bundle access_class must be restricted"
        )
    if document["split"] != SEALED_SPLIT:
        raise CustodianReplayError("prediction bundle split must be sealed-blind")
    for field in ("dataset_id", "revision"):
        _string(
            document[field],
            f"prediction bundle.{field}",
            maximum_characters=MAX_SHORT_STRING_CHARACTERS,
        )
    for field in (
        "input_projection_sha256",
        "candidate_lock_sha256",
        "input_export_receipt_sha256",
        "raw_predictions_sha256",
        "execution_envelope_sha256",
        "items_sha256",
    ):
        _sha256(document[field], f"prediction bundle.{field}")
    adapter = _string(
        document["hypothesis_adapter_version"],
        "prediction bundle.hypothesis_adapter_version",
    )
    if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CustodianReplayError(
            "prediction bundle hypothesis_adapter_version is unsupported"
        )
    items = _validate_prediction_items(
        document["items"],
        hypothesis_adapter_version=adapter,
    )
    item_count = _bounded_nonnegative_integer(
        document["item_count"],
        "prediction bundle.item_count",
        maximum=MAX_PREDICTION_ITEMS,
    )
    if item_count != len(items):
        raise CustodianReplayError(
            "prediction bundle item_count does not match items"
        )
    actual_items_digest = sha256_bytes(canonical_json_bytes(items))
    if actual_items_digest != document["items_sha256"]:
        raise CustodianReplayError(
            "prediction bundle items_sha256 does not match items"
        )


def canonical_prediction_bundle_bytes(document: Mapping[str, Any]) -> bytes:
    validate_prediction_bundle(document)
    return canonical_json_bytes(document)


def load_prediction_bundle(path: Path) -> LoadedArtifact:
    payload = _regular_file_bytes(path, MAX_ARTIFACT_BYTES, "prediction bundle")
    document = _parse_json_document(
        payload,
        "prediction bundle",
        require_canonical=True,
    )
    validate_prediction_bundle(document)
    return LoadedArtifact(document, payload, sha256_bytes(payload))


def validate_prediction_handoff(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    predictions: LoadedArtifact,
) -> None:
    """Validate the complete reference-free input/lock/prediction chain."""

    validate_decode_handoff(sealed_input, candidate_lock)
    projection = sealed_input.document
    lock = candidate_lock.document
    bundle = predictions.document
    validate_prediction_bundle(bundle)
    for field in ("dataset_id", "revision", "split"):
        if bundle[field] != projection[field]:
            raise CustodianReplayError(
                f"prediction bundle {field} does not match sealed input"
            )
    if bundle["input_projection_sha256"] != sealed_input.sha256:
        raise CustodianReplayError(
            "prediction bundle input projection does not match sealed input bytes"
        )
    if bundle["candidate_lock_sha256"] != candidate_lock.sha256:
        raise CustodianReplayError(
            "prediction bundle does not bind the supplied candidate lock"
        )
    if bundle["hypothesis_adapter_version"] != lock[
        "hypothesis_adapter_version"
    ]:
        raise CustodianReplayError(
            "prediction bundle hypothesis adapter does not match candidate lock"
        )
    _validate_prediction_id_subsequence(
        [str(item["id"]) for item in bundle["items"]],
        _decode_item_ids(projection),
    )


def _validate_execution_candidate_facts(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    execution_envelope: LoadedExecutionEnvelope,
    *,
    input_export_receipt_sha256: str,
) -> None:
    """Bind runner-owned execution facts to the custodian-owned handoff."""

    validate_decode_handoff(sealed_input, candidate_lock)
    try:
        validate_execution_envelope(execution_envelope.document)
    except ExecutionEnvelopeError as exc:
        raise CustodianReplayError(str(exc)) from exc
    envelope = execution_envelope.document
    lock = candidate_lock.document
    candidate = lock["candidate"]
    try:
        execution_plan = validate_sealed_candidate_execution(
            candidate,
            lock["hypothesis_adapter_version"],
            REPOSITORY_ROOT,
        )
    except SealedCandidateContractError as exc:
        raise CustodianReplayError(str(exc)) from exc
    if (
        envelope["measurement"]["warmup_run_count"]
        != execution_plan.baseline_config.warmup_runs
    ):
        raise CustodianReplayError(
            "execution envelope warmup count does not match candidate command"
        )
    projection = sealed_input.document
    root_facts = {
        "experiment_id": candidate["experiment_id"],
        "dataset_id": projection["dataset_id"],
        "revision": projection["revision"],
        "split": SEALED_SPLIT,
    }
    for field, expected in root_facts.items():
        if envelope[field] != expected:
            raise CustodianReplayError(
                f"execution envelope {field} does not match candidate handoff"
            )
    binding_facts = {
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": lock["candidate_freeze_sha256"],
        "decode_item_ids_sha256": lock["decode_item_ids_sha256"],
        "input_export_receipt_sha256": input_export_receipt_sha256,
        "hypothesis_adapter_version": lock["hypothesis_adapter_version"],
    }
    bindings = envelope["bindings"]
    for field, expected in binding_facts.items():
        if bindings[field] != expected:
            raise CustodianReplayError(
                f"execution envelope {field} does not match candidate handoff"
            )
    expected_items = projection["items"]
    execution_items = envelope["items"]
    if len(execution_items) != len(expected_items):
        raise CustodianReplayError(
            "execution envelope item count does not match sealed input"
        )
    for index, (execution_item, input_item) in enumerate(
        zip(execution_items, expected_items, strict=True)
    ):
        if execution_item["id"] != input_item["id"]:
            raise CustodianReplayError(
                f"execution envelope item {index} does not match sealed input order"
            )
        if not math.isclose(
            execution_item["audio_duration_seconds"],
            input_item["duration_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise CustodianReplayError(
                f"execution envelope item {index} duration does not match sealed input"
            )
    runner = envelope["runner"]
    expected_runner_facts = {
        "code_commit": candidate["code_commit"],
        "effective_config_sha256": candidate["config_sha256"],
        "command": candidate["command"],
        "models": candidate["models"],
        "hardware": candidate["hardware"],
    }
    for field, expected in expected_runner_facts.items():
        if runner[field] != expected:
            raise CustodianReplayError(
                f"execution runner {field} does not match candidate lock"
            )


def validate_raw_execution_handoff(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    input_export_receipt: LoadedArtifact,
    raw_predictions: LoadedPredictionItems,
    execution_envelope: LoadedExecutionEnvelope,
) -> None:
    """Validate raw prediction bytes and timing evidence before freezing."""

    validate_input_export_receipt_handoff(
        sealed_input,
        candidate_lock,
        input_export_receipt,
    )
    _validate_execution_candidate_facts(
        sealed_input,
        candidate_lock,
        execution_envelope,
        input_export_receipt_sha256=input_export_receipt.sha256,
    )
    try:
        validate_execution_envelope_for_predictions(
            execution_envelope.document,
            raw_predictions.items,
            raw_predictions_sha256=raw_predictions.sha256,
        )
    except ExecutionEnvelopeError as exc:
        raise CustodianReplayError(str(exc)) from exc


def validate_frozen_execution_handoff(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    input_export_receipt: LoadedArtifact,
    predictions: LoadedArtifact,
    execution_envelope: LoadedExecutionEnvelope,
) -> None:
    """Validate the frozen prediction/envelope chain before sealed scoring."""

    validate_input_export_receipt_handoff(
        sealed_input,
        candidate_lock,
        input_export_receipt,
    )
    validate_prediction_handoff(sealed_input, candidate_lock, predictions)
    _validate_execution_candidate_facts(
        sealed_input,
        candidate_lock,
        execution_envelope,
        input_export_receipt_sha256=input_export_receipt.sha256,
    )
    bundle = predictions.document
    if bundle["input_export_receipt_sha256"] != input_export_receipt.sha256:
        raise CustodianReplayError(
            "prediction bundle does not bind the supplied input export receipt"
        )
    if bundle["execution_envelope_sha256"] != execution_envelope.sha256:
        raise CustodianReplayError(
            "prediction bundle does not bind the supplied execution envelope"
        )
    bindings = execution_envelope.document["bindings"]
    if bundle["raw_predictions_sha256"] != bindings["raw_predictions_sha256"]:
        raise CustodianReplayError(
            "prediction bundle raw prediction hash does not match execution envelope"
        )
    if bundle["items_sha256"] != bindings["prediction_items_sha256"]:
        raise CustodianReplayError(
            "prediction bundle items do not match execution envelope"
        )
    try:
        validate_execution_envelope_for_predictions(
            execution_envelope.document,
            bundle["items"],
            raw_predictions_sha256=bundle["raw_predictions_sha256"],
        )
    except ExecutionEnvelopeError as exc:
        raise CustodianReplayError(str(exc)) from exc


def preflight_replay_artifacts(
    descriptor: CollectionDescriptor,
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    predictions: LoadedArtifact,
) -> None:
    """Check all reference-free locks before a caller opens sealed references."""

    validate_prediction_handoff(sealed_input, candidate_lock, predictions)
    projection = sealed_input.document
    lock = candidate_lock.document
    bundle = predictions.document

    common_expected = {
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "split": SEALED_SPLIT,
    }
    for field, expected_value in common_expected.items():
        if projection.get(field) != expected_value:
            raise CustodianReplayError(
                f"sealed input {field} does not match collection descriptor"
            )
        if lock.get(field) != expected_value:
            raise CustodianReplayError(
                f"candidate lock {field} does not match collection descriptor"
            )
        if bundle.get(field) != expected_value:
            raise CustodianReplayError(
                f"prediction bundle {field} does not match collection descriptor"
            )
    expected_input_digest = descriptor.blind["input_projection_sha256"]
    if sealed_input.sha256 != expected_input_digest:
        raise CustodianReplayError(
            "sealed input bytes do not match descriptor input projection hash"
        )
    if lock["data_sha256"] != descriptor.raw_sha256:
        raise CustodianReplayError(
            "candidate lock data_sha256 does not match descriptor"
        )
    if lock["candidate"]["normalizer_version"] != descriptor.normalizer_version:
        raise CustodianReplayError(
            "candidate lock normalizer_version does not match descriptor"
        )


def validate_replay_collection(
    collection: ValidatedCollection,
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
) -> None:
    """Rebuild collection-owned identities after the reference-free preflight."""

    rebuilt = build_sealed_input_projection(collection)
    if rebuilt != sealed_input.payload:
        raise CustodianReplayError(
            "sealed input bytes do not match the validated collection"
        )
    if collection.summary.get("data_sha256") != candidate_lock.document[
        "data_sha256"
    ]:
        raise CustodianReplayError(
            "validated collection data identity does not match candidate lock"
        )
    try:
        scoped_digest = record_input_sha256(_sealed_records(collection))
    except RecordIdentityError as exc:
        raise CustodianReplayError(str(exc)) from exc
    if scoped_digest != candidate_lock.document["record_input_sha256"]:
        raise CustodianReplayError(
            "sealed scoring record identity does not match candidate lock"
        )


def _validate_receipt_common(document: Mapping[str, Any], context: str) -> None:
    if document["schema_version"] != RECEIPT_SCHEMA_VERSION or isinstance(
        document["schema_version"], bool
    ):
        raise CustodianReplayError(f"{context} schema_version is unsupported")
    if document["state"] != "complete":
        raise CustodianReplayError(f"{context} state must be complete")
    if document["access_class"] != "restricted":
        raise CustodianReplayError(f"{context} access_class must be restricted")
    experiment_id = _string(
        document["experiment_id"],
        f"{context}.experiment_id",
        maximum_characters=MAX_EXPERIMENT_ID_CHARACTERS,
    )
    if EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
        raise CustodianReplayError(
            f"{context}.experiment_id has an invalid format"
        )
    for field in ("dataset_id", "revision"):
        _string(
            document[field],
            f"{context}.{field}",
            maximum_characters=MAX_SHORT_STRING_CHARACTERS,
        )


def _validate_candidate_registration(
    document: Mapping[str, Any],
    context: str,
    experiment_id: str,
) -> None:
    commit = _string(
        document["candidate_registration_commit"],
        f"{context}.candidate_registration_commit",
    )
    if GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise CustodianReplayError(
            f"{context}.candidate_registration_commit must be a full Git commit"
        )
    manifest_path = _string(
        document["candidate_manifest_path"],
        f"{context}.candidate_manifest_path",
        maximum_characters=4096,
    )
    expected_path = f"experiments/manifests/{experiment_id}.json"
    if manifest_path != expected_path:
        raise CustodianReplayError(
            f"{context}.candidate_manifest_path does not match experiment_id"
        )
    _sha256(
        document["candidate_manifest_sha256"],
        f"{context}.candidate_manifest_sha256",
    )


def _validate_scorer_runtime(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise CustodianReplayError("score receipt scorer_runtime must be an object")
    _strict_fields(value, SCORER_RUNTIME_FIELDS, "score receipt scorer_runtime")
    if value["python_implementation"] != "cpython":
        raise CustodianReplayError("score receipt scorer_runtime requires CPython")
    python_version = _string(
        value["python_version"],
        "score receipt scorer_runtime.python_version",
        maximum_characters=64,
    )
    if PYTHON_VERSION_PATTERN.fullmatch(python_version) is None:
        raise CustodianReplayError("score receipt scorer Python version is invalid")
    cache_tag = _string(
        value["python_cache_tag"],
        "score receipt scorer_runtime.python_cache_tag",
        maximum_characters=64,
    )
    if PYTHON_CACHE_TAG_PATTERN.fullmatch(cache_tag) is None:
        raise CustodianReplayError("score receipt scorer cache tag is invalid")
    for field in ("dependency_lock_sha256", "installed_dependencies_sha256"):
        _sha256(value[field], f"score receipt scorer_runtime.{field}")
    count = value["installed_dependency_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= 10_000
    ):
        raise CustodianReplayError(
            "score receipt scorer installed dependency count is invalid"
        )
    unicode_version = _string(
        value["unicode_version"],
        "score receipt scorer_runtime.unicode_version",
        maximum_characters=32,
    )
    if UNICODE_VERSION_PATTERN.fullmatch(unicode_version) is None:
        raise CustodianReplayError("score receipt scorer Unicode version is invalid")


def validate_custodian_receipt(document: Any) -> None:
    """Validate one restricted, deterministic custodian completion receipt."""

    if not isinstance(document, Mapping):
        raise CustodianReplayError("custodian receipt must be a JSON object")
    kind = document.get("kind")
    if kind == INPUT_EXPORT_RECEIPT_KIND:
        _strict_fields(document, INPUT_EXPORT_RECEIPT_FIELDS, "input export receipt")
        context = "input export receipt"
        _validate_receipt_common(document, context)
        _validate_candidate_registration(document, context, document["experiment_id"])
        if document["split"] != SEALED_SPLIT:
            raise CustodianReplayError(f"{context} split must be sealed-blind")
        if _bounded_nonnegative_integer(
            document["decode_item_count"],
            f"{context}.decode_item_count",
            maximum=MAX_PREDICTION_ITEMS,
        ) == 0:
            raise CustodianReplayError(
                f"{context}.decode_item_count must be greater than zero"
            )
        for field in (
            "input_projection_sha256",
            "candidate_lock_sha256",
            "candidate_freeze_sha256",
        ):
            _sha256(document[field], f"{context}.{field}")
        return

    if kind == PREDICTION_FREEZE_RECEIPT_KIND:
        _strict_fields(
            document,
            PREDICTION_FREEZE_RECEIPT_FIELDS,
            "prediction freeze receipt",
        )
        context = "prediction freeze receipt"
        _validate_receipt_common(document, context)
        _validate_candidate_registration(document, context, document["experiment_id"])
        if document["split"] != SEALED_SPLIT:
            raise CustodianReplayError(f"{context} split must be sealed-blind")
        expected_count = _bounded_nonnegative_integer(
            document["expected_decode_item_count"],
            f"{context}.expected_decode_item_count",
            maximum=MAX_PREDICTION_ITEMS,
        )
        prediction_count = _bounded_nonnegative_integer(
            document["prediction_item_count"],
            f"{context}.prediction_item_count",
            maximum=MAX_PREDICTION_ITEMS,
        )
        missing_count = _bounded_nonnegative_integer(
            document["missing_prediction_count"],
            f"{context}.missing_prediction_count",
            maximum=MAX_PREDICTION_ITEMS,
        )
        if expected_count == 0:
            raise CustodianReplayError(
                f"{context}.expected_decode_item_count must be greater than zero"
            )
        if prediction_count + missing_count != expected_count:
            raise CustodianReplayError(
                f"{context} prediction and missing counts must equal expected count"
            )
        adapter = _string(
            document["hypothesis_adapter_version"],
            f"{context}.hypothesis_adapter_version",
        )
        if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
            raise CustodianReplayError(
                f"{context}.hypothesis_adapter_version is unsupported"
            )
        for field in (
            "input_projection_sha256",
            "candidate_lock_sha256",
            "candidate_freeze_sha256",
            "prediction_artifact_sha256",
            "prediction_items_sha256",
            "input_export_receipt_sha256",
            "raw_predictions_sha256",
            "execution_envelope_sha256",
            "runner_source_sha256",
        ):
            _sha256(document[field], f"{context}.{field}")
        runner_code_commit = _string(
            document["runner_code_commit"],
            f"{context}.runner_code_commit",
        )
        if not GIT_COMMIT_PATTERN.fullmatch(runner_code_commit):
            raise CustodianReplayError(
                f"{context}.runner_code_commit must be a full Git commit"
            )
        return

    if kind == CUSTODIAN_SCORE_RECEIPT_KIND:
        _strict_fields(
            document,
            CUSTODIAN_SCORE_RECEIPT_FIELDS,
            "custodian score receipt",
        )
        context = "custodian score receipt"
        _validate_receipt_common(document, context)
        _validate_candidate_registration(document, context, document["experiment_id"])
        scope = document["evaluation_scope"]
        if not isinstance(scope, Mapping):
            raise CustodianReplayError(f"{context}.evaluation_scope must be an object")
        _strict_fields(
            scope,
            frozenset({"kind", "split"}),
            f"{context}.evaluation_scope",
        )
        if dict(scope) != {"kind": "split", "split": SEALED_SPLIT}:
            raise CustodianReplayError(
                f"{context}.evaluation_scope must select sealed-blind"
            )
        for field in (
            "data_sha256",
            "input_projection_sha256",
            "record_input_sha256",
            "prediction_input_sha256",
            "candidate_lock_sha256",
            "candidate_freeze_sha256",
            "prediction_artifact_sha256",
            "prediction_items_sha256",
            "input_export_receipt_sha256",
            "prediction_freeze_receipt_sha256",
            "execution_envelope_sha256",
            "runner_source_sha256",
            "core_sha256",
        ):
            _sha256(document[field], f"{context}.{field}")
        runner_code_commit = _string(
            document["runner_code_commit"],
            f"{context}.runner_code_commit",
        )
        if not GIT_COMMIT_PATTERN.fullmatch(runner_code_commit):
            raise CustodianReplayError(
                f"{context}.runner_code_commit must be a full Git commit"
            )
        scorer_code_commit = _string(
            document["scorer_code_commit"],
            f"{context}.scorer_code_commit",
        )
        if not GIT_COMMIT_PATTERN.fullmatch(scorer_code_commit):
            raise CustodianReplayError(
                f"{context}.scorer_code_commit must be a full Git commit"
            )
        _sha256(
            document["scorer_source_sha256"],
            f"{context}.scorer_source_sha256",
        )
        _validate_scorer_runtime(document["scorer_runtime"])
        if document["record_identity_version"] != RECORD_IDENTITY_VERSION:
            raise CustodianReplayError(
                f"{context}.record_identity_version is unsupported"
            )
        adapter = _string(
            document["hypothesis_adapter_version"],
            f"{context}.hypothesis_adapter_version",
        )
        if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
            raise CustodianReplayError(
                f"{context}.hypothesis_adapter_version is unsupported"
            )
        if document["core_schema_version"] != CORE_SCHEMA_VERSION or isinstance(
            document["core_schema_version"], bool
        ):
            raise CustodianReplayError(
                f"{context}.core_schema_version is unsupported"
            )
        public_release = document["public_release"]
        if not isinstance(public_release, Mapping):
            raise CustodianReplayError(f"{context}.public_release must be an object")
        _strict_fields(
            public_release,
            PUBLIC_RELEASE_FIELDS,
            f"{context}.public_release",
        )
        if dict(public_release) != {
            "state": "withheld",
            "summary_sha256": None,
            "reason_code": "release_policy_not_implemented",
        }:
            raise CustodianReplayError(
                f"{context}.public_release must remain withheld"
            )
        return

    raise CustodianReplayError("custodian receipt kind is unsupported")


def canonical_custodian_receipt_bytes(document: Mapping[str, Any]) -> bytes:
    """Return canonical bytes for a validated restricted receipt."""

    validate_custodian_receipt(document)
    return canonical_json_bytes(document)


def load_custodian_receipt(path: Path) -> LoadedArtifact:
    payload = _regular_file_bytes(path, MAX_ARTIFACT_BYTES, "custodian receipt")
    document = _parse_json_document(
        payload,
        "custodian receipt",
        require_canonical=True,
    )
    validate_custodian_receipt(document)
    return LoadedArtifact(document, payload, sha256_bytes(payload))


def validate_input_export_receipt_handoff(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    input_export_receipt: LoadedArtifact,
) -> None:
    """Require the durable export completion marker for one input/lock pair."""

    validate_decode_handoff(sealed_input, candidate_lock)
    receipt = input_export_receipt.document
    validate_custodian_receipt(receipt)
    if receipt["kind"] != INPUT_EXPORT_RECEIPT_KIND:
        raise CustodianReplayError(
            "sealed replay requires an input export receipt"
        )
    lock = candidate_lock.document
    expected = {
        "experiment_id": lock["candidate"]["experiment_id"],
        "dataset_id": sealed_input.document["dataset_id"],
        "revision": sealed_input.document["revision"],
        "split": SEALED_SPLIT,
        "decode_item_count": lock["decode_item_count"],
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": lock["candidate_freeze_sha256"],
        "candidate_registration_commit": lock["candidate_registration_commit"],
        "candidate_manifest_path": lock["candidate_manifest_path"],
        "candidate_manifest_sha256": lock["candidate_manifest_sha256"],
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise CustodianReplayError(
                f"input export receipt {field} does not match replay artifacts"
            )


def validate_prediction_freeze_receipt_handoff(
    sealed_input: LoadedArtifact,
    candidate_lock: LoadedArtifact,
    input_export_receipt: LoadedArtifact,
    predictions: LoadedArtifact,
    execution_envelope: LoadedExecutionEnvelope,
    prediction_receipt: LoadedArtifact,
) -> None:
    """Validate the complete reference-free completion chain before scoring."""

    validate_frozen_execution_handoff(
        sealed_input,
        candidate_lock,
        input_export_receipt,
        predictions,
        execution_envelope,
    )
    receipt = prediction_receipt.document
    validate_custodian_receipt(receipt)
    if receipt["kind"] != PREDICTION_FREEZE_RECEIPT_KIND:
        raise CustodianReplayError(
            "sealed scoring requires a prediction freeze receipt"
        )
    lock = candidate_lock.document
    bundle = predictions.document
    envelope = execution_envelope.document
    expected_count = lock["decode_item_count"]
    expected = {
        "experiment_id": lock["candidate"]["experiment_id"],
        "dataset_id": sealed_input.document["dataset_id"],
        "revision": sealed_input.document["revision"],
        "split": SEALED_SPLIT,
        "expected_decode_item_count": expected_count,
        "prediction_item_count": bundle["item_count"],
        "missing_prediction_count": expected_count - bundle["item_count"],
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": lock["candidate_freeze_sha256"],
        "candidate_registration_commit": lock["candidate_registration_commit"],
        "candidate_manifest_path": lock["candidate_manifest_path"],
        "candidate_manifest_sha256": lock["candidate_manifest_sha256"],
        "hypothesis_adapter_version": bundle["hypothesis_adapter_version"],
        "prediction_artifact_sha256": predictions.sha256,
        "prediction_items_sha256": bundle["items_sha256"],
        "input_export_receipt_sha256": input_export_receipt.sha256,
        "raw_predictions_sha256": bundle["raw_predictions_sha256"],
        "execution_envelope_sha256": execution_envelope.sha256,
        "runner_code_commit": envelope["runner"]["code_commit"],
        "runner_source_sha256": envelope["runner"]["source_sha256"],
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise CustodianReplayError(
                f"prediction freeze receipt {field} does not match replay artifacts"
            )


def validate_terminal_manifest_for_receipt(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    core_report: Mapping[str, Any],
    execution_envelope: Mapping[str, Any],
    input_export_receipt: Mapping[str, Any],
    prediction_freeze_receipt: Mapping[str, Any],
) -> None:
    """Bind terminal accuracy, performance, and lineage to sealed evidence."""

    violations = validate_manifest(dict(manifest), "terminal candidate manifest")
    if violations:
        raise CustodianReplayError(violations[0])
    if manifest.get("decision") == "planned":
        raise CustodianReplayError("terminal candidate manifest must contain a result")
    validate_custodian_receipt(receipt)
    if receipt["kind"] != CUSTODIAN_SCORE_RECEIPT_KIND:
        raise CustodianReplayError("terminal candidate requires a score receipt")
    validate_custodian_receipt(input_export_receipt)
    if input_export_receipt["kind"] != INPUT_EXPORT_RECEIPT_KIND:
        raise CustodianReplayError(
            "terminal candidate requires an input export receipt"
        )
    validate_custodian_receipt(prediction_freeze_receipt)
    if prediction_freeze_receipt["kind"] != PREDICTION_FREEZE_RECEIPT_KIND:
        raise CustodianReplayError(
            "terminal candidate requires a prediction freeze receipt"
        )
    try:
        validate_execution_envelope(execution_envelope)
        execution_payload = canonical_execution_envelope_bytes(execution_envelope)
    except ExecutionEnvelopeError as exc:
        raise CustodianReplayError(str(exc)) from exc
    execution_sha256 = sha256_bytes(execution_payload)
    input_export_receipt_sha256 = sha256_bytes(
        canonical_custodian_receipt_bytes(input_export_receipt)
    )
    prediction_receipt_sha256 = sha256_bytes(
        canonical_custodian_receipt_bytes(prediction_freeze_receipt)
    )
    if receipt["execution_envelope_sha256"] != execution_sha256:
        raise CustodianReplayError(
            "terminal execution envelope does not match score receipt"
        )
    if receipt["input_export_receipt_sha256"] != input_export_receipt_sha256:
        raise CustodianReplayError(
            "terminal input export receipt does not match score receipt"
        )
    if receipt["prediction_freeze_receipt_sha256"] != prediction_receipt_sha256:
        raise CustodianReplayError(
            "terminal prediction freeze receipt does not match score receipt"
        )
    if prediction_freeze_receipt["execution_envelope_sha256"] != execution_sha256:
        raise CustodianReplayError(
            "prediction freeze receipt does not bind the execution envelope"
        )
    if (
        prediction_freeze_receipt["input_export_receipt_sha256"]
        != input_export_receipt_sha256
        or execution_envelope["bindings"]["input_export_receipt_sha256"]
        != input_export_receipt_sha256
    ):
        raise CustodianReplayError(
            "terminal evidence does not bind the input export receipt"
        )
    for field in ("runner_code_commit", "runner_source_sha256"):
        expected = execution_envelope["runner"][
            "code_commit" if field == "runner_code_commit" else "source_sha256"
        ]
        if receipt[field] != expected or prediction_freeze_receipt[field] != expected:
            raise CustodianReplayError(
                f"terminal {field} does not match execution evidence"
            )
    if receipt["experiment_id"] != manifest["experiment_id"]:
        raise CustodianReplayError(
            "terminal manifest experiment_id does not match score receipt"
        )
    if candidate_manifest_freeze_sha256(manifest) != receipt[
        "candidate_freeze_sha256"
    ]:
        raise CustodianReplayError(
            "terminal manifest candidate facts do not match score receipt"
        )
    envelope_bindings = execution_envelope["bindings"]
    envelope_runner = execution_envelope["runner"]
    envelope_candidate_facts = {
        "experiment_id": manifest["experiment_id"],
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "candidate_freeze_sha256": receipt["candidate_freeze_sha256"],
        "candidate_lock_sha256": receipt["candidate_lock_sha256"],
        "input_projection_sha256": receipt["input_projection_sha256"],
        "hypothesis_adapter_version": receipt["hypothesis_adapter_version"],
    }
    for field in ("experiment_id", "dataset_id", "revision"):
        if execution_envelope[field] != envelope_candidate_facts[field]:
            raise CustodianReplayError(
                f"terminal execution {field} does not match score receipt"
            )
    for field in (
        "candidate_freeze_sha256",
        "candidate_lock_sha256",
        "input_projection_sha256",
        "hypothesis_adapter_version",
    ):
        if envelope_bindings[field] != envelope_candidate_facts[field]:
            raise CustodianReplayError(
                f"terminal execution {field} does not match score receipt"
            )
    input_export_score_facts = (
        "experiment_id",
        "dataset_id",
        "revision",
        "input_projection_sha256",
        "candidate_lock_sha256",
        "candidate_freeze_sha256",
        *CANDIDATE_REGISTRATION_FIELD_NAMES,
    )
    for field in input_export_score_facts:
        if input_export_receipt[field] != receipt[field]:
            raise CustodianReplayError(
                f"terminal input export receipt {field} does not match score receipt"
            )
    if (
        input_export_receipt["split"] != SEALED_SPLIT
        or input_export_receipt["decode_item_count"]
        != execution_envelope["measurement"]["counts"]["decode_item_count"]
    ):
        raise CustodianReplayError(
            "terminal input export receipt decode scope does not match execution"
        )
    execution_counts = execution_envelope["measurement"]["counts"]
    freeze_execution_facts = {
        "expected_decode_item_count": execution_counts["decode_item_count"],
        "prediction_item_count": execution_counts["prediction_item_count"],
        "missing_prediction_count": execution_counts["missing_prediction_count"],
        "prediction_items_sha256": envelope_bindings[
            "prediction_items_sha256"
        ],
        "raw_predictions_sha256": envelope_bindings["raw_predictions_sha256"],
    }
    for field, expected in freeze_execution_facts.items():
        if prediction_freeze_receipt[field] != expected:
            raise CustodianReplayError(
                f"terminal prediction freeze receipt {field} does not match execution"
            )
    freeze_score_facts = (
        "experiment_id",
        "dataset_id",
        "revision",
        "input_projection_sha256",
        "candidate_lock_sha256",
        "candidate_freeze_sha256",
        *CANDIDATE_REGISTRATION_FIELD_NAMES,
        "hypothesis_adapter_version",
        "prediction_artifact_sha256",
        "prediction_items_sha256",
        "input_export_receipt_sha256",
        "execution_envelope_sha256",
        "runner_code_commit",
        "runner_source_sha256",
    )
    for field in freeze_score_facts:
        if prediction_freeze_receipt[field] != receipt[field]:
            raise CustodianReplayError(
                f"terminal prediction freeze receipt {field} does not match score receipt"
            )
    if envelope_runner["code_commit"] != manifest["code_commit"]:
        raise CustodianReplayError(
            "terminal execution runner commit does not match manifest"
        )
    if receipt["scorer_code_commit"] != manifest["code_commit"]:
        raise CustodianReplayError(
            "terminal scorer commit does not match manifest"
        )
    runner_manifest_facts = {
        "effective_config_sha256": manifest["config_sha256"],
        "command": manifest["command"],
        "models": manifest["models"],
        "hardware": manifest["hardware"],
    }
    for field, expected in runner_manifest_facts.items():
        if envelope_runner[field] != expected:
            raise CustodianReplayError(
                f"terminal execution runner {field} does not match manifest"
            )

    expected_data_version = f"{receipt['dataset_id']}-{receipt['revision']}"
    if manifest["task_id"] != "EVAL-01":
        raise CustodianReplayError("terminal manifest task_id must be EVAL-01")
    if manifest["data_sha256"] != receipt["data_sha256"]:
        raise CustodianReplayError(
            "terminal manifest data_sha256 does not match score receipt"
        )
    if manifest["eval_data_version"] != expected_data_version:
        raise CustodianReplayError(
            "terminal manifest eval_data_version does not match score receipt"
        )

    try:
        validate_core_report(core_report)
        core_payload = canonical_core_bytes(core_report)
    except CoreReportValidationError as exc:
        raise CustodianReplayError(str(exc)) from exc
    if sha256_bytes(core_payload) != receipt["core_sha256"]:
        raise CustodianReplayError(
            "terminal core bytes do not match score receipt"
        )

    core_provenance = core_report["provenance"]
    core_scoring = core_report["scoring"]
    core_scope = core_report["configuration"]["evaluation_scope"]
    receipt_core_facts = {
        "data_sha256": core_provenance["data_sha256"],
        "record_identity_version": core_provenance["record_identity_version"],
        "record_input_sha256": core_provenance["record_input_sha256"],
        "prediction_input_sha256": core_provenance["prediction_input_sha256"],
        "hypothesis_adapter_version": core_scoring[
            "hypothesis_adapter_version"
        ],
        "evaluation_scope": core_scope,
        "core_schema_version": core_report["schema_version"],
    }
    if any(receipt[field] != value for field, value in receipt_core_facts.items()):
        raise CustodianReplayError(
            "score receipt facts do not match the restricted core"
        )
    if manifest["normalizer_version"] != core_scoring["normalizer_version"]:
        raise CustodianReplayError(
            "terminal manifest normalizer_version does not match restricted core"
        )
    _validate_candidate_command_adapter(
        manifest,
        receipt["hypothesis_adapter_version"],
    )

    metrics = manifest["metrics"]
    if set(metrics) != TERMINAL_METRIC_FIELDS:
        raise CustodianReplayError(
            "terminal metrics must contain exactly the sealed EVAL-01 fields"
        )
    cer = core_report["aggregate"]["cer"]
    counts = core_report["counts"]
    if cer["reference_units"] == 0:
        raise CustodianReplayError(
            "a terminal experiment manifest requires nonzero core CER reference units"
        )
    exact_metrics = {
        "substitutions": cer["substitutions"],
        "deletions": cer["deletions"],
        "insertions": cer["insertions"],
        "reference_units": cer["reference_units"],
        "utterance_count": counts["utterance_count"],
        "failed_count": counts["failed_count"],
        "excluded_count": counts["excluded_count"],
    }
    for field, expected in exact_metrics.items():
        actual = metrics.get(field)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected
        ):
            raise CustodianReplayError(
                f"terminal metrics.{field} does not match restricted core"
            )
    expected_cer = cer["errors"] / cer["reference_units"]
    if not math.isclose(
        metrics["content_cer"],
        expected_cer,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise CustodianReplayError(
            "terminal metrics.content_cer does not match restricted core"
        )

    mer = core_report["aggregate"]["mer"]
    if "mer" not in metrics:
        raise CustodianReplayError(
            "terminal metrics.mer is required for a sealed score receipt"
        )
    expected_mer = (
        None
        if mer["reference_units"] == 0
        else mer["errors"] / mer["reference_units"]
    )
    terminal_mer = metrics["mer"]
    if expected_mer is None:
        mer_matches = terminal_mer is None
    else:
        mer_matches = isinstance(terminal_mer, (int, float)) and not isinstance(
            terminal_mer, bool
        ) and math.isclose(
            terminal_mer,
            expected_mer,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    if not mer_matches:
        raise CustodianReplayError(
            "terminal metrics.mer does not match restricted core"
        )

    measurement = execution_envelope["measurement"]
    execution_counts = measurement["counts"]
    if execution_counts["decode_item_count"] != counts["scored_count"]:
        raise CustodianReplayError(
            "execution decode count does not match restricted core"
        )
    if execution_counts["failed_count"] != counts["failed_count"]:
        raise CustodianReplayError(
            "execution failed count does not match restricted core"
        )
    required_execution_metrics = {
        "rtf_p50": measurement["rtf_p50"],
        "rtf_p95": measurement["rtf_p95"],
        "peak_rss_mb": peak_rss_mib(measurement["peak_rss_bytes"]),
        "rtf_attempted_count": execution_counts["total_attempt_count"],
        "retried_count": execution_counts["retried_item_count"],
        "model_load_seconds": measurement["model_load_ns"] / 1_000_000_000,
        "cold_inference_seconds": measurement["cold_inference_ns"]
        / 1_000_000_000,
        "cold_start_seconds": measurement["cold_start_ns"] / 1_000_000_000,
        "warm_wall_seconds": measurement["measured_wall_ns"] / 1_000_000_000,
        "warm_audio_seconds": measurement["measured_audio_seconds"],
    }
    for field, expected in required_execution_metrics.items():
        actual = metrics.get(field)
        if isinstance(expected, int):
            matches = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == expected
            )
        else:
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            )
        if not matches:
            raise CustodianReplayError(
                f"terminal metrics.{field} does not match execution envelope"
            )

    receipt_sha256 = sha256_bytes(canonical_custodian_receipt_bytes(receipt))
    required_artifacts = (
        ("other", receipt["input_projection_sha256"], "sealed input projection"),
        ("other", receipt["candidate_lock_sha256"], "candidate lock"),
        (
            "other",
            receipt["input_export_receipt_sha256"],
            "input export receipt",
        ),
        ("prediction", receipt["prediction_artifact_sha256"], "prediction bundle"),
        (
            "report",
            receipt["execution_envelope_sha256"],
            "execution envelope",
        ),
        (
            "other",
            receipt["prediction_freeze_receipt_sha256"],
            "prediction freeze receipt",
        ),
        ("report", receipt["core_sha256"], "restricted core"),
        ("other", receipt_sha256, "score receipt"),
    )
    artifacts = manifest["artifacts"]
    for expected_kind, expected_sha256, label in required_artifacts:
        if not any(
            artifact["kind"] == expected_kind
            and artifact["sha256"] == expected_sha256
            for artifact in artifacts
        ):
            raise CustodianReplayError(
                f"terminal artifacts must bind the {label} with kind "
                f"{expected_kind!r}"
            )


def _safe_output_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("output paths must be pathlib.Path instances")
    absolute = Path(os.path.abspath(path))
    parent = absolute.parent
    resolved_parent = parent.resolve()
    if resolved_parent != parent:
        raise CustodianReplayError("output parent must not contain symlinks")
    if not resolved_parent.is_dir():
        raise CustodianReplayError("output parent directory does not exist")
    parent_mode = resolved_parent.stat().st_mode & 0o777
    if parent_mode != 0o700:
        raise CustodianReplayError(
            "output parent directory mode must be exactly 0700"
        )
    resolved = resolved_parent / absolute.name
    if os.path.lexists(resolved):
        raise CustodianReplayError("refusing to overwrite an existing output")
    return resolved


def validate_output_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Preflight distinct, non-existing outputs without publishing anything."""

    if isinstance(paths, (str, bytes, bytearray)) or not isinstance(
        paths, Sequence
    ):
        raise TypeError("output paths must be an ordered sequence")
    if len(paths) == 0:
        raise CustodianReplayError("at least one output path is required")
    resolved: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        path = _safe_output_path(raw_path)
        if path in seen_paths:
            raise CustodianReplayError("output paths must be distinct")
        seen_paths.add(path)
        resolved.append(path)
    if len({path.parent for path in resolved}) != 1:
        raise CustodianReplayError(
            "all outputs for one transition must share one private directory"
        )
    return tuple(resolved)


def validate_restricted_input_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Require one exact 0700 directory containing exact 0600 regular inputs."""

    if isinstance(paths, (str, bytes, bytearray)) or not isinstance(
        paths, Sequence
    ):
        raise TypeError("restricted input paths must be an ordered sequence")
    if not paths:
        raise CustodianReplayError("at least one restricted input is required")
    resolved: list[Path] = []
    for raw_path in paths:
        if not isinstance(raw_path, Path):
            raise TypeError("restricted inputs must be pathlib.Path instances")
        absolute = Path(os.path.abspath(raw_path))
        parent = absolute.parent
        if parent.resolve() != parent or not parent.is_dir():
            raise CustodianReplayError(
                "restricted input parent must be a non-symlink directory"
            )
        try:
            metadata = absolute.lstat()
        except OSError as exc:
            raise CustodianReplayError("restricted input is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CustodianReplayError("restricted input must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise CustodianReplayError("restricted input mode must be exactly 0600")
        resolved.append(absolute)
    parents = {path.parent for path in resolved}
    if len(parents) != 1:
        raise CustodianReplayError(
            "restricted inputs must share one restricted directory"
        )
    parent = next(iter(parents))
    if stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise CustodianReplayError(
            "restricted input parent mode must be exactly 0700"
        )
    return tuple(resolved)


def validate_restricted_transition_paths(
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Keep one transition's existing evidence and new outputs in one vault."""

    resolved_inputs = validate_restricted_input_paths(input_paths)
    resolved_outputs = validate_output_paths(output_paths)
    if resolved_inputs[0].parent != resolved_outputs[0].parent:
        raise CustodianReplayError(
            "restricted inputs and outputs must share one private directory"
        )
    return resolved_inputs, resolved_outputs


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_atomic_outputs(outputs: Sequence[tuple[Path, bytes]]) -> None:
    """Publish complete mode-0600 artifacts without overwriting evidence."""

    if isinstance(outputs, (str, bytes, bytearray)) or not isinstance(
        outputs, Sequence
    ):
        raise TypeError("outputs must be an ordered sequence of path/payload pairs")
    if len(outputs) == 0:
        raise CustodianReplayError("at least one output is required")
    pairs: list[tuple[Path, bytes]] = []
    for index, pair in enumerate(outputs):
        if (
            isinstance(pair, (str, bytes, bytearray))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(f"outputs[{index}] must be one path/payload pair")
        raw_path, payload = pair
        if not isinstance(payload, bytes):
            raise TypeError("output payloads must be bytes")
        pairs.append((raw_path, payload))
    resolved_paths = validate_output_paths([path for path, _ in pairs])
    resolved = [
        (resolved_path, payload)
        for resolved_path, (_, payload) in zip(resolved_paths, pairs, strict=True)
    ]

    temporary_paths: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for target, payload in resolved:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            temporary_paths.append((target, temporary))
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as destination:
                    descriptor = -1
                    destination.write(payload)
                    destination.flush()
                    os.fsync(destination.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

        for target, temporary in temporary_paths:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise CustodianReplayError(
                    "refusing to overwrite an output created during replay"
                ) from exc
            published.append(target)
            # Ordered callers place their receipt or runner envelope completion
            # marker last. Persist each predecessor before that marker can
            # itself become durable.
            _fsync_directory(target.parent)
        for _, temporary in temporary_paths:
            temporary.unlink()
        _fsync_directory(resolved[0][0].parent)
    except BaseException:
        # Remove the completion marker first. If that removal fails, retain all
        # predecessor artifacts rather than leave a visible marker that points
        # at evidence this rollback subsequently deleted.
        for target in reversed(published):
            try:
                target.unlink()
            except OSError:
                break
        for _, temporary in temporary_paths:
            try:
                temporary.unlink()
            except OSError:
                pass
        for parent in {path.parent for path in published}:
            try:
                _fsync_directory(parent)
            except OSError:
                pass
        raise


__all__ = [
    "CANDIDATE_LOCK_KIND",
    "CANDIDATE_LOCK_SCHEMA_VERSION",
    "CUSTODIAN_SCORE_RECEIPT_KIND",
    "INPUT_EXPORT_RECEIPT_KIND",
    "PREDICTION_BUNDLE_KIND",
    "PREDICTION_BUNDLE_SCHEMA_VERSION",
    "PREDICTION_FREEZE_RECEIPT_KIND",
    "RECEIPT_SCHEMA_VERSION",
    "SEALED_SPLIT",
    "SCORER_SOURCE_PATHS",
    "CustodianReplayError",
    "LoadedArtifact",
    "LoadedPredictionItems",
    "build_candidate_lock",
    "build_prediction_bundle",
    "candidate_freeze_projection",
    "candidate_freeze_sha256",
    "candidate_manifest_freeze_sha256",
    "canonical_candidate_lock_bytes",
    "canonical_custodian_receipt_bytes",
    "canonical_prediction_bundle_bytes",
    "load_candidate_lock",
    "load_custodian_receipt",
    "load_planned_candidate_manifest",
    "load_prediction_items_jsonl",
    "load_prediction_items_jsonl_artifact",
    "load_prediction_bundle",
    "load_restricted_core_report",
    "load_sealed_input_projection",
    "load_terminal_candidate_manifest",
    "parse_sealed_input_projection",
    "preflight_replay_artifacts",
    "scorer_code_identity",
    "validate_candidate_lock",
    "validate_candidate_request",
    "validate_custodian_receipt",
    "validate_decode_handoff",
    "validate_frozen_execution_handoff",
    "validate_input_export_receipt_handoff",
    "validate_prediction_freeze_receipt_handoff",
    "validate_prediction_bundle",
    "validate_prediction_handoff",
    "validate_raw_execution_handoff",
    "validate_registered_candidate_binding",
    "validate_restricted_input_paths",
    "validate_restricted_transition_paths",
    "validate_output_paths",
    "validate_replay_collection",
    "validate_terminal_manifest_for_receipt",
    "write_atomic_outputs",
]
