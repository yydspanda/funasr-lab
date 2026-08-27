"""Strict, replayable dataset collection contract for EVAL-01.

The validator is intentionally standard-library only. It validates private
collection bytes when they are locally available, but never downloads data and
never imports FunASR.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from eval.normalizers import NORMALIZER_VERSION
from eval.record_identity import DEFAULT_SLICE_FIELDS
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import RecordIdentityError
from eval.record_identity import record_input_sha256
from eval.scoring import MER_TOKENIZER_VERSION


SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
REVISION_PATTERN = re.compile(r"^v[0-9]+\.[0-9]+(?:[._-][A-Za-z0-9.-]+)?$")
MONTH_PATTERN = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")
REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SCENARIO_TAXONOMY_VERSION = "asr-scenario-v1"
SEALED_INPUT_PROJECTION_KIND = "asr-sealed-audio-input"
SEALED_REFERENCE_PROJECTION_KIND = "asr-sealed-reference-input"
FLOATING_REVISIONS = frozenset({"head", "latest", "main", "master", "trunk"})
ALLOWED_SPLITS = ("smoke", "dev", "sealed-blind")
ISOLATION_KEYS = (
    "speaker_id",
    "session_id",
    "source_recording_id",
    "lineage_group_id",
    "dedup_cluster_id",
    "audio_sha256",
    "source_audio_sha256",
)
SCENARIO_TAGS = frozenset(
    {
        "accent:mandarin",
        "accent:regional",
        "content:domain-term",
        "environment:far-field",
        "environment:meeting",
        "environment:near-field",
        "language:zh",
        "language:zh-en",
        "noise:clean",
        "noise:stationary",
        "noise:transient",
        "signal:clipped",
        "signal:low-volume",
        "signal:non-speech",
        "signal:silence",
        "speech:long-utterance",
        "speech:overlap",
    }
)
ACCESS_CLASSES = frozenset({"public", "internal", "restricted", "sealed"})
RIGHTS_BASES = frozenset(
    {"explicit-consent", "contract", "licensed", "public-license", "synthetic"}
)
CONSENT_STATUSES = frozenset({"verified", "not-required"})
ALLOWED_USES = frozenset({"asr-evaluation", "asr-training"})
SOURCE_KINDS = frozenset(
    {"first-party-recording", "licensed-dataset", "public-dataset", "synthetic"}
)

DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "dataset_id",
        "revision",
        "state",
        "record_schema_version",
        "normalizer_version",
        "mer_tokenizer_version",
        "scenario_taxonomy_version",
        "manifests",
        "split_policy",
        "provenance_groups",
        "rights_groups",
        "dedup",
        "blind",
    }
)
MANIFEST_FIELDS = frozenset(
    {"split", "path", "sha256", "record_count", "reference_access"}
)
SPLIT_POLICY_FIELDS = frozenset({"allowed", "isolation_keys"})
PROVENANCE_FIELDS = frozenset(
    {
        "id",
        "source_id",
        "source_kind",
        "source_revision",
        "collection_period",
        "annotation_protocol_sha256",
        "transform_recipe_sha256",
    }
)
RIGHTS_FIELDS = frozenset(
    {
        "id",
        "basis",
        "consent_status",
        "allowed_uses",
        "access_class",
        "evidence_sha256",
        "reviewed_at",
    }
)
DEDUP_FIELDS = frozenset(
    {
        "method",
        "version",
        "threshold",
        "config_sha256",
        "report_path",
        "report_sha256",
        "status",
    }
)
BLIND_FIELDS = frozenset(
    {
        "split",
        "reference_access",
        "custodian_role",
        "sealed_at",
        "unlock_policy",
        "input_projection_sha256",
        "reference_projection_sha256",
    }
)
DEDUP_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "dataset_id",
        "revision",
        "method",
        "version",
        "threshold",
        "config_sha256",
        "manifest_inventory_sha256",
        "record_count",
        "record_cluster_inventory_sha256",
        "status",
    }
)
RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "data_revision",
        "id",
        "audio",
        "audio_sha256",
        "duration_seconds",
        "sample_rate",
        "channels",
        "sample_width_bits",
        "raw_text",
        "reference_sha256",
        "normalizer_version",
        "mer_tokenizer_version",
        "speaker_id",
        "session_id",
        "source_recording_id",
        "lineage_group_id",
        "dedup_cluster_id",
        "lineage_kind",
        "derived_from_id",
        "source_audio_sha256",
        "transform_recipe_sha256",
        "split",
        "provenance_group_id",
        "rights_group_id",
        "access_class",
        "scenario_taxonomy_version",
        "scenario_tags",
        "evaluation_status",
        "exclusion_reason",
    }
)


class CollectionValidationError(ValueError):
    """Raised when a collection cannot be frozen or replayed."""


class DuplicateJsonKeyError(ValueError):
    """Raised when JSON contains an ambiguous duplicate object key."""


class InvalidJsonConstantError(ValueError):
    """Raised when JSON contains a non-standard NaN or infinity literal."""


@dataclass(frozen=True)
class ManifestSpec:
    split: str
    path: str
    sha256: str
    record_count: int
    reference_access: str


@dataclass(frozen=True)
class CollectionDescriptor:
    path: Path
    raw_sha256: str
    dataset_id: str
    revision: str
    normalizer_version: str
    mer_tokenizer_version: str
    manifests: tuple[ManifestSpec, ...]
    provenance_groups: Mapping[str, Mapping[str, Any]]
    rights_groups: Mapping[str, Mapping[str, Any]]
    dedup: Mapping[str, Any]
    blind: Mapping[str, Any]


@dataclass(frozen=True)
class _ValidatedRecord:
    record: dict[str, Any]
    source: str
    actual_duration: float


@dataclass(frozen=True)
class ValidatedCollection:
    """Fully checked collection records in frozen descriptor/manifest order."""

    summary: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    _sealed_input_projection: bytes


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKeyError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise InvalidJsonConstantError(f"non-standard JSON constant {value!r}")


def canonical_json_bytes(value: Any) -> bytes:
    document = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (document + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _strict_fields(
    document: Mapping[str, Any], expected: frozenset[str], source: str
) -> None:
    keys = set(document)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise CollectionValidationError(
            f"{source}: missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise CollectionValidationError(
            f"{source}: unknown field(s): {', '.join(unknown)}"
        )


def _string(document: Mapping[str, Any], field: str, source: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise CollectionValidationError(f"{source}: {field} must be a non-empty string")
    if value != value.strip():
        raise CollectionValidationError(
            f"{source}: {field} must not contain surrounding whitespace"
        )
    return value


def _identifier(document: Mapping[str, Any], field: str, source: str) -> str:
    value = _string(document, field, source)
    if ID_PATTERN.fullmatch(value) is None:
        raise CollectionValidationError(
            f"{source}: {field} must match {ID_PATTERN.pattern}"
        )
    return value


def _sha256(document: Mapping[str, Any], field: str, source: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CollectionValidationError(
            f"{source}: {field} must use sha256:<64 lowercase hex chars>"
        )
    digest = value.removeprefix("sha256:")
    if len(set(digest)) == 1:
        raise CollectionValidationError(f"{source}: {field} looks like a placeholder")
    return value


def _positive_integer(document: Mapping[str, Any], field: str, source: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CollectionValidationError(f"{source}: {field} must be a positive integer")
    return value


def _positive_number(document: Mapping[str, Any], field: str, source: str) -> float:
    value = document.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CollectionValidationError(
            f"{source}: {field} must be a positive finite number"
        )
    return float(value)


def _unit_interval_number(
    document: Mapping[str, Any], field: str, source: str
) -> float:
    value = document.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or value > 1
    ):
        raise CollectionValidationError(
            f"{source}: {field} must be a finite number from 0 through 1"
        )
    return float(value)


def _string_list(
    document: Mapping[str, Any],
    field: str,
    source: str,
    *,
    allowed: frozenset[str] | None = None,
) -> list[str]:
    value = document.get(field)
    if not isinstance(value, list) or not value:
        raise CollectionValidationError(
            f"{source}: {field} must be a non-empty array"
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise CollectionValidationError(
            f"{source}: {field} must contain non-empty strings"
        )
    if len(set(value)) != len(value):
        raise CollectionValidationError(f"{source}: {field} contains duplicates")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise CollectionValidationError(
                f"{source}: {field} contains unknown value(s): {', '.join(unknown)}"
            )
    return value


def _validate_date(value: str, source: str, field: str) -> None:
    try:
        dt.date.fromisoformat(value)
    except ValueError as error:
        raise CollectionValidationError(
            f"{source}: {field} must be an ISO YYYY-MM-DD date"
        ) from error


def _validate_collection_period(value: str, source: str) -> None:
    parts = value.split("/")
    if len(parts) not in (1, 2) or any(MONTH_PATTERN.fullmatch(part) is None for part in parts):
        raise CollectionValidationError(
            f"{source}: collection_period must be YYYY-MM or YYYY-MM/YYYY-MM"
        )
    if len(parts) == 2 and parts[1] < parts[0]:
        raise CollectionValidationError(
            f"{source}: collection_period end must not precede its start"
        )


def _resolve_file(root: Path, logical_path: str, source: str, field: str) -> Path:
    relative = Path(logical_path)
    if relative.is_absolute():
        raise CollectionValidationError(f"{source}: {field} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise CollectionValidationError(
            f"{source}: {field} escapes its configured root"
        ) from error
    if not resolved.is_file():
        raise CollectionValidationError(
            f"{source}: {field} does not exist: {logical_path}"
        )
    return resolved


def _json_document(path: Path, *, canonical: bool) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CollectionValidationError(f"cannot read {path}: {error}") from error
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CollectionValidationError(f"{path}: must be UTF-8") from error
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        DuplicateJsonKeyError,
        InvalidJsonConstantError,
    ) as error:
        raise CollectionValidationError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise CollectionValidationError(f"{path}: document must be a JSON object")
    if canonical and payload != canonical_json_bytes(document):
        raise CollectionValidationError(
            f"{path}: collection descriptor must use canonical JSON bytes"
        )
    return document, payload


def _parse_manifest_specs(document: Mapping[str, Any], source: str) -> tuple[ManifestSpec, ...]:
    raw = document.get("manifests")
    if not isinstance(raw, list) or len(raw) != len(ALLOWED_SPLITS):
        raise CollectionValidationError(
            f"{source}: manifests must contain exactly {len(ALLOWED_SPLITS)} entries"
        )
    specs: list[ManifestSpec] = []
    for index, entry in enumerate(raw):
        entry_source = f"{source}: manifests[{index}]"
        if not isinstance(entry, dict):
            raise CollectionValidationError(f"{entry_source}: must be an object")
        _strict_fields(entry, MANIFEST_FIELDS, entry_source)
        split = _string(entry, "split", entry_source)
        if split not in ALLOWED_SPLITS:
            raise CollectionValidationError(f"{entry_source}: unknown split {split!r}")
        access = _string(entry, "reference_access", entry_source)
        if access not in ACCESS_CLASSES:
            raise CollectionValidationError(
                f"{entry_source}: unknown reference_access {access!r}"
            )
        if split == "sealed-blind" and access != "sealed":
            raise CollectionValidationError(
                f"{entry_source}: sealed-blind reference_access must be 'sealed'"
            )
        specs.append(
            ManifestSpec(
                split=split,
                path=_string(entry, "path", entry_source),
                sha256=_sha256(entry, "sha256", entry_source),
                record_count=_positive_integer(entry, "record_count", entry_source),
                reference_access=access,
            )
        )
    if tuple(spec.split for spec in specs) != ALLOWED_SPLITS:
        raise CollectionValidationError(
            f"{source}: manifests must use canonical split order {ALLOWED_SPLITS}"
        )
    if len({spec.path for spec in specs}) != len(specs):
        raise CollectionValidationError(f"{source}: manifest paths must be unique")
    return tuple(specs)


def _parse_provenance_groups(
    document: Mapping[str, Any], source: str
) -> dict[str, Mapping[str, Any]]:
    raw = document.get("provenance_groups")
    if not isinstance(raw, list) or not raw:
        raise CollectionValidationError(
            f"{source}: provenance_groups must be a non-empty array"
        )
    groups: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(raw):
        entry_source = f"{source}: provenance_groups[{index}]"
        if not isinstance(entry, dict):
            raise CollectionValidationError(f"{entry_source}: must be an object")
        _strict_fields(entry, PROVENANCE_FIELDS, entry_source)
        group_id = _identifier(entry, "id", entry_source)
        if group_id in groups:
            raise CollectionValidationError(
                f"{entry_source}: duplicate provenance group {group_id!r}"
            )
        _identifier(entry, "source_id", entry_source)
        source_kind = _string(entry, "source_kind", entry_source)
        if source_kind not in SOURCE_KINDS:
            raise CollectionValidationError(
                f"{entry_source}: unknown source_kind {source_kind!r}"
            )
        source_revision = _string(entry, "source_revision", entry_source)
        if source_revision.casefold().rsplit("/", 1)[-1] in FLOATING_REVISIONS:
            raise CollectionValidationError(
                f"{entry_source}: source_revision must be immutable, not a floating "
                "branch token"
            )
        period = _string(entry, "collection_period", entry_source)
        _validate_collection_period(period, entry_source)
        _sha256(entry, "annotation_protocol_sha256", entry_source)
        _sha256(entry, "transform_recipe_sha256", entry_source)
        groups[group_id] = entry
    return groups


def _parse_rights_groups(
    document: Mapping[str, Any], source: str
) -> dict[str, Mapping[str, Any]]:
    raw = document.get("rights_groups")
    if not isinstance(raw, list) or not raw:
        raise CollectionValidationError(
            f"{source}: rights_groups must be a non-empty array"
        )
    groups: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(raw):
        entry_source = f"{source}: rights_groups[{index}]"
        if not isinstance(entry, dict):
            raise CollectionValidationError(f"{entry_source}: must be an object")
        _strict_fields(entry, RIGHTS_FIELDS, entry_source)
        group_id = _identifier(entry, "id", entry_source)
        if group_id in groups:
            raise CollectionValidationError(
                f"{entry_source}: duplicate rights group {group_id!r}"
            )
        basis = _string(entry, "basis", entry_source)
        if basis not in RIGHTS_BASES:
            raise CollectionValidationError(
                f"{entry_source}: unknown rights basis {basis!r}"
            )
        consent = _string(entry, "consent_status", entry_source)
        if consent not in CONSENT_STATUSES:
            raise CollectionValidationError(
                f"{entry_source}: consent_status must be verified or not-required"
            )
        uses = _string_list(
            entry, "allowed_uses", entry_source, allowed=ALLOWED_USES
        )
        if "asr-evaluation" not in uses:
            raise CollectionValidationError(
                f"{entry_source}: allowed_uses must include asr-evaluation"
            )
        access = _string(entry, "access_class", entry_source)
        if access not in ACCESS_CLASSES:
            raise CollectionValidationError(
                f"{entry_source}: unknown access_class {access!r}"
            )
        _sha256(entry, "evidence_sha256", entry_source)
        reviewed_at = _string(entry, "reviewed_at", entry_source)
        _validate_date(reviewed_at, entry_source, "reviewed_at")
        groups[group_id] = entry
    return groups


def load_collection_descriptor(
    descriptor_path: Path, *, require_frozen: bool = True
) -> CollectionDescriptor:
    document, payload = _json_document(descriptor_path, canonical=True)
    source = str(descriptor_path)
    _strict_fields(document, DESCRIPTOR_FIELDS, source)
    if document.get("schema_version") != 1:
        raise CollectionValidationError(f"{source}: schema_version must be integer 1")
    if document.get("kind") != "asr-evaluation-collection":
        raise CollectionValidationError(
            f"{source}: kind must be 'asr-evaluation-collection'"
        )
    dataset_id = _identifier(document, "dataset_id", source)
    revision = _string(document, "revision", source)
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise CollectionValidationError(
            f"{source}: revision must be an immutable v<major>.<minor> value"
        )
    state = _string(document, "state", source)
    if state not in {"schema-only", "frozen"}:
        raise CollectionValidationError(f"{source}: state must be schema-only or frozen")
    if require_frozen and state != "frozen":
        raise CollectionValidationError(
            f"{source}: state is {state!r}; schema-only examples are not evidence"
        )
    if document.get("record_schema_version") != 1:
        raise CollectionValidationError(
            f"{source}: record_schema_version must be integer 1"
        )
    normalizer_version = _string(document, "normalizer_version", source)
    if normalizer_version != NORMALIZER_VERSION:
        raise CollectionValidationError(
            f"{source}: normalizer_version must be {NORMALIZER_VERSION!r}"
        )
    mer_tokenizer_version = _string(document, "mer_tokenizer_version", source)
    if mer_tokenizer_version != MER_TOKENIZER_VERSION:
        raise CollectionValidationError(
            f"{source}: mer_tokenizer_version must be {MER_TOKENIZER_VERSION!r}"
        )
    taxonomy_version = _string(document, "scenario_taxonomy_version", source)
    if taxonomy_version != SCENARIO_TAXONOMY_VERSION:
        raise CollectionValidationError(
            f"{source}: scenario_taxonomy_version must be "
            f"{SCENARIO_TAXONOMY_VERSION!r}"
        )

    policy = document.get("split_policy")
    if not isinstance(policy, dict):
        raise CollectionValidationError(f"{source}: split_policy must be an object")
    _strict_fields(policy, SPLIT_POLICY_FIELDS, f"{source}: split_policy")
    if policy.get("allowed") != list(ALLOWED_SPLITS):
        raise CollectionValidationError(
            f"{source}: split_policy.allowed must equal {list(ALLOWED_SPLITS)}"
        )
    if policy.get("isolation_keys") != list(ISOLATION_KEYS):
        raise CollectionValidationError(
            f"{source}: split_policy.isolation_keys must equal {list(ISOLATION_KEYS)}"
        )

    manifests = _parse_manifest_specs(document, source)
    provenance_groups = _parse_provenance_groups(document, source)
    rights_groups = _parse_rights_groups(document, source)

    dedup = document.get("dedup")
    if not isinstance(dedup, dict):
        raise CollectionValidationError(f"{source}: dedup must be an object")
    _strict_fields(dedup, DEDUP_FIELDS, f"{source}: dedup")
    _string(dedup, "method", f"{source}: dedup")
    _string(dedup, "version", f"{source}: dedup")
    _unit_interval_number(dedup, "threshold", f"{source}: dedup")
    _sha256(dedup, "config_sha256", f"{source}: dedup")
    _string(dedup, "report_path", f"{source}: dedup")
    _sha256(dedup, "report_sha256", f"{source}: dedup")
    if dedup.get("status") != "reviewed":
        raise CollectionValidationError(
            f"{source}: dedup.status must be 'reviewed' before freeze"
        )

    blind = document.get("blind")
    if not isinstance(blind, dict):
        raise CollectionValidationError(f"{source}: blind must be an object")
    _strict_fields(blind, BLIND_FIELDS, f"{source}: blind")
    if blind.get("split") != "sealed-blind":
        raise CollectionValidationError(
            f"{source}: blind.split must be 'sealed-blind'"
        )
    if blind.get("reference_access") != "sealed":
        raise CollectionValidationError(
            f"{source}: blind.reference_access must be 'sealed'"
        )
    _identifier(blind, "custodian_role", f"{source}: blind")
    sealed_at = _string(blind, "sealed_at", f"{source}: blind")
    if not sealed_at.endswith("Z"):
        raise CollectionValidationError(
            f"{source}: blind.sealed_at must be an ISO UTC timestamp ending in Z"
        )
    try:
        dt.datetime.fromisoformat(sealed_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise CollectionValidationError(
            f"{source}: blind.sealed_at must be an ISO UTC timestamp"
        ) from error
    if blind.get("unlock_policy") != "candidate-config-command-hashes-frozen":
        raise CollectionValidationError(
            f"{source}: blind.unlock_policy must freeze candidate/config/command hashes"
        )
    _sha256(blind, "input_projection_sha256", f"{source}: blind")
    _sha256(blind, "reference_projection_sha256", f"{source}: blind")

    return CollectionDescriptor(
        path=descriptor_path,
        raw_sha256=sha256_bytes(payload),
        dataset_id=dataset_id,
        revision=revision,
        normalizer_version=normalizer_version,
        mer_tokenizer_version=mer_tokenizer_version,
        manifests=manifests,
        provenance_groups=provenance_groups,
        rights_groups=rights_groups,
        dedup=dedup,
        blind=blind,
    )


def _wav_identity(path: Path, source: str) -> tuple[int, int, int, float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width_bits = wav_file.getsampwidth() * 8
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
    except (OSError, EOFError, wave.Error) as error:
        raise CollectionValidationError(
            f"{source}: audio is not a readable PCM WAV: {error}"
        ) from error
    if compression != "NONE" or frame_count <= 0 or sample_rate <= 0:
        raise CollectionValidationError(
            f"{source}: audio must be non-empty uncompressed PCM WAV"
        )
    return sample_rate, channels, sample_width_bits, frame_count / sample_rate


def _manifest_inventory_sha256(manifests: tuple[ManifestSpec, ...]) -> str:
    inventory = "".join(
        f"{spec.sha256.removeprefix('sha256:')}  {spec.path}\n"
        for spec in manifests
    )
    return sha256_bytes(inventory.encode("utf-8"))


def _record_cluster_inventory_sha256(records: list[dict[str, Any]]) -> str:
    inventory = [
        {
            "id": record["id"],
            "dedup_cluster_id": record["dedup_cluster_id"],
        }
        for record in records
    ]
    return sha256_bytes(canonical_json_bytes(inventory))


def _sealed_manifest(descriptor: CollectionDescriptor) -> ManifestSpec:
    return next(spec for spec in descriptor.manifests if spec.split == "sealed-blind")


def _sealed_input_projection_document(
    descriptor: CollectionDescriptor,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = _sealed_manifest(descriptor)
    sealed_records = [record for record in records if record["split"] == manifest.split]
    decode_records = [
        record
        for record in sealed_records
        if record["evaluation_status"] == "included"
    ]
    return {
        "schema_version": 2,
        "kind": SEALED_INPUT_PROJECTION_KIND,
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "split": manifest.split,
        "manifest_sha256": manifest.sha256,
        "manifest_record_count": manifest.record_count,
        "item_count": len(decode_records),
        "items": [
            {
                "id": record["id"],
                "split": record["split"],
                "audio": record["audio"],
                "audio_sha256": record["audio_sha256"],
                "duration_seconds": record["duration_seconds"],
                "sample_rate": record["sample_rate"],
                "channels": record["channels"],
                "sample_width_bits": record["sample_width_bits"],
            }
            for record in decode_records
        ],
    }


def _sealed_reference_projection_document(
    descriptor: CollectionDescriptor,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = _sealed_manifest(descriptor)
    sealed_records = [record for record in records if record["split"] == manifest.split]
    return {
        "schema_version": 2,
        "kind": SEALED_REFERENCE_PROJECTION_KIND,
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "split": manifest.split,
        "manifest_sha256": manifest.sha256,
        "manifest_record_count": manifest.record_count,
        "item_count": len(sealed_records),
        "items": [
            {
                "id": record["id"],
                "split": record["split"],
                "raw_text": record["raw_text"],
                "reference_sha256": record["reference_sha256"],
                "normalizer_version": record["normalizer_version"],
                "mer_tokenizer_version": record["mer_tokenizer_version"],
                "scenario_taxonomy_version": record[
                    "scenario_taxonomy_version"
                ],
                "scenario_tags": list(record["scenario_tags"]),
                "evaluation_status": record["evaluation_status"],
                "exclusion_reason": record["exclusion_reason"],
            }
            for record in sealed_records
        ],
    }


def _validate_dedup_report(
    descriptor: CollectionDescriptor,
    collection_root: Path,
    manifest_inventory_sha256: str,
    record_cluster_inventory_sha256: str,
    record_count: int,
) -> str:
    report_path = _resolve_file(
        collection_root,
        str(descriptor.dedup["report_path"]),
        str(descriptor.path),
        "dedup.report_path",
    )
    document, payload = _json_document(report_path, canonical=True)
    source = str(report_path)
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != descriptor.dedup["report_sha256"]:
        raise CollectionValidationError(
            f"{source}: dedup report hash mismatch; "
            f"expected {descriptor.dedup['report_sha256']}, got {actual_sha256}"
        )
    _strict_fields(document, DEDUP_REPORT_FIELDS, source)
    if document.get("schema_version") != 1:
        raise CollectionValidationError(f"{source}: schema_version must be integer 1")
    if document.get("kind") != "asr-dedup-review":
        raise CollectionValidationError(f"{source}: kind must be 'asr-dedup-review'")
    if document.get("dataset_id") != descriptor.dataset_id:
        raise CollectionValidationError(
            f"{source}: dataset_id does not match collection descriptor"
        )
    if document.get("revision") != descriptor.revision:
        raise CollectionValidationError(
            f"{source}: revision does not match collection descriptor"
        )
    for field in ("method", "version"):
        value = _string(document, field, source)
        if value != descriptor.dedup[field]:
            raise CollectionValidationError(
                f"{source}: {field} does not match collection descriptor"
            )
    threshold = _unit_interval_number(document, "threshold", source)
    if threshold != float(descriptor.dedup["threshold"]):
        raise CollectionValidationError(
            f"{source}: threshold does not match collection descriptor"
        )
    config_sha256 = _sha256(document, "config_sha256", source)
    if config_sha256 != descriptor.dedup["config_sha256"]:
        raise CollectionValidationError(
            f"{source}: config_sha256 does not match collection descriptor"
        )
    if (
        _sha256(document, "manifest_inventory_sha256", source)
        != manifest_inventory_sha256
    ):
        raise CollectionValidationError(
            f"{source}: manifest_inventory_sha256 does not match ordered manifests"
        )
    if _positive_integer(document, "record_count", source) != record_count:
        raise CollectionValidationError(
            f"{source}: record_count does not match validated records"
        )
    if (
        _sha256(document, "record_cluster_inventory_sha256", source)
        != record_cluster_inventory_sha256
    ):
        raise CollectionValidationError(
            f"{source}: record_cluster_inventory_sha256 does not match records"
        )
    if document.get("status") != "reviewed":
        raise CollectionValidationError(f"{source}: status must be 'reviewed'")
    return actual_sha256


def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(record)
    copied["scenario_tags"] = list(record["scenario_tags"])
    return copied


def _load_manifest_records(
    descriptor: CollectionDescriptor,
    spec: ManifestSpec,
    collection_root: Path,
    audio_root: Path,
) -> list[_ValidatedRecord]:
    manifest_path = _resolve_file(
        collection_root, spec.path, str(descriptor.path), "manifest path"
    )
    payload = manifest_path.read_bytes()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != spec.sha256:
        raise CollectionValidationError(
            f"{manifest_path}: manifest hash mismatch; "
            f"expected {spec.sha256}, got {actual_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CollectionValidationError(f"{manifest_path}: must be UTF-8") from error

    records: list[_ValidatedRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        source = f"{manifest_path}:{line_number}"
        if not line.strip():
            raise CollectionValidationError(
                f"{source}: blank JSONL records are not allowed"
            )
        try:
            record = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (
            json.JSONDecodeError,
            DuplicateJsonKeyError,
            InvalidJsonConstantError,
        ) as error:
            raise CollectionValidationError(
                f"{source}: invalid JSON object: {error}"
            ) from error
        if not isinstance(record, dict):
            raise CollectionValidationError(f"{source}: record must be an object")
        _strict_fields(record, RECORD_FIELDS, source)
        if record.get("schema_version") != 1:
            raise CollectionValidationError(f"{source}: schema_version must be integer 1")
        if record.get("dataset_id") != descriptor.dataset_id:
            raise CollectionValidationError(
                f"{source}: dataset_id does not match collection descriptor"
            )
        if record.get("data_revision") != descriptor.revision:
            raise CollectionValidationError(
                f"{source}: data_revision does not match collection descriptor"
            )
        _identifier(record, "id", source)
        split = _string(record, "split", source)
        if split != spec.split:
            raise CollectionValidationError(
                f"{source}: split {split!r} does not match manifest split {spec.split!r}"
            )
        if record.get("normalizer_version") != descriptor.normalizer_version:
            raise CollectionValidationError(
                f"{source}: normalizer_version does not match collection descriptor"
            )
        if record.get("mer_tokenizer_version") != descriptor.mer_tokenizer_version:
            raise CollectionValidationError(
                f"{source}: mer_tokenizer_version does not match collection descriptor"
            )
        if record.get("scenario_taxonomy_version") != SCENARIO_TAXONOMY_VERSION:
            raise CollectionValidationError(
                f"{source}: scenario_taxonomy_version must be "
                f"{SCENARIO_TAXONOMY_VERSION!r}"
            )
        for field in (
            "speaker_id",
            "session_id",
            "source_recording_id",
            "lineage_group_id",
            "dedup_cluster_id",
            "provenance_group_id",
            "rights_group_id",
        ):
            _identifier(record, field, source)

        audio = _string(record, "audio", source)
        audio_path = _resolve_file(audio_root, audio, source, "audio")
        audio_sha256 = _sha256(record, "audio_sha256", source)
        actual_audio_sha256 = sha256_file(audio_path)
        if actual_audio_sha256 != audio_sha256:
            raise CollectionValidationError(
                f"{source}: audio_sha256 mismatch; "
                f"expected {audio_sha256}, got {actual_audio_sha256}"
            )
        duration = _positive_number(record, "duration_seconds", source)
        sample_rate = _positive_integer(record, "sample_rate", source)
        channels = _positive_integer(record, "channels", source)
        sample_width_bits = _positive_integer(record, "sample_width_bits", source)
        actual_rate, actual_channels, actual_width, actual_duration = _wav_identity(
            audio_path, source
        )
        if (
            sample_rate != 16_000
            or actual_rate != 16_000
            or channels != 1
            or actual_channels != 1
            or sample_width_bits != 16
            or actual_width != 16
        ):
            raise CollectionValidationError(
                f"{source}: evaluation audio must be 16 kHz mono signed 16-bit PCM"
            )
        if not math.isclose(duration, actual_duration, rel_tol=0, abs_tol=1 / actual_rate):
            raise CollectionValidationError(
                f"{source}: duration_seconds mismatch; "
                f"manifest={duration}, actual={actual_duration}"
            )

        raw_text = record.get("raw_text")
        if not isinstance(raw_text, str):
            raise CollectionValidationError(f"{source}: raw_text must be a string")
        reference_sha256 = _sha256(record, "reference_sha256", source)
        if reference_sha256 != sha256_bytes(raw_text.encode("utf-8")):
            raise CollectionValidationError(
                f"{source}: reference_sha256 does not match raw_text"
            )

        provenance_id = record["provenance_group_id"]
        provenance = descriptor.provenance_groups.get(provenance_id)
        if provenance is None:
            raise CollectionValidationError(
                f"{source}: unknown provenance_group_id {provenance_id!r}"
            )
        rights_id = record["rights_group_id"]
        rights = descriptor.rights_groups.get(rights_id)
        if rights is None:
            raise CollectionValidationError(
                f"{source}: unknown rights_group_id {rights_id!r}"
            )
        access = _string(record, "access_class", source)
        if access not in ACCESS_CLASSES:
            raise CollectionValidationError(
                f"{source}: unknown access_class {access!r}"
            )
        if access != spec.reference_access or access != rights["access_class"]:
            raise CollectionValidationError(
                f"{source}: access_class must match manifest and rights group"
            )
        if split == "sealed-blind" and access != "sealed":
            raise CollectionValidationError(
                f"{source}: sealed-blind records must use sealed access"
            )

        scenario_tags = _string_list(
            record, "scenario_tags", source, allowed=SCENARIO_TAGS
        )
        if not any(tag.startswith("language:") for tag in scenario_tags):
            raise CollectionValidationError(
                f"{source}: scenario_tags must include at least one language:* tag"
            )
        evaluation_status = _string(record, "evaluation_status", source)
        exclusion_reason = record.get("exclusion_reason")
        if evaluation_status == "included":
            if exclusion_reason is not None:
                raise CollectionValidationError(
                    f"{source}: included record exclusion_reason must be null"
                )
        elif evaluation_status == "excluded":
            if (
                not isinstance(exclusion_reason, str)
                or REASON_CODE_PATTERN.fullmatch(exclusion_reason) is None
            ):
                raise CollectionValidationError(
                    f"{source}: excluded record exclusion_reason must be a stable "
                    "lowercase snake_case code"
                )
        else:
            raise CollectionValidationError(
                f"{source}: evaluation_status must be included or excluded"
            )

        lineage_kind = _string(record, "lineage_kind", source)
        derived_from_id = record.get("derived_from_id")
        source_audio_sha256 = _sha256(record, "source_audio_sha256", source)
        transform_sha256 = _sha256(record, "transform_recipe_sha256", source)
        if transform_sha256 != provenance["transform_recipe_sha256"]:
            raise CollectionValidationError(
                f"{source}: transform_recipe_sha256 does not match provenance group"
            )
        if lineage_kind == "source":
            if derived_from_id is not None:
                raise CollectionValidationError(
                    f"{source}: source lineage derived_from_id must be null"
                )
            if source_audio_sha256 != audio_sha256:
                raise CollectionValidationError(
                    f"{source}: source lineage source_audio_sha256 must equal audio_sha256"
                )
        elif lineage_kind == "derived":
            if (
                not isinstance(derived_from_id, str)
                or ID_PATTERN.fullmatch(derived_from_id) is None
            ):
                raise CollectionValidationError(
                    f"{source}: derived lineage needs a stable derived_from_id"
                )
        else:
            raise CollectionValidationError(
                f"{source}: lineage_kind must be source or derived"
            )

        records.append(
            _ValidatedRecord(
                record=record,
                source=source,
                actual_duration=actual_duration,
            )
        )

    if len(records) != spec.record_count:
        raise CollectionValidationError(
            f"{manifest_path}: record_count mismatch; "
            f"expected {spec.record_count}, got {len(records)}"
        )
    return records


def load_validated_collection(
    descriptor_path: Path,
    collection_root: Path,
    audio_root: Path | None = None,
) -> ValidatedCollection:
    """Custodian load of a frozen collection, including sealed references."""

    descriptor = load_collection_descriptor(descriptor_path)
    audio_root = collection_root if audio_root is None else audio_root

    validated_records: list[_ValidatedRecord] = []
    for spec in descriptor.manifests:
        validated_records.extend(
            _load_manifest_records(descriptor, spec, collection_root, audio_root)
        )
    all_records = [validated.record for validated in validated_records]

    utterance_ids: dict[str, str] = {}
    audio_hash_ids: dict[str, str] = {}
    isolation: dict[str, dict[str, str]] = {key: {} for key in ISOLATION_KEYS}
    records_by_id: dict[str, _ValidatedRecord] = {}
    used_provenance: set[str] = set()
    used_rights: set[str] = set()
    for validated in validated_records:
        record = validated.record
        source = validated.source
        utterance_id = str(record["id"])
        if utterance_id in utterance_ids:
            raise CollectionValidationError(
                f"{source}: duplicate utterance id {utterance_id!r}; "
                f"first seen at {utterance_ids[utterance_id]}"
            )
        utterance_ids[utterance_id] = source
        records_by_id[utterance_id] = validated

        audio_sha256 = str(record["audio_sha256"])
        if audio_sha256 in audio_hash_ids:
            raise CollectionValidationError(
                f"{source}: duplicate audio_sha256 also used by "
                f"{audio_hash_ids[audio_sha256]!r}"
            )
        audio_hash_ids[audio_sha256] = utterance_id

        split = str(record["split"])
        for key in ISOLATION_KEYS:
            value = str(record[key])
            previous_split = isolation[key].get(value)
            if previous_split is not None and previous_split != split:
                raise CollectionValidationError(
                    f"{source}: {key} {value!r} crosses splits "
                    f"{previous_split!r} and {split!r}"
                )
            isolation[key][value] = split
        used_provenance.add(str(record["provenance_group_id"]))
        used_rights.add(str(record["rights_group_id"]))

    for validated in validated_records:
        record = validated.record
        if record["lineage_kind"] != "derived":
            continue
        parent = records_by_id.get(str(record["derived_from_id"]))
        if parent is None:
            continue
        for field in (
            "split",
            "speaker_id",
            "session_id",
            "source_recording_id",
            "lineage_group_id",
            "source_audio_sha256",
        ):
            if record[field] != parent.record[field]:
                raise CollectionValidationError(
                    f"{validated.source}: derived record and parent disagree on {field}"
                )

    unused_provenance = sorted(set(descriptor.provenance_groups) - used_provenance)
    unused_rights = sorted(set(descriptor.rights_groups) - used_rights)
    if unused_provenance:
        raise CollectionValidationError(
            "collection has unused provenance group(s): "
            + ", ".join(unused_provenance)
        )
    if unused_rights:
        raise CollectionValidationError(
            "collection has unused rights group(s): " + ", ".join(unused_rights)
        )

    manifest_inventory_sha256 = _manifest_inventory_sha256(descriptor.manifests)
    record_cluster_inventory_sha256 = _record_cluster_inventory_sha256(all_records)
    actual_dedup_sha256 = _validate_dedup_report(
        descriptor,
        collection_root,
        manifest_inventory_sha256,
        record_cluster_inventory_sha256,
        len(all_records),
    )

    sealed_input_projection = canonical_json_bytes(
        _sealed_input_projection_document(descriptor, all_records)
    )
    sealed_reference_projection = canonical_json_bytes(
        _sealed_reference_projection_document(descriptor, all_records)
    )
    input_projection_sha256 = sha256_bytes(sealed_input_projection)
    reference_projection_sha256 = sha256_bytes(sealed_reference_projection)
    if input_projection_sha256 != descriptor.blind["input_projection_sha256"]:
        raise CollectionValidationError(
            f"{descriptor.path}: blind.input_projection_sha256 does not match "
            "validated sealed records"
        )
    if reference_projection_sha256 != descriptor.blind["reference_projection_sha256"]:
        raise CollectionValidationError(
            f"{descriptor.path}: blind.reference_projection_sha256 does not match "
            "validated sealed records"
        )

    try:
        scoring_record_input_sha256 = record_input_sha256(
            all_records,
            slice_fields=DEFAULT_SLICE_FIELDS,
        )
    except RecordIdentityError as error:
        raise CollectionValidationError(
            f"collection scoring record identity is invalid: {error}"
        ) from error

    split_counts = {split: 0 for split in ALLOWED_SPLITS}
    scenario_counts = {tag: 0 for tag in sorted(SCENARIO_TAGS)}
    access_counts = {access: 0 for access in sorted(ACCESS_CLASSES)}
    excluded_count = 0
    audio_seconds = 0.0
    for validated in validated_records:
        record = validated.record
        split_counts[str(record["split"])] += 1
        access_counts[str(record["access_class"])] += 1
        for tag in record["scenario_tags"]:
            scenario_counts[str(tag)] += 1
        if record["evaluation_status"] == "excluded":
            excluded_count += 1
        audio_seconds += validated.actual_duration

    scenario_counts = {
        tag: count for tag, count in scenario_counts.items() if count > 0
    }
    access_counts = {
        access: count for access, count in access_counts.items() if count > 0
    }
    summary = {
        "schema_version": 1,
        "kind": "asr-collection-validation",
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "normalizer_version": descriptor.normalizer_version,
        "mer_tokenizer_version": descriptor.mer_tokenizer_version,
        "scenario_taxonomy_version": SCENARIO_TAXONOMY_VERSION,
        "data_sha256": descriptor.raw_sha256,
        "manifest_inventory_sha256": manifest_inventory_sha256,
        "record_identity_version": RECORD_IDENTITY_VERSION,
        "record_input_sha256": scoring_record_input_sha256,
        "dedup_report_sha256": actual_dedup_sha256,
        "dedup": {
            "method": descriptor.dedup["method"],
            "version": descriptor.dedup["version"],
            "threshold": descriptor.dedup["threshold"],
            "config_sha256": descriptor.dedup["config_sha256"],
            "manifest_inventory_sha256": manifest_inventory_sha256,
            "record_count": len(all_records),
            "record_cluster_inventory_sha256": record_cluster_inventory_sha256,
            "report_sha256": actual_dedup_sha256,
        },
        "input_projection_sha256": input_projection_sha256,
        "reference_projection_sha256": reference_projection_sha256,
        "record_count": len(all_records),
        "included_count": len(all_records) - excluded_count,
        "excluded_count": excluded_count,
        "audio_seconds": round(audio_seconds, 9),
        "split_counts": split_counts,
        "scenario_counts": scenario_counts,
        "access_counts": access_counts,
    }
    return ValidatedCollection(
        summary=summary,
        records=tuple(_copy_record(record) for record in all_records),
        _sealed_input_projection=sealed_input_projection,
    )


def build_sealed_input_projection(collection: ValidatedCollection) -> bytes:
    """Return canonical audio-only bytes for a sealed-blind decode runner."""

    if not isinstance(collection, ValidatedCollection):
        raise TypeError("collection must be a ValidatedCollection")
    return bytes(collection._sealed_input_projection)


def validate_collection(
    descriptor_path: Path,
    collection_root: Path,
    audio_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one frozen collection and return a canonicalizable summary."""

    return load_validated_collection(
        descriptor_path,
        collection_root,
        audio_root,
    ).summary
