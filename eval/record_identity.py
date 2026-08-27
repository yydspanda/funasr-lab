"""Shared ordered-record identity for collection validation and core scoring."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SLICE_FIELDS = ("split", "scenario_tags")
RECORD_IDENTITY_VERSION = "eval-core-record-input-v1"


class RecordIdentityError(ValueError):
    """Raised when records cannot form one unambiguous scoring identity."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        document = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecordIdentityError(f"record identity is not canonical JSON: {exc}") from exc
    return (document + "\n").encode("utf-8")


def normalize_slice_fields(slice_fields: Sequence[str]) -> tuple[str, ...]:
    if isinstance(slice_fields, (str, bytes, bytearray)) or not isinstance(
        slice_fields, Sequence
    ):
        raise RecordIdentityError("slice_fields must be a sequence of strings")
    requested: list[str] = []
    for field in slice_fields:
        if not isinstance(field, str) or not field:
            raise RecordIdentityError("slice_fields must contain non-empty strings")
        requested.append(field)
    if not requested:
        raise RecordIdentityError("slice_fields must contain at least one field")
    if len(set(requested)) != len(requested):
        raise RecordIdentityError("slice_fields must not contain duplicates")
    return tuple(sorted(requested))


def normalized_slice_values(
    record: Mapping[str, Any], field: str, context: str
) -> tuple[str, ...]:
    if field not in record:
        raise RecordIdentityError(f"{context} is missing slice field {field!r}")
    raw_value = record[field]
    if isinstance(raw_value, str):
        values = (raw_value,)
    elif isinstance(raw_value, Sequence) and not isinstance(
        raw_value, (str, bytes, bytearray)
    ):
        values = tuple(raw_value)
    else:
        raise RecordIdentityError(
            f"{context}.{field} must be a string or sequence of strings"
        )
    if not values:
        raise RecordIdentityError(f"{context}.{field} must not be empty")
    for value in values:
        if not isinstance(value, str) or not value:
            raise RecordIdentityError(
                f"{context}.{field} must contain non-empty strings"
            )
    if len(set(values)) != len(values):
        raise RecordIdentityError(
            f"{context}.{field} must not contain duplicate values"
        )
    return tuple(sorted(values))


def record_input_projection(
    records: Sequence[Mapping[str, Any]],
    *,
    slice_fields: Sequence[str] = DEFAULT_SLICE_FIELDS,
) -> list[dict[str, Any]]:
    """Project exactly the ordered record fields consumed by core scoring."""

    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise RecordIdentityError("records must be an ordered sequence")
    if not records:
        raise RecordIdentityError("records must contain at least one item")
    normalized_fields = normalize_slice_fields(slice_fields)
    projection: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        context = f"records[{index}]"
        if not isinstance(record, Mapping):
            raise RecordIdentityError(f"{context} must be a mapping")
        utterance_id = record.get("id")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise RecordIdentityError(f"{context}.id must be a non-empty string")
        raw_text = record.get("raw_text")
        if not isinstance(raw_text, str):
            raise RecordIdentityError(f"{context}.raw_text must be a string")
        evaluation_status = record.get("evaluation_status", "included")
        if evaluation_status not in {"included", "excluded"}:
            raise RecordIdentityError(
                f"{context}.evaluation_status must be included or excluded"
            )
        exclusion_reason = record.get("exclusion_reason")
        if evaluation_status == "included" and exclusion_reason is not None:
            raise RecordIdentityError(
                f"{context}.exclusion_reason must be null for an included record"
            )
        if evaluation_status == "excluded" and (
            not isinstance(exclusion_reason, str) or not exclusion_reason
        ):
            raise RecordIdentityError(
                f"{context}.exclusion_reason must identify an excluded record"
            )
        projection.append(
            {
                "id": utterance_id,
                "raw_text": raw_text,
                "evaluation_status": evaluation_status,
                "exclusion_reason": exclusion_reason,
                "slices": [
                    {
                        "dimension": field,
                        "values": list(normalized_slice_values(record, field, context)),
                    }
                    for field in normalized_fields
                ],
            }
        )
    return projection


def record_input_sha256(
    records: Sequence[Mapping[str, Any]],
    *,
    slice_fields: Sequence[str] = DEFAULT_SLICE_FIELDS,
) -> str:
    """Hash the canonical ordered projection consumed by core scoring."""

    payload = canonical_json_bytes(
        record_input_projection(records, slice_fields=slice_fields)
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "DEFAULT_SLICE_FIELDS",
    "RECORD_IDENTITY_VERSION",
    "RecordIdentityError",
    "canonical_json_bytes",
    "normalize_slice_fields",
    "normalized_slice_values",
    "record_input_projection",
    "record_input_sha256",
]
