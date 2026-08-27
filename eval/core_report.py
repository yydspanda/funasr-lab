"""Deterministic, performance-free evaluation core reports.

The core report binds frozen records to pure prediction text and scoring
results.  Runtime measurements, paths, commands, exception messages, and wall
clock timestamps belong in a separate execution envelope so replaying the same
scoring inputs produces identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from eval.normalizers import NORMALIZER_VERSION
from eval.normalizers import normalize_content
from eval.record_identity import DEFAULT_SLICE_FIELDS
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import RecordIdentityError
from eval.record_identity import normalized_slice_values
from eval.record_identity import record_input_sha256
from eval.scoring import ALIGNMENT_VERSION
from eval.scoring import MER_TOKENIZER_VERSION
from eval.scoring import EditCounts
from eval.scoring import ScoringResult
from eval.scoring import cer_score
from eval.scoring import cer_units
from eval.scoring import mer_score
from eval.scoring import mixed_units

if TYPE_CHECKING:
    from eval.collection import ValidatedCollection


CORE_SCHEMA_VERSION = 2
CORE_KIND = "asr-evaluation-core"
CORE_SUMMARY_KIND = "asr-evaluation-core-summary"
RATE_DECIMAL_PLACES = 12

IDENTITY_HYPOTHESIS_ADAPTER_VERSION = "identity-v1"
SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION = "sensevoice-control-tags-v1"
HYPOTHESIS_ADAPTER_VERSIONS = frozenset(
    {
        IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
        SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
    }
)
SENSEVOICE_TAG_PATTERN = re.compile(r"<\|[^<>]*?\|>")

SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
ITEM_STATUSES = frozenset({"ok", "failed", "empty", "excluded"})
_CORE_SLICE_FIELDS = tuple(sorted(DEFAULT_SLICE_FIELDS))


class CoreReportValidationError(ValueError):
    """Raised when core-report inputs or a report violate the frozen contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    document = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (document + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CoreReportValidationError(f"{context} must be a mapping")
    return value


def _require_string(
    value: Any,
    context: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise CoreReportValidationError(f"{context} must be a string")
    if not allow_empty and not value:
        raise CoreReportValidationError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: Any, context: str) -> str:
    digest = _require_string(value, context)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise CoreReportValidationError(
            f"{context} must use sha256:<64 lowercase hex characters>"
        )
    return digest


def _require_reason_code(value: Any, context: str) -> str:
    reason_code = _require_string(value, context)
    if REASON_CODE_PATTERN.fullmatch(reason_code) is None:
        raise CoreReportValidationError(
            f"{context} must be a stable lowercase snake_case code"
        )
    return reason_code


def _require_nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoreReportValidationError(
            f"{context} must be a non-negative integer"
        )
    return value


def _validate_evaluation_scope(
    value: Any, context: str = "evaluation_scope"
) -> Mapping[str, Any]:
    """Validate the versioned collection-owned evaluation scope."""

    scope = _require_mapping(value, context)
    kind = scope.get("kind")
    if kind == "collection":
        if set(scope) != {"kind"}:
            raise CoreReportValidationError(
                f"{context} collection scope must contain exactly kind"
            )
        return scope
    if kind == "split":
        if set(scope) != {"kind", "split"}:
            raise CoreReportValidationError(
                f"{context} split scope must contain exactly kind and split"
            )
        split = _require_string(scope["split"], f"{context}.split")
        # Delayed import preserves the lightweight scoring/identity dependency
        # direction while making the split vocabulary collection-owned.
        from eval.collection import ALLOWED_SPLITS

        if split not in ALLOWED_SPLITS:
            raise CoreReportValidationError(
                f"{context}.split must be one of {list(ALLOWED_SPLITS)!r}"
            )
        return scope
    raise CoreReportValidationError(
        f"{context}.kind must be collection or split"
    )


def _decimal_ratio(numerator: int, denominator: int) -> str:
    """Render one deterministic half-up decimal without floating point."""

    scale = 10**RATE_DECIMAL_PLACES
    scaled, remainder = divmod(numerator * scale, denominator)
    if remainder * 2 >= denominator:
        scaled += 1
    whole, fraction = divmod(scaled, scale)
    if fraction == 0:
        return str(whole)
    fraction_text = f"{fraction:0{RATE_DECIMAL_PLACES}d}".rstrip("0")
    return f"{whole}.{fraction_text}"


def _metric_document(result: ScoringResult) -> dict[str, Any]:
    counts = result.counts
    errors = counts.total
    if result.reference_units == 0:
        rate = None
        rate_decimal = None
    else:
        rate = {
            "numerator": errors,
            "denominator": result.reference_units,
        }
        rate_decimal = _decimal_ratio(errors, result.reference_units)
    return {
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "errors": errors,
        "reference_units": result.reference_units,
        "rate": rate,
        "rate_decimal": rate_decimal,
    }


def _sum_metric_documents(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = EditCounts()
    reference_units = 0
    for metric in metrics:
        counts += EditCounts(
            substitutions=int(metric["substitutions"]),
            deletions=int(metric["deletions"]),
            insertions=int(metric["insertions"]),
        )
        reference_units += int(metric["reference_units"])
    return _metric_document(
        ScoringResult(counts=counts, reference_units=reference_units)
    )


def _text_views(raw_text: str, display_text: str) -> dict[str, Any]:
    return {
        "raw": raw_text,
        "content": normalize_content(display_text),
        "display": display_text,
        "cer_units": list(cer_units(display_text)),
        "mer_units": list(mixed_units(display_text)),
    }


def adapt_hypothesis(raw_text: str, adapter_version: str) -> str:
    """Derive the only scoring/display hypothesis allowed by an adapter."""

    if adapter_version == IDENTITY_HYPOTHESIS_ADAPTER_VERSION:
        return raw_text
    if adapter_version == SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION:
        return SENSEVOICE_TAG_PATTERN.sub("", raw_text).strip()
    raise CoreReportValidationError(
        f"unsupported hypothesis_adapter_version {adapter_version!r}"
    )


def _item_count_document(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    statuses = [str(item["status"]) for item in items]
    scored_items = [item for item in items if item["status"] != "excluded"]
    return {
        "utterance_count": len(items),
        "scored_count": len(scored_items),
        "ok_count": statuses.count("ok"),
        "failed_count": statuses.count("failed"),
        "empty_count": statuses.count("empty"),
        "excluded_count": statuses.count("excluded"),
        "zero_reference_count": sum(
            1
            for item in scored_items
            if item["cer"]["reference_units"] == 0
        ),
    }


def _aggregate_document(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored_items = [item for item in items if item["status"] != "excluded"]
    return {
        "cer": _sum_metric_documents([item["cer"] for item in scored_items]),
        "mer": _sum_metric_documents([item["mer"] for item in scored_items]),
    }


def _slice_reports(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    memberships: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in items:
        for membership in item["slices"]:
            key = (str(membership["dimension"]), str(membership["value"]))
            memberships.setdefault(key, []).append(item)

    reports: list[dict[str, Any]] = []
    for (dimension, value), members in sorted(memberships.items()):
        reports.append(
            {
                "dimension": dimension,
                "value": value,
                "item_ids": [item["id"] for item in members],
                "counts": _item_count_document(members),
                "aggregate": _aggregate_document(members),
            }
        )
    return reports


def _prediction_projection(item: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis = item["hypothesis"]
    if hypothesis is None:
        return {
            "id": item["id"],
            "status": "excluded",
            "reason_code": item["reason_code"],
        }
    return {
        "id": item["id"],
        "raw_text": hypothesis["raw"],
        "display_text": hypothesis["display"],
        "status": item["status"],
        "reason_code": item["reason_code"],
    }


def _record_identity_inputs(
    items: Sequence[Mapping[str, Any]], slice_fields: Sequence[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        values_by_dimension = {field: [] for field in slice_fields}
        for membership in item["slices"]:
            values_by_dimension[str(membership["dimension"])].append(
                str(membership["value"])
            )
        excluded = item["status"] == "excluded"
        records.append(
            {
                "id": item["id"],
                "raw_text": item["reference"]["raw"],
                "evaluation_status": "excluded" if excluded else "included",
                "exclusion_reason": item["reason_code"] if excluded else None,
                **values_by_dimension,
            }
        )
    return records


def _build_core_report(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    data_sha256: str,
    expected_record_input_sha256: str,
    evaluation_scope: Mapping[str, Any],
    hypothesis_adapter_version: str = IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
) -> dict[str, Any]:
    """Build a deterministic core report from already joined collection inputs.

    Records require ``id``, ``raw_text``, and every selected slice field.  A
    prediction requires ``id`` and exact decoder ``raw_text``.  A frozen,
    report-level hypothesis adapter derives display/scoring text; callers may
    not supply ``display_text``.  Missing predictions become an empty failed
    hypothesis with the stable ``missing_prediction`` reason.  Duplicate or
    extra predictions are fatal.  Prediction fields outside ``id``,
    ``raw_text``, ``status``, and ``reason_code`` are execution-envelope
    metadata and are deliberately ignored by both the core document and its
    prediction-input hash.

    A record may be predeclared with ``evaluation_status: "excluded"`` and a
    stable snake-case ``exclusion_reason``.  Excluded records remain in counts
    and slices but have no hypothesis or scoring denominator, and accepting a
    prediction for one is an error.  Predictions cannot dynamically exclude an
    item.  Frozen reference ``raw_text`` is always the reference
    display/scoring text; the hypothesis adapter alone derives decoder-cleaned
    display text.
    """

    data_digest = _require_sha256(data_sha256, "data_sha256")
    expected_record_digest = _require_sha256(
        expected_record_input_sha256, "expected_record_input_sha256"
    )
    scope = _validate_evaluation_scope(evaluation_scope)
    adapter_version = _require_string(
        hypothesis_adapter_version, "hypothesis_adapter_version"
    )
    if adapter_version not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CoreReportValidationError(
            f"unsupported hypothesis_adapter_version {adapter_version!r}"
        )
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise CoreReportValidationError("records must be an ordered sequence")
    if isinstance(predictions, (str, bytes, bytearray)) or not isinstance(
        predictions, Sequence
    ):
        raise CoreReportValidationError("predictions must be a sequence")

    normalized_slice_fields = _CORE_SLICE_FIELDS
    try:
        actual_record_digest = record_input_sha256(
            records, slice_fields=DEFAULT_SLICE_FIELDS
        )
    except RecordIdentityError as exc:
        raise CoreReportValidationError(str(exc)) from exc
    if actual_record_digest != expected_record_digest:
        raise CoreReportValidationError(
            "expected_record_input_sha256 does not match the ordered scoring records"
        )

    ordered_records: list[tuple[str, Mapping[str, Any]]] = []
    record_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        context = f"records[{index}]"
        record = _require_mapping(raw_record, context)
        utterance_id = _require_string(record.get("id"), f"{context}.id")
        if utterance_id in record_ids:
            raise CoreReportValidationError(
                f"duplicate record id {utterance_id!r}"
            )
        record_ids.add(utterance_id)
        _require_string(
            record.get("raw_text"), f"{context}.raw_text", allow_empty=True
        )
        ordered_records.append((utterance_id, record))

    if not ordered_records:
        raise CoreReportValidationError("records must contain at least one item")

    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_prediction in enumerate(predictions):
        context = f"predictions[{index}]"
        prediction = _require_mapping(raw_prediction, context)
        utterance_id = _require_string(prediction.get("id"), f"{context}.id")
        if utterance_id in prediction_by_id:
            raise CoreReportValidationError(
                f"duplicate prediction id {utterance_id!r}"
            )
        prediction_by_id[utterance_id] = prediction

    extra_ids = sorted(set(prediction_by_id) - record_ids)
    if extra_ids:
        raise CoreReportValidationError(
            "predictions contain ids absent from frozen records: "
            + ", ".join(extra_ids)
        )

    items: list[dict[str, Any]] = []
    for index, (utterance_id, record) in enumerate(ordered_records):
        context = f"records[{index}]"
        reference_raw = _require_string(
            record.get("raw_text"), f"{context}.raw_text", allow_empty=True
        )
        reference_display = reference_raw
        memberships = [
            {"dimension": field, "value": value}
            for field in normalized_slice_fields
            for value in normalized_slice_values(record, field, context)
        ]
        memberships.sort(
            key=lambda membership: (membership["dimension"], membership["value"])
        )

        evaluation_status = record.get("evaluation_status", "included")
        if evaluation_status not in {"included", "excluded"}:
            raise CoreReportValidationError(
                f"{context}.evaluation_status must be included or excluded"
            )
        exclusion_reason = record.get("exclusion_reason")
        if evaluation_status == "included" and exclusion_reason is not None:
            raise CoreReportValidationError(
                f"{context}.exclusion_reason must be null for an included record"
            )
        if evaluation_status == "excluded":
            reason_code = _require_reason_code(
                exclusion_reason,
                f"{context}.exclusion_reason",
            )
            if utterance_id in prediction_by_id:
                raise CoreReportValidationError(
                    f"excluded record {utterance_id!r} must not have a prediction"
                )
            items.append(
                {
                    "id": utterance_id,
                    "status": "excluded",
                    "reason_code": reason_code,
                    "slices": memberships,
                    "reference": _text_views(reference_raw, reference_display),
                    "hypothesis": None,
                    "cer": None,
                    "mer": None,
                }
            )
            continue

        prediction = prediction_by_id.get(utterance_id)
        if prediction is None:
            hypothesis_raw = ""
            hypothesis_display = ""
            status = "failed"
            reason_code: str | None = "missing_prediction"
        else:
            prediction_context = f"prediction[{utterance_id!r}]"
            if "display_text" in prediction:
                raise CoreReportValidationError(
                    f"{prediction_context}.display_text is caller-controlled; "
                    "select a versioned hypothesis adapter instead"
                )
            hypothesis_raw = _require_string(
                prediction.get("raw_text"),
                f"{prediction_context}.raw_text",
                allow_empty=True,
            )
            hypothesis_display = adapt_hypothesis(
                hypothesis_raw, adapter_version
            )
            raw_status = prediction.get("status")
            if raw_status is None:
                status = "empty" if not hypothesis_display else "ok"
            else:
                status = _require_string(raw_status, f"{prediction_context}.status")
                if status not in ITEM_STATUSES - {"excluded"}:
                    raise CoreReportValidationError(
                        f"{prediction_context}.status must be ok, failed, or empty"
                    )

            raw_reason_code = prediction.get("reason_code")
            if status == "ok":
                if not hypothesis_display:
                    raise CoreReportValidationError(
                        f"{prediction_context}: ok prediction must not be empty"
                    )
                if raw_reason_code is not None:
                    raise CoreReportValidationError(
                        f"{prediction_context}: ok prediction must not have reason_code"
                    )
                reason_code = None
            elif status == "empty":
                if hypothesis_display:
                    raise CoreReportValidationError(
                        f"{prediction_context}: empty prediction must adapt to "
                        "empty text"
                    )
                reason_code = (
                    "empty_hypothesis"
                    if raw_reason_code is None
                    else _require_reason_code(
                        raw_reason_code, f"{prediction_context}.reason_code"
                    )
                )
            else:
                if hypothesis_raw or hypothesis_display:
                    raise CoreReportValidationError(
                        f"{prediction_context}: failed prediction must contain no text"
                    )
                reason_code = _require_reason_code(
                    raw_reason_code, f"{prediction_context}.reason_code"
                )

        reference = _text_views(reference_raw, reference_display)
        hypothesis = _text_views(hypothesis_raw, hypothesis_display)
        items.append(
            {
                "id": utterance_id,
                "status": status,
                "reason_code": reason_code,
                "slices": memberships,
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": _metric_document(
                    cer_score(reference_display, hypothesis_display)
                ),
                "mer": _metric_document(
                    mer_score(reference_display, hypothesis_display)
                ),
            }
        )

    prediction_input_sha256 = _sha256_bytes(
        _canonical_json_bytes([_prediction_projection(item) for item in items])
    )
    report = {
        "schema_version": CORE_SCHEMA_VERSION,
        "kind": CORE_KIND,
        "access_class": "restricted",
        "provenance": {
            "data_sha256": data_digest,
            "record_identity_version": RECORD_IDENTITY_VERSION,
            "record_input_sha256": actual_record_digest,
            "prediction_input_sha256": prediction_input_sha256,
        },
        "scoring": {
            "normalizer_version": NORMALIZER_VERSION,
            "mer_tokenizer_version": MER_TOKENIZER_VERSION,
            "alignment_version": ALIGNMENT_VERSION,
            "hypothesis_adapter_version": adapter_version,
            "rate_decimal_places": RATE_DECIMAL_PLACES,
        },
        "configuration": {
            "slice_fields": list(normalized_slice_fields),
            "evaluation_scope": dict(scope),
        },
        "counts": _item_count_document(items),
        "aggregate": _aggregate_document(items),
        "slices": _slice_reports(items),
        "items": items,
    }
    validate_core_report(report)
    return report


def _validated_collection_inputs(
    collection: Any,
) -> tuple[Sequence[Mapping[str, Any]], str, str]:
    """Extract and verify the atomic collection inputs consumed by scoring."""

    # Delayed to keep record identity/scoring usable without importing the
    # collection loader and to make the dependency direction explicit.
    from eval.collection import ValidatedCollection

    if not isinstance(collection, ValidatedCollection):
        raise TypeError("collection must be an eval.collection.ValidatedCollection")
    summary = _require_mapping(collection.summary, "collection.summary")
    data_digest = _require_sha256(
        summary.get("data_sha256"), "collection.summary.data_sha256"
    )
    if summary.get("record_identity_version") != RECORD_IDENTITY_VERSION:
        raise CoreReportValidationError(
            "collection.summary.record_identity_version is unsupported"
        )
    expected_record_digest = _require_sha256(
        summary.get("record_input_sha256"),
        "collection.summary.record_input_sha256",
    )
    records = collection.records
    try:
        actual_record_digest = record_input_sha256(
            records, slice_fields=DEFAULT_SLICE_FIELDS
        )
    except RecordIdentityError as exc:
        raise CoreReportValidationError(str(exc)) from exc
    if actual_record_digest != expected_record_digest:
        raise CoreReportValidationError(
            "collection.summary.record_input_sha256 does not match collection.records"
        )
    return records, data_digest, expected_record_digest


def _scoped_collection_inputs(
    collection: Any,
    evaluation_scope: Mapping[str, Any],
) -> tuple[Sequence[Mapping[str, Any]], str, str]:
    """Select ordered scoring records only through a validated collection."""

    all_records, data_digest, collection_record_digest = (
        _validated_collection_inputs(collection)
    )
    scope = _validate_evaluation_scope(evaluation_scope)
    if scope["kind"] == "collection":
        return all_records, data_digest, collection_record_digest

    split = str(scope["split"])
    selected_records = tuple(
        record for record in all_records if record.get("split") == split
    )
    if not selected_records:
        raise CoreReportValidationError(
            f"validated collection contains no records for split {split!r}"
        )
    try:
        selected_record_digest = record_input_sha256(
            selected_records, slice_fields=DEFAULT_SLICE_FIELDS
        )
    except RecordIdentityError as exc:
        raise CoreReportValidationError(str(exc)) from exc
    return selected_records, data_digest, selected_record_digest


def validate_core_report_for_collection(
    report: Mapping[str, Any], collection: ValidatedCollection
) -> None:
    """Validate a core report and bind it to one validated collection."""

    validate_core_report(report)
    evaluation_scope = report["configuration"]["evaluation_scope"]
    records, data_digest, record_digest = _scoped_collection_inputs(
        collection, evaluation_scope
    )
    provenance = report["provenance"]
    if provenance["data_sha256"] != data_digest:
        raise CoreReportValidationError(
            "core provenance.data_sha256 does not match collection.summary"
        )
    if provenance["record_identity_version"] != RECORD_IDENTITY_VERSION:
        raise CoreReportValidationError(
            "core record identity version does not match the collection contract"
        )
    if provenance["record_input_sha256"] != record_digest:
        raise CoreReportValidationError(
            "core provenance.record_input_sha256 does not match the selected "
            "collection scope"
        )

    # validate_core_report binds the digest to the emitted items; this explicit
    # recomputation documents that the same digest also covers collection order.
    try:
        collection_record_digest = record_input_sha256(
            records, slice_fields=DEFAULT_SLICE_FIELDS
        )
    except RecordIdentityError as exc:  # defensive; extracted above
        raise CoreReportValidationError(str(exc)) from exc
    if provenance["record_input_sha256"] != collection_record_digest:
        raise CoreReportValidationError(
            "core record inputs do not match the selected validated collection scope"
        )


def build_core_report(
    collection: ValidatedCollection,
    predictions: Sequence[Mapping[str, Any]],
    *,
    hypothesis_adapter_version: str = IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
) -> dict[str, Any]:
    """Build a core report atomically from one validated collection.

    Dataset identity and ordered scoring-record identity are taken only from
    ``collection.summary`` and ``collection.records``.  Callers cannot pair an
    unrelated data digest, record sequence, or reduced slice configuration.
    """

    records, data_digest, record_digest = _validated_collection_inputs(collection)
    evaluation_scope = {"kind": "collection"}
    report = _build_core_report(
        records,
        predictions,
        data_sha256=data_digest,
        expected_record_input_sha256=record_digest,
        evaluation_scope=evaluation_scope,
        hypothesis_adapter_version=hypothesis_adapter_version,
    )
    validate_core_report_for_collection(report, collection)
    return report


def build_split_core_report(
    collection: ValidatedCollection,
    predictions: Sequence[Mapping[str, Any]],
    *,
    split: str,
    hypothesis_adapter_version: str = IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
) -> dict[str, Any]:
    """Build a core report for one collection-owned split in manifest order.

    ``predictions`` must cover only records in the selected split; prediction
    IDs from another split are rejected rather than silently discarded.
    Dataset identity remains the full collection descriptor hash, while record
    identity binds only the ordered records selected by ``split``.
    """

    evaluation_scope = {"kind": "split", "split": split}
    records, data_digest, record_digest = _scoped_collection_inputs(
        collection, evaluation_scope
    )
    report = _build_core_report(
        records,
        predictions,
        data_sha256=data_digest,
        expected_record_input_sha256=record_digest,
        evaluation_scope=evaluation_scope,
        hypothesis_adapter_version=hypothesis_adapter_version,
    )
    validate_core_report_for_collection(report, collection)
    return report


def _validate_metric(metric: Any, context: str) -> Mapping[str, Any]:
    document = _require_mapping(metric, context)
    expected_keys = {
        "substitutions",
        "deletions",
        "insertions",
        "errors",
        "reference_units",
        "rate",
        "rate_decimal",
    }
    if set(document) != expected_keys:
        raise CoreReportValidationError(
            f"{context} must contain exactly {sorted(expected_keys)!r}"
        )
    substitutions = _require_nonnegative_integer(
        document["substitutions"], f"{context}.substitutions"
    )
    deletions = _require_nonnegative_integer(
        document["deletions"], f"{context}.deletions"
    )
    insertions = _require_nonnegative_integer(
        document["insertions"], f"{context}.insertions"
    )
    errors = _require_nonnegative_integer(document["errors"], f"{context}.errors")
    reference_units = _require_nonnegative_integer(
        document["reference_units"], f"{context}.reference_units"
    )
    if errors != substitutions + deletions + insertions:
        raise CoreReportValidationError(
            f"{context}.errors must equal substitutions + deletions + insertions"
        )

    rate = document["rate"]
    rate_decimal = document["rate_decimal"]
    if reference_units == 0:
        if rate is not None or rate_decimal is not None:
            raise CoreReportValidationError(
                f"{context}: zero-reference rate and rate_decimal must be null"
            )
    else:
        rate_document = _require_mapping(rate, f"{context}.rate")
        if set(rate_document) != {"numerator", "denominator"}:
            raise CoreReportValidationError(
                f"{context}.rate must contain exactly numerator and denominator"
            )
        _require_nonnegative_integer(
            rate_document["numerator"], f"{context}.rate.numerator"
        )
        denominator = _require_nonnegative_integer(
            rate_document["denominator"], f"{context}.rate.denominator"
        )
        if denominator == 0:
            raise CoreReportValidationError(
                f"{context}.rate.denominator must be greater than zero"
            )
        expected_rate = {"numerator": errors, "denominator": reference_units}
        if dict(rate_document) != expected_rate:
            raise CoreReportValidationError(
                f"{context}.rate must equal the exact errors/reference_units ratio"
            )
        expected_decimal = _decimal_ratio(errors, reference_units)
        if rate_decimal != expected_decimal:
            raise CoreReportValidationError(
                f"{context}.rate_decimal must be {expected_decimal!r}"
            )
    return document


def _validate_views(views: Any, context: str) -> Mapping[str, Any]:
    document = _require_mapping(views, context)
    expected_keys = {"raw", "content", "display", "cer_units", "mer_units"}
    if set(document) != expected_keys:
        raise CoreReportValidationError(
            f"{context} must contain exactly {sorted(expected_keys)!r}"
        )
    raw = _require_string(document["raw"], f"{context}.raw", allow_empty=True)
    display = _require_string(
        document["display"], f"{context}.display", allow_empty=True
    )
    _require_string(document["content"], f"{context}.content", allow_empty=True)
    if document["content"] != normalize_content(display):
        raise CoreReportValidationError(
            f"{context}.content does not match the frozen normalizer"
        )
    if document["cer_units"] != list(cer_units(display)):
        raise CoreReportValidationError(
            f"{context}.cer_units do not match the frozen CER unitizer"
        )
    if document["mer_units"] != list(mixed_units(display)):
        raise CoreReportValidationError(
            f"{context}.mer_units do not match the frozen MER unitizer"
        )
    # Reading raw here is intentional: it is preserved but is not itself scored.
    del raw
    return document


def validate_core_report(report: Mapping[str, Any]) -> None:
    """Validate structure plus scoring/count/aggregate semantics."""

    document = _require_mapping(report, "report")
    expected_top_level = {
        "schema_version",
        "kind",
        "access_class",
        "provenance",
        "scoring",
        "configuration",
        "counts",
        "aggregate",
        "slices",
        "items",
    }
    if set(document) != expected_top_level:
        raise CoreReportValidationError(
            "report contains missing or unknown top-level fields"
        )
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], int)
        or document["schema_version"] != CORE_SCHEMA_VERSION
    ):
        raise CoreReportValidationError("unsupported core report schema_version")
    if document["kind"] != CORE_KIND:
        raise CoreReportValidationError("unsupported core report kind")
    if document["access_class"] != "restricted":
        raise CoreReportValidationError(
            "full core reports must have access_class='restricted'"
        )

    provenance = _require_mapping(document["provenance"], "provenance")
    if set(provenance) != {
        "data_sha256",
        "record_identity_version",
        "record_input_sha256",
        "prediction_input_sha256",
    }:
        raise CoreReportValidationError("provenance contains missing or unknown fields")
    _require_sha256(provenance["data_sha256"], "provenance.data_sha256")
    if provenance["record_identity_version"] != RECORD_IDENTITY_VERSION:
        raise CoreReportValidationError(
            "provenance.record_identity_version is unsupported"
        )
    _require_sha256(
        provenance["record_input_sha256"],
        "provenance.record_input_sha256",
    )
    _require_sha256(
        provenance["prediction_input_sha256"],
        "provenance.prediction_input_sha256",
    )

    scoring = _require_mapping(document["scoring"], "scoring")
    if set(scoring) != {
        "normalizer_version",
        "mer_tokenizer_version",
        "alignment_version",
        "hypothesis_adapter_version",
        "rate_decimal_places",
    }:
        raise CoreReportValidationError("scoring contains missing or unknown fields")
    expected_scoring = {
        "normalizer_version": NORMALIZER_VERSION,
        "mer_tokenizer_version": MER_TOKENIZER_VERSION,
        "alignment_version": ALIGNMENT_VERSION,
    }
    if any(scoring[name] != value for name, value in expected_scoring.items()):
        raise CoreReportValidationError("scoring versions do not match this evaluator")
    adapter_version = scoring["hypothesis_adapter_version"]
    if adapter_version not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CoreReportValidationError("scoring hypothesis adapter is unsupported")
    if scoring["rate_decimal_places"] != RATE_DECIMAL_PLACES:
        raise CoreReportValidationError("scoring rate decimal policy is unsupported")

    configuration = _require_mapping(document["configuration"], "configuration")
    if set(configuration) != {"slice_fields", "evaluation_scope"}:
        raise CoreReportValidationError(
            "configuration must contain exactly slice_fields and evaluation_scope"
        )
    slice_fields = configuration["slice_fields"]
    if slice_fields != list(_CORE_SLICE_FIELDS):
        raise CoreReportValidationError(
            "configuration.slice_fields must equal the frozen collection slice contract"
        )
    evaluation_scope = _validate_evaluation_scope(
        configuration["evaluation_scope"], "configuration.evaluation_scope"
    )

    raw_items = document["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise CoreReportValidationError("items must be a non-empty list")
    items: list[Mapping[str, Any]] = []
    item_ids: set[str] = set()
    expected_item_keys = {
        "id",
        "status",
        "reason_code",
        "slices",
        "reference",
        "hypothesis",
        "cer",
        "mer",
    }
    for index, raw_item in enumerate(raw_items):
        context = f"items[{index}]"
        item = _require_mapping(raw_item, context)
        if set(item) != expected_item_keys:
            raise CoreReportValidationError(
                f"{context} contains missing or unknown fields"
            )
        utterance_id = _require_string(item["id"], f"{context}.id")
        if utterance_id in item_ids:
            raise CoreReportValidationError(f"duplicate item id {utterance_id!r}")
        item_ids.add(utterance_id)
        status = _require_string(item["status"], f"{context}.status")
        if status not in ITEM_STATUSES:
            raise CoreReportValidationError(f"{context}.status is invalid")
        reason_code = item["reason_code"]
        if status == "ok":
            if reason_code is not None:
                raise CoreReportValidationError(
                    f"{context}: ok item must not have reason_code"
                )
        else:
            _require_reason_code(reason_code, f"{context}.reason_code")

        memberships = item["slices"]
        if not isinstance(memberships, list):
            raise CoreReportValidationError(f"{context}.slices must be a list")
        expected_memberships: list[tuple[str, str]] = []
        for membership_index, raw_membership in enumerate(memberships):
            membership = _require_mapping(
                raw_membership, f"{context}.slices[{membership_index}]"
            )
            if set(membership) != {"dimension", "value"}:
                raise CoreReportValidationError(
                    f"{context}.slices[{membership_index}] is malformed"
                )
            dimension = _require_string(
                membership["dimension"],
                f"{context}.slices[{membership_index}].dimension",
            )
            value = _require_string(
                membership["value"],
                f"{context}.slices[{membership_index}].value",
            )
            if dimension not in slice_fields:
                raise CoreReportValidationError(
                    f"{context}.slices contains an unconfigured dimension"
                )
            expected_memberships.append((dimension, value))
        if expected_memberships != sorted(set(expected_memberships)):
            raise CoreReportValidationError(
                f"{context}.slices must be unique and sorted"
            )
        if {dimension for dimension, _ in expected_memberships} != set(slice_fields):
            raise CoreReportValidationError(
                f"{context}.slices must include every configured dimension"
            )
        if evaluation_scope["kind"] == "split":
            split_values = [
                value
                for dimension, value in expected_memberships
                if dimension == "split"
            ]
            if split_values != [evaluation_scope["split"]]:
                raise CoreReportValidationError(
                    f"{context}.slices do not match the configured split scope"
                )

        reference = _validate_views(item["reference"], f"{context}.reference")
        if reference["display"] != reference["raw"]:
            raise CoreReportValidationError(
                f"{context}.reference display must equal frozen raw text"
            )
        if status == "excluded":
            if any(item[field] is not None for field in ("hypothesis", "cer", "mer")):
                raise CoreReportValidationError(
                    f"{context}: excluded item must not have hypothesis or metrics"
                )
        else:
            hypothesis = _validate_views(
                item["hypothesis"], f"{context}.hypothesis"
            )
            if hypothesis["display"] != adapt_hypothesis(
                hypothesis["raw"], str(adapter_version)
            ):
                raise CoreReportValidationError(
                    f"{context}.hypothesis display does not match the frozen adapter"
                )
            cer = _validate_metric(item["cer"], f"{context}.cer")
            mer = _validate_metric(item["mer"], f"{context}.mer")
            expected_cer = _metric_document(
                cer_score(reference["display"], hypothesis["display"])
            )
            expected_mer = _metric_document(
                mer_score(reference["display"], hypothesis["display"])
            )
            if dict(cer) != expected_cer:
                raise CoreReportValidationError(
                    f"{context}.cer does not match the item text"
                )
            if dict(mer) != expected_mer:
                raise CoreReportValidationError(
                    f"{context}.mer does not match the item text"
                )
            if status in {"failed", "empty"} and hypothesis["display"]:
                raise CoreReportValidationError(
                    f"{context}: failed or empty hypothesis must score as empty"
                )
            if status == "failed" and hypothesis["raw"]:
                raise CoreReportValidationError(
                    f"{context}: failed item must not contain raw exception text"
                )
            if status == "ok" and not hypothesis["display"]:
                raise CoreReportValidationError(
                    f"{context}: ok hypothesis must not be empty"
                )
        items.append(item)

    counts = _require_mapping(document["counts"], "counts")
    expected_counts = _item_count_document(items)
    for name in expected_counts:
        if name not in counts:
            raise CoreReportValidationError(f"counts is missing {name}")
        _require_nonnegative_integer(counts[name], f"counts.{name}")
    if dict(counts) != expected_counts:
        raise CoreReportValidationError("counts do not match items")

    aggregate = _require_mapping(document["aggregate"], "aggregate")
    if set(aggregate) != {"cer", "mer"}:
        raise CoreReportValidationError("aggregate must contain exactly cer and mer")
    _validate_metric(aggregate["cer"], "aggregate.cer")
    _validate_metric(aggregate["mer"], "aggregate.mer")
    if dict(aggregate) != _aggregate_document(items):
        raise CoreReportValidationError("aggregate metrics do not match items")

    slices = document["slices"]
    if not isinstance(slices, list):
        raise CoreReportValidationError("slices must be a list")
    if slices != _slice_reports(items):
        raise CoreReportValidationError(
            "slices do not deterministically match item memberships and metrics"
        )

    try:
        expected_record_digest = record_input_sha256(
            _record_identity_inputs(items, slice_fields),
            slice_fields=slice_fields,
        )
    except RecordIdentityError as exc:
        raise CoreReportValidationError(str(exc)) from exc
    if provenance["record_input_sha256"] != expected_record_digest:
        raise CoreReportValidationError(
            "provenance.record_input_sha256 does not match item record inputs"
        )

    expected_prediction_digest = _sha256_bytes(
        _canonical_json_bytes([_prediction_projection(item) for item in items])
    )
    if provenance["prediction_input_sha256"] != expected_prediction_digest:
        raise CoreReportValidationError(
            "provenance.prediction_input_sha256 does not match item inputs"
        )


def canonical_core_bytes(report: Mapping[str, Any]) -> bytes:
    """Return validated canonical UTF-8 JSON bytes for the core report."""

    validate_core_report(report)
    return _canonical_json_bytes(report)


def core_report_sha256(report: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of validated canonical core bytes."""

    return _sha256_bytes(canonical_core_bytes(report))


def _core_summary_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CORE_SCHEMA_VERSION,
        "kind": CORE_SUMMARY_KIND,
        "access_class": "restricted",
        "core_sha256": core_report_sha256(report),
        "scoring": deepcopy(report["scoring"]),
        "configuration": deepcopy(report["configuration"]),
        "counts": deepcopy(report["counts"]),
        "aggregate": deepcopy(report["aggregate"]),
        "slices": [
            {
                "dimension": slice_report["dimension"],
                "value": slice_report["value"],
                "counts": deepcopy(slice_report["counts"]),
                "aggregate": deepcopy(slice_report["aggregate"]),
            }
            for slice_report in report["slices"]
        ],
    }


def build_core_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project a restricted core into a restricted text-free aggregate.

    This document never authorizes release of sealed-set metrics. A future
    release workflow must separately enforce the frozen one-candidate and
    minimum-cell policies and create a distinct public artifact.
    """

    validate_core_report(report)
    summary = _core_summary_projection(report)
    validate_core_summary(summary, source_core=report)
    return summary


def _validate_summary_counts(value: Any, context: str) -> Mapping[str, Any]:
    document = _require_mapping(value, context)
    expected_fields = {
        "utterance_count",
        "scored_count",
        "ok_count",
        "failed_count",
        "empty_count",
        "excluded_count",
        "zero_reference_count",
    }
    if set(document) != expected_fields:
        raise CoreReportValidationError(
            f"{context} contains missing or unknown fields"
        )
    counts = {
        field: _require_nonnegative_integer(document[field], f"{context}.{field}")
        for field in expected_fields
    }
    if counts["scored_count"] != (
        counts["ok_count"] + counts["failed_count"] + counts["empty_count"]
    ):
        raise CoreReportValidationError(
            f"{context}.scored_count does not match item-status counts"
        )
    if counts["utterance_count"] != (
        counts["scored_count"] + counts["excluded_count"]
    ):
        raise CoreReportValidationError(
            f"{context}.utterance_count does not match scored/excluded counts"
        )
    if counts["zero_reference_count"] > counts["scored_count"]:
        raise CoreReportValidationError(
            f"{context}.zero_reference_count exceeds scored_count"
        )
    return document


def _validate_summary_scoring(value: Any) -> Mapping[str, Any]:
    scoring = _require_mapping(value, "scoring")
    if set(scoring) != {
        "normalizer_version",
        "mer_tokenizer_version",
        "alignment_version",
        "hypothesis_adapter_version",
        "rate_decimal_places",
    }:
        raise CoreReportValidationError("scoring contains missing or unknown fields")
    if scoring["normalizer_version"] != NORMALIZER_VERSION:
        raise CoreReportValidationError("summary normalizer_version is unsupported")
    if scoring["mer_tokenizer_version"] != MER_TOKENIZER_VERSION:
        raise CoreReportValidationError("summary MER tokenizer is unsupported")
    if scoring["alignment_version"] != ALIGNMENT_VERSION:
        raise CoreReportValidationError("summary alignment version is unsupported")
    if scoring["hypothesis_adapter_version"] not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise CoreReportValidationError("summary hypothesis adapter is unsupported")
    if scoring["rate_decimal_places"] != RATE_DECIMAL_PLACES:
        raise CoreReportValidationError("summary rate decimal policy is unsupported")
    return scoring


def validate_core_summary(
    summary: Mapping[str, Any],
    *,
    source_core: Mapping[str, Any] | None = None,
) -> None:
    """Validate a restricted summary and optionally bind it to its full core.

    Validation proves shape and core parity, not sealed-result release
    authorization.
    """

    document = _require_mapping(summary, "summary")
    if set(document) != {
        "schema_version",
        "kind",
        "access_class",
        "core_sha256",
        "scoring",
        "configuration",
        "counts",
        "aggregate",
        "slices",
    }:
        raise CoreReportValidationError(
            "core summary contains missing or unknown top-level fields"
        )
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], int)
        or document["schema_version"] != CORE_SCHEMA_VERSION
    ):
        raise CoreReportValidationError("unsupported core summary schema_version")
    if document["kind"] != CORE_SUMMARY_KIND:
        raise CoreReportValidationError("unsupported core summary kind")
    if document["access_class"] != "restricted":
        raise CoreReportValidationError(
            "core summary access_class must be restricted"
        )
    _require_sha256(document["core_sha256"], "core_sha256")

    _validate_summary_scoring(document["scoring"])

    configuration = _require_mapping(document["configuration"], "configuration")
    if set(configuration) != {"slice_fields", "evaluation_scope"}:
        raise CoreReportValidationError(
            "core summary configuration must contain exactly slice_fields and "
            "evaluation_scope"
        )
    slice_fields = configuration["slice_fields"]
    if slice_fields != list(_CORE_SLICE_FIELDS):
        raise CoreReportValidationError(
            "core summary slice_fields must equal the frozen collection "
            "slice contract"
        )
    evaluation_scope = _validate_evaluation_scope(
        configuration["evaluation_scope"], "configuration.evaluation_scope"
    )

    _validate_summary_counts(document["counts"], "counts")
    aggregate = _require_mapping(document["aggregate"], "aggregate")
    if set(aggregate) != {"cer", "mer"}:
        raise CoreReportValidationError("aggregate must contain exactly cer and mer")
    _validate_metric(aggregate["cer"], "aggregate.cer")
    _validate_metric(aggregate["mer"], "aggregate.mer")

    raw_slices = document["slices"]
    if not isinstance(raw_slices, list):
        raise CoreReportValidationError("slices must be a list")
    slice_keys: list[tuple[str, str]] = []
    for index, raw_slice in enumerate(raw_slices):
        context = f"slices[{index}]"
        slice_report = _require_mapping(raw_slice, context)
        if set(slice_report) != {"dimension", "value", "counts", "aggregate"}:
            raise CoreReportValidationError(
                f"{context} contains blind-unsafe or unknown fields"
            )
        dimension = _require_string(
            slice_report["dimension"], f"{context}.dimension"
        )
        value = _require_string(slice_report["value"], f"{context}.value")
        if dimension not in slice_fields:
            raise CoreReportValidationError(
                f"{context}.dimension is not configured"
            )
        slice_keys.append((dimension, value))
        _validate_summary_counts(slice_report["counts"], f"{context}.counts")
        slice_aggregate = _require_mapping(
            slice_report["aggregate"], f"{context}.aggregate"
        )
        if set(slice_aggregate) != {"cer", "mer"}:
            raise CoreReportValidationError(
                f"{context}.aggregate must contain exactly cer and mer"
            )
        _validate_metric(slice_aggregate["cer"], f"{context}.aggregate.cer")
        _validate_metric(slice_aggregate["mer"], f"{context}.aggregate.mer")
    if slice_keys != sorted(set(slice_keys)):
        raise CoreReportValidationError(
            "core summary slices must be unique and sorted"
        )
    if evaluation_scope["kind"] == "split":
        split_values = [
            value for dimension, value in slice_keys if dimension == "split"
        ]
        if split_values != [evaluation_scope["split"]]:
            raise CoreReportValidationError(
                "core summary slices do not match the configured split scope"
            )

    if source_core is not None:
        validate_core_report(source_core)
        expected_summary = _core_summary_projection(source_core)
        if dict(document) != expected_summary:
            raise CoreReportValidationError(
                "core summary does not match the restricted source core"
            )


def canonical_core_summary_bytes(summary: Mapping[str, Any]) -> bytes:
    """Return canonical bytes for a validated restricted core summary."""

    validate_core_summary(summary)
    return _canonical_json_bytes(summary)


def core_summary_sha256(summary: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 identity of a restricted core summary."""

    return _sha256_bytes(canonical_core_summary_bytes(summary))


__all__ = [
    "ALIGNMENT_VERSION",
    "CORE_KIND",
    "CORE_SCHEMA_VERSION",
    "CORE_SUMMARY_KIND",
    "IDENTITY_HYPOTHESIS_ADAPTER_VERSION",
    "SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION",
    "CoreReportValidationError",
    "adapt_hypothesis",
    "build_core_report",
    "build_core_summary",
    "build_split_core_report",
    "canonical_core_bytes",
    "canonical_core_summary_bytes",
    "core_summary_sha256",
    "core_report_sha256",
    "validate_core_report",
    "validate_core_report_for_collection",
    "validate_core_summary",
]
