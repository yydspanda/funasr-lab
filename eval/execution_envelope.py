"""Canonical restricted runtime evidence for one sealed offline decode.

The trusted decoder writes raw prediction JSONL first and publishes this
text-free envelope last.  The envelope binds both the raw bytes and canonical
prediction items.  Custodian freeze validates and extends that one-way chain;
it never rewrites runner evidence.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.collection import canonical_json_bytes, sha256_bytes
from eval.core_report import (
    CoreReportValidationError,
    HYPOTHESIS_ADAPTER_VERSIONS,
    adapt_hypothesis,
)

EXECUTION_SCHEMA_VERSION = 1
EXECUTION_ENVELOPE_KIND = "asr-evaluation-execution-envelope"
OFFLINE_MODE = "offline-single-stream"
SEALED_SPLIT = "sealed-blind"
PERCENTILE_METHOD = "linear-interpolation-n-minus-one-v1"
CLOCK_VERSION = "python-perf-counter-ns-v1"
RSS_VERSION = "linux-rusage-self-maxrss-kib-v1"
RSS_SCOPE = "fresh-process-rusage-self"
RTF_POPULATION = "all-measured-attempts"

MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_ITEMS = 1_000_000
MAX_NS = 2**63 - 1
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
EXPERIMENT_ID = re.compile(r"^EXP-[0-9]{8}-[0-9]{3}(?:-[a-z0-9-]+)?$")
ROLE = re.compile(r"^[a-z][a-z0-9_-]*$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REASON = re.compile(r"^[a-z][a-z0-9_]*$")
PYTHON_VERSION = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$"
)
PYTHON_CACHE_TAG = re.compile(r"^[A-Za-z0-9_-]+$")
UNICODE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{6})?Z$"
)

ROOT_FIELDS = {
    "schema_version", "kind", "state", "access_class", "mode",
    "experiment_id", "dataset_id", "revision", "split", "bindings",
    "runner", "measurement", "items",
}
BINDING_FIELDS = {
    "input_projection_sha256", "candidate_lock_sha256",
    "candidate_freeze_sha256", "decode_item_ids_sha256",
    "input_export_receipt_sha256",
    "raw_predictions_sha256", "prediction_items_sha256",
    "hypothesis_adapter_version",
}
RUNNER_FIELDS = {
    "code_commit", "source_sha256", "effective_config_sha256", "models",
    "command", "hardware", "runtime",
}
MEASUREMENT_FIELDS = {
    "clock_version", "rss_version", "rss_scope", "rtf_population",
    "percentile_method", "started_at_utc", "finished_at_utc",
    "model_load_ns", "cold_inference_ns",
    "cold_start_ns", "warmup_run_count", "warmup_wall_ns",
    "measured_wall_ns", "measured_audio_seconds", "rtf_p50", "rtf_p95",
    "peak_rss_bytes", "counts",
}
COUNT_FIELDS = {
    "decode_item_count", "prediction_item_count", "missing_prediction_count",
    "total_attempt_count", "retried_item_count", "ok_count", "empty_count",
    "failed_count",
}
ITEM_FIELDS = {
    "id", "audio_duration_seconds", "wall_time_ns", "attempt_count",
    "status", "reason_code",
}
OBSERVATION_FIELDS = {
    "experiment_id", "dataset_id", "revision", "split",
    "candidate_freeze_sha256", "candidate_lock_sha256",
    "input_projection_sha256", "hypothesis_adapter_version", "config_sha256",
    "models", "command", "hardware", "runtime", "runner_code_commit",
    "runner_source_sha256", "raw_predictions_sha256",
    "prediction_items_sha256", "prediction_item_count",
    "started_at_utc", "finished_at_utc", "measurement_contract",
    "model_load_ns", "cold_attempt",
    "warmup_attempts", "decode_attempts", "peak_rss_bytes",
}
CONTRACT_FIELDS = {
    "clock_version", "rss_version", "rss_scope", "rtf_population", "warmup_runs",
}
ATTEMPT_FIELDS = {
    "id", "attempt_index", "elapsed_ns", "audio_duration_seconds", "status",
    "reason_code",
}
PREDICTION_FIELDS = {"id", "raw_text", "status", "reason_code"}
MODEL_FIELDS = {"role", "identifier", "revision", "sha256"}
COMMAND_FIELDS = {"working_directory", "argv", "environment"}
HARDWARE_REQUIRED = {
    "host_id", "os", "cpu_model", "logical_cpu_count", "memory_bytes", "device",
    "accelerator",
}
HARDWARE_FIELDS = HARDWARE_REQUIRED
RUNTIME_FIELDS = {
    "python_implementation", "python_version", "python_cache_tag",
    "dependency_lock_sha256", "installed_dependencies_sha256",
    "installed_dependency_count",
    "unicode_version",
}


class ExecutionEnvelopeError(ValueError):
    """Execution evidence violates the frozen v1 contract."""


class _DuplicateKey(ValueError):
    pass


class _InvalidConstant(ValueError):
    pass


@dataclass(frozen=True)
class LoadedExecutionEnvelope:
    document: dict[str, Any]
    payload: bytes
    sha256: str


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise _InvalidConstant(f"non-standard JSON constant {value!r}")


def _map(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionEnvelopeError(f"{context} must be an object")
    return value


def _seq(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ExecutionEnvelopeError(f"{context} must be an ordered sequence")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], context: str) -> None:
    if set(value) != fields:
        raise ExecutionEnvelopeError(f"{context} must contain exactly {sorted(fields)!r}")


def _text(value: Any, context: str, maximum: int = 1024, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and (not value or value != value.strip())):
        raise ExecutionEnvelopeError(f"{context} must be a valid string")
    if len(value) > maximum:
        raise ExecutionEnvelopeError(f"{context} exceeds the character limit")
    return value


def _digest(value: Any, context: str) -> str:
    result = _text(value, context, 71)
    if SHA256.fullmatch(result) is None:
        raise ExecutionEnvelopeError(f"{context} must be sha256:<64 lowercase hex>")
    return result


def _utc_timestamp(value: Any, context: str) -> tuple[str, datetime]:
    result = _text(value, context, 27)
    if UTC_TIMESTAMP.fullmatch(result) is None:
        raise ExecutionEnvelopeError(
            f"{context} must be strict UTC YYYY-MM-DDTHH:MM:SS(.ffffff)Z"
        )
    pattern = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in result else "%Y-%m-%dT%H:%M:%SZ"
    try:
        parsed = datetime.strptime(result, pattern).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ExecutionEnvelopeError(f"{context} is not a valid UTC timestamp") from exc
    return result, parsed


def _int(value: Any, context: str, minimum: int = 0, maximum: int = MAX_NS) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ExecutionEnvelopeError(f"{context} must be an integer in range")
    return value


def _positive(value: Any, context: str) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0
    ):
        raise ExecutionEnvelopeError(f"{context} must be a positive finite number")
    return float(value)


def _round(value: float) -> float:
    return round(value, 9)


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    raw = _seq(values, "percentile values")
    if not raw or isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise ExecutionEnvelopeError("invalid percentile input")
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ExecutionEnvelopeError("quantile must be between zero and one")
    ordered = sorted(_positive(item, "percentile value") for item in raw)
    position = (len(ordered) - 1) * float(quantile)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def peak_rss_mib(peak_rss_bytes: int) -> float:
    return _round(_int(peak_rss_bytes, "peak_rss_bytes", 1) / (1024 * 1024))


def _status_reason(status: Any, reason: Any, context: str) -> tuple[str, str | None]:
    if not isinstance(status, str) or status not in {"ok", "empty", "failed"}:
        raise ExecutionEnvelopeError(f"{context}.status is invalid")
    if status == "ok":
        if reason is not None:
            raise ExecutionEnvelopeError(f"{context}.reason_code must be null")
        return status, None
    result = _text(reason, f"{context}.reason_code", 128)
    if REASON.fullmatch(result) is None:
        raise ExecutionEnvelopeError(f"{context}.reason_code is invalid")
    return status, result


def _models(value: Any) -> list[dict[str, Any]]:
    raw = _seq(value, "runner.models")
    if not raw or len(raw) > 32:
        raise ExecutionEnvelopeError("runner.models has invalid length")
    result, roles = [], set()
    for index, item_value in enumerate(raw):
        item = _map(item_value, f"runner.models[{index}]")
        _exact(item, MODEL_FIELDS, f"runner.models[{index}]")
        role = _text(item["role"], "model role", 64)
        if ROLE.fullmatch(role) is None or role in roles:
            raise ExecutionEnvelopeError("model roles must be valid and unique")
        roles.add(role)
        _text(item["identifier"], "model identifier")
        _text(item["revision"], "model revision")
        _digest(item["sha256"], "model sha256")
        result.append(deepcopy(dict(item)))
    return result


def _command(value: Any) -> dict[str, Any]:
    command = _map(value, "runner.command")
    _exact(command, COMMAND_FIELDS, "runner.command")
    _text(command["working_directory"], "working directory", 4096)
    argv = _seq(command["argv"], "runner.command.argv")
    if not 2 <= len(argv) <= 1024:
        raise ExecutionEnvelopeError("runner.command.argv has invalid length")
    for argument in argv:
        _text(argument, "command argument", 16_384)
    environment = _map(command["environment"], "runner.command.environment")
    if len(environment) > 256:
        raise ExecutionEnvelopeError("runner.command.environment is too large")
    for name, item in environment.items():
        if not isinstance(name, str) or ENV_NAME.fullmatch(name) is None:
            raise ExecutionEnvelopeError("invalid environment name")
        _text(item, f"environment.{name}", 8192, empty=True)
    return deepcopy(dict(command))


def _hardware(value: Any) -> dict[str, Any]:
    hardware = _map(value, "runner.hardware")
    _exact(hardware, HARDWARE_FIELDS, "runner.hardware")
    for field in ("host_id", "os", "cpu_model", "device"):
        _text(hardware[field], f"hardware.{field}")
    _int(hardware["logical_cpu_count"], "logical_cpu_count", 1, 1_048_576)
    _int(hardware["memory_bytes"], "memory_bytes", 1, 2**60)
    if hardware.get("accelerator") is not None:
        _text(hardware["accelerator"], "hardware.accelerator")
    if hardware["device"] != "cpu" or hardware.get("accelerator") is not None:
        raise ExecutionEnvelopeError("execution envelope v1 supports CPU only")
    return deepcopy(dict(hardware))


def _runtime(value: Any) -> dict[str, Any]:
    runtime = _map(value, "runner.runtime")
    _exact(runtime, RUNTIME_FIELDS, "runner.runtime")
    if runtime["python_implementation"] != "cpython":
        raise ExecutionEnvelopeError("runner.runtime requires CPython")
    python_version = _text(
        runtime["python_version"], "runner.runtime.python_version", 64
    )
    if PYTHON_VERSION.fullmatch(python_version) is None:
        raise ExecutionEnvelopeError("runner.runtime.python_version is invalid")
    cache_tag = _text(
        runtime["python_cache_tag"], "runner.runtime.python_cache_tag", 64
    )
    if PYTHON_CACHE_TAG.fullmatch(cache_tag) is None:
        raise ExecutionEnvelopeError("runner.runtime.python_cache_tag is invalid")
    unicode_version = _text(
        runtime["unicode_version"],
        "runner.runtime.unicode_version",
        32,
    )
    if UNICODE_VERSION.fullmatch(unicode_version) is None:
        raise ExecutionEnvelopeError("runner.runtime.unicode_version is invalid")
    return {
        "python_implementation": "cpython",
        "python_version": python_version,
        "python_cache_tag": cache_tag,
        "dependency_lock_sha256": _digest(
            runtime["dependency_lock_sha256"],
            "runner.runtime.dependency_lock_sha256",
        ),
        "installed_dependencies_sha256": _digest(
            runtime["installed_dependencies_sha256"],
            "runner.runtime.installed_dependencies_sha256",
        ),
        "installed_dependency_count": _int(
            runtime["installed_dependency_count"],
            "runner.runtime.installed_dependency_count",
            1,
            10_000,
        ),
        "unicode_version": unicode_version,
    }


def _runner(value: Any) -> dict[str, Any]:
    runner = _map(value, "runner")
    _exact(runner, RUNNER_FIELDS, "runner")
    commit = _text(runner["code_commit"], "runner.code_commit", 40)
    if GIT_COMMIT.fullmatch(commit) is None:
        raise ExecutionEnvelopeError("runner.code_commit must be a full commit")
    return {
        "code_commit": commit,
        "source_sha256": _digest(runner["source_sha256"], "runner.source_sha256"),
        "effective_config_sha256": _digest(
            runner["effective_config_sha256"], "runner.effective_config_sha256"
        ),
        "models": _models(runner["models"]),
        "command": _command(runner["command"]),
        "hardware": _hardware(runner["hardware"]),
        "runtime": _runtime(runner["runtime"]),
    }


def _predictions(value: Any, adapter: str) -> list[dict[str, Any]]:
    raw = _seq(value, "prediction_items")
    if len(raw) > MAX_ITEMS:
        raise ExecutionEnvelopeError("too many prediction items")
    result, seen, total = [], set(), 0
    for index, item_value in enumerate(raw):
        item = _map(item_value, f"prediction_items[{index}]")
        _exact(item, PREDICTION_FIELDS, f"prediction_items[{index}]")
        utterance_id = _text(item["id"], "prediction id", 512)
        if utterance_id in seen:
            raise ExecutionEnvelopeError("duplicate prediction id")
        seen.add(utterance_id)
        text = _text(item["raw_text"], "prediction raw_text", 16_384, empty=True)
        total += len(text)
        if total > 1_000_000:
            raise ExecutionEnvelopeError("prediction text exceeds total limit")
        status, reason = _status_reason(item["status"], item["reason_code"], "prediction")
        if status == "failed" and text:
            raise ExecutionEnvelopeError("failed prediction text must be empty")
        try:
            display = adapt_hypothesis(text, adapter)
        except CoreReportValidationError as exc:
            raise ExecutionEnvelopeError(str(exc)) from exc
        if (status == "ok" and not display) or (status == "empty" and display):
            raise ExecutionEnvelopeError("prediction status conflicts with adapted text")
        result.append(
            {
                "id": utterance_id,
                "raw_text": text,
                "status": status,
                "reason_code": reason,
            }
        )
    return result


def _attempt(value: Any, context: str, index: int) -> dict[str, Any]:
    attempt = _map(value, context)
    _exact(attempt, ATTEMPT_FIELDS, context)
    if _int(attempt["attempt_index"], f"{context}.attempt_index", 0, MAX_ITEMS) != index:
        raise ExecutionEnvelopeError(f"{context}.attempt_index does not match order")
    status, reason = _status_reason(attempt["status"], attempt["reason_code"], context)
    return {
        "id": _text(attempt["id"], f"{context}.id", 512),
        "elapsed_ns": _int(attempt["elapsed_ns"], f"{context}.elapsed_ns", 1),
        "audio_duration_seconds": _positive(
            attempt["audio_duration_seconds"], f"{context}.audio_duration_seconds"
        ),
        "status": status,
        "reason_code": reason,
    }


def _statuses(
    ids: Sequence[str],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str | None]]:
    positions = {item: index for index, item in enumerate(ids)}
    result, seen_positions = {}, []
    for prediction in predictions:
        utterance_id = str(prediction["id"])
        if utterance_id not in positions:
            raise ExecutionEnvelopeError("prediction id absent from decode attempts")
        seen_positions.append(positions[utterance_id])
        result[utterance_id] = (str(prediction["status"]), prediction["reason_code"])
    if seen_positions != sorted(seen_positions):
        raise ExecutionEnvelopeError("predictions must preserve decode order")
    for utterance_id in ids:
        result.setdefault(utterance_id, ("failed", "missing_prediction"))
    return result


def _measurement(
    started_at_utc: str,
    finished_at_utc: str,
    model_load_ns: int,
    cold_ns: int,
    warmup_wall_ns: int,
    warmup_count: int,
    items: Sequence[Mapping[str, Any]],
    prediction_count: int,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    measured_wall = sum(int(item["wall_time_ns"]) for item in items)
    if max(measured_wall, model_load_ns + cold_ns, warmup_wall_ns) > MAX_NS:
        raise ExecutionEnvelopeError("duration overflow")
    audio = _round(math.fsum(float(item["audio_duration_seconds"]) for item in items))
    rtfs = [
        (int(item["wall_time_ns"]) / 1_000_000_000) / float(item["audio_duration_seconds"])
        for item in items
    ]
    statuses = [str(item["status"]) for item in items]
    return {
        "clock_version": CLOCK_VERSION,
        "rss_version": RSS_VERSION,
        "rss_scope": RSS_SCOPE,
        "rtf_population": RTF_POPULATION,
        "percentile_method": PERCENTILE_METHOD,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "model_load_ns": model_load_ns,
        "cold_inference_ns": cold_ns,
        "cold_start_ns": model_load_ns + cold_ns,
        "warmup_run_count": warmup_count,
        "warmup_wall_ns": warmup_wall_ns,
        "measured_wall_ns": measured_wall,
        "measured_audio_seconds": audio,
        "rtf_p50": _round(linear_percentile(rtfs, 0.50)),
        "rtf_p95": _round(linear_percentile(rtfs, 0.95)),
        "peak_rss_bytes": peak_rss_bytes,
        "counts": {
            "decode_item_count": len(items),
            "prediction_item_count": prediction_count,
            "missing_prediction_count": len(items) - prediction_count,
            "total_attempt_count": len(items),
            "retried_item_count": 0,
            "ok_count": statuses.count("ok"),
            "empty_count": statuses.count("empty"),
            "failed_count": statuses.count("failed"),
        },
    }


def _ids_sha(items: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes([str(item["id"]) for item in items]))


def build_execution_envelope(
    observation: Mapping[str, Any],
    prediction_items: Sequence[Mapping[str, Any]],
    *,
    input_export_receipt_sha256: str,
) -> dict[str, Any]:
    """Convert the decoder's strict in-memory facts into final runner evidence."""

    facts = _map(observation, "runner observation")
    _exact(facts, OBSERVATION_FIELDS, "runner observation")
    if facts["split"] != SEALED_SPLIT:
        raise ExecutionEnvelopeError("runner observation split must be sealed-blind")
    adapter = _text(facts["hypothesis_adapter_version"], "hypothesis adapter", 64)
    if adapter not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise ExecutionEnvelopeError("unsupported hypothesis adapter")
    predictions = _predictions(prediction_items, adapter)
    if _int(
        facts["prediction_item_count"],
        "prediction_item_count",
        0,
        MAX_ITEMS,
    ) != len(predictions):
        raise ExecutionEnvelopeError("prediction_item_count mismatch")
    prediction_sha = sha256_bytes(canonical_json_bytes(predictions))
    if (
        _digest(facts["prediction_items_sha256"], "prediction_items_sha256")
        != prediction_sha
    ):
        raise ExecutionEnvelopeError("prediction_items_sha256 mismatch")

    contract = _map(facts["measurement_contract"], "measurement_contract")
    _exact(contract, CONTRACT_FIELDS, "measurement_contract")
    expected = {
        "clock_version": CLOCK_VERSION, "rss_version": RSS_VERSION,
        "rss_scope": RSS_SCOPE, "rtf_population": RTF_POPULATION,
    }
    if any(contract[field] != value for field, value in expected.items()):
        raise ExecutionEnvelopeError("measurement contract is unsupported")
    attempts_raw = _seq(facts["decode_attempts"], "decode_attempts")
    if not attempts_raw or len(attempts_raw) > MAX_ITEMS:
        raise ExecutionEnvelopeError("decode_attempts has invalid length")
    attempts = [_attempt(item, f"decode_attempts[{i}]", i) for i, item in enumerate(attempts_raw)]
    ids = [str(item["id"]) for item in attempts]
    if len(ids) != len(set(ids)):
        raise ExecutionEnvelopeError("decode attempt ids must be unique")
    statuses = _statuses(ids, predictions)
    items = []
    for attempt in attempts:
        status_reason = (attempt["status"], attempt["reason_code"])
        if status_reason != statuses[str(attempt["id"])]:
            raise ExecutionEnvelopeError("decode status does not match complete predictions")
        items.append({
            "id": attempt["id"], "audio_duration_seconds": attempt["audio_duration_seconds"],
            "wall_time_ns": attempt["elapsed_ns"], "attempt_count": 1,
            "status": attempt["status"], "reason_code": attempt["reason_code"],
        })
    cold = _attempt(facts["cold_attempt"], "cold_attempt", 0)
    if cold["id"] != ids[0]:
        raise ExecutionEnvelopeError("cold attempt must use first decode item")
    warmup_raw = _seq(facts["warmup_attempts"], "warmup_attempts")
    warmups = [_attempt(item, f"warmup_attempts[{i}]", i) for i, item in enumerate(warmup_raw)]
    if any(item["id"] != ids[0] for item in warmups):
        raise ExecutionEnvelopeError("warmups must use first decode item")
    warmup_count = _int(contract["warmup_runs"], "warmup_runs", 0, MAX_ITEMS)
    if warmup_count != len(warmups):
        raise ExecutionEnvelopeError("warmup count mismatch")
    runner = _runner({
        "code_commit": facts["runner_code_commit"],
        "source_sha256": facts["runner_source_sha256"],
        "effective_config_sha256": facts["config_sha256"],
        "models": facts["models"], "command": facts["command"],
        "hardware": facts["hardware"], "runtime": facts["runtime"],
    })
    started_at_utc, started_at = _utc_timestamp(
        facts["started_at_utc"], "started_at_utc"
    )
    finished_at_utc, finished_at = _utc_timestamp(
        facts["finished_at_utc"], "finished_at_utc"
    )
    if finished_at < started_at:
        raise ExecutionEnvelopeError("finished_at_utc precedes started_at_utc")
    envelope = {
        "schema_version": 1, "kind": EXECUTION_ENVELOPE_KIND, "state": "frozen",
        "access_class": "restricted", "mode": OFFLINE_MODE,
        "experiment_id": facts["experiment_id"], "dataset_id": facts["dataset_id"],
        "revision": facts["revision"], "split": SEALED_SPLIT,
        "bindings": {
            "input_projection_sha256": facts["input_projection_sha256"],
            "candidate_lock_sha256": facts["candidate_lock_sha256"],
            "candidate_freeze_sha256": facts["candidate_freeze_sha256"],
            "decode_item_ids_sha256": _ids_sha(items),
            "input_export_receipt_sha256": _digest(
                input_export_receipt_sha256,
                "input_export_receipt_sha256",
            ),
            "raw_predictions_sha256": facts["raw_predictions_sha256"],
            "prediction_items_sha256": prediction_sha,
            "hypothesis_adapter_version": adapter,
        },
        "runner": runner,
        "measurement": _measurement(
            started_at_utc, finished_at_utc,
            _int(facts["model_load_ns"], "model_load_ns", 1), int(cold["elapsed_ns"]),
            sum(int(item["elapsed_ns"]) for item in warmups), len(warmups), items,
            len(predictions), _int(facts["peak_rss_bytes"], "peak_rss_bytes", 1),
        ),
        "items": items,
    }
    validate_execution_envelope_for_predictions(
        envelope, predictions, raw_predictions_sha256=str(facts["raw_predictions_sha256"])
    )
    return envelope


def _items(value: Any) -> list[dict[str, Any]]:
    raw = _seq(value, "items")
    if not raw or len(raw) > MAX_ITEMS:
        raise ExecutionEnvelopeError("items has invalid length")
    result, seen = [], set()
    for index, item_value in enumerate(raw):
        item = _map(item_value, f"items[{index}]")
        _exact(item, ITEM_FIELDS, f"items[{index}]")
        utterance_id = _text(item["id"], "item id", 512)
        if utterance_id in seen:
            raise ExecutionEnvelopeError("duplicate item id")
        seen.add(utterance_id)
        status, reason = _status_reason(item["status"], item["reason_code"], "item")
        result.append({
            "id": utterance_id,
            "audio_duration_seconds": _positive(item["audio_duration_seconds"], "audio duration"),
            "wall_time_ns": _int(item["wall_time_ns"], "wall_time_ns", 1),
            "attempt_count": _int(item["attempt_count"], "attempt_count", 1, 1),
            "status": status, "reason_code": reason,
        })
    return result


def validate_execution_envelope(document: Any) -> None:
    envelope = _map(document, "execution envelope")
    _exact(envelope, ROOT_FIELDS, "execution envelope")
    _int(envelope["schema_version"], "schema_version", 1, 1)
    constants = {
        "schema_version": 1, "kind": EXECUTION_ENVELOPE_KIND, "state": "frozen",
        "access_class": "restricted", "mode": OFFLINE_MODE, "split": SEALED_SPLIT,
    }
    if any(envelope[field] != value for field, value in constants.items()):
        raise ExecutionEnvelopeError("execution envelope constants are invalid")
    experiment_id = _text(envelope["experiment_id"], "experiment_id", 128)
    if EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ExecutionEnvelopeError("experiment_id is invalid")
    _text(envelope["dataset_id"], "dataset_id", 256)
    _text(envelope["revision"], "revision", 256)
    bindings = _map(envelope["bindings"], "bindings")
    _exact(bindings, BINDING_FIELDS, "bindings")
    for field in BINDING_FIELDS - {"hypothesis_adapter_version"}:
        _digest(bindings[field], f"bindings.{field}")
    if bindings["hypothesis_adapter_version"] not in HYPOTHESIS_ADAPTER_VERSIONS:
        raise ExecutionEnvelopeError("unsupported bindings adapter")
    _runner(envelope["runner"])
    items = _items(envelope["items"])
    if bindings["decode_item_ids_sha256"] != _ids_sha(items):
        raise ExecutionEnvelopeError("decode item identity mismatch")
    measurement = _map(envelope["measurement"], "measurement")
    _exact(measurement, MEASUREMENT_FIELDS, "measurement")
    fixed = {
        "clock_version": CLOCK_VERSION, "rss_version": RSS_VERSION,
        "rss_scope": RSS_SCOPE, "rtf_population": RTF_POPULATION,
        "percentile_method": PERCENTILE_METHOD,
    }
    if any(measurement[field] != value for field, value in fixed.items()):
        raise ExecutionEnvelopeError("measurement contract is invalid")
    started_at_utc, started_at = _utc_timestamp(
        measurement["started_at_utc"], "measurement.started_at_utc"
    )
    finished_at_utc, finished_at = _utc_timestamp(
        measurement["finished_at_utc"], "measurement.finished_at_utc"
    )
    if finished_at < started_at:
        raise ExecutionEnvelopeError("measurement finished_at_utc precedes started_at_utc")
    counts = _map(measurement["counts"], "measurement.counts")
    _exact(counts, COUNT_FIELDS, "measurement.counts")
    for field in COUNT_FIELDS:
        minimum = 1 if field in {"decode_item_count", "total_attempt_count"} else 0
        _int(counts[field], f"measurement.counts.{field}", minimum, MAX_ITEMS)
    prediction_count = _int(counts["prediction_item_count"], "prediction count", 0, MAX_ITEMS)
    if prediction_count > len(items):
        raise ExecutionEnvelopeError("prediction count exceeds decode item count")
    warmup_count = _int(measurement["warmup_run_count"], "warmup count", 0, MAX_ITEMS)
    warmup_wall = _int(measurement["warmup_wall_ns"], "warmup wall")
    if (warmup_count == 0) != (warmup_wall == 0):
        raise ExecutionEnvelopeError("warmup count/wall mismatch")
    expected = _measurement(
        started_at_utc, finished_at_utc,
        _int(measurement["model_load_ns"], "model load", 1),
        _int(measurement["cold_inference_ns"], "cold inference", 1),
        warmup_wall, warmup_count, items, prediction_count,
        _int(measurement["peak_rss_bytes"], "peak RSS", 1),
    )
    if dict(measurement) != expected:
        raise ExecutionEnvelopeError("measurement does not match execution items")


def validate_execution_envelope_for_predictions(
    envelope: Mapping[str, Any], prediction_items: Sequence[Mapping[str, Any]], *,
    raw_predictions_sha256: str,
) -> None:
    validate_execution_envelope(envelope)
    bindings = envelope["bindings"]
    if bindings["raw_predictions_sha256"] != _digest(raw_predictions_sha256, "raw predictions"):
        raise ExecutionEnvelopeError("raw prediction bytes mismatch")
    predictions = _predictions(prediction_items, str(bindings["hypothesis_adapter_version"]))
    if bindings["prediction_items_sha256"] != sha256_bytes(canonical_json_bytes(predictions)):
        raise ExecutionEnvelopeError("prediction items mismatch")
    statuses = _statuses([str(item["id"]) for item in envelope["items"]], predictions)
    for item in envelope["items"]:
        if (item["status"], item["reason_code"]) != statuses[str(item["id"])]:
            raise ExecutionEnvelopeError("complete prediction status mismatch")
    if envelope["measurement"]["counts"]["prediction_item_count"] != len(predictions):
        raise ExecutionEnvelopeError("prediction count mismatch")


def canonical_execution_envelope_bytes(document: Mapping[str, Any]) -> bytes:
    validate_execution_envelope(document)
    return canonical_json_bytes(document)


def load_execution_envelope(path: Path) -> LoadedExecutionEnvelope:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ExecutionEnvelopeError("O_NOFOLLOW is required")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise ExecutionEnvelopeError("cannot read execution envelope") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_ARTIFACT_BYTES:
            raise ExecutionEnvelopeError("execution envelope file is invalid")
        payload = bytearray()
        while len(payload) <= MAX_ARTIFACT_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_ARTIFACT_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > MAX_ARTIFACT_BYTES:
            raise ExecutionEnvelopeError("execution envelope exceeds size limit")
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or (
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
            raise ExecutionEnvelopeError(
                "execution envelope changed while it was read"
            )
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ExecutionEnvelopeError(
            "execution envelope path changed while it was read"
        ) from exc
    if (
        not stat.S_ISREG(path_after.st_mode)
        or (path_after.st_dev, path_after.st_ino)
        != (after.st_dev, after.st_ino)
    ):
        raise ExecutionEnvelopeError(
            "execution envelope path changed while it was read"
        )
    raw = bytes(payload)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ExecutionEnvelopeError("execution envelope must not contain BOM")
    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, _InvalidConstant) as exc:
        raise ExecutionEnvelopeError(f"invalid execution envelope JSON: {exc}") from exc
    validate_execution_envelope(document)
    if raw != canonical_json_bytes(document):
        raise ExecutionEnvelopeError("execution envelope must use canonical JSON")
    return LoadedExecutionEnvelope(dict(document), raw, sha256_bytes(raw))


__all__ = [
    "CLOCK_VERSION", "EXECUTION_ENVELOPE_KIND", "EXECUTION_SCHEMA_VERSION",
    "ExecutionEnvelopeError", "LoadedExecutionEnvelope", "OFFLINE_MODE",
    "PERCENTILE_METHOD", "RSS_SCOPE", "RSS_VERSION", "RTF_POPULATION",
    "build_execution_envelope", "canonical_execution_envelope_bytes",
    "linear_percentile", "load_execution_envelope", "peak_rss_mib",
    "validate_execution_envelope", "validate_execution_envelope_for_predictions",
]
