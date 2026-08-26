"""Reproducible, downstream-owned offline ASR baseline evaluation.

The module deliberately imports FunASR only inside the default model factory so
dataset validation, scoring, hashing, and focused tests never download a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from eval.normalizers import NORMALIZER_VERSION
from eval.normalizers import normalize_content


SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SENSEVOICE_TAG_PATTERN = re.compile(r"<\|[^<>]*?\|>")
FLOATING_REVISION_NAMES = {"head", "latest", "main", "master", "trunk"}


@dataclass(frozen=True)
class TrackSpec:
    name: str
    hub: str
    model_identifier: str
    generate_options: Mapping[str, Any]


TRACKS: dict[str, TrackSpec] = {
    "paraformer": TrackSpec(
        name="paraformer",
        hub="ms",
        model_identifier=(
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-"
            "common-vocab8404-pytorch"
        ),
        generate_options={"batch_size": 1},
    ),
    "sensevoice": TrackSpec(
        name="sensevoice",
        hub="ms",
        model_identifier="iic/SenseVoiceSmall",
        generate_options={"batch_size": 1, "language": "zh", "use_itn": False},
    ),
}


class BaselineError(Exception):
    """Base class for actionable baseline errors."""


class DatasetValidationError(BaselineError):
    """Raised before inference when the frozen dataset identity is invalid."""


class BaselineExecutionError(BaselineError):
    """Raised when the model cannot produce a meaningful baseline report."""


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSONL record contains an ambiguous duplicate key."""


@dataclass(frozen=True)
class DatasetItem:
    utterance_id: str
    audio: str
    audio_path: Path
    audio_sha256: str
    duration_seconds: float
    raw_text: str
    reference_sha256: str
    speaker_id: str
    session_id: str
    split: str
    data_version: str


@dataclass(frozen=True)
class FrozenDataset:
    manifest_path: Path
    manifest_sha256: str
    data_version: str
    items: tuple[DatasetItem, ...]


@dataclass(frozen=True)
class BaselineConfig:
    track: str
    model_revision: str
    device: str = "cpu"
    ncpu: int = 4
    warmup_runs: int = 1
    seed: int = 0


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKeyError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the exact UTF-8 representation used for config and report hashes."""

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


def model_directory_sha256(model_path: Path) -> str:
    """Hash a resolved model bundle using the manifest inventory contract.

    Files are sorted by their POSIX relative paths. Each inventory line is
    encoded exactly as ``<file-sha256><two spaces><relative-path>\n`` before the
    inventory itself is hashed.
    """

    if not model_path.is_dir():
        raise BaselineExecutionError(
            f"resolved model_path is not a directory: {model_path}"
        )
    files = sorted(
        (path for path in model_path.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(model_path).as_posix(),
    )
    if not files:
        raise BaselineExecutionError(f"resolved model bundle is empty: {model_path}")
    inventory_lines: list[str] = []
    for path in files:
        file_digest = sha256_file(path).removeprefix("sha256:")
        relative_path = path.relative_to(model_path).as_posix()
        inventory_lines.append(f"{file_digest}  {relative_path}\n")
    return sha256_bytes("".join(inventory_lines).encode("utf-8"))


def validate_immutable_revision(revision: str) -> str:
    """Reject well-known floating branch names before AutoModel is constructed."""

    if not isinstance(revision, str) or not revision:
        raise BaselineExecutionError("model revision must be a non-empty string")
    if revision != revision.strip():
        raise BaselineExecutionError("model revision must not contain surrounding whitespace")
    lowered = revision.lower()
    path_parts = [part for part in lowered.split("/") if part]
    if lowered in FLOATING_REVISION_NAMES or any(
        part in FLOATING_REVISION_NAMES for part in path_parts
    ):
        raise BaselineExecutionError(
            f"model revision {revision!r} is floating; use an immutable tag or commit"
        )
    return revision


def strip_sensevoice_tags(text: str) -> str:
    """Remove SenseVoice control tags without adding display-only emoji."""

    return SENSEVOICE_TAG_PATTERN.sub("", text).strip()


def text_views(track: str, raw_text: str) -> dict[str, str]:
    if track not in TRACKS:
        raise BaselineExecutionError(f"unknown baseline track: {track}")
    display_text = strip_sensevoice_tags(raw_text) if track == "sensevoice" else raw_text
    return {
        "raw": raw_text,
        "content": normalize_content(display_text),
        "display": display_text,
    }


def reference_views(raw_text: str) -> dict[str, str]:
    return {
        "raw": raw_text,
        "content": normalize_content(raw_text),
        "display": raw_text,
    }


def cer_components(reference: str, hypothesis: str) -> EditCounts:
    """Return deterministic character-level Levenshtein components.

    Equal-cost paths prefer a diagonal operation, then deletion, then insertion.
    This fixes component accounting even when more than one minimum alignment
    exists.
    """

    previous = [EditCounts(insertions=index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_character in enumerate(reference, start=1):
        current = [EditCounts(deletions=ref_index)]
        for hyp_index, hyp_character in enumerate(hypothesis, start=1):
            diagonal = previous[hyp_index - 1]
            if ref_character != hyp_character:
                diagonal = diagonal + EditCounts(substitutions=1)
            deletion = previous[hyp_index] + EditCounts(deletions=1)
            insertion = current[hyp_index - 1] + EditCounts(insertions=1)
            candidates = (diagonal, deletion, insertion)
            current.append(
                min(
                    enumerate(candidates),
                    key=lambda entry: (entry[1].total, entry[0]),
                )[1]
            )
        previous = current
    return previous[-1]


def percentile(values: Sequence[float], quantile: float) -> float:
    """Compute a deterministic linear-interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _require_string(document: Mapping[str, Any], field: str, source: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{source}: {field} must be a non-empty string")
    return value


def _require_sha256(document: Mapping[str, Any], field: str, source: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise DatasetValidationError(
            f"{source}: {field} must use sha256:<64 lowercase hex chars>"
        )
    return value


def _require_positive_number(
    document: Mapping[str, Any], field: str, source: str
) -> float:
    value = document.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DatasetValidationError(f"{source}: {field} must be a positive finite number")
    return float(value)


def _require_integer(document: Mapping[str, Any], field: str, source: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DatasetValidationError(f"{source}: {field} must be an integer")
    return value


def _resolve_audio_path(audio: str, repo_root: Path, source: str) -> Path:
    relative_path = Path(audio)
    if relative_path.is_absolute():
        raise DatasetValidationError(f"{source}: audio must be repository-relative")
    root = repo_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DatasetValidationError(
            f"{source}: audio path escapes the repository root"
        ) from exc
    if not resolved.is_file():
        raise DatasetValidationError(f"{source}: audio file does not exist: {audio}")
    return resolved


def _read_wav_identity(path: Path, source: str) -> tuple[int, int, float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise DatasetValidationError(f"{source}: audio is not a readable PCM WAV: {exc}") from exc
    if compression != "NONE":
        raise DatasetValidationError(f"{source}: compressed WAV is not supported")
    if frame_count <= 0:
        raise DatasetValidationError(f"{source}: audio must contain at least one frame")
    return sample_rate, channels, frame_count / sample_rate


def load_frozen_dataset(manifest_path: Path, repo_root: Path) -> FrozenDataset:
    """Parse and verify a speaker/session-aware, ordered baseline JSONL file."""

    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise DatasetValidationError(f"cannot read dataset manifest: {exc}") from exc
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("dataset manifest must be UTF-8") from exc

    items: list[DatasetItem] = []
    utterance_ids: set[str] = set()
    data_versions: set[str] = set()
    for line_number, line in enumerate(manifest_text.splitlines(), start=1):
        source = f"{manifest_path}:{line_number}"
        if not line.strip():
            raise DatasetValidationError(f"{source}: blank JSONL records are not allowed")
        try:
            document = json.loads(line, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, DuplicateJsonKeyError) as exc:
            raise DatasetValidationError(f"{source}: invalid JSON object: {exc}") from exc
        if not isinstance(document, dict):
            raise DatasetValidationError(f"{source}: record must be a JSON object")

        utterance_id = _require_string(document, "id", source)
        if utterance_id in utterance_ids:
            raise DatasetValidationError(f"{source}: duplicate utterance id {utterance_id!r}")
        utterance_ids.add(utterance_id)

        audio = _require_string(document, "audio", source)
        audio_path = _resolve_audio_path(audio, repo_root, source)
        audio_sha256 = _require_sha256(document, "audio_sha256", source)
        actual_audio_sha256 = sha256_file(audio_path)
        if audio_sha256 != actual_audio_sha256:
            raise DatasetValidationError(
                f"{source}: audio_sha256 mismatch for {audio!r}; "
                f"expected {audio_sha256}, got {actual_audio_sha256}"
            )

        sample_rate = _require_integer(document, "sample_rate", source)
        channels = _require_integer(document, "channels", source)
        duration_seconds = _require_positive_number(document, "duration_seconds", source)
        actual_rate, actual_channels, actual_duration = _read_wav_identity(audio_path, source)
        if sample_rate != 16_000 or actual_rate != 16_000:
            raise DatasetValidationError(
                f"{source}: baseline audio must be 16 kHz; "
                f"manifest={sample_rate}, actual={actual_rate}"
            )
        if channels != 1 or actual_channels != 1:
            raise DatasetValidationError(
                f"{source}: baseline audio must be mono; "
                f"manifest={channels}, actual={actual_channels}"
            )
        if not math.isclose(
            duration_seconds,
            actual_duration,
            rel_tol=0,
            abs_tol=1 / actual_rate,
        ):
            raise DatasetValidationError(
                f"{source}: duration_seconds mismatch; "
                f"manifest={duration_seconds}, actual={actual_duration}"
            )

        raw_text = document.get("raw_text")
        if not isinstance(raw_text, str):
            raise DatasetValidationError(f"{source}: raw_text must be a string")
        reference_sha256 = _require_sha256(document, "reference_sha256", source)
        actual_reference_sha256 = sha256_bytes(raw_text.encode("utf-8"))
        if reference_sha256 != actual_reference_sha256:
            raise DatasetValidationError(
                f"{source}: reference_sha256 does not match raw_text"
            )

        normalizer_version = _require_string(document, "normalizer_version", source)
        if normalizer_version != NORMALIZER_VERSION:
            raise DatasetValidationError(
                f"{source}: normalizer_version must be {NORMALIZER_VERSION!r}, "
                f"got {normalizer_version!r}"
            )
        speaker_id = _require_string(document, "speaker_id", source)
        session_id = _require_string(document, "session_id", source)
        split = _require_string(document, "split", source)
        data_version = _require_string(document, "data_version", source)
        data_versions.add(data_version)
        items.append(
            DatasetItem(
                utterance_id=utterance_id,
                audio=audio,
                audio_path=audio_path,
                audio_sha256=audio_sha256,
                duration_seconds=actual_duration,
                raw_text=raw_text,
                reference_sha256=reference_sha256,
                speaker_id=speaker_id,
                session_id=session_id,
                split=split,
                data_version=data_version,
            )
        )

    if not items:
        raise DatasetValidationError("dataset manifest must contain at least one record")
    if len(data_versions) != 1:
        raise DatasetValidationError(
            "dataset manifest must contain exactly one data_version, got "
            + ", ".join(sorted(data_versions))
        )
    reference_units = sum(len(reference_views(item.raw_text)["content"]) for item in items)
    if reference_units == 0:
        raise DatasetValidationError(
            "dataset manifest must contain at least one normalized reference character"
        )
    return FrozenDataset(
        manifest_path=manifest_path,
        manifest_sha256=sha256_bytes(manifest_bytes),
        data_version=next(iter(data_versions)),
        items=tuple(items),
    )


def validate_config(config: BaselineConfig) -> TrackSpec:
    if config.track not in TRACKS:
        raise BaselineExecutionError(f"unknown baseline track: {config.track}")
    validate_immutable_revision(config.model_revision)
    if config.device != "cpu":
        raise BaselineExecutionError(
            "BASE-01 only permits device='cpu'; GPU evaluation requires a separate baseline"
        )
    if not isinstance(config.ncpu, int) or isinstance(config.ncpu, bool) or config.ncpu <= 0:
        raise BaselineExecutionError("ncpu must be a positive integer")
    if (
        not isinstance(config.warmup_runs, int)
        or isinstance(config.warmup_runs, bool)
        or config.warmup_runs < 0
    ):
        raise BaselineExecutionError("warmup_runs must be a non-negative integer")
    if config.seed != 0 or isinstance(config.seed, bool):
        raise BaselineExecutionError("BASE-01 fixes seed to integer 0")
    return TRACKS[config.track]


def effective_config(config: BaselineConfig) -> dict[str, Any]:
    spec = validate_config(config)
    return {
        "schema_version": 1,
        "track": spec.name,
        "model": {
            "role": "asr",
            "hub": spec.hub,
            "identifier": spec.model_identifier,
            "revision": config.model_revision,
        },
        "runtime": {
            "device": config.device,
            "precision": "fp32",
            "ncpu": config.ncpu,
            "batch_size": 1,
            "warmup_runs_after_cold": config.warmup_runs,
            "rtf_population": "all_warm_attempts",
            "seed": config.seed,
            "disable_update": True,
            "check_latest": False,
        },
        "pipeline": {
            "vad": None,
            "punctuation": None,
            "itn": False,
            "normalizer_version": NORMALIZER_VERSION,
        },
        "generate_options": dict(spec.generate_options),
    }


def default_model_factory(**kwargs: Any) -> Any:
    """Construct AutoModel only for an explicitly requested real execution."""

    from funasr import AutoModel

    return AutoModel(**kwargs)


def _model_kwargs(spec: TrackSpec, config: BaselineConfig) -> dict[str, Any]:
    return {
        "model": spec.model_identifier,
        "model_revision": config.model_revision,
        "hub": spec.hub,
        "device": config.device,
        "ncpu": config.ncpu,
        "disable_update": True,
        "disable_pbar": True,
        "check_latest": False,
        "seed": config.seed,
    }


def _generate(model: Any, spec: TrackSpec, item: DatasetItem) -> str:
    result = model.generate(
        input=str(item.audio_path),
        **dict(spec.generate_options),
    )
    if not isinstance(result, list) or len(result) != 1:
        raise BaselineExecutionError("AutoModel.generate must return one result object")
    first = result[0]
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        raise BaselineExecutionError("AutoModel result must contain a string text field")
    return first["text"]


def _positive_elapsed(started: float, finished: float, label: str) -> float:
    elapsed = finished - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise BaselineExecutionError(f"{label} elapsed time must be positive")
    return elapsed


def peak_rss_mb() -> float:
    """Return process high-water RSS for the Linux/WSL baseline environment."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _metric(value: float) -> float:
    return round(value, 9)


def run_offline_baseline(
    dataset: FrozenDataset,
    config: BaselineConfig,
    *,
    model_factory: Callable[..., Any] = default_model_factory,
    clock: Callable[[], float] = time.perf_counter,
    rss_reader: Callable[[], float] = peak_rss_mb,
    model_bundle_hasher: Callable[[Path], str] = model_directory_sha256,
    command: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run cold and warm offline inference and return a report document."""

    spec = validate_config(config)
    config_document = effective_config(config)
    config_sha256 = sha256_bytes(canonical_json_bytes(config_document))

    model_started = clock()
    try:
        model = model_factory(**_model_kwargs(spec, config))
    except Exception as exc:
        raise BaselineExecutionError(
            f"failed to construct pinned {spec.name} model: {type(exc).__name__}: {exc}"
        ) from exc
    model_finished = clock()
    model_load_seconds = _positive_elapsed(model_started, model_finished, "model load")
    resolved_model_path = getattr(model, "model_path", None)
    if not isinstance(resolved_model_path, (str, os.PathLike)):
        raise BaselineExecutionError(
            "AutoModel must expose the resolved model bundle through model_path"
        )
    try:
        model_sha256 = model_bundle_hasher(Path(resolved_model_path))
    except BaselineError:
        raise
    except Exception as exc:
        raise BaselineExecutionError(
            f"failed to hash resolved model bundle: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(model_sha256, str) or SHA256_PATTERN.fullmatch(model_sha256) is None:
        raise BaselineExecutionError(
            "model bundle hasher must return sha256:<64 lowercase hex chars>"
        )

    first_item = dataset.items[0]
    cold_started = clock()
    try:
        _generate(model, spec, first_item)
    except Exception as exc:
        raise BaselineExecutionError(
            f"cold inference failed for {first_item.utterance_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    cold_finished = clock()
    cold_inference_seconds = _positive_elapsed(cold_started, cold_finished, "cold inference")

    for warmup_index in range(config.warmup_runs):
        try:
            _generate(model, spec, first_item)
        except Exception as exc:
            raise BaselineExecutionError(
                f"warmup {warmup_index + 1} failed for {first_item.utterance_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    aggregate = EditCounts()
    item_reports: list[dict[str, Any]] = []
    attempted_rtfs: list[float] = []
    successful_rtfs: list[float] = []
    failed_count = 0
    warm_wall_seconds = 0.0
    warm_audio_seconds = 0.0
    reference_units = 0

    for item in dataset.items:
        started = clock()
        raw_hypothesis = ""
        failure_reason: str | None = None
        try:
            raw_hypothesis = _generate(model, spec, item)
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
        finished = clock()
        elapsed = _positive_elapsed(started, finished, f"warm inference {item.utterance_id}")
        rtf = elapsed / item.duration_seconds
        attempted_rtfs.append(rtf)
        warm_wall_seconds += elapsed
        warm_audio_seconds += item.duration_seconds

        reference = reference_views(item.raw_text)
        hypothesis = text_views(config.track, raw_hypothesis)
        if failure_reason is None and reference["content"] and not hypothesis["content"]:
            failure_reason = "empty content hypothesis"
        status = "failed" if failure_reason is not None else "ok"
        if status == "failed":
            failed_count += 1
        else:
            successful_rtfs.append(rtf)

        counts = cer_components(reference["content"], hypothesis["content"])
        aggregate += counts
        reference_units += len(reference["content"])
        item_reports.append(
            {
                "id": item.utterance_id,
                "audio": item.audio,
                "audio_sha256": item.audio_sha256,
                "reference_sha256": item.reference_sha256,
                "duration_seconds": _metric(item.duration_seconds),
                "speaker_id": item.speaker_id,
                "session_id": item.session_id,
                "split": item.split,
                "status": status,
                "failure_reason": failure_reason,
                "reference": reference,
                "hypothesis": hypothesis,
                "metrics": {
                    "substitutions": counts.substitutions,
                    "deletions": counts.deletions,
                    "insertions": counts.insertions,
                    "reference_units": len(reference["content"]),
                    "wall_seconds": _metric(elapsed),
                    "rtf": _metric(rtf),
                },
            }
        )

    if not successful_rtfs:
        raise BaselineExecutionError("all warm inference items failed")
    measured_rss = rss_reader()
    if not isinstance(measured_rss, (int, float)) or not math.isfinite(measured_rss):
        raise BaselineExecutionError("peak RSS must be a finite number")
    if measured_rss <= 0:
        raise BaselineExecutionError("peak RSS must be greater than zero")

    metrics = {
        "content_cer": aggregate.total / reference_units,
        "substitutions": aggregate.substitutions,
        "deletions": aggregate.deletions,
        "insertions": aggregate.insertions,
        "reference_units": reference_units,
        "utterance_count": len(dataset.items),
        "failed_count": failed_count,
        "excluded_count": 0,
        "retried_count": 0,
        "rtf_attempted_count": len(attempted_rtfs),
        "rtf_successful_count": len(successful_rtfs),
        "model_load_seconds": _metric(model_load_seconds),
        "cold_inference_seconds": _metric(cold_inference_seconds),
        "cold_start_seconds": _metric(model_load_seconds + cold_inference_seconds),
        "warm_wall_seconds": _metric(warm_wall_seconds),
        "warm_audio_seconds": _metric(warm_audio_seconds),
        "rtf_p50": _metric(percentile(attempted_rtfs, 0.50)),
        "rtf_p95": _metric(percentile(attempted_rtfs, 0.95)),
        "successful_rtf_p50": _metric(percentile(successful_rtfs, 0.50)),
        "successful_rtf_p95": _metric(percentile(successful_rtfs, 0.95)),
        "peak_rss_mb": _metric(float(measured_rss)),
    }
    model_provenance = dict(config_document["model"])
    model_provenance["sha256"] = model_sha256
    return {
        "schema_version": 1,
        "kind": "offline-baseline-report",
        "track": spec.name,
        "provenance": {
            "data_sha256": dataset.manifest_sha256,
            "config_sha256": config_sha256,
            "normalizer_version": NORMALIZER_VERSION,
            "model": model_provenance,
            "command": dict(command) if command is not None else None,
        },
        "dataset": {
            "manifest": str(dataset.manifest_path),
            "data_version": dataset.data_version,
            "utterance_count": len(dataset.items),
        },
        "configuration": config_document,
        "metrics": metrics,
        "items": item_reports,
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> str:
    """Write one canonical report without overwriting existing evidence."""

    payload = canonical_json_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError as exc:
        raise BaselineExecutionError(
            f"refusing to overwrite existing report: {output_path}"
        ) from exc
    return sha256_bytes(payload)


def command_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Capture the non-secret environment variables that can affect CPU timing."""

    environment = os.environ if source is None else source
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "MODELSCOPE_CACHE",
        "PYTHONHASHSEED",
    )
    return {name: environment[name] for name in names if name in environment}
