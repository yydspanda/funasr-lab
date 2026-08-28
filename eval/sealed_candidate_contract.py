"""Pure, reference-free contract for one sealed EVAL-01 runner command."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.core_report import IDENTITY_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import BaselineError
from eval.offline_baseline import TRACKS
from eval.offline_baseline import canonical_json_bytes
from eval.offline_baseline import effective_config
from eval.offline_baseline import sha256_bytes


MODEL_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_INTEGER_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)$")
MAX_SEALED_NCPU = 4096
MAX_SEALED_WARMUP_RUNS = 100
MAX_SEALED_SEED = 4_294_967_295

TRACK_ADAPTERS = {
    "paraformer": IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
    "sensevoice": SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
}
TRACK_MODEL_CLASSES = {
    "paraformer": "Paraformer",
    "sensevoice": "SenseVoiceSmall",
}
TRACK_COMPONENT_PROFILES = {
    "paraformer": {
        "model": "Paraformer",
        "tokenizer": "CharTokenizer",
        "frontend": "WavFrontend",
        "specaug": "SpecAugLFR",
        "normalize": None,
        "encoder": "SANMEncoder",
        "decoder": "ParaformerSANMDecoder",
        "predictor": "CifPredictorV2",
    },
    "sensevoice": {
        "model": "SenseVoiceSmall",
        "tokenizer": "SentencepiecesTokenizer",
        "frontend": "WavFrontend",
        "specaug": "SpecAugLFR",
        "normalize": None,
        "encoder": "SenseVoiceEncoderSmall",
        "decoder": None,
        "predictor": None,
    },
}

RUNNER_ARGV_PREFIX = (
    ".venv/bin/python",
    "-P",
    "-S",
    "scripts/run_sealed_asr_candidate.py",
    "run",
)
RUNNER_OPTION_ORDER = (
    "--input-projection",
    "--candidate-lock",
    "--input-receipt",
    "--audio-root",
    "--track",
    "--model-revision",
    "--device",
    "--ncpu",
    "--warmup-runs",
    "--seed",
    "--hypothesis-adapter-version",
    "--output-raw-predictions",
    "--output-execution-envelope",
)
RUNNER_REQUIRED_ENVIRONMENT_NAMES = frozenset(
    {
        "CRC32C_SW_MODE",
        "HYDRA_FULL_ERROR",
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "MKL_NUM_THREADS",
        "MODELSCOPE_CACHE",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "TORCHINDUCTOR_CACHE_DIR",
    }
)


class SealedCandidateContractError(ValueError):
    """Raised before sealed references are opened or a model is constructed."""


@dataclass(frozen=True)
class FrozenExecutionPlan:
    """Validated command facts shared by the custodian and decoder."""

    baseline_config: BaselineConfig
    hypothesis_adapter_version: str
    input_projection: str
    candidate_lock: str
    input_receipt: str
    audio_root: str
    output_raw_predictions: str
    output_execution_envelope: str


def _canonical_integer(
    value: str,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, str) or CANONICAL_INTEGER_PATTERN.fullmatch(value) is None:
        raise SealedCandidateContractError(
            f"candidate command {field} must be a canonical decimal integer"
        )
    maximum_text = str(maximum)
    if len(value) > len(maximum_text) or (
        len(value) == len(maximum_text) and value > maximum_text
    ):
        raise SealedCandidateContractError(
            f"candidate command {field} must be at most {maximum}"
        )
    parsed = int(value)
    if parsed < minimum:
        raise SealedCandidateContractError(
            f"candidate command {field} must be at least {minimum}"
        )
    return parsed


def _command_options(candidate: Mapping[str, Any]) -> dict[str, str]:
    command = candidate.get("command")
    if not isinstance(command, Mapping) or command.get("working_directory") != ".":
        raise SealedCandidateContractError(
            "candidate command working_directory must be exactly '.'"
        )
    argv = command.get("argv")
    expected_length = len(RUNNER_ARGV_PREFIX) + 2 * len(RUNNER_OPTION_ORDER)
    if (
        not isinstance(argv, list)
        or len(argv) != expected_length
        or tuple(argv[: len(RUNNER_ARGV_PREFIX)]) != RUNNER_ARGV_PREFIX
    ):
        raise SealedCandidateContractError(
            "candidate command must use the complete canonical sealed runner argv"
        )

    options: dict[str, str] = {}
    cursor = len(RUNNER_ARGV_PREFIX)
    for expected_option in RUNNER_OPTION_ORDER:
        option = argv[cursor]
        value = argv[cursor + 1]
        if option != expected_option or not isinstance(value, str) or not value:
            raise SealedCandidateContractError(
                "candidate command must use the complete canonical sealed runner argv"
            )
        if value.startswith("--") or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise SealedCandidateContractError(
                f"candidate command {expected_option} value is unsafe"
            )
        options[expected_option] = value
        cursor += 2
    return options


def _validate_environment(
    environment: Any,
    *,
    repository_root: Path,
    ncpu: int,
) -> None:
    if not isinstance(environment, Mapping) or set(environment) != set(
        RUNNER_REQUIRED_ENVIRONMENT_NAMES
    ):
        raise SealedCandidateContractError(
            "candidate command environment must contain exactly the sealed CPU names"
        )
    if any(not isinstance(value, str) for value in environment.values()):
        raise SealedCandidateContractError(
            "candidate command environment values must be strings"
        )
    fixed_values = {
        "CRC32C_SW_MODE": "auto",
        "HYDRA_FULL_ERROR": "1",
        "KMP_DUPLICATE_LIB_OK": "True",
        "KMP_INIT_AT_FORK": "FALSE",
        "PYTHONHASHSEED": "0",
    }
    for name, expected in fixed_values.items():
        if environment[name] != expected:
            raise SealedCandidateContractError(
                f"candidate command environment {name} must be exactly {expected!r}"
            )
    expected_threads = str(ncpu)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if environment[name] != expected_threads:
            raise SealedCandidateContractError(
                f"candidate command environment {name} must equal ncpu"
            )

    root = repository_root.resolve()
    expected_caches = {
        "MODELSCOPE_CACHE": root / ".cache/modelscope",
        "TORCHINDUCTOR_CACHE_DIR": root / ".cache/torchinductor",
    }
    for name, expected in expected_caches.items():
        configured = Path(environment[name])
        if not configured.is_absolute() or configured != expected:
            raise SealedCandidateContractError(
                f"candidate command environment {name} must be the exact "
                "repository-local path without aliases"
            )


def validate_sealed_candidate_execution(
    candidate: Mapping[str, Any],
    requested_adapter: str,
    repository_root: Path,
) -> FrozenExecutionPlan:
    """Validate every candidate fact needed by the sealed CPU v1 runner."""

    if not isinstance(candidate, Mapping):
        raise SealedCandidateContractError("candidate must be a mapping")
    options = _command_options(candidate)
    track = options["--track"]
    if track not in TRACKS:
        raise SealedCandidateContractError("candidate command track is unsupported")
    revision = options["--model-revision"]
    if MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        raise SealedCandidateContractError(
            "candidate model revision must be a full immutable snapshot commit"
        )
    if options["--device"] != "cpu":
        raise SealedCandidateContractError("candidate command device must be cpu")
    ncpu = _canonical_integer(
        options["--ncpu"],
        "ncpu",
        minimum=1,
        maximum=MAX_SEALED_NCPU,
    )
    warmup_runs = _canonical_integer(
        options["--warmup-runs"],
        "warmup-runs",
        minimum=0,
        maximum=MAX_SEALED_WARMUP_RUNS,
    )
    seed = _canonical_integer(
        options["--seed"],
        "seed",
        minimum=0,
        maximum=MAX_SEALED_SEED,
    )
    if seed != 0 or candidate.get("seed") != seed:
        raise SealedCandidateContractError("candidate seed must be integer 0")

    adapter = options["--hypothesis-adapter-version"]
    expected_adapter = TRACK_ADAPTERS[track]
    if adapter != expected_adapter or requested_adapter != expected_adapter:
        raise SealedCandidateContractError(
            "candidate hypothesis adapter does not match the fixed model track"
        )

    hardware = candidate.get("hardware")
    hardware_fields = {
        "host_id",
        "os",
        "cpu_model",
        "logical_cpu_count",
        "memory_bytes",
        "device",
        "accelerator",
    }
    if not isinstance(hardware, Mapping) or set(hardware) != hardware_fields:
        raise SealedCandidateContractError(
            "candidate hardware must contain the exact sealed CPU fields"
        )
    logical_cpu_count = hardware.get("logical_cpu_count")
    memory_bytes = hardware.get("memory_bytes")
    if (
        hardware.get("device") != "cpu"
        or hardware.get("accelerator") is not None
        or any(
            not isinstance(hardware.get(name), str) or not hardware.get(name)
            for name in ("host_id", "os", "cpu_model")
        )
        or isinstance(logical_cpu_count, bool)
        or not isinstance(logical_cpu_count, int)
        or logical_cpu_count < ncpu
        or isinstance(memory_bytes, bool)
        or not isinstance(memory_bytes, int)
        or memory_bytes <= 0
    ):
        raise SealedCandidateContractError(
            "candidate hardware must provide the declared sealed CPU capacity"
        )

    command = candidate["command"]
    _validate_environment(
        command.get("environment"),
        repository_root=repository_root,
        ncpu=ncpu,
    )

    baseline_config = BaselineConfig(
        track=track,
        model_revision=revision,
        device="cpu",
        ncpu=ncpu,
        warmup_runs=warmup_runs,
        seed=seed,
    )
    try:
        config_document = effective_config(baseline_config)
    except BaselineError as exc:
        raise SealedCandidateContractError(str(exc)) from exc
    expected_config_sha256 = sha256_bytes(canonical_json_bytes(config_document))
    if candidate.get("config_sha256") != expected_config_sha256:
        raise SealedCandidateContractError(
            "candidate config_sha256 does not match the sealed effective config"
        )
    if candidate.get("normalizer_version") != config_document["pipeline"][
        "normalizer_version"
    ]:
        raise SealedCandidateContractError(
            "candidate normalizer_version does not match the sealed config"
        )

    models = candidate.get("models")
    expected_spec = TRACKS[track]
    if (
        not isinstance(models, list)
        or len(models) != 1
        or not isinstance(models[0], Mapping)
        or models[0].get("role") != "asr"
        or models[0].get("identifier") != expected_spec.model_identifier
        or models[0].get("revision") != revision
    ):
        raise SealedCandidateContractError(
            "candidate model component does not match the sealed model track"
        )

    return FrozenExecutionPlan(
        baseline_config=baseline_config,
        hypothesis_adapter_version=adapter,
        input_projection=options["--input-projection"],
        candidate_lock=options["--candidate-lock"],
        input_receipt=options["--input-receipt"],
        audio_root=options["--audio-root"],
        output_raw_predictions=options["--output-raw-predictions"],
        output_execution_envelope=options["--output-execution-envelope"],
    )


__all__ = [
    "FrozenExecutionPlan",
    "MAX_SEALED_NCPU",
    "MAX_SEALED_SEED",
    "MAX_SEALED_WARMUP_RUNS",
    "MODEL_REVISION_PATTERN",
    "RUNNER_ARGV_PREFIX",
    "RUNNER_OPTION_ORDER",
    "RUNNER_REQUIRED_ENVIRONMENT_NAMES",
    "SealedCandidateContractError",
    "TRACK_ADAPTERS",
    "TRACK_COMPONENT_PROFILES",
    "TRACK_MODEL_CLASSES",
    "validate_sealed_candidate_execution",
]
