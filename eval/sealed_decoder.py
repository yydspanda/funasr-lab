"""Controlled, reference-free decoder for one sealed EVAL candidate.

The runner accepts only the audio-only projection and its custodian-owned
candidate lock.  It deliberately has no descriptor, collection-manifest,
reference-text, normalization, scoring, or core-report input.  Real model
construction is delayed until every reference-free identity and every audio
file has been checked.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import stat
import subprocess
import sys
import time
import unicodedata
import wave
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from eval.core_report import CoreReportValidationError
from eval.core_report import adapt_hypothesis
from eval.custodian_replay import LoadedArtifact
from eval.custodian_replay import load_candidate_lock
from eval.custodian_replay import load_sealed_input_projection
from eval.custodian_replay import validate_decode_handoff
from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import BaselineError
from eval.offline_baseline import TRACKS
from eval.offline_baseline import canonical_json_bytes
from eval.offline_baseline import effective_config
from eval.offline_baseline import model_directory_sha256
from eval.offline_baseline import sha256_bytes
from eval.sealed_candidate_contract import MODEL_REVISION_PATTERN
from eval.sealed_candidate_contract import RUNNER_REQUIRED_ENVIRONMENT_NAMES
from eval.sealed_candidate_contract import SealedCandidateContractError
from eval.sealed_candidate_contract import TRACK_COMPONENT_PROFILES
from eval.sealed_candidate_contract import TRACK_MODEL_CLASSES
from eval.sealed_candidate_contract import validate_sealed_candidate_execution


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_GIT_PATH = Path("/usr/bin/git")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

CLOCK_VERSION = "python-perf-counter-ns-v1"
RSS_VERSION = "linux-rusage-self-maxrss-kib-v1"
RSS_SCOPE = "fresh-process-rusage-self"
RTF_POPULATION = "all-measured-attempts"
SEALED_PYCACHE_PREFIX = "/dev/null/asr-lab-sealed-pycache"

MAX_HYPOTHESIS_CHARACTERS = 16_384
MAX_TOTAL_HYPOTHESIS_CHARACTERS = 1_000_000
MAX_RUNNER_SOURCE_BYTES = 4 * 1024 * 1024
MAX_AUDIO_BYTES_PER_ITEM = 2 * 1024 * 1024 * 1024
LAB_CPU_LOCK_PATH = "requirements/lab-cpu.lock"
LOCKED_REQUIREMENT_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)$"
)
MODEL_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
PYTHON_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$")
PYTHON_CACHE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
RUNTIME_IDENTITY_FIELDS = frozenset(
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
MODEL_STACK_MODULE_ROOTS = frozenset(
    {
        "funasr",
        "kaldiio",
        "librosa",
        "modelscope",
        "numba",
        "numpy",
        "scipy",
        "sklearn",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers",
    }
)

ALLOWED_ENVIRONMENT_NAMES = tuple(sorted(RUNNER_REQUIRED_ENVIRONMENT_NAMES))
FORBIDDEN_STARTUP_ENVIRONMENT_NAMES = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONPYCACHEPREFIX",
        "PYTHONOPTIMIZE",
        "PYTHONINSPECT",
        "PYTHONMALLOC",
        "PYTHONTRACEMALLOC",
        "PYTHONWARNINGS",
        "PYTHONINTMAXSTRDIGITS",
        "PYTHONUTF8",
        "PYTHONCOERCECLOCALE",
        "PYTHONIOENCODING",
        "PYTHONDEVMODE",
        "PYTHONWARNDEFAULTENCODING",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
        "GOMP_CPU_AFFINITY",
    }
)
FORBIDDEN_STARTUP_ENVIRONMENT_PREFIXES = (
    "LD_",
    "FUNASR_",
    "TORCH_",
)

# These files own the runner behavior or a validation/configuration primitive
# it executes before constructing the pinned upstream model.  A sealed run
# refuses local bytes that differ from the candidate's committed code revision.
RUNNER_SOURCE_PATHS = (
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
    LAB_CPU_LOCK_PATH,
    "scripts/check_experiment_manifests.py",
    "scripts/__init__.py",
    "scripts/run_sealed_asr_candidate.py",
)


class SealedDecoderError(ValueError):
    """Raised before evidence is published when a sealed run is invalid."""


@dataclass(frozen=True)
class SealedDecoderConfig:
    """The complete supported v1 decode configuration."""

    track: str
    model_revision: str
    device: str
    ncpu: int
    warmup_runs: int
    seed: int
    hypothesis_adapter_version: str


@dataclass(frozen=True)
class SealedAudioItem:
    """One verified audio-only input item."""

    utterance_id: str
    audio_root: Path
    relative_audio: str
    audio_sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width_bits: int


@dataclass(frozen=True)
class SealedDecoderResult:
    """Reference-free predictions plus facts for the execution envelope."""

    prediction_items: tuple[dict[str, object], ...]
    observation: dict[str, object]


def allowed_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Capture only non-secret environment variables allowed by the contract."""

    environment = os.environ if source is None else source
    return {
        name: environment[name]
        for name in ALLOWED_ENVIRONMENT_NAMES
        if name in environment
    }


def _reject_unsafe_startup_environment(environment: Mapping[str, str]) -> None:
    for name, value in environment.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise SealedDecoderError("process environment must contain string pairs")
        if not value:
            continue
        if (
            (name.startswith("PYTHON") and name != "PYTHONHASHSEED")
            or name in FORBIDDEN_STARTUP_ENVIRONMENT_NAMES
            or name.startswith(FORBIDDEN_STARTUP_ENVIRONMENT_PREFIXES)
            or (
                name.startswith("KMP_")
                and name not in {"KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK"}
            )
        ):
            raise SealedDecoderError(
                f"sealed decoder forbids non-empty startup environment {name}"
            )


@contextmanager
def sanitized_process_environment(
    environment: Mapping[str, str],
) -> Iterator[None]:
    """Expose only the frozen allowlist while model code is importable."""

    if any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in environment.items()
    ):
        raise SealedDecoderError("sanitized environment must contain string pairs")
    if not set(environment) <= set(ALLOWED_ENVIRONMENT_NAMES):
        raise SealedDecoderError("sanitized environment contains a non-allowlisted name")
    previous = dict(os.environ)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    os.environ.clear()
    os.environ.update(environment)
    sys.dont_write_bytecode = True
    # ``dont_write_bytecode`` alone still permits reads from a checkout-local
    # cache.  Redirect lookup to a path below /dev/null so later FunASR imports
    # must execute the source bytes already verified against the candidate.
    sys.pycache_prefix = SEALED_PYCACHE_PREFIX
    try:
        yield
    finally:
        sys.pycache_prefix = previous_pycache_prefix
        sys.dont_write_bytecode = previous_dont_write_bytecode
        os.environ.clear()
        os.environ.update(previous)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def runtime_identity(repository_root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """Validate the project interpreter and exact lab-cpu installation."""

    root = repository_root.resolve()
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 11):
        raise SealedDecoderError("sealed decoder requires CPython 3.11")
    expected_prefix = root / ".venv"
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise SealedDecoderError("sealed decoder must use the repository-local .venv")

    lock_path = root / LAB_CPU_LOCK_PATH
    lock_payload = _read_bounded_regular_file(
        lock_path,
        MAX_RUNNER_SOURCE_BYTES,
        "lab CPU dependency lock",
    )
    try:
        lock_text = lock_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SealedDecoderError("lab CPU dependency lock must use UTF-8") from exc
    locked: dict[str, str] = {}
    editable_project_seen = False
    for line_number, raw_line in enumerate(lock_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or raw_line[:1].isspace():
            continue
        if line == "-e .":
            if editable_project_seen:
                raise SealedDecoderError("lab CPU dependency lock repeats '-e .'")
            editable_project_seen = True
            continue
        match = LOCKED_REQUIREMENT_PATTERN.fullmatch(line)
        if match is None:
            raise SealedDecoderError(
                f"lab CPU dependency lock line {line_number} is unsupported"
            )
        name = _canonical_distribution_name(match.group(1))
        if name in locked:
            raise SealedDecoderError("lab CPU dependency lock repeats a distribution")
        locked[name] = match.group(2)
    if not editable_project_seen or not locked:
        raise SealedDecoderError("lab CPU dependency lock is incomplete")

    installed: dict[str, tuple[str, importlib_metadata.Distribution]] = {}
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise SealedDecoderError("installed distribution has no stable name")
        name = _canonical_distribution_name(raw_name)
        if name in installed:
            raise SealedDecoderError(f"installed distribution {name!r} is ambiguous")
        installed[name] = (distribution.version, distribution)
    expected_names = set(locked) | {"funasr"}
    if set(installed) != expected_names:
        raise SealedDecoderError(
            "installed distributions do not exactly match requirements/lab-cpu.lock"
        )
    for name, expected_version in locked.items():
        if installed[name][0] != expected_version:
            raise SealedDecoderError(
                f"installed distribution {name!r} does not match lab-cpu.lock"
            )

    funasr_distribution = installed["funasr"][1]
    try:
        direct_url = json.loads(funasr_distribution.read_text("direct_url.json") or "")
    except (json.JSONDecodeError, TypeError) as exc:
        raise SealedDecoderError(
            "installed funasr must expose editable direct_url metadata"
        ) from exc
    if not isinstance(direct_url, Mapping):
        raise SealedDecoderError(
            "installed funasr direct_url metadata must be an object"
        )
    parsed_url = urlparse(str(direct_url.get("url", "")))
    directory_info = direct_url.get("dir_info")
    editable = (
        isinstance(directory_info, Mapping)
        and directory_info.get("editable") is True
    )
    source_path = Path(unquote(parsed_url.path)).resolve()
    if (
        parsed_url.scheme != "file"
        or parsed_url.netloc
        or not editable
        or source_path != root
    ):
        raise SealedDecoderError(
            "installed funasr must be the repository-local editable distribution"
        )

    inventory = [
        {"name": name, "version": installed[name][0]}
        for name in sorted(installed)
    ]
    cache_tag = sys.implementation.cache_tag
    if not isinstance(cache_tag, str) or not cache_tag:
        raise SealedDecoderError("Python cache tag is unavailable")
    return {
        "python_implementation": "cpython",
        "python_version": platform.python_version(),
        "python_cache_tag": cache_tag,
        "dependency_lock_sha256": sha256_bytes(lock_payload),
        "installed_dependencies_sha256": sha256_bytes(
            canonical_json_bytes(inventory)
        ),
        "installed_dependency_count": len(inventory),
        "unicode_version": unicodedata.unidata_version,
    }


def _validated_runtime_identity(value: Mapping[str, object]) -> dict[str, object]:
    runtime = dict(value)
    if set(runtime) != RUNTIME_IDENTITY_FIELDS:
        raise SealedDecoderError("runtime identity fields are invalid")
    if runtime["python_implementation"] != "cpython":
        raise SealedDecoderError("runtime identity requires CPython")
    python_version = runtime["python_version"]
    cache_tag = runtime["python_cache_tag"]
    if (
        not isinstance(python_version, str)
        or len(python_version) > 64
        or PYTHON_VERSION_PATTERN.fullmatch(python_version) is None
    ):
        raise SealedDecoderError("runtime Python version is invalid")
    if (
        not isinstance(cache_tag, str)
        or len(cache_tag) > 64
        or PYTHON_CACHE_TAG_PATTERN.fullmatch(cache_tag) is None
    ):
        raise SealedDecoderError("runtime Python cache tag is invalid")
    for field in ("dependency_lock_sha256", "installed_dependencies_sha256"):
        digest = runtime[field]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise SealedDecoderError(f"runtime {field} is invalid")
    count = runtime["installed_dependency_count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10_000:
        raise SealedDecoderError("runtime installed dependency count is invalid")
    unicode_version = runtime["unicode_version"]
    if (
        not isinstance(unicode_version, str)
        or re.fullmatch(r"^[0-9]+\.[0-9]+\.[0-9]+$", unicode_version) is None
    ):
        raise SealedDecoderError("runtime Unicode version is invalid")
    return runtime


def _validate_reproducibility_environment(
    environment: Mapping[str, str],
    repository_root: Path,
    ncpu: int,
) -> None:
    def validate_cache_path(name: str, relative: Path) -> None:
        expected = repository_root.resolve() / relative
        configured = environment.get(name)
        if configured is None or Path(configured) != expected:
            raise SealedDecoderError(
                f"{name} must be the exact repository-local {relative.as_posix()} path"
            )
        current = repository_root.resolve()
        for component in relative.parts:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                break
            except OSError as exc:
                raise SealedDecoderError(f"cannot inspect {name} path") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise SealedDecoderError(
                    f"{name} path must contain only real directories"
                )

    if environment.get("PYTHONHASHSEED") != "0":
        raise SealedDecoderError("PYTHONHASHSEED must be exactly 0")
    if environment.get("HYDRA_FULL_ERROR") != "1":
        raise SealedDecoderError("HYDRA_FULL_ERROR must be exactly 1")
    fixed_import_environment = {
        "CRC32C_SW_MODE": "auto",
        "KMP_DUPLICATE_LIB_OK": "True",
        "KMP_INIT_AT_FORK": "FALSE",
    }
    for name, expected in fixed_import_environment.items():
        if environment.get(name) != expected:
            raise SealedDecoderError(f"{name} must be exactly {expected!r}")
    validate_cache_path("MODELSCOPE_CACHE", Path(".cache/modelscope"))
    validate_cache_path(
        "TORCHINDUCTOR_CACHE_DIR", Path(".cache/torchinductor")
    )
    expected_threads = str(ncpu)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if environment.get(name) != expected_threads:
            raise SealedDecoderError(
                f"{name} must be explicitly set to ncpu={expected_threads}"
            )


def _cpu_model() -> str:
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as source:
            for line in source:
                name, separator, value = line.partition(":")
                if separator and name.strip() in {"model name", "Hardware"}:
                    model = value.strip()
                    if model:
                        return model
    except OSError as exc:
        raise SealedDecoderError("cannot identify the CPU model") from exc
    fallback = platform.processor().strip()
    if not fallback:
        raise SealedDecoderError("cannot identify the CPU model")
    return fallback


def _installed_memory_bytes() -> int:
    try:
        page_count = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError) as exc:
        raise SealedDecoderError("cannot identify installed memory") from exc
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_count <= 0
        or page_size <= 0
    ):
        raise SealedDecoderError("installed memory identity is invalid")
    return page_count * page_size


def _integer_ranges(values: Sequence[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return "unavailable"
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _cpu_affinity() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return ()
    try:
        values = tuple(sorted(int(value) for value in getter(0)))
    except (OSError, TypeError, ValueError):
        return ()
    return values


def _cgroup_cpu_max() -> str:
    cgroup_root = Path("/sys/fs/cgroup")
    relative = Path(".")
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                relative = Path(path.removeprefix("/"))
                break
        candidate = (cgroup_root / relative / "cpu.max").resolve()
        candidate.relative_to(cgroup_root)
        value = " ".join(candidate.read_text(encoding="ascii").split())
    except (OSError, UnicodeDecodeError, ValueError):
        return "unavailable"
    if re.fullmatch(r"(?:max|[0-9]+) [0-9]+", value) is None:
        return "unavailable"
    return value


def _cpu_governors(cpu_ids: Sequence[int]) -> str:
    governors: set[str] = set()
    for cpu_id in cpu_ids:
        path = Path(
            f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor"
        )
        try:
            value = path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if re.fullmatch(r"[A-Za-z0-9_-]+", value):
            governors.add(value)
    return ",".join(sorted(governors)) if governors else "unavailable"


def _operating_system_identity() -> str:
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    distribution = "-".join(
        value
        for value in (
            str(release.get("ID", "")).strip(),
            str(release.get("VERSION_ID", "")).strip(),
        )
        if value
    ) or "unavailable"
    affinity = _cpu_affinity()
    identity = (
        f"{platform.system()} {platform.release()} {platform.machine()}; "
        f"distro={distribution}; affinity={_integer_ranges(affinity)}; "
        f"cpu.max={_cgroup_cpu_max()}; governor={_cpu_governors(affinity)}"
    )
    if len(identity) > 1024:
        raise SealedDecoderError("operating system identity exceeds the contract limit")
    return identity


def hardware_identity(device: str = "cpu") -> dict[str, object]:
    """Return the actual v1 hardware identity compared with the candidate lock."""

    if device != "cpu":
        raise SealedDecoderError("sealed decoder v1 only supports device 'cpu'")
    host_id = platform.node().strip()
    logical_cpu_count = os.cpu_count()
    if not host_id:
        raise SealedDecoderError("cannot identify a stable host_id")
    if (
        isinstance(logical_cpu_count, bool)
        or not isinstance(logical_cpu_count, int)
        or logical_cpu_count <= 0
    ):
        raise SealedDecoderError("cannot identify logical CPU count")
    os_identity = _operating_system_identity()
    if not os_identity:
        raise SealedDecoderError("cannot identify the operating system")
    return {
        "host_id": host_id,
        "os": os_identity,
        "cpu_model": _cpu_model(),
        "logical_cpu_count": logical_cpu_count,
        "memory_bytes": _installed_memory_bytes(),
        "device": "cpu",
        "accelerator": None,
    }


def describe_runtime(
    ncpu: int,
    *,
    device: str = "cpu",
    environment_source: Mapping[str, str] | None = None,
    current_directory: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    hardware_reader: Callable[[str], Mapping[str, object]] = hardware_identity,
    runtime_identity_reader: Callable[[Path], Mapping[str, object]] = runtime_identity,
) -> dict[str, object]:
    """Describe pre-registerable runtime facts without loading audio or a model."""

    if isinstance(ncpu, bool) or not isinstance(ncpu, int) or ncpu <= 0:
        raise SealedDecoderError("ncpu must be a positive integer")
    root = repository_root.resolve()
    cwd = (Path.cwd() if current_directory is None else current_directory).resolve()
    if cwd != root:
        raise SealedDecoderError("sealed decoder must run from the repository root")
    complete_environment = os.environ if environment_source is None else environment_source
    _reject_unsafe_startup_environment(complete_environment)
    environment = allowed_environment(complete_environment)
    _validate_reproducibility_environment(environment, root, ncpu)
    hardware = dict(hardware_reader(device))
    if hardware.get("device") != device:
        raise SealedDecoderError("described hardware device does not match request")
    runtime = _validated_runtime_identity(runtime_identity_reader(root))
    return {
        "working_directory": ".",
        "environment": environment,
        "hardware": hardware,
        "runtime": runtime,
    }


def peak_rss_bytes() -> int:
    """Return Linux RUSAGE_SELF high-water RSS as integer bytes."""

    if not sys_platform_is_linux():
        raise SealedDecoderError(
            "sealed decoder v1 RSS measurement requires Linux"
        )
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SealedDecoderError("RUSAGE_SELF peak RSS is invalid")
    if not math.isfinite(float(value)) or value <= 0:
        raise SealedDecoderError("RUSAGE_SELF peak RSS must be positive")
    return int(math.ceil(float(value) * 1024))


def sys_platform_is_linux() -> bool:
    return platform.system() == "Linux"


def _read_bounded_regular_file(path: Path, maximum: int, context: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise SealedDecoderError(f"cannot safely read {context}: O_NOFOLLOW unavailable")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise SealedDecoderError(f"cannot read {context}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SealedDecoderError(f"{context} must be a regular file")
        if before.st_size > maximum:
            raise SealedDecoderError(f"{context} exceeds the {maximum}-byte limit")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) > maximum:
            raise SealedDecoderError(f"{context} exceeds the {maximum}-byte limit")
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
            raise SealedDecoderError(f"{context} changed while it was read")
        try:
            path_after = path.lstat()
        except OSError as exc:
            raise SealedDecoderError(
                f"{context} path changed while it was read"
            ) from exc
        if (
            not stat.S_ISREG(path_after.st_mode)
            or (path_after.st_dev, path_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise SealedDecoderError(
                f"{context} path changed while it was read"
            )
        return bytes(payload)
    finally:
        os.close(descriptor)


def _committed_funasr_paths(
    root: Path,
    code_commit: str,
    git_environment: Mapping[str, str],
) -> tuple[str, ...]:
    try:
        payload = subprocess.run(
            [
                str(TRUSTED_GIT_PATH),
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                code_commit,
                "--",
                "funasr",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            env=dict(git_environment),
        ).stdout
        decoded = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError) as exc:
        raise SealedDecoderError("cannot enumerate committed funasr source") from exc
    paths = tuple(item for item in decoded.rstrip("\x00").split("\x00") if item)
    if not paths or len(paths) != len(set(paths)):
        raise SealedDecoderError("committed funasr source inventory is invalid")
    for relative_path in paths:
        posix_path = PurePosixPath(relative_path)
        if (
            not posix_path.parts
            or posix_path.parts[0] != "funasr"
            or any(part in {"", ".", ".."} for part in posix_path.parts)
        ):
            raise SealedDecoderError("committed funasr source path is unsafe")
    return tuple(sorted(paths))


def _reject_alternative_funasr_source(
    root: Path,
    committed_paths: Sequence[str],
) -> None:
    funasr_root = root / "funasr"
    if funasr_root.is_symlink() or not funasr_root.is_dir():
        raise SealedDecoderError("local funasr source root must be a regular directory")
    expected = set(committed_paths)
    for path in funasr_root.rglob("*"):
        if path.is_symlink():
            raise SealedDecoderError("local funasr source must not contain symlinks")
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path in expected:
            continue
        if path.parent.name == "__pycache__" and path.suffix == ".pyc":
            # The sanitized import context cannot read these normal local
            # compileall/test artefacts because its cache prefix is redirected.
            continue
        raise SealedDecoderError(
            f"local funasr source contains uncommitted runtime file {relative_path}"
        )


def runner_source_identity(
    code_commit: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, str]:
    """Bind exact runner bytes to the candidate's committed code revision."""

    if not isinstance(code_commit, str) or GIT_COMMIT_PATTERN.fullmatch(
        code_commit
    ) is None:
        raise SealedDecoderError("candidate code_commit must be a full Git commit")
    root = repository_root.resolve()
    try:
        git_metadata = TRUSTED_GIT_PATH.lstat()
    except OSError as exc:
        raise SealedDecoderError("trusted Git executable is unavailable") from exc
    if (
        stat.S_ISLNK(git_metadata.st_mode)
        or not stat.S_ISREG(git_metadata.st_mode)
        or stat.S_IMODE(git_metadata.st_mode) & 0o022
    ):
        raise SealedDecoderError("trusted Git executable is unsafe")
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
        raise SealedDecoderError("cannot inspect the trusted Git object view") from exc
    common_path = Path(common_directory)
    try:
        common_metadata = common_path.lstat()
    except OSError as exc:
        raise SealedDecoderError("Git common directory is unavailable") from exc
    if (
        not common_path.is_absolute()
        or stat.S_ISLNK(common_metadata.st_mode)
        or not stat.S_ISDIR(common_metadata.st_mode)
    ):
        raise SealedDecoderError("Git common directory is unsafe")
    if replacement_refs:
        raise SealedDecoderError("Git replacement refs are forbidden")
    if os.path.lexists(common_path / "info/grafts"):
        raise SealedDecoderError("Git grafts are forbidden")
    try:
        resolved = subprocess.run(
            [str(TRUSTED_GIT_PATH), "rev-parse", "--verify", f"{code_commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SealedDecoderError("candidate code_commit does not resolve") from exc
    if resolved != code_commit:
        raise SealedDecoderError("candidate code_commit did not resolve exactly")

    funasr_paths = _committed_funasr_paths(root, code_commit, git_environment)
    _reject_alternative_funasr_source(root, funasr_paths)
    source_paths = tuple(sorted(set(RUNNER_SOURCE_PATHS) | set(funasr_paths)))
    inventory: list[dict[str, str]] = []
    for relative_path in source_paths:
        payload = _read_bounded_regular_file(
            root / relative_path,
            MAX_RUNNER_SOURCE_BYTES,
            f"runner source {relative_path}",
        )
        try:
            committed = subprocess.run(
                [str(TRUSTED_GIT_PATH), "show", f"{code_commit}:{relative_path}"],
                cwd=root,
                check=True,
                capture_output=True,
                env=git_environment,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SealedDecoderError(
                f"cannot load committed runner source {relative_path}"
            ) from exc
        if payload != committed:
            raise SealedDecoderError(
                f"runner source {relative_path} differs from candidate code_commit; "
                "commit the runner before a sealed decode"
            )
        inventory.append(
            {"path": relative_path, "sha256": sha256_bytes(payload)}
        )
    return code_commit, sha256_bytes(canonical_json_bytes(inventory))


def _audio_path_parts(relative_audio: str, context: str) -> tuple[str, ...]:
    """Return one canonical relative POSIX audio path as safe components."""

    if not isinstance(relative_audio, str):
        raise SealedDecoderError(f"{context}.audio must be a safe relative path")
    posix_path = PurePosixPath(relative_audio)
    if (
        posix_path.is_absolute()
        or not posix_path.parts
        or posix_path.as_posix() != relative_audio
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise SealedDecoderError(f"{context}.audio must be a safe relative path")
    return posix_path.parts


def _open_audio_descriptor(
    audio_root: Path,
    relative_audio: str,
    context: str,
) -> tuple[int, tuple[int, ...], tuple[str, ...], Path]:
    """Open a leaf beneath one verified root without following any component."""

    root = Path(os.path.abspath(audio_root))
    try:
        root_before = root.lstat()
    except OSError as exc:
        raise SealedDecoderError(
            "audio_root must be an existing non-symlink directory"
        ) from exc
    if (
        root.resolve() != root
        or stat.S_ISLNK(root_before.st_mode)
        or not stat.S_ISDIR(root_before.st_mode)
    ):
        raise SealedDecoderError(
            "audio_root must be an existing non-symlink directory"
        )
    parts = _audio_path_parts(relative_audio, context)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise SealedDecoderError(
            "safe audio loading requires O_NOFOLLOW and O_DIRECTORY"
        )
    directory_flags = (
        os.O_RDONLY
        | no_follow
        | directory_only
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_descriptors: list[int] = []
    leaf_descriptor: int | None = None
    try:
        directory_descriptors.append(os.open(root, directory_flags))
        root_opened = os.fstat(directory_descriptors[0])
        root_after = root.lstat()
        expected_root_identity = (root_before.st_dev, root_before.st_ino)
        if (
            not stat.S_ISDIR(root_opened.st_mode)
            or expected_root_identity
            != (root_opened.st_dev, root_opened.st_ino)
            or expected_root_identity
            != (root_after.st_dev, root_after.st_ino)
        ):
            raise SealedDecoderError("audio_root changed while it was opened")
        for component in parts[:-1]:
            child_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptors[-1],
            )
            try:
                child_metadata = os.fstat(child_descriptor)
            except OSError:
                os.close(child_descriptor)
                raise
            if not stat.S_ISDIR(child_metadata.st_mode):
                os.close(child_descriptor)
                raise SealedDecoderError(
                    f"{context}.audio parent must be a directory"
                )
            directory_descriptors.append(child_descriptor)
        leaf_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptors[-1],
        )
    except SealedDecoderError:
        if leaf_descriptor is not None:
            os.close(leaf_descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)
        raise
    except OSError as exc:
        if leaf_descriptor is not None:
            os.close(leaf_descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        raise SealedDecoderError(f"cannot safely open {context}.audio") from exc
    return leaf_descriptor, tuple(directory_descriptors), parts, root


def _validate_open_audio_binding(
    root: Path,
    directory_descriptors: tuple[int, ...],
    parts: tuple[str, ...],
    leaf_metadata: os.stat_result,
    context: str,
) -> None:
    """Rebind the opened descriptor chain to its current lexical path."""

    try:
        root_path_metadata = root.lstat()
        root_descriptor_metadata = os.fstat(directory_descriptors[0])
        if (
            not stat.S_ISDIR(root_path_metadata.st_mode)
            or (root_path_metadata.st_dev, root_path_metadata.st_ino)
            != (root_descriptor_metadata.st_dev, root_descriptor_metadata.st_ino)
        ):
            raise SealedDecoderError(
                f"{context}.audio path changed while it was verified"
            )
        for index, component in enumerate(parts[:-1]):
            path_metadata = os.stat(
                component,
                dir_fd=directory_descriptors[index],
                follow_symlinks=False,
            )
            child_metadata = os.fstat(directory_descriptors[index + 1])
            if (
                not stat.S_ISDIR(path_metadata.st_mode)
                or (path_metadata.st_dev, path_metadata.st_ino)
                != (child_metadata.st_dev, child_metadata.st_ino)
            ):
                raise SealedDecoderError(
                    f"{context}.audio path changed while it was verified"
                )
        leaf_path_metadata = os.stat(
            parts[-1],
            dir_fd=directory_descriptors[-1],
            follow_symlinks=False,
        )
    except SealedDecoderError:
        raise
    except OSError as exc:
        raise SealedDecoderError(
            f"{context}.audio path changed while it was verified"
        ) from exc
    if (
        not stat.S_ISREG(leaf_path_metadata.st_mode)
        or (leaf_path_metadata.st_dev, leaf_path_metadata.st_ino)
        != (leaf_metadata.st_dev, leaf_metadata.st_ino)
    ):
        raise SealedDecoderError(
            f"{context}.audio path changed while it was verified"
        )


def _verified_wav_buffer(
    audio_root: Path,
    relative_audio: str,
    *,
    expected_sha256: str,
    expected_duration_seconds: float,
    expected_sample_rate: int,
    expected_channels: int,
    expected_sample_width_bits: int,
    context: str,
) -> io.BytesIO:
    """Return one verified buffer populated from one O_NOFOLLOW descriptor."""

    descriptor, directory_descriptors, parts, root = _open_audio_descriptor(
        audio_root,
        relative_audio,
        context,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SealedDecoderError(f"{context}.audio must be a regular file")
        if before.st_size > MAX_AUDIO_BYTES_PER_ITEM:
            raise SealedDecoderError(f"{context}.audio exceeds the byte limit")
        payload = io.BytesIO()
        digest = hashlib.sha256()
        total_bytes = 0
        while total_bytes <= MAX_AUDIO_BYTES_PER_ITEM:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    MAX_AUDIO_BYTES_PER_ITEM + 1 - total_bytes,
                ),
            )
            if not chunk:
                break
            payload.write(chunk)
            digest.update(chunk)
            total_bytes += len(chunk)
        if total_bytes > MAX_AUDIO_BYTES_PER_ITEM:
            raise SealedDecoderError(f"{context}.audio exceeds the byte limit")
        actual_sha256 = f"sha256:{digest.hexdigest()}"
        if actual_sha256 != expected_sha256:
            raise SealedDecoderError(f"{context}.audio_sha256 mismatch")

        try:
            payload.seek(0)
            with wave.open(payload, "rb") as wav_file:
                compression = wav_file.getcomptype()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width_bits = wav_file.getsampwidth() * 8
                frame_count = wav_file.getnframes()
        except (OSError, EOFError, wave.Error) as exc:
            raise SealedDecoderError(
                f"{context}.audio is not a readable PCM WAV"
            ) from exc
        payload.seek(0)
        after = os.fstat(descriptor)
        _validate_open_audio_binding(
            root,
            directory_descriptors,
            parts,
            after,
            context,
        )
    finally:
        os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)

    stable_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if stable_identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SealedDecoderError(f"{context}.audio changed while it was verified")
    if compression != "NONE" or frame_count <= 0 or sample_rate <= 0:
        raise SealedDecoderError(f"{context}.audio must be non-empty PCM WAV")
    declared_identity = (
        expected_sample_rate,
        expected_channels,
        expected_sample_width_bits,
    )
    actual_identity = (sample_rate, channels, sample_width_bits)
    if declared_identity != actual_identity:
        raise SealedDecoderError(f"{context}.audio WAV identity mismatch")
    if actual_identity != (16_000, 1, 16):
        raise SealedDecoderError(
            f"{context}.audio must be 16 kHz mono signed 16-bit PCM"
        )
    actual_duration = frame_count / sample_rate
    if not math.isclose(
        expected_duration_seconds,
        actual_duration,
        rel_tol=0,
        abs_tol=1 / sample_rate,
    ):
        raise SealedDecoderError(f"{context}.audio duration mismatch")
    return payload


def _verified_audio_item(
    raw_item: Mapping[str, Any],
    audio_root: Path,
    index: int,
) -> SealedAudioItem:
    context = f"sealed input item[{index}]"
    relative_audio = str(raw_item["audio"])
    frozen_duration = float(raw_item["duration_seconds"])
    with _verified_wav_buffer(
        audio_root,
        relative_audio,
        expected_sha256=str(raw_item["audio_sha256"]),
        expected_duration_seconds=frozen_duration,
        expected_sample_rate=int(raw_item["sample_rate"]),
        expected_channels=int(raw_item["channels"]),
        expected_sample_width_bits=int(raw_item["sample_width_bits"]),
        context=context,
    ):
        pass
    return SealedAudioItem(
        utterance_id=str(raw_item["id"]),
        audio_root=Path(os.path.abspath(audio_root)),
        relative_audio=relative_audio,
        audio_sha256=str(raw_item["audio_sha256"]),
        duration_seconds=frozen_duration,
        sample_rate=int(raw_item["sample_rate"]),
        channels=int(raw_item["channels"]),
        sample_width_bits=int(raw_item["sample_width_bits"]),
    )


def _attempt_audio_buffer(item: SealedAudioItem) -> io.BytesIO:
    return _verified_wav_buffer(
        item.audio_root,
        item.relative_audio,
        expected_sha256=item.audio_sha256,
        expected_duration_seconds=item.duration_seconds,
        expected_sample_rate=item.sample_rate,
        expected_channels=item.channels,
        expected_sample_width_bits=item.sample_width_bits,
        context=f"sealed input item {item.utterance_id!r}",
    )


def load_verified_audio_items(
    sealed_input: LoadedArtifact,
    audio_root: Path,
) -> tuple[SealedAudioItem, ...]:
    """Verify every audio hash and WAV identity without opening references."""

    return tuple(
        _verified_audio_item(item, audio_root, index)
        for index, item in enumerate(sealed_input.document["items"])
    )


def _validate_local_model_tree(model_path: Path) -> None:
    if model_path.is_symlink() or not model_path.is_dir():
        raise SealedDecoderError("local model snapshot must be a non-symlink directory")
    files = 0
    for path in model_path.rglob("*"):
        if path.is_symlink():
            raise SealedDecoderError("local model snapshot must not contain symlinks")
        try:
            path_stat = path.stat()
        except OSError as exc:
            raise SealedDecoderError("cannot inspect local model snapshot") from exc
        if stat.S_ISDIR(path_stat.st_mode):
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise SealedDecoderError(
                "local model snapshot may contain only directories and regular files"
            )
        files += 1
    if files == 0:
        raise SealedDecoderError("local model snapshot is empty")


def _model_bundle_digest(
    model_path: Path,
    model_bundle_hasher: Callable[[Path], str],
    *,
    context: str,
) -> str:
    _validate_local_model_tree(model_path)
    try:
        digest = model_bundle_hasher(model_path)
    except Exception as exc:
        raise SealedDecoderError(f"failed to hash {context}") from exc
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise SealedDecoderError(f"{context} hash is invalid")
    _validate_local_model_tree(model_path)
    return digest


def resolve_local_model_snapshot(
    repository_root: Path,
    identifier: str,
    revision: str,
    expected_sha256: str,
    model_bundle_hasher: Callable[[Path], str] = model_directory_sha256,
) -> Path:
    """Resolve the exact already-cached ModelScope revision and lock hash."""

    if MODEL_IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise SealedDecoderError("candidate model identifier is unsafe")
    if (
        not isinstance(revision, str)
        or MODEL_REVISION_PATTERN.fullmatch(revision) is None
    ):
        raise SealedDecoderError(
            "sealed decoder model revision must be a full immutable snapshot commit"
        )
    if (
        not isinstance(expected_sha256, str)
        or SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise SealedDecoderError("candidate model hash is invalid")
    root = repository_root.resolve()
    encoded_identifier = identifier.replace("/", "--")
    snapshots_root = (
        root / ".cache" / "modelscope" / "models" / encoded_identifier / "snapshots"
    )
    current = root
    for component in snapshots_root.relative_to(root).parts:
        current /= component
        try:
            component_stat = current.lstat()
        except OSError as exc:
            raise SealedDecoderError(
                "pinned model snapshot is absent; sealed runs cannot download"
            ) from exc
        if stat.S_ISLNK(component_stat.st_mode) or not stat.S_ISDIR(
            component_stat.st_mode
        ):
            raise SealedDecoderError("model cache path must contain only real directories")

    snapshot = snapshots_root / revision
    try:
        snapshot_stat = snapshot.lstat()
    except OSError as exc:
        raise SealedDecoderError(
            "pinned model revision is absent; sealed runs cannot download"
        ) from exc
    if stat.S_ISLNK(snapshot_stat.st_mode) or not stat.S_ISDIR(snapshot_stat.st_mode):
        raise SealedDecoderError("pinned model revision must be a real directory")
    digest = _model_bundle_digest(
        snapshot,
        model_bundle_hasher,
        context=f"cached {identifier}@{revision} model snapshot",
    )
    if digest != expected_sha256:
        raise SealedDecoderError(
            "pinned model revision content does not match the candidate model hash"
        )
    return snapshot


def _assert_model_stack_not_loaded() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name.partition(".")[0] in MODEL_STACK_MODULE_ROOTS
    )
    if loaded:
        raise SealedDecoderError(
            "ASR model stack was imported before environment sanitization"
        )


def _reject_repository_model_stack_shadows(repository_root: Path) -> None:
    """Reject checkout files that would precede locked model distributions."""

    root = repository_root.resolve()
    for search_root in (root, root / "scripts"):
        if search_root == root / "scripts" and not search_root.exists():
            continue
        if search_root.is_symlink() or not search_root.is_dir():
            raise SealedDecoderError("model import search root is unsafe")
        try:
            entries = tuple(search_root.iterdir())
        except OSError as exc:
            raise SealedDecoderError("cannot inspect the model import search root") from exc
        for module_root in MODEL_STACK_MODULE_ROOTS:
            for entry in entries:
                if search_root == root and module_root == "funasr" and entry == root / "funasr":
                    continue
                if entry.name == module_root or (
                    entry.name.startswith(f"{module_root}.")
                    and entry.suffix.lower() in {".py", ".pyc", ".so", ".pyd"}
                ):
                    raise SealedDecoderError(
                        f"repository model import shadow is forbidden: {entry}"
                    )


def _assert_only_verified_repository_modules(repository_root: Path) -> None:
    """Reject loaded checkout code outside the committed runner inventory."""

    root = repository_root.resolve()
    locked_environment_root = (root / ".venv").resolve()
    exact_sources = set(RUNNER_SOURCE_PATHS)
    for module_name, module in tuple(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, (str, os.PathLike)):
            continue
        try:
            module_path = Path(raw_path).resolve(strict=True)
        except OSError as exc:
            try:
                Path(raw_path).absolute().relative_to(root)
            except ValueError:
                continue
            raise SealedDecoderError(
                f"loaded repository module {module_name!r} is unavailable"
            ) from exc
        try:
            relative_path = module_path.relative_to(root).as_posix()
        except ValueError:
            continue
        try:
            module_path.relative_to(locked_environment_root)
        except ValueError:
            pass
        else:
            # runtime_identity separately freezes the exact lock-compatible
            # installed distribution inventory for this environment.
            continue
        if relative_path in exact_sources:
            continue
        if relative_path.startswith("funasr/") and "/__pycache__/" not in relative_path:
            # runner_source_identity verifies every committed FunASR file and
            # rejects uncommitted runtime files before model construction.
            continue
        raise SealedDecoderError(
            f"loaded repository module {module_name!r} is outside the verified "
            "runner source inventory"
        )


def _validate_loaded_funasr_origin(repository_root: Path) -> None:
    module = sys.modules.get("funasr")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise SealedDecoderError("loaded funasr package has no stable source path")
    expected_root = (repository_root / "funasr").resolve()
    try:
        Path(module_file).resolve(strict=True).relative_to(expected_root)
    except (OSError, ValueError) as exc:
        raise SealedDecoderError(
            "loaded funasr package is not the committed repository source"
        ) from exc


def _reported_model_path(model: Any, expected_path: Path, context: str) -> Path:
    value = getattr(model, "model_path", None)
    if not isinstance(value, (str, os.PathLike)):
        raise SealedDecoderError(f"{context} must expose resolved model_path")
    reported = Path(value)
    try:
        resolved = reported.resolve(strict=True)
    except OSError as exc:
        raise SealedDecoderError(f"{context} model_path is unavailable") from exc
    if not reported.is_absolute() or reported != expected_path or resolved != expected_path:
        raise SealedDecoderError(
            f"{context} model_path differs from the pinned local snapshot"
        )
    return reported


def default_model_factory(**kwargs: Any) -> Any:
    """Construct the pinned upstream model only after all preflight checks."""

    from funasr import AutoModel

    return AutoModel(**kwargs)


def _validate_pinned_model_bundle_contract(
    config: SealedDecoderConfig,
    local_model_path: Path,
) -> None:
    """Reject a pinned bundle that can select a different registered model class."""

    config_path = local_model_path / "config.yaml"
    weights_path = local_model_path / "model.pt"
    for path, label in (
        (config_path, "model config"),
        (weights_path, "model weights"),
    ):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SealedDecoderError(f"pinned {label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SealedDecoderError(f"pinned {label} must be a regular file")
        if path.resolve() != path:
            raise SealedDecoderError(f"pinned {label} path must not contain aliases")

    payload = _read_bounded_regular_file(
        config_path,
        MAX_RUNNER_SOURCE_BYTES,
        "pinned model config",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SealedDecoderError("pinned model config must use UTF-8") from exc
    try:
        import yaml

        root_node = yaml.compose(text, Loader=yaml.SafeLoader)
    except Exception as exc:
        raise SealedDecoderError("pinned model config is invalid YAML") from exc
    if root_node is None or not isinstance(root_node, yaml.MappingNode):
        raise SealedDecoderError("pinned model config must be a YAML mapping")
    top_level: dict[str, Any] = {}
    for key_node, value_node in root_node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            raise SealedDecoderError("pinned model config keys must be scalars")
        key = key_node.value
        if key == "<<":
            raise SealedDecoderError("pinned model config must not use YAML merge keys")
        if key in top_level:
            raise SealedDecoderError(
                f"pinned model config has duplicate top-level key {key!r}"
            )
        top_level[key] = value_node
    for selector, expected_class in TRACK_COMPONENT_PROFILES[config.track].items():
        selector_node = top_level.get(selector)
        if expected_class is None:
            if selector_node is not None and (
                not isinstance(selector_node, yaml.ScalarNode)
                or selector_node.tag != "tag:yaml.org,2002:null"
            ):
                raise SealedDecoderError(
                    f"pinned model config selector {selector!r} must be null or absent"
                )
            continue
        if (
            not isinstance(selector_node, yaml.ScalarNode)
            or selector_node.tag != "tag:yaml.org,2002:str"
            or selector_node.value != expected_class
        ):
            raise SealedDecoderError(
                f"pinned model config selector {selector!r} does not match "
                "the sealed track"
            )


def _validate_constructed_model_contract(
    model: Any,
    config: SealedDecoderConfig,
    local_model_path: Path,
) -> None:
    """Verify the default AutoModel resolved one CPU FP32 ASR model only."""

    for attribute in ("vad_model", "punc_model", "spk_model"):
        if getattr(model, attribute, None) is not None:
            raise SealedDecoderError(
                "constructed ASR model loaded an undeclared auxiliary model"
            )
    resolved = getattr(model, "kwargs", None)
    if not isinstance(resolved, Mapping):
        raise SealedDecoderError("constructed ASR model has no resolved kwargs")
    exact_values = {
        "model": TRACK_MODEL_CLASSES[config.track],
        "model_path": str(local_model_path),
        "config": str(local_model_path / "config.yaml"),
        "init_param": str(local_model_path / "model.pt"),
        "device": "cpu",
        "ngpu": 0,
        "batch_size": 1,
        "fp16": False,
        "bf16": False,
        "trust_remote_code": False,
        "output_dir": None,
        "lm_weight": 0.0,
        "lm_file": None,
        "token_lists": [],
        "seg_dicts": [],
        "vad_model": None,
        "punc_model": None,
        "spk_model": None,
    }
    if any(resolved.get(name) != value for name, value in exact_values.items()):
        raise SealedDecoderError(
            "constructed ASR model resolved outside the sealed runtime contract"
        )
    resolved_tokenizer_conf = resolved.get("tokenizer_conf")
    if (
        not isinstance(resolved_tokenizer_conf, Mapping)
        or resolved_tokenizer_conf.get("non_linguistic_symbols") is not None
    ):
        raise SealedDecoderError(
            "constructed ASR tokenizer resolved outside the sealed runtime contract"
        )

    resource_names = {
        "bpemodel",
        "cmvn_file",
        "config",
        "init_param",
        "jieba_usr_dict",
        "non_linguistic_symbols",
        "model_path",
        "seg_dict",
        "seg_dict_file",
        "token_list",
        "init_param_path",
        "filters_path",
        "vocab_path",
    }

    def validate_resources(value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_name, child in value.items():
                name = str(raw_name)
                if name == "remote_code" and child not in (None, ""):
                    raise SealedDecoderError(
                        "constructed ASR model resolved remote code"
                    )
                if name in resource_names and isinstance(child, (str, os.PathLike)):
                    resource = Path(child)
                    try:
                        resolved_resource = resource.resolve(strict=True)
                    except OSError as exc:
                        raise SealedDecoderError(
                            f"constructed ASR model resource {name} is unavailable"
                        ) from exc
                    try:
                        resolved_resource.relative_to(local_model_path)
                    except ValueError as exc:
                        raise SealedDecoderError(
                            f"constructed ASR model resource {name} escapes the snapshot"
                        ) from exc
                    if not resource.is_absolute() or resource != resolved_resource:
                        raise SealedDecoderError(
                            f"constructed ASR model resource {name} uses an alias"
                        )
                validate_resources(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    validate_resources(child)

    validate_resources(resolved)

    expected_components = TRACK_COMPONENT_PROFILES[config.track]
    for name in ("tokenizer", "frontend"):
        component = resolved.get(name)
        if (
            component is None
            or component.__class__.__name__ != expected_components[name]
        ):
            raise SealedDecoderError(
                f"constructed ASR model {name} class does not match the sealed track"
            )

    primary = getattr(model, "model", None)
    if primary is None or primary.__class__.__name__ != TRACK_MODEL_CLASSES[config.track]:
        raise SealedDecoderError(
            "constructed ASR model class does not match the sealed track"
        )
    for name in ("specaug", "normalize", "encoder", "decoder", "predictor"):
        expected_component = expected_components[name]
        component = getattr(primary, name, None)
        if expected_component is None:
            if component is not None:
                raise SealedDecoderError(
                    f"constructed ASR model {name} must be absent"
                )
        elif (
            component is None
            or component.__class__.__name__ != expected_component
        ):
            raise SealedDecoderError(
                f"constructed ASR model {name} class does not match the sealed track"
            )
    floating_tensor_count = 0
    for collection_name in ("parameters", "buffers"):
        reader = getattr(primary, collection_name, None)
        if not callable(reader):
            raise SealedDecoderError(
                "constructed ASR model does not expose tensor state"
            )
        try:
            tensors = reader()
            for tensor in tensors:
                is_floating_point = getattr(tensor, "is_floating_point", None)
                if callable(is_floating_point) and is_floating_point():
                    floating_tensor_count += 1
                    if str(getattr(tensor, "dtype", "")) != "torch.float32":
                        raise SealedDecoderError(
                            "constructed ASR model floating tensors must be FP32"
                        )
        except SealedDecoderError:
            raise
        except Exception as exc:
            raise SealedDecoderError(
                "cannot inspect constructed ASR model tensor state"
            ) from exc
    if floating_tensor_count == 0:
        raise SealedDecoderError(
            "constructed ASR model exposes no floating tensor state"
        )


def _model_kwargs(
    config: SealedDecoderConfig,
    local_model_path: Path,
) -> dict[str, object]:
    spec = TRACKS[config.track]
    config_path = local_model_path / "config.yaml"
    weights_path = local_model_path / "model.pt"
    tokenizer_conf: dict[str, object] = {
        "token_list": str(local_model_path / "tokens.json"),
        "non_linguistic_symbols": None,
    }
    if config.track == "paraformer":
        tokenizer_conf["seg_dict"] = str(local_model_path / "seg_dict")
    else:
        tokenizer_conf["bpemodel"] = str(
            local_model_path / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        )
    return {
        "model": str(local_model_path),
        "model_path": str(local_model_path),
        "config": str(config_path),
        "init_param": str(weights_path),
        "model_revision": config.model_revision,
        "hub": spec.hub,
        "device": config.device,
        "ncpu": config.ncpu,
        "ngpu": 0,
        "batch_size": 1,
        "fp16": False,
        "bf16": False,
        "trust_remote_code": False,
        "output_dir": None,
        "lm_weight": 0.0,
        "lm_file": None,
        "tokenizer": TRACK_COMPONENT_PROFILES[config.track]["tokenizer"],
        "frontend": TRACK_COMPONENT_PROFILES[config.track]["frontend"],
        "token_lists": [],
        "seg_dicts": [],
        "vad_model": None,
        "punc_model": None,
        "spk_model": None,
        "tokenizer_conf": tokenizer_conf,
        "frontend_conf": {"cmvn_file": str(local_model_path / "am.mvn")},
        "disable_update": True,
        "disable_pbar": True,
        "check_latest": False,
        "seed": config.seed,
    }


def _positive_elapsed_ns(started: Any, finished: Any, context: str) -> int:
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
    ):
        raise SealedDecoderError(f"{context} clock values must be integer nanoseconds")
    elapsed = finished - started
    if elapsed <= 0:
        raise SealedDecoderError(f"{context} elapsed_ns must be positive")
    return elapsed


def _decode_once(
    model: Any,
    item: SealedAudioItem,
    config: SealedDecoderConfig,
    clock: Callable[[], int],
) -> tuple[dict[str, object], int]:
    # Keep storage, hashing, and WAV validation outside the decode timer.  The
    # model still receives the exact in-memory bytes that passed this attempt's
    # no-follow identity check.
    audio_buffer = _attempt_audio_buffer(item)
    started = clock()
    raw_text = ""
    status = "failed"
    reason_code: str | None = "decoder_exception"
    try:
        with audio_buffer:
            result = model.generate(
                input=audio_buffer,
                **dict(TRACKS[config.track].generate_options),
            )
    except Exception:
        pass
    else:
        if (
            isinstance(result, list)
            and len(result) == 1
            and isinstance(result[0], Mapping)
            and isinstance(result[0].get("text"), str)
        ):
            candidate_text = str(result[0]["text"])
            if len(candidate_text) > MAX_HYPOTHESIS_CHARACTERS:
                reason_code = "hypothesis_too_long"
            else:
                try:
                    adapted = adapt_hypothesis(
                        candidate_text,
                        config.hypothesis_adapter_version,
                    )
                except CoreReportValidationError as exc:
                    raise SealedDecoderError(str(exc)) from exc
                raw_text = candidate_text
                if adapted:
                    status = "ok"
                    reason_code = None
                else:
                    status = "empty"
                    reason_code = "empty_hypothesis"
        else:
            reason_code = "invalid_model_output"
    finished = clock()
    elapsed_ns = _positive_elapsed_ns(started, finished, "decode attempt")
    return (
        {
            "id": item.utterance_id,
            "raw_text": raw_text,
            "status": status,
            "reason_code": reason_code,
        },
        elapsed_ns,
    )


def _attempt_fact(
    prediction: Mapping[str, object],
    item: SealedAudioItem,
    elapsed_ns: int,
    *,
    attempt_index: int,
) -> dict[str, object]:
    return {
        "id": item.utterance_id,
        "attempt_index": attempt_index,
        "elapsed_ns": elapsed_ns,
        "audio_duration_seconds": item.duration_seconds,
        "status": prediction["status"],
        "reason_code": prediction["reason_code"],
    }


def raw_prediction_jsonl_bytes(
    prediction_items: Sequence[Mapping[str, object]],
) -> bytes:
    """Serialize only the four existing prediction fields, one canonical line each."""

    payload = bytearray()
    total_characters = 0
    exact_fields = {"id", "raw_text", "status", "reason_code"}
    for index, prediction in enumerate(prediction_items):
        if set(prediction) != exact_fields:
            raise SealedDecoderError(
                f"prediction_items[{index}] must contain exactly the four raw fields"
            )
        raw_text = prediction["raw_text"]
        if not isinstance(raw_text, str):
            raise SealedDecoderError(f"prediction_items[{index}].raw_text must be a string")
        total_characters += len(raw_text)
        if total_characters > MAX_TOTAL_HYPOTHESIS_CHARACTERS:
            raise SealedDecoderError("prediction text exceeds the total character limit")
        payload.extend(canonical_json_bytes(dict(prediction)))
    return bytes(payload)


def _validated_candidate_facts(
    candidate_lock: LoadedArtifact,
    config: SealedDecoderConfig,
    actual_command: Mapping[str, object],
    actual_hardware: Mapping[str, object],
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = candidate_lock.document
    candidate = lock["candidate"]
    try:
        execution_plan = validate_sealed_candidate_execution(
            candidate,
            lock["hypothesis_adapter_version"],
            repository_root,
        )
    except SealedCandidateContractError as exc:
        raise SealedDecoderError(str(exc)) from exc
    expected_config = SealedDecoderConfig(
        track=execution_plan.baseline_config.track,
        model_revision=execution_plan.baseline_config.model_revision,
        device=execution_plan.baseline_config.device,
        ncpu=execution_plan.baseline_config.ncpu,
        warmup_runs=execution_plan.baseline_config.warmup_runs,
        seed=execution_plan.baseline_config.seed,
        hypothesis_adapter_version=execution_plan.hypothesis_adapter_version,
    )
    if config != expected_config:
        raise SealedDecoderError(
            "actual decoder configuration does not match the candidate command"
        )

    try:
        baseline_config = BaselineConfig(
            track=config.track,
            model_revision=config.model_revision,
            device=config.device,
            ncpu=config.ncpu,
            warmup_runs=config.warmup_runs,
            seed=config.seed,
        )
        config_document = effective_config(baseline_config)
    except BaselineError as exc:
        raise SealedDecoderError(str(exc)) from exc
    config_sha256 = sha256_bytes(canonical_json_bytes(config_document))
    if candidate["config_sha256"] != config_sha256:
        raise SealedDecoderError(
            "actual effective config does not match candidate config_sha256"
        )
    if candidate["seed"] != config.seed:
        raise SealedDecoderError("actual seed does not match candidate lock")
    if candidate["normalizer_version"] != config_document["pipeline"][
        "normalizer_version"
    ]:
        raise SealedDecoderError(
            "actual normalizer contract does not match candidate lock"
        )
    if dict(actual_command) != candidate["command"]:
        raise SealedDecoderError("actual complete command does not match candidate lock")
    if dict(actual_hardware) != candidate["hardware"]:
        raise SealedDecoderError("actual hardware identity does not match candidate lock")

    models = candidate["models"]
    spec = TRACKS[config.track]
    if not isinstance(models, list) or len(models) != 1:
        raise SealedDecoderError(
            "sealed decoder v1 requires exactly one ASR model component"
        )
    model = models[0]
    if (
        model.get("role") != "asr"
        or model.get("identifier") != spec.model_identifier
        or model.get("revision") != config.model_revision
    ):
        raise SealedDecoderError(
            "candidate model identity does not match the fixed model track"
        )
    return config_document, model


def execute_sealed_candidate(
    input_projection_path: Path,
    candidate_lock_path: Path,
    audio_root: Path,
    config: SealedDecoderConfig,
    *,
    actual_argv: Sequence[str],
    environment_source: Mapping[str, str] | None = None,
    current_directory: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    hardware_reader: Callable[[str], Mapping[str, object]] = hardware_identity,
    source_identity_reader: Callable[[str, Path], tuple[str, str]] = (
        runner_source_identity
    ),
    runtime_identity_reader: Callable[[Path], Mapping[str, object]] = runtime_identity,
    model_snapshot_resolver: Callable[
        [Path, str, str, str, Callable[[Path], str]], Path
    ] = resolve_local_model_snapshot,
    model_factory: Callable[..., Any] = default_model_factory,
    model_bundle_hasher: Callable[[Path], str] = model_directory_sha256,
    clock: Callable[[], int] = time.perf_counter_ns,
    rss_reader: Callable[[], int] = peak_rss_bytes,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> SealedDecoderResult:
    """Run one fully measured pass without reading a reference or descriptor."""

    # The lock/input relationship is always the first content validation.  No
    # audio is opened and no model is imported before this succeeds.
    try:
        sealed_input = load_sealed_input_projection(input_projection_path)
        candidate_lock = load_candidate_lock(candidate_lock_path)
        validate_decode_handoff(sealed_input, candidate_lock)
    except (OSError, ValueError) as exc:
        raise SealedDecoderError(str(exc)) from exc

    if isinstance(actual_argv, (str, bytes, bytearray)) or not isinstance(
        actual_argv, Sequence
    ):
        raise SealedDecoderError("actual argv must be an ordered sequence")
    argv = list(actual_argv)
    if len(argv) < 2 or any(not isinstance(item, str) or not item for item in argv):
        raise SealedDecoderError("actual argv must be complete non-empty strings")
    if argv[:3] != [
        ".venv/bin/python",
        "-P",
        "-S",
    ] or argv[3:5] != [
        "scripts/run_sealed_asr_candidate.py",
        "run",
    ]:
        raise SealedDecoderError(
            "sealed evidence requires direct .venv runner script execution"
        )
    root = repository_root.resolve()
    cwd = (Path.cwd() if current_directory is None else current_directory).resolve()
    if cwd != root:
        raise SealedDecoderError("sealed decoder must run from the repository root")
    complete_environment = os.environ if environment_source is None else environment_source
    _reject_unsafe_startup_environment(complete_environment)
    observed_environment = allowed_environment(complete_environment)
    _validate_reproducibility_environment(observed_environment, root, config.ncpu)
    actual_command: dict[str, object] = {
        "working_directory": ".",
        "argv": argv,
        "environment": observed_environment,
    }
    observed_hardware = dict(hardware_reader(config.device))
    config_document, expected_model = _validated_candidate_facts(
        candidate_lock,
        config,
        actual_command,
        observed_hardware,
        root,
    )
    _reject_repository_model_stack_shadows(root)
    uses_default_model_factory = model_factory is default_model_factory
    if uses_default_model_factory:
        _assert_model_stack_not_loaded()
    with sanitized_process_environment(observed_environment):
        observed_runtime = _validated_runtime_identity(runtime_identity_reader(root))
        source_commit, source_sha256 = source_identity_reader(
            candidate_lock.document["candidate"]["code_commit"],
            root,
        )
        if source_commit != candidate_lock.document["candidate"]["code_commit"]:
            raise SealedDecoderError(
                "runner source identity does not match candidate code_commit"
            )
        if not isinstance(source_sha256, str) or SHA256_PATTERN.fullmatch(
            source_sha256
        ) is None:
            raise SealedDecoderError("runner source inventory hash is invalid")

        audio_items = load_verified_audio_items(sealed_input, audio_root)
        if not audio_items:
            raise SealedDecoderError("sealed input contains no decode-eligible audio")

        try:
            selected_model_path = Path(
                model_snapshot_resolver(
                    root,
                    str(expected_model["identifier"]),
                    str(expected_model["revision"]),
                    str(expected_model["sha256"]),
                    model_bundle_hasher,
                )
            )
        except SealedDecoderError:
            raise
        except Exception as exc:
            raise SealedDecoderError("failed to resolve the pinned local model") from exc
        try:
            canonical_model_path = selected_model_path.resolve(strict=True)
        except OSError as exc:
            raise SealedDecoderError("pinned local model path is unavailable") from exc
        if not selected_model_path.is_absolute() or canonical_model_path != selected_model_path:
            raise SealedDecoderError(
                "pinned local model path must be absolute and free of aliases"
            )
        preconstruction_model_sha256 = _model_bundle_digest(
            canonical_model_path,
            model_bundle_hasher,
            context="preconstruction model bundle",
        )
        if preconstruction_model_sha256 != expected_model["sha256"]:
            raise SealedDecoderError(
                "pinned local model does not match candidate model identity"
            )
        _validate_pinned_model_bundle_contract(config, canonical_model_path)

        if uses_default_model_factory:
            _assert_model_stack_not_loaded()
        started_at = utc_now()
        if not isinstance(started_at, datetime) or started_at.tzinfo is None:
            raise SealedDecoderError("started_at_utc must be timezone-aware")
        model_started = clock()
        try:
            model = model_factory(**_model_kwargs(config, canonical_model_path))
        except Exception as exc:
            raise SealedDecoderError("failed to construct the pinned ASR model") from exc
        model_finished = clock()
        model_load_ns = _positive_elapsed_ns(
            model_started,
            model_finished,
            "model load",
        )
        if uses_default_model_factory:
            _validate_loaded_funasr_origin(root)
            _validate_constructed_model_contract(
                model,
                config,
                canonical_model_path,
            )
        _assert_only_verified_repository_modules(root)
        returned_model_path = _reported_model_path(
            model,
            canonical_model_path,
            "constructed ASR model",
        )
        constructed_model_sha256 = _model_bundle_digest(
            returned_model_path,
            model_bundle_hasher,
            context="constructed model bundle",
        )
        if constructed_model_sha256 != expected_model["sha256"]:
            raise SealedDecoderError(
                "constructed model bundle does not match candidate model identity"
            )

        first_item = audio_items[0]
        cold_prediction, cold_elapsed_ns = _decode_once(
            model,
            first_item,
            config,
            clock,
        )
        if cold_prediction["status"] == "failed":
            raise SealedDecoderError(
                "cold inference failed; refusing to publish performance evidence"
            )
        cold_attempt = _attempt_fact(
            cold_prediction,
            first_item,
            cold_elapsed_ns,
            attempt_index=0,
        )

        warmup_attempts: list[dict[str, object]] = []
        for warmup_index in range(config.warmup_runs):
            warmup_prediction, warmup_elapsed_ns = _decode_once(
                model,
                first_item,
                config,
                clock,
            )
            if warmup_prediction["status"] == "failed":
                raise SealedDecoderError(
                    "warmup inference failed; refusing to publish performance evidence"
                )
            warmup_attempts.append(
                _attempt_fact(
                    warmup_prediction,
                    first_item,
                    warmup_elapsed_ns,
                    attempt_index=warmup_index,
                )
            )

        prediction_items: list[dict[str, object]] = []
        decode_attempts: list[dict[str, object]] = []
        total_hypothesis_characters = 0
        for attempt_index, item in enumerate(audio_items):
            prediction, elapsed_ns = _decode_once(model, item, config, clock)
            total_hypothesis_characters += len(str(prediction["raw_text"]))
            if total_hypothesis_characters > MAX_TOTAL_HYPOTHESIS_CHARACTERS:
                raise SealedDecoderError(
                    "prediction text exceeds the total character limit"
                )
            prediction_items.append(prediction)
            decode_attempts.append(
                _attempt_fact(
                    prediction,
                    item,
                    elapsed_ns,
                    attempt_index=attempt_index,
                )
            )

        # Sample the process high-water mark immediately after the measured
        # pass.  The mandatory post-measurement model inventory below is an
        # integrity check, not part of the stated RSS or wall-clock scope.
        peak_bytes = rss_reader()
        if (
            isinstance(peak_bytes, bool)
            or not isinstance(peak_bytes, int)
            or peak_bytes <= 0
        ):
            raise SealedDecoderError("peak RSS must be a positive integer byte count")
        finished_at = utc_now()
        if not isinstance(finished_at, datetime) or finished_at.tzinfo is None:
            raise SealedDecoderError("finished_at_utc must be timezone-aware")
        if finished_at < started_at:
            raise SealedDecoderError("finished_at_utc precedes started_at_utc")

        final_model_path = _reported_model_path(
            model,
            canonical_model_path,
            "post-measurement ASR model",
        )
        final_model_sha256 = _model_bundle_digest(
            final_model_path,
            model_bundle_hasher,
            context="post-measurement model bundle",
        )
        if final_model_sha256 != expected_model["sha256"]:
            raise SealedDecoderError(
                "model bundle changed during the sealed decode"
            )
        if uses_default_model_factory:
            _validate_constructed_model_contract(
                model,
                config,
                canonical_model_path,
            )
        if dict(os.environ) != observed_environment:
            raise SealedDecoderError(
                "sanitized process environment changed during the sealed decode"
            )
        if not sys.dont_write_bytecode or sys.pycache_prefix != SEALED_PYCACHE_PREFIX:
            raise SealedDecoderError(
                "sealed import safety flags changed during the sealed decode"
            )
        if _validated_runtime_identity(runtime_identity_reader(root)) != observed_runtime:
            raise SealedDecoderError(
                "runner runtime identity changed during the sealed decode"
            )
        _reject_repository_model_stack_shadows(root)
        _assert_only_verified_repository_modules(root)
        if source_identity_reader(
            candidate_lock.document["candidate"]["code_commit"],
            root,
        ) != (source_commit, source_sha256):
            raise SealedDecoderError(
                "runner source identity changed during the sealed decode"
            )
    raw_payload = raw_prediction_jsonl_bytes(prediction_items)
    prediction_items_sha256 = sha256_bytes(
        canonical_json_bytes(prediction_items)
    )
    observation: dict[str, object] = {
        "experiment_id": candidate_lock.document["candidate"]["experiment_id"],
        "dataset_id": sealed_input.document["dataset_id"],
        "revision": sealed_input.document["revision"],
        "split": sealed_input.document["split"],
        "candidate_freeze_sha256": candidate_lock.document[
            "candidate_freeze_sha256"
        ],
        "candidate_lock_sha256": candidate_lock.sha256,
        "input_projection_sha256": sealed_input.sha256,
        "hypothesis_adapter_version": config.hypothesis_adapter_version,
        "config_sha256": candidate_lock.document["candidate"]["config_sha256"],
        "models": [dict(expected_model)],
        "command": actual_command,
        "hardware": observed_hardware,
        "runtime": observed_runtime,
        "runner_code_commit": source_commit,
        "runner_source_sha256": source_sha256,
        "raw_predictions_sha256": sha256_bytes(raw_payload),
        "prediction_items_sha256": prediction_items_sha256,
        "prediction_item_count": len(prediction_items),
        "started_at_utc": started_at.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "finished_at_utc": finished_at.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "measurement_contract": {
            "clock_version": CLOCK_VERSION,
            "rss_version": RSS_VERSION,
            "rss_scope": RSS_SCOPE,
            "rtf_population": RTF_POPULATION,
            "warmup_runs": config.warmup_runs,
        },
        "model_load_ns": model_load_ns,
        "cold_attempt": cold_attempt,
        "warmup_attempts": warmup_attempts,
        "decode_attempts": decode_attempts,
        "peak_rss_bytes": peak_bytes,
    }
    return SealedDecoderResult(tuple(prediction_items), observation)


__all__ = [
    "ALLOWED_ENVIRONMENT_NAMES",
    "CLOCK_VERSION",
    "RSS_SCOPE",
    "RSS_VERSION",
    "RTF_POPULATION",
    "RUNNER_SOURCE_PATHS",
    "SEALED_PYCACHE_PREFIX",
    "SealedAudioItem",
    "SealedDecoderConfig",
    "SealedDecoderError",
    "SealedDecoderResult",
    "_assert_only_verified_repository_modules",
    "allowed_environment",
    "describe_runtime",
    "execute_sealed_candidate",
    "hardware_identity",
    "load_verified_audio_items",
    "peak_rss_bytes",
    "raw_prediction_jsonl_bytes",
    "resolve_local_model_snapshot",
    "runner_source_identity",
    "runtime_identity",
]
