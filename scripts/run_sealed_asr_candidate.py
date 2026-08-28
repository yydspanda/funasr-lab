#!/usr/bin/env python3
"""Run one committed, pinned ASR candidate on reference-free sealed audio."""

from __future__ import annotations

import sys


SEALED_PYCACHE_PREFIX = "/dev/null/asr-lab-sealed-pycache"
_DIRECT_ENTRYPOINT_SECURED = False
if __name__ == "__main__":
    if __spec__ is not None:
        raise SystemExit("sealed commands require direct script execution")
    # The script directory and PYTHONPATH precede the standard library during
    # normal script startup. Reject externally supplied import roots and
    # remove the script directory before importing argparse/pathlib.
    sys.dont_write_bytecode = True
    sys.pycache_prefix = SEALED_PYCACHE_PREFIX
    _startup_posix = sys.modules.get("posix")
    if _startup_posix is None or getattr(
        getattr(_startup_posix, "__spec__", None), "origin", None
    ) != "built-in":
        raise SystemExit("sealed commands require CPython's built-in posix module")
    for _name in (
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
    ):
        if _startup_posix.environ.get(_name.encode("ascii")):
            raise SystemExit(
                f"sealed commands forbid non-empty startup environment {_name}"
            )
    if _startup_posix.environ.get(b"PYTHONHASHSEED") != b"0":
        raise SystemExit("sealed commands require startup PYTHONHASHSEED=0")
    for _key, _value in _startup_posix.environ.items():
        if (
            _value
            and _key.startswith(b"PYTHON")
            and _key != b"PYTHONHASHSEED"
        ):
            _display_name = _key.decode("ascii", "backslashreplace")
            raise SystemExit(
                "sealed commands forbid non-empty startup environment "
                f"{_display_name}"
            )
        if _value and (
            _key.startswith((b"LD_", b"FUNASR_", b"TORCH_"))
            or (
                _key.startswith(b"KMP_")
                and _key not in {b"KMP_DUPLICATE_LIB_OK", b"KMP_INIT_AT_FORK"}
            )
            or _key in {b"OMP_DYNAMIC", b"MKL_DYNAMIC", b"GOMP_CPU_AFFINITY"}
        ):
            _display_name = _key.decode("ascii", "backslashreplace")
            raise SystemExit(
                "sealed commands forbid non-empty startup environment "
                f"{_display_name}"
            )
    for _key, _expected in (
        (b"PATH", b"/usr/bin:/bin"),
        (b"HOME", b"/dev/null"),
        (b"LANG", b"C"),
        (b"LC_ALL", b"C"),
    ):
        if _startup_posix.environ.get(_key) != _expected:
            _display_name = _key.decode("ascii")
            raise SystemExit(
                f"sealed commands require exact startup environment {_display_name}"
            )
    if sys.flags.utf8_mode != 1 or sys.getfilesystemencoding() != "utf-8":
        raise SystemExit("sealed commands require the fixed UTF-8 runtime mode")
    _allowed_startup_names = {
        b"PATH",
        b"HOME",
        b"LANG",
        b"LC_ALL",
        b"PYTHONHASHSEED",
        b"MODELSCOPE_CACHE",
        b"HYDRA_FULL_ERROR",
        b"CRC32C_SW_MODE",
        b"KMP_DUPLICATE_LIB_OK",
        b"KMP_INIT_AT_FORK",
        b"TORCHINDUCTOR_CACHE_DIR",
        b"OMP_NUM_THREADS",
        b"MKL_NUM_THREADS",
        b"OPENBLAS_NUM_THREADS",
        b"NUMEXPR_NUM_THREADS",
    }
    for _key, _value in _startup_posix.environ.items():
        if _value and _key not in _allowed_startup_names:
            _display_name = _key.decode("ascii", "backslashreplace")
            raise SystemExit(
                f"sealed runner forbids unexpected startup environment {_display_name}"
            )
    if not sys.flags.no_site or not sys.flags.safe_path:
        raise SystemExit("sealed commands require Python flags -P -S")

import argparse
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_PACKAGES = ("eval", "eval/normalizers", "scripts")
_BOOTSTRAP_SOURCES = (
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
    "scripts/__init__.py",
    "scripts/check_experiment_manifests.py",
    "scripts/run_sealed_asr_candidate.py",
)
_IMPORTABLE_ALTERNATIVE_SUFFIXES = frozenset({".py", ".pyc", ".so", ".pyd"})
_ALLOWED_ROOT_IMPORT_NAMES = frozenset(
    {
        "asr_lab",
        "benchmark_vllm",
        "eval",
        "funasr",
        "fun_text_processing",
        "runtime",
        "scripts",
        "setup",
    }
)


def _reject_import_alternatives(expected: Path) -> None:
    """Reject a package/module that could shadow one fixed source path."""

    try:
        metadata = expected.lstat()
    except OSError as exc:
        raise SystemExit(f"sealed import source is unavailable: {expected}") from exc
    expected_kind_ok = (
        stat.S_ISDIR(metadata.st_mode)
        if expected.suffix == ""
        else stat.S_ISREG(metadata.st_mode)
    )
    if stat.S_ISLNK(metadata.st_mode) or not expected_kind_ok:
        raise SystemExit(f"sealed import source is unsafe: {expected}")
    stem = expected.name if expected.suffix == "" else expected.stem
    try:
        siblings = tuple(expected.parent.iterdir())
    except OSError as exc:
        raise SystemExit(f"cannot inspect sealed import source: {expected}") from exc
    for sibling in siblings:
        if sibling == expected or sibling.name == "__pycache__":
            continue
        if sibling.name == stem and sibling.is_dir():
            raise SystemExit(f"sealed import shadow is forbidden: {sibling}")
        if (
            sibling.name.startswith(f"{stem}.")
            and sibling.suffix.lower() in _IMPORTABLE_ALTERNATIVE_SUFFIXES
        ):
            raise SystemExit(f"sealed import shadow is forbidden: {sibling}")


def _reject_unexpected_root_imports() -> None:
    """Reject checkout entries that could shadow stdlib or locked packages."""

    try:
        entries = tuple(REPOSITORY_ROOT.iterdir())
    except OSError as exc:
        raise SystemExit("cannot inspect sealed repository import root") from exc
    for entry in entries:
        name: str | None = None
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise SystemExit(f"cannot inspect repository import entry: {entry}") from exc
        if stat.S_ISREG(metadata.st_mode):
            if entry.suffix.lower() in _IMPORTABLE_ALTERNATIVE_SUFFIXES:
                name = entry.name.partition(".")[0]
        elif stat.S_ISLNK(metadata.st_mode):
            if entry.suffix.lower() in _IMPORTABLE_ALTERNATIVE_SUFFIXES or (
                entry.name.isidentifier() and entry.is_dir()
            ):
                raise SystemExit(
                    f"sealed repository import symlink is forbidden: {entry}"
                )
        elif stat.S_ISDIR(metadata.st_mode):
            try:
                package_entries = tuple(entry.iterdir())
            except OSError as exc:
                raise SystemExit(
                    f"cannot inspect repository package candidate: {entry}"
                ) from exc
            if any(
                child.name == "__init__.py"
                or (
                    child.name.startswith("__init__.")
                    and child.suffix.lower() in _IMPORTABLE_ALTERNATIVE_SUFFIXES
                )
                for child in package_entries
            ):
                name = entry.name
        if name is not None and name not in _ALLOWED_ROOT_IMPORT_NAMES:
            raise SystemExit(f"sealed repository import shadow is forbidden: {entry}")


def _secure_project_imports() -> None:
    """Establish source-only project imports before importing any eval module."""

    for name in ("PYTHONPYCACHEPREFIX", "PYTHONOPTIMIZE"):
        if not os.environ.get(name):
            continue
        raise SystemExit(
            f"sealed commands forbid non-empty startup environment {name}"
        )
    if sys.flags.optimize != 0:
        raise SystemExit("sealed commands require Python optimization level 0")
    if (
        os.environ.get("PYTHONHASHSEED") != "0"
        or sys.flags.hash_randomization != 0
    ):
        raise SystemExit("sealed commands require an effective PYTHONHASHSEED=0")
    try:
        process_argv = [
            item.decode("utf-8")
            for item in Path("/proc/self/cmdline")
            .read_bytes()
            .rstrip(b"\x00")
            .split(b"\x00")
        ]
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit("cannot verify sealed process argv") from exc
    allowed_python = {
        ".venv/bin/python",
        str(REPOSITORY_ROOT / ".venv/bin/python"),
    }
    if (
        len(process_argv) < 5
        or process_argv[0] not in allowed_python
        or process_argv[1:4]
        != ["-P", "-S", "scripts/run_sealed_asr_candidate.py"]
        or process_argv[4] not in {"describe-runtime", "run"}
    ):
        raise SystemExit("sealed runner requires exact direct process argv")
    if any(
        name == "eval"
        or name.startswith("eval.")
        or name == "scripts"
        or name.startswith("scripts.")
        for name in sys.modules
    ):
        raise SystemExit("sealed project modules were loaded before bootstrap")
    sys.dont_write_bytecode = True
    sys.pycache_prefix = SEALED_PYCACHE_PREFIX
    expected_python = REPOSITORY_ROOT / ".venv/bin/python"
    if Path(sys.executable) != expected_python:
        raise SystemExit("sealed commands require repository .venv/bin/python")
    site_packages = REPOSITORY_ROOT / ".venv/lib/python3.11/site-packages"
    if site_packages.is_symlink() or not site_packages.is_dir():
        raise SystemExit("sealed site-packages directory is unsafe")
    sys.prefix = str(REPOSITORY_ROOT / ".venv")
    sys.exec_prefix = sys.prefix
    sys.path.append(str(site_packages))
    _reject_unexpected_root_imports()
    root_text = str(REPOSITORY_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    for relative in _BOOTSTRAP_PACKAGES:
        _reject_import_alternatives(REPOSITORY_ROOT / relative)
    for relative in _BOOTSTRAP_SOURCES:
        _reject_import_alternatives(REPOSITORY_ROOT / relative)


if __name__ == "__main__":
    _secure_project_imports()
    _DIRECT_ENTRYPOINT_SECURED = True
elif str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.custodian_replay import CustodianReplayError
from eval.custodian_replay import load_candidate_lock
from eval.custodian_replay import load_custodian_receipt
from eval.custodian_replay import load_sealed_input_projection
from eval.custodian_replay import validate_input_export_receipt_handoff
from eval.custodian_replay import validate_registered_candidate_binding
from eval.custodian_replay import validate_output_paths
from eval.custodian_replay import write_atomic_outputs
from eval.execution_envelope import ExecutionEnvelopeError
from eval.execution_envelope import build_execution_envelope
from eval.execution_envelope import canonical_execution_envelope_bytes
from eval.sealed_decoder import SealedDecoderConfig
from eval.sealed_decoder import SealedDecoderError
from eval.sealed_decoder import SealedDecoderResult
from eval.sealed_decoder import SEALED_PYCACHE_PREFIX as DECODER_PYCACHE_PREFIX
from eval.sealed_decoder import describe_runtime
from eval.sealed_decoder import execute_sealed_candidate
from eval.sealed_decoder import raw_prediction_jsonl_bytes
from eval.offline_baseline import TRACKS
from eval.offline_baseline import canonical_json_bytes
from eval.offline_baseline import sha256_bytes


if DECODER_PYCACHE_PREFIX != SEALED_PYCACHE_PREFIX:
    raise RuntimeError("sealed runner pycache prefixes disagree")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled reference-free decoder or describe its runtime "
            "facts without downloading a model."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser(
        "describe-runtime",
        help="Print pre-registerable hardware/environment facts without a model.",
    )
    describe_parser.add_argument("--device", choices=("cpu",), required=True)
    describe_parser.add_argument("--ncpu", type=int, required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Decode one sealed input and publish raw predictions plus its envelope.",
    )
    run_parser.add_argument("--input-projection", type=Path, required=True)
    run_parser.add_argument("--candidate-lock", type=Path, required=True)
    run_parser.add_argument("--input-receipt", type=Path, required=True)
    run_parser.add_argument("--audio-root", type=Path, required=True)
    run_parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    run_parser.add_argument("--model-revision", required=True)
    run_parser.add_argument("--device", choices=("cpu",), required=True)
    run_parser.add_argument("--ncpu", type=int, required=True)
    run_parser.add_argument("--warmup-runs", type=int, required=True)
    run_parser.add_argument("--seed", type=int, choices=(0,), required=True)
    run_parser.add_argument(
        "--hypothesis-adapter-version",
        choices=("identity-v1", "sensevoice-control-tags-v1"),
        required=True,
    )
    run_parser.add_argument("--output-raw-predictions", type=Path, required=True)
    run_parser.add_argument("--output-execution-envelope", type=Path, required=True)
    return parser


def _current_process_argv() -> list[str]:
    """Read the actual Linux argv rather than reconstructing parser defaults."""

    try:
        payload = Path("/proc/self/cmdline").read_bytes()
        values = payload.rstrip(b"\x00").split(b"\x00")
        argv = [value.decode("utf-8") for value in values]
    except (OSError, UnicodeDecodeError) as exc:
        raise SealedDecoderError("cannot capture the complete process argv") from exc
    if len(argv) < 2 or any(not value for value in argv):
        raise SealedDecoderError("captured process argv is incomplete")
    return argv


@contextmanager
def _silence_decoder_streams() -> Iterator[None]:
    """Discard model/library output while preserving runner diagnostics."""

    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "wb") as sink:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(sink.fileno(), stdout_fd)
            os.dup2(sink.fileno(), stderr_fd)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _validate_sealed_output_paths(paths: list[Path]) -> tuple[Path, ...]:
    resolved = validate_output_paths(paths)
    parent_mode = stat.S_IMODE(resolved[0].parent.stat().st_mode)
    if parent_mode != 0o700:
        raise CustodianReplayError("output parent directory mode must be exactly 0700")
    return resolved


def _runner_envelope(
    result: SealedDecoderResult,
    *,
    input_export_receipt_sha256: str,
) -> dict[str, object]:
    envelope = build_execution_envelope(
        result.observation,
        result.prediction_items,
        input_export_receipt_sha256=input_export_receipt_sha256,
    )
    if envelope["bindings"]["prediction_items_sha256"] != result.observation[
        "prediction_items_sha256"
    ]:
        raise SealedDecoderError(
            "execution envelope prediction identity differs from runner observation"
        )
    return envelope


def _publish_result(
    result: SealedDecoderResult,
    raw_predictions_path: Path,
    execution_envelope_path: Path,
    *,
    input_export_receipt_sha256: str,
) -> None:
    _validate_sealed_output_paths(
        [raw_predictions_path, execution_envelope_path]
    )
    raw_payload = raw_prediction_jsonl_bytes(result.prediction_items)
    if sha256_bytes(raw_payload) != result.observation["raw_predictions_sha256"]:
        raise SealedDecoderError("raw prediction bytes differ from runner observation")
    envelope_payload = canonical_execution_envelope_bytes(
        _runner_envelope(
            result,
            input_export_receipt_sha256=input_export_receipt_sha256,
        )
    )
    # The ordered atomic writer persists the raw artifact first and the final
    # canonical envelope last as this execution's completion marker.
    write_atomic_outputs(
        [
            (raw_predictions_path, raw_payload),
            (execution_envelope_path, envelope_payload),
        ]
    )


def _run(args: argparse.Namespace, *, actual_argv: list[str]) -> None:
    _validate_sealed_output_paths(
        [args.output_raw_predictions, args.output_execution_envelope]
    )
    sealed_input = load_sealed_input_projection(args.input_projection)
    candidate_lock = load_candidate_lock(args.candidate_lock)
    input_export_receipt = load_custodian_receipt(args.input_receipt)
    validate_input_export_receipt_handoff(
        sealed_input,
        candidate_lock,
        input_export_receipt,
    )
    validate_registered_candidate_binding(candidate_lock.document)
    config = SealedDecoderConfig(
        track=args.track,
        model_revision=args.model_revision,
        device=args.device,
        ncpu=args.ncpu,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )
    result = execute_sealed_candidate(
        args.input_projection,
        args.candidate_lock,
        args.audio_root,
        config,
        actual_argv=actual_argv,
    )
    _publish_result(
        result,
        args.output_raw_predictions,
        args.output_execution_envelope,
        input_export_receipt_sha256=input_export_receipt.sha256,
    )


def main(argv: list[str] | None = None) -> int:
    if not _DIRECT_ENTRYPOINT_SECURED:
        raise SystemExit("sealed runner main requires its secured direct entrypoint")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "describe-runtime":
            document = describe_runtime(args.ncpu, device=args.device)
            sys.stdout.buffer.write(canonical_json_bytes(document))
        else:
            # FunASR and its dependencies may print progress or warnings even
            # when their progress bars are disabled. None of that output is
            # evidence, and model text must not escape through process streams.
            with _silence_decoder_streams():
                _run(args, actual_argv=_current_process_argv())
    except (
        CustodianReplayError,
        ExecutionEnvelopeError,
        SealedDecoderError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
