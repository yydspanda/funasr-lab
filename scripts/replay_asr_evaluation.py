#!/usr/bin/env python3
"""Export or score one sealed EVAL-01 custodian replay.

The command is deliberately split into a reference-free decoder handoff,
prediction freezing, and restricted scoring. It never exposes sealed
references or metrics on stdout, and it never imports or starts a FunASR model.
"""

from __future__ import annotations

import sys


SEALED_PYCACHE_PREFIX = "/dev/null/asr-lab-sealed-pycache"
_DIRECT_ENTRYPOINT_SECURED = False
if __name__ == "__main__":
    if __spec__ is not None:
        raise SystemExit("sealed commands require direct script execution")
    # Keep checkout/PYTHONPATH modules from shadowing even the standard library
    # before the project-specific import guard has a chance to run.
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
            _key.startswith((b"LD_", b"FUNASR_", b"TORCH_", b"KMP_"))
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
    }
    for _key, _value in _startup_posix.environ.items():
        if _value and _key not in _allowed_startup_names:
            _display_name = _key.decode("ascii", "backslashreplace")
            raise SystemExit(
                f"custodian forbids unexpected startup environment {_display_name}"
            )
    if not sys.flags.no_site or not sys.flags.safe_path:
        raise SystemExit("sealed commands require Python flags -P -S")

import argparse
import os
import stat
from pathlib import Path


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
    "requirements/lab-cpu.lock",
    "scripts/__init__.py",
    "scripts/check_experiment_manifests.py",
    "scripts/replay_asr_evaluation.py",
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
        raise SystemExit("cannot verify custodian process argv") from exc
    allowed_python = {
        ".venv/bin/python",
        str(REPOSITORY_ROOT / ".venv/bin/python"),
    }
    if (
        len(process_argv) < 5
        or process_argv[0] not in allowed_python
        or process_argv[1:4]
        != ["-P", "-S", "scripts/replay_asr_evaluation.py"]
        or process_argv[4]
        not in {"export-input", "freeze-predictions", "score", "validate-terminal"}
    ):
        raise SystemExit("custodian requires exact direct process argv")
    if any(
        name == "eval"
        or name.startswith("eval.")
        or name == "scripts"
        or name.startswith("scripts.")
        for name in sys.modules
    ):
        raise SystemExit("custodian project modules were loaded before bootstrap")
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

from eval.collection import CollectionValidationError
from eval.collection import build_sealed_input_projection
from eval.collection import load_collection_descriptor
from eval.collection import load_validated_collection
from eval.collection import sha256_bytes
from eval.core_report import CORE_SCHEMA_VERSION
from eval.core_report import CoreReportValidationError
from eval.core_report import IDENTITY_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import build_split_core_report
from eval.core_report import canonical_core_bytes
from eval.custodian_replay import CUSTODIAN_SCORE_RECEIPT_KIND
from eval.custodian_replay import INPUT_EXPORT_RECEIPT_KIND
from eval.custodian_replay import PREDICTION_FREEZE_RECEIPT_KIND
from eval.custodian_replay import RECEIPT_SCHEMA_VERSION
from eval.custodian_replay import SEALED_SPLIT
from eval.custodian_replay import CustodianReplayError
from eval.custodian_replay import LoadedArtifact
from eval.custodian_replay import build_candidate_lock
from eval.custodian_replay import build_prediction_bundle
from eval.custodian_replay import canonical_candidate_lock_bytes
from eval.custodian_replay import canonical_custodian_receipt_bytes
from eval.custodian_replay import canonical_prediction_bundle_bytes
from eval.custodian_replay import load_candidate_lock
from eval.custodian_replay import load_custodian_receipt
from eval.custodian_replay import load_planned_candidate_manifest
from eval.custodian_replay import load_prediction_bundle
from eval.custodian_replay import load_prediction_items_jsonl_artifact
from eval.custodian_replay import load_restricted_core_report
from eval.custodian_replay import load_sealed_input_projection
from eval.custodian_replay import load_terminal_candidate_manifest
from eval.custodian_replay import parse_sealed_input_projection
from eval.custodian_replay import preflight_replay_artifacts
from eval.custodian_replay import scorer_code_identity
from eval.custodian_replay import validate_candidate_request
from eval.custodian_replay import validate_decode_handoff
from eval.custodian_replay import validate_prediction_freeze_receipt_handoff
from eval.custodian_replay import validate_output_paths
from eval.custodian_replay import validate_prediction_handoff
from eval.custodian_replay import validate_raw_execution_handoff
from eval.custodian_replay import validate_replay_collection
from eval.custodian_replay import validate_restricted_input_paths
from eval.custodian_replay import validate_restricted_transition_paths
from eval.custodian_replay import validate_registered_candidate_binding
from eval.custodian_replay import validate_terminal_manifest_for_receipt
from eval.custodian_replay import write_atomic_outputs
from eval.execution_envelope import ExecutionEnvelopeError
from eval.execution_envelope import load_execution_envelope
from eval.sealed_decoder import SealedDecoderError
from eval.sealed_decoder import load_verified_audio_items
from eval.sealed_decoder import runner_source_identity
from eval.sealed_decoder import runtime_identity
from eval.sealed_candidate_contract import validate_sealed_candidate_execution
from eval.sealed_candidate_contract import SealedCandidateContractError


def _scorer_runtime_identity() -> dict[str, object]:
    return runtime_identity(REPOSITORY_ROOT)


def _validated_runner_source_identity(
    candidate: dict[str, object],
    execution_envelope: dict[str, object],
) -> tuple[str, str]:
    expected = runner_source_identity(
        str(candidate["code_commit"]),
        REPOSITORY_ROOT,
    )
    runner = execution_envelope["runner"]
    observed = (runner["code_commit"], runner["source_sha256"])
    if observed != expected:
        raise CustodianReplayError(
            "execution envelope runner source does not match candidate commit"
        )
    return expected


def _scorer_identity_for_candidate(
    candidate: dict[str, object],
    scorer_identity: tuple[str, str] | None,
    context: str,
) -> tuple[str, str]:
    """Require every custodian transition to use the candidate's frozen code."""

    expected_commit = candidate["code_commit"]
    current_identity = scorer_code_identity(code_commit=expected_commit)
    if scorer_identity is not None and scorer_identity != current_identity:
        raise CustodianReplayError(
            f"{context} scorer identity changed after transition startup"
        )
    if current_identity[0] != expected_commit:
        raise CustodianReplayError(
            f"{context} scorer commit does not match candidate code_commit"
        )
    return current_identity


def _collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Root for descriptor manifest and dedup-report paths.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Root for record audio paths; defaults to --collection-root.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline, sealed-reference ASR custodian workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-input",
        help="Bind a planned candidate and export reference-free decode input.",
    )
    _collection_arguments(export_parser)
    export_parser.add_argument("--candidate-manifest", type=Path, required=True)
    export_parser.add_argument(
        "--candidate-registration-commit",
        required=True,
        help="Full Git commit that first registered the exact planned manifest bytes.",
    )
    export_parser.add_argument(
        "--hypothesis-adapter-version",
        choices=(
            IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
            SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
        ),
        required=True,
    )
    export_parser.add_argument("--output-input", type=Path, required=True)
    export_parser.add_argument("--output-candidate-lock", type=Path, required=True)
    export_parser.add_argument("--output-receipt", type=Path, required=True)

    freeze_parser = subparsers.add_parser(
        "freeze-predictions",
        help="Freeze reference-free decoder JSONL into a canonical bundle.",
    )
    freeze_parser.add_argument("--input-projection", type=Path, required=True)
    freeze_parser.add_argument("--candidate-lock", type=Path, required=True)
    freeze_parser.add_argument("--input-receipt", type=Path, required=True)
    freeze_parser.add_argument("--raw-predictions", type=Path, required=True)
    freeze_parser.add_argument("--execution-envelope", type=Path, required=True)
    freeze_parser.add_argument(
        "--hypothesis-adapter-version",
        choices=(
            IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
            SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
        ),
        required=True,
    )
    freeze_parser.add_argument("--output-predictions", type=Path, required=True)
    freeze_parser.add_argument("--output-receipt", type=Path, required=True)

    score_parser = subparsers.add_parser(
        "score",
        help="Score one frozen sealed prediction bundle inside custodian scope.",
    )
    _collection_arguments(score_parser)
    score_parser.add_argument("--input-projection", type=Path, required=True)
    score_parser.add_argument("--candidate-lock", type=Path, required=True)
    score_parser.add_argument("--input-receipt", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--execution-envelope", type=Path, required=True)
    score_parser.add_argument("--prediction-receipt", type=Path, required=True)
    score_parser.add_argument("--output-core", type=Path, required=True)
    score_parser.add_argument("--output-receipt", type=Path, required=True)

    terminal_parser = subparsers.add_parser(
        "validate-terminal",
        help="Silently validate one private terminal evidence chain.",
    )
    terminal_parser.add_argument("--input-receipt", type=Path, required=True)
    terminal_parser.add_argument("--prediction-receipt", type=Path, required=True)
    terminal_parser.add_argument("--score-receipt", type=Path, required=True)
    terminal_parser.add_argument("--core", type=Path, required=True)
    terminal_parser.add_argument("--execution-envelope", type=Path, required=True)
    terminal_parser.add_argument("--terminal-manifest", type=Path, required=True)
    return parser


def _export_input(
    args: argparse.Namespace,
    *,
    startup_scorer_identity: tuple[str, str] | None = None,
) -> dict[str, object]:
    # This preflight sees only public/restricted metadata.  It must pass before
    # load_validated_collection opens any sealed reference manifest.
    validate_output_paths(
        [args.output_input, args.output_candidate_lock, args.output_receipt]
    )
    descriptor = load_collection_descriptor(args.descriptor)
    effective_audio_root = (
        args.collection_root if args.audio_root is None else args.audio_root
    )
    candidate_manifest = load_planned_candidate_manifest(
        args.candidate_manifest,
        args.candidate_registration_commit,
    )
    scorer_identity = _scorer_identity_for_candidate(
        candidate_manifest.document,
        startup_scorer_identity,
        "input export",
    )
    if candidate_manifest.document["data_sha256"] != descriptor.raw_sha256:
        raise CustodianReplayError(
            "candidate data_sha256 does not match collection descriptor"
        )
    if (
        candidate_manifest.document["normalizer_version"]
        != descriptor.normalizer_version
    ):
        raise CustodianReplayError(
            "candidate normalizer_version does not match collection descriptor"
        )
    candidate = validate_candidate_request(
        descriptor,
        candidate_manifest.document,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )
    execution_plan = validate_sealed_candidate_execution(
        candidate,
        args.hypothesis_adapter_version,
        REPOSITORY_ROOT,
    )

    actual_audio_root = Path(os.path.abspath(effective_audio_root))
    if actual_audio_root.resolve() != actual_audio_root or not actual_audio_root.is_dir():
        raise CustodianReplayError(
            "audio_root must be an existing non-symlink directory"
        )

    def command_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    actual_handoff_paths = tuple(
        Path(os.path.abspath(path))
        for path in (
            args.output_input,
            args.output_candidate_lock,
            args.output_receipt,
        )
    )
    planned_handoff_paths = tuple(
        Path(os.path.abspath(command_path(value)))
        for value in (
            execution_plan.input_projection,
            execution_plan.candidate_lock,
            execution_plan.input_receipt,
        )
    )
    if planned_handoff_paths != actual_handoff_paths:
        raise CustodianReplayError(
            "candidate runner handoff paths do not match export outputs"
        )
    planned_audio_root = Path(
        os.path.abspath(command_path(execution_plan.audio_root))
    )
    if planned_audio_root != actual_audio_root:
        raise CustodianReplayError(
            "candidate runner audio_root does not match export audio_root"
        )
    validate_output_paths(
        [
            *actual_handoff_paths,
            command_path(execution_plan.output_raw_predictions),
            command_path(execution_plan.output_execution_envelope),
        ]
    )

    collection = load_validated_collection(
        args.descriptor,
        args.collection_root,
        actual_audio_root,
    )
    sealed_input = parse_sealed_input_projection(
        build_sealed_input_projection(collection)
    )
    # Prove that the exact reference-free handoff is openable by the sealed
    # runner before publishing any part of the three-file transition.
    load_verified_audio_items(sealed_input, actual_audio_root)
    candidate_lock = build_candidate_lock(
        descriptor,
        collection,
        sealed_input,
        candidate_manifest,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )
    candidate_lock_payload = canonical_candidate_lock_bytes(candidate_lock)
    candidate_lock_sha256 = sha256_bytes(candidate_lock_payload)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": INPUT_EXPORT_RECEIPT_KIND,
        "state": "complete",
        "access_class": "restricted",
        "experiment_id": candidate_lock["candidate"]["experiment_id"],
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "split": SEALED_SPLIT,
        "decode_item_count": candidate_lock["decode_item_count"],
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock_sha256,
        "candidate_freeze_sha256": candidate_lock["candidate_freeze_sha256"],
        "candidate_registration_commit": candidate_lock[
            "candidate_registration_commit"
        ],
        "candidate_manifest_path": candidate_lock["candidate_manifest_path"],
        "candidate_manifest_sha256": candidate_lock[
            "candidate_manifest_sha256"
        ],
    }
    receipt_payload = canonical_custodian_receipt_bytes(receipt)
    if (
        scorer_code_identity(code_commit=scorer_identity[0])
        != scorer_identity
    ):
        raise CustodianReplayError(
            "custodian source identity changed during input export"
        )
    write_atomic_outputs(
        [
            (args.output_input, sealed_input.payload),
            (args.output_candidate_lock, candidate_lock_payload),
            (args.output_receipt, receipt_payload),
        ]
    )
    return receipt


def _freeze_predictions(
    args: argparse.Namespace,
    *,
    startup_scorer_identity: tuple[str, str] | None = None,
) -> dict[str, object]:
    validate_restricted_transition_paths(
        [
            args.input_projection,
            args.candidate_lock,
            args.input_receipt,
            args.raw_predictions,
            args.execution_envelope,
        ],
        [args.output_predictions, args.output_receipt],
    )
    sealed_input = load_sealed_input_projection(args.input_projection)
    candidate_lock = load_candidate_lock(args.candidate_lock)
    validate_registered_candidate_binding(candidate_lock.document)
    scorer_identity = _scorer_identity_for_candidate(
        candidate_lock.document["candidate"],
        startup_scorer_identity,
        "prediction freeze",
    )
    input_export_receipt = load_custodian_receipt(args.input_receipt)
    validate_decode_handoff(sealed_input, candidate_lock)
    if args.hypothesis_adapter_version != candidate_lock.document[
        "hypothesis_adapter_version"
    ]:
        raise CustodianReplayError(
            "requested hypothesis adapter does not match candidate lock"
        )
    raw_predictions = load_prediction_items_jsonl_artifact(args.raw_predictions)
    execution_envelope = load_execution_envelope(args.execution_envelope)
    validate_raw_execution_handoff(
        sealed_input,
        candidate_lock,
        input_export_receipt,
        raw_predictions,
        execution_envelope,
    )
    runner_identity = _validated_runner_source_identity(
        candidate_lock.document["candidate"],
        execution_envelope.document,
    )
    bundle = build_prediction_bundle(
        sealed_input,
        candidate_lock.sha256,
        raw_predictions.items,
        input_export_receipt_sha256=input_export_receipt.sha256,
        raw_predictions_sha256=raw_predictions.sha256,
        execution_envelope_sha256=execution_envelope.sha256,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )
    bundle_payload = canonical_prediction_bundle_bytes(bundle)
    loaded_bundle = LoadedArtifact(
        document=bundle,
        payload=bundle_payload,
        sha256=sha256_bytes(bundle_payload),
    )
    validate_prediction_handoff(sealed_input, candidate_lock, loaded_bundle)
    expected_count = candidate_lock.document["decode_item_count"]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": PREDICTION_FREEZE_RECEIPT_KIND,
        "state": "complete",
        "access_class": "restricted",
        "experiment_id": candidate_lock.document["candidate"]["experiment_id"],
        "dataset_id": sealed_input.document["dataset_id"],
        "revision": sealed_input.document["revision"],
        "split": SEALED_SPLIT,
        "expected_decode_item_count": expected_count,
        "prediction_item_count": bundle["item_count"],
        "missing_prediction_count": expected_count - bundle["item_count"],
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": candidate_lock.document[
            "candidate_freeze_sha256"
        ],
        "candidate_registration_commit": candidate_lock.document[
            "candidate_registration_commit"
        ],
        "candidate_manifest_path": candidate_lock.document[
            "candidate_manifest_path"
        ],
        "candidate_manifest_sha256": candidate_lock.document[
            "candidate_manifest_sha256"
        ],
        "hypothesis_adapter_version": bundle["hypothesis_adapter_version"],
        "prediction_artifact_sha256": loaded_bundle.sha256,
        "prediction_items_sha256": bundle["items_sha256"],
        "input_export_receipt_sha256": input_export_receipt.sha256,
        "raw_predictions_sha256": raw_predictions.sha256,
        "execution_envelope_sha256": execution_envelope.sha256,
        "runner_code_commit": execution_envelope.document["runner"][
            "code_commit"
        ],
        "runner_source_sha256": execution_envelope.document["runner"][
            "source_sha256"
        ],
    }
    receipt_payload = canonical_custodian_receipt_bytes(receipt)
    if (
        scorer_code_identity(code_commit=scorer_identity[0])
        != scorer_identity
    ):
        raise CustodianReplayError(
            "custodian source identity changed during prediction freeze"
        )
    if _validated_runner_source_identity(
        candidate_lock.document["candidate"],
        execution_envelope.document,
    ) != runner_identity:
        raise CustodianReplayError(
            "runner source identity changed during prediction freeze"
        )
    write_atomic_outputs(
        [
            (args.output_predictions, bundle_payload),
            (args.output_receipt, receipt_payload),
        ]
    )
    return receipt


def _score(
    args: argparse.Namespace,
    *,
    startup_scorer_identity: tuple[str, str] | None = None,
    startup_scorer_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    scorer_runtime = _scorer_runtime_identity()
    if startup_scorer_runtime is not None and scorer_runtime != startup_scorer_runtime:
        raise CustodianReplayError(
            "scorer runtime identity changed after process startup"
        )
    # No sealed reference is opened until all externally supplied artifacts
    # agree with the descriptor and one frozen candidate.
    validate_restricted_transition_paths(
        [
            args.input_projection,
            args.candidate_lock,
            args.input_receipt,
            args.predictions,
            args.execution_envelope,
            args.prediction_receipt,
        ],
        [args.output_core, args.output_receipt],
    )
    descriptor = load_collection_descriptor(args.descriptor)
    sealed_input = load_sealed_input_projection(args.input_projection)
    candidate_lock = load_candidate_lock(args.candidate_lock)
    validate_registered_candidate_binding(candidate_lock.document)
    scorer_code_commit, scorer_source_sha256 = _scorer_identity_for_candidate(
        candidate_lock.document["candidate"],
        startup_scorer_identity,
        "scoring",
    )
    input_export_receipt = load_custodian_receipt(args.input_receipt)
    predictions = load_prediction_bundle(args.predictions)
    execution_envelope = load_execution_envelope(args.execution_envelope)
    prediction_receipt = load_custodian_receipt(args.prediction_receipt)
    validate_prediction_freeze_receipt_handoff(
        sealed_input,
        candidate_lock,
        input_export_receipt,
        predictions,
        execution_envelope,
        prediction_receipt,
    )
    runner_identity = _validated_runner_source_identity(
        candidate_lock.document["candidate"],
        execution_envelope.document,
    )
    preflight_replay_artifacts(
        descriptor,
        sealed_input,
        candidate_lock,
        predictions,
    )
    effective_audio_root = (
        args.collection_root if args.audio_root is None else args.audio_root
    )
    collection = load_validated_collection(
        args.descriptor,
        args.collection_root,
        effective_audio_root,
    )
    validate_replay_collection(collection, sealed_input, candidate_lock)
    report = build_split_core_report(
        collection,
        predictions.document["items"],
        split=SEALED_SPLIT,
        hypothesis_adapter_version=predictions.document[
            "hypothesis_adapter_version"
        ],
    )
    if report["provenance"]["record_input_sha256"] != candidate_lock.document[
        "record_input_sha256"
    ]:
        raise CustodianReplayError(
            "core record identity does not match candidate lock"
        )
    core_payload = canonical_core_bytes(report)
    core_sha256 = sha256_bytes(core_payload)
    if scorer_code_identity(code_commit=scorer_code_commit) != (
        scorer_code_commit,
        scorer_source_sha256,
    ):
        raise CustodianReplayError("scorer source identity changed during scoring")
    if _scorer_runtime_identity() != scorer_runtime:
        raise CustodianReplayError("scorer runtime identity changed during scoring")
    if _validated_runner_source_identity(
        candidate_lock.document["candidate"],
        execution_envelope.document,
    ) != runner_identity:
        raise CustodianReplayError("runner source identity changed during scoring")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": CUSTODIAN_SCORE_RECEIPT_KIND,
        "state": "complete",
        "access_class": "restricted",
        "experiment_id": candidate_lock.document["candidate"]["experiment_id"],
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "evaluation_scope": report["configuration"]["evaluation_scope"],
        "data_sha256": report["provenance"]["data_sha256"],
        "input_projection_sha256": sealed_input.sha256,
        "record_identity_version": report["provenance"][
            "record_identity_version"
        ],
        "record_input_sha256": report["provenance"]["record_input_sha256"],
        "hypothesis_adapter_version": report["scoring"][
            "hypothesis_adapter_version"
        ],
        "prediction_input_sha256": report["provenance"][
            "prediction_input_sha256"
        ],
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": candidate_lock.document[
            "candidate_freeze_sha256"
        ],
        "candidate_registration_commit": candidate_lock.document[
            "candidate_registration_commit"
        ],
        "candidate_manifest_path": candidate_lock.document[
            "candidate_manifest_path"
        ],
        "candidate_manifest_sha256": candidate_lock.document[
            "candidate_manifest_sha256"
        ],
        "prediction_artifact_sha256": predictions.sha256,
        "prediction_items_sha256": predictions.document["items_sha256"],
        "input_export_receipt_sha256": input_export_receipt.sha256,
        "prediction_freeze_receipt_sha256": prediction_receipt.sha256,
        "execution_envelope_sha256": execution_envelope.sha256,
        "runner_code_commit": execution_envelope.document["runner"][
            "code_commit"
        ],
        "runner_source_sha256": execution_envelope.document["runner"][
            "source_sha256"
        ],
        "scorer_code_commit": scorer_code_commit,
        "scorer_source_sha256": scorer_source_sha256,
        "scorer_runtime": scorer_runtime,
        "core_schema_version": CORE_SCHEMA_VERSION,
        "core_sha256": core_sha256,
        "public_release": {
            "state": "withheld",
            "summary_sha256": None,
            "reason_code": "release_policy_not_implemented",
        },
    }
    receipt_payload = canonical_custodian_receipt_bytes(receipt)
    write_atomic_outputs(
        [
            (args.output_core, core_payload),
            (args.output_receipt, receipt_payload),
        ]
    )
    return receipt


def _validate_terminal(
    args: argparse.Namespace,
    *,
    startup_scorer_runtime: dict[str, object] | None = None,
    startup_scorer_identity: tuple[str, str] | None = None,
) -> None:
    scorer_runtime = _scorer_runtime_identity()
    if startup_scorer_runtime is not None and scorer_runtime != startup_scorer_runtime:
        raise CustodianReplayError(
            "terminal scorer runtime changed after process startup"
        )
    validate_restricted_input_paths(
        [
            args.terminal_manifest,
            args.input_receipt,
            args.prediction_receipt,
            args.score_receipt,
            args.core,
            args.execution_envelope,
        ]
    )
    terminal_manifest = load_terminal_candidate_manifest(args.terminal_manifest)
    scorer_identity = scorer_code_identity(
        code_commit=terminal_manifest["code_commit"]
    )
    if (
        startup_scorer_identity is not None
        and startup_scorer_identity != scorer_identity
    ):
        raise CustodianReplayError(
            "terminal validator source identity changed after process startup"
        )
    input_receipt = load_custodian_receipt(args.input_receipt).document
    prediction_receipt = load_custodian_receipt(args.prediction_receipt).document
    score_receipt = load_custodian_receipt(args.score_receipt).document
    validate_registered_candidate_binding(score_receipt)
    if scorer_identity != (
        score_receipt["scorer_code_commit"],
        score_receipt["scorer_source_sha256"],
    ):
        raise CustodianReplayError(
            "terminal validator source does not match score receipt"
        )
    if score_receipt.get("scorer_runtime") != scorer_runtime:
        raise CustodianReplayError(
            "terminal scorer runtime does not match score receipt"
        )
    core_report = load_restricted_core_report(args.core)
    execution_envelope = load_execution_envelope(
        args.execution_envelope
    ).document
    runner_identity = _validated_runner_source_identity(
        terminal_manifest,
        execution_envelope,
    )
    validate_terminal_manifest_for_receipt(
        terminal_manifest,
        score_receipt,
        core_report,
        execution_envelope,
        input_receipt,
        prediction_receipt,
    )
    if (
        scorer_code_identity(code_commit=scorer_identity[0])
        != scorer_identity
    ):
        raise CustodianReplayError(
            "terminal validator source identity changed during validation"
        )
    if _validated_runner_source_identity(
        terminal_manifest,
        execution_envelope,
    ) != runner_identity:
        raise CustodianReplayError(
            "terminal runner source identity changed during validation"
        )


def main(argv: list[str] | None = None) -> int:
    if not _DIRECT_ENTRYPOINT_SECURED:
        raise SystemExit("custodian main requires its secured direct entrypoint")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        startup_scorer_runtime = (
            _scorer_runtime_identity()
            if args.command in {"score", "validate-terminal"}
            else None
        )
        if args.command == "export-input":
            _export_input(args)
        elif args.command == "freeze-predictions":
            _freeze_predictions(args)
        elif args.command == "score":
            _score(
                args,
                startup_scorer_runtime=startup_scorer_runtime,
            )
        else:
            _validate_terminal(
                args,
                startup_scorer_runtime=startup_scorer_runtime,
            )
    except (
        CollectionValidationError,
        CoreReportValidationError,
        CustodianReplayError,
        ExecutionEnvelopeError,
        SealedCandidateContractError,
        SealedDecoderError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
