#!/usr/bin/env python3
"""Run one pinned BASE-01 offline track on a frozen JSONL corpus.

A real invocation may download the explicitly pinned ModelScope model when it
is not already cached. Dataset validation completes before FunASR is imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import BaselineError
from eval.offline_baseline import TRACKS
from eval.offline_baseline import command_environment
from eval.offline_baseline import load_frozen_dataset
from eval.offline_baseline import run_offline_baseline
from eval.offline_baseline import write_report


def _expanded_argv(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/run_offline_baseline.py",
        "--track",
        args.track,
        "--dataset-manifest",
        str(args.dataset_manifest),
        "--model-revision",
        args.model_revision,
        "--output-report",
        str(args.output_report),
        "--device",
        args.device,
        "--ncpu",
        str(args.ncpu),
        "--warmup-runs",
        str(args.warmup_runs),
        "--seed",
        str(args.seed),
    ]


def _validated_working_directory(current_directory: Path | None = None) -> str:
    actual = (Path.cwd() if current_directory is None else current_directory).resolve()
    expected = REPOSITORY_ROOT.resolve()
    if actual != expected:
        raise BaselineError(
            "run_offline_baseline.py must run from the repository root; "
            f"expected {expected}, got {actual}"
        )
    return "."


def _validated_command_environment(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = command_environment(source)
    expected_cache = (REPOSITORY_ROOT / ".cache/modelscope").resolve()
    configured_cache = environment.get("MODELSCOPE_CACHE")
    if configured_cache is None or Path(configured_cache).resolve() != expected_cache:
        raise BaselineError(
            "MODELSCOPE_CACHE must resolve to the repository-local cache at "
            f"{expected_cache}"
        )
    if environment.get("PYTHONHASHSEED") != "0":
        raise BaselineError(
            "PYTHONHASHSEED must be set to 0 before starting the baseline process"
        )
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pinned, ASR-only FunASR offline baseline.",
    )
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--ncpu", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        working_directory = _validated_working_directory()
        environment = _validated_command_environment()
        dataset = load_frozen_dataset(args.dataset_manifest, REPOSITORY_ROOT)
        config = BaselineConfig(
            track=args.track,
            model_revision=args.model_revision,
            device=args.device,
            ncpu=args.ncpu,
            warmup_runs=args.warmup_runs,
            seed=args.seed,
        )
        command = {
            "working_directory": working_directory,
            "argv": _expanded_argv(args),
            "environment": environment,
        }
        report = run_offline_baseline(dataset, config, command=command)
        report_sha256 = write_report(report, args.output_report)
    except BaselineError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "kind": "offline-baseline-complete",
                "track": args.track,
                "report": str(args.output_report),
                "report_sha256": report_sha256,
                "data_sha256": report["provenance"]["data_sha256"],
                "config_sha256": report["provenance"]["config_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
