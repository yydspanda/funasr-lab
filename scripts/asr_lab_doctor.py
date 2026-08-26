#!/usr/bin/env python3
"""Check the lightweight prerequisites for working in this downstream fork."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_GIB = 30
ROADMAP_PATH = REPO_ROOT / ".notes" / "asr" / "delivery-roadmap.md"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _command_output(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _baseline_control() -> tuple[str, str]:
    try:
        text = ROADMAP_PATH.read_text(encoding="utf-8")
    except OSError:
        return "", ""
    ref_match = re.search(
        r"^- \*\*Baseline Ref:\*\* `([^`]+)`$", text, re.MULTILINE
    )
    commit_match = re.search(
        r"^- \*\*Baseline Commit:\*\* `([0-9a-f]{40})`$", text, re.MULTILINE
    )
    return (
        ref_match.group(1) if ref_match else "",
        commit_match.group(1) if commit_match else "",
    )


def collect_checks() -> list[Check]:
    """Return deterministic prerequisite checks without downloading anything."""

    checks: list[Check] = []
    version = sys.version_info
    python_ok = (version.major, version.minor) >= (3, 10)
    python_status = "pass" if python_ok else "fail"
    python_detail = f"{version.major}.{version.minor}.{version.micro}"
    if python_ok and version.minor != 11:
        python_status = "warn"
        python_detail += "; project bootstrap uses Python 3.11"
    checks.append(Check("python", python_status, python_detail))

    for executable in ("git", "uv", "ffmpeg"):
        path = shutil.which(executable)
        checks.append(
            Check(executable, "pass" if path else "fail", path or "not found")
        )

    cmake = shutil.which("cmake")
    checks.append(
        Check(
            "cmake",
            "pass" if cmake else "warn",
            cmake or "not found; required only for C++/ONNX runtime work",
        )
    )

    free_gib = shutil.disk_usage(REPO_ROOT).free / (1024**3)
    checks.append(
        Check(
            "disk",
            "pass" if free_gib >= MIN_FREE_GIB else "fail",
            f"{free_gib:.1f} GiB free; minimum {MIN_FREE_GIB} GiB",
        )
    )

    origin = _command_output("git", "remote", "get-url", "origin")
    upstream = _command_output("git", "remote", "get-url", "upstream")
    checks.append(
        Check(
            "origin",
            "pass" if "yydspanda/funasr-lab" in origin else "fail",
            origin or "missing",
        )
    )
    checks.append(
        Check(
            "upstream",
            "pass" if "modelscope/FunASR" in upstream else "fail",
            upstream or "missing",
        )
    )

    baseline_ref, baseline_commit = _baseline_control()
    baseline = (
        _command_output("git", "rev-parse", f"{baseline_ref}^{{commit}}")
        if baseline_ref
        else ""
    )
    baseline_ok = bool(baseline and baseline == baseline_commit)
    checks.append(
        Check(
            "baseline",
            "pass" if baseline_ok else "fail",
            (
                f"{baseline_ref} -> {baseline[:12]}"
                if baseline_ok
                else "Roadmap Baseline Ref/Commit is missing, unresolved, or mismatched"
            ),
        )
    )

    lock_path = REPO_ROOT / "requirements" / "lab-cpu.lock"
    checks.append(
        Check(
            "environment-lock",
            "pass" if lock_path.is_file() else "fail",
            str(lock_path.relative_to(REPO_ROOT)),
        )
    )
    return checks


def main() -> int:
    checks = collect_checks()
    if "--json" in sys.argv:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            print(f"[{check.status.upper():4}] {check.name}: {check.detail}")

    failures = [check for check in checks if check.status == "fail"]
    if failures:
        print(f"\nDoctor found {len(failures)} blocking issue(s).", file=sys.stderr)
        return 1
    print("\nDoctor passed. Warnings are non-blocking for the baseline stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
