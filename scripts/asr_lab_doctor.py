#!/usr/bin/env python3
"""Check the lightweight prerequisites for working in this downstream fork."""

from __future__ import annotations

import json
import os
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
BASE_INPUT_PATH = REPO_ROOT / "requirements" / "lab-cpu.in"
BASE_LOCK_PATH = REPO_ROOT / "requirements" / "lab-cpu.lock"
BASE_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PROJECT_VERSION_PATH = REPO_ROOT / "funasr" / "version.txt"
BASE_PROBE_MARKER = "__ASR_LAB_BASE_ENV__="
PINNED_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^;\s]+)$"
)
IMPORT_NAME_OVERRIDES: dict[str, str] = {}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class BaseRequirement:
    distribution: str
    module: str
    expected_version: str


def _command_output(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_repo_funasr_module(module_file: object, repo_root: Path) -> bool:
    if not isinstance(module_file, str):
        return False
    try:
        Path(module_file).resolve().relative_to((repo_root / "funasr").resolve())
    except (OSError, ValueError):
        return False
    return True


def _read_locked_versions(lock_path: Path) -> tuple[dict[str, str], str]:
    try:
        lock_lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {}, f"cannot read BASE lock: {error}"

    versions: dict[str, str] = {}
    for raw_line in lock_lines:
        match = PINNED_REQUIREMENT_RE.fullmatch(raw_line.strip())
        if match:
            versions[_canonical_distribution(match.group(1))] = match.group(2)
    if not versions:
        return {}, "BASE lock contains no exact resolved versions"
    return versions, ""


def _read_base_requirements(
    input_path: Path = BASE_INPUT_PATH,
    lock_path: Path = BASE_LOCK_PATH,
    project_version_path: Path = PROJECT_VERSION_PATH,
) -> tuple[list[BaseRequirement], str]:
    """Resolve direct BASE imports to their exact locked versions."""

    try:
        input_lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], f"cannot read BASE requirement files: {error}"

    locked_versions, lock_error = _read_locked_versions(lock_path)
    if lock_error:
        return [], lock_error

    requirements: list[BaseRequirement] = []
    for raw_line in input_lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "-e .":
            try:
                expected_version = project_version_path.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError as error:
                return [], f"cannot read project version: {error}"
            if not expected_version:
                return [], "project version is empty"
            requirements.append(
                BaseRequirement("funasr", "funasr", expected_version)
            )
            continue

        match = PINNED_REQUIREMENT_RE.fullmatch(line)
        if not match:
            return [], f"unsupported direct BASE requirement: {line}"
        distribution = _canonical_distribution(match.group(1))
        expected_version = locked_versions.get(distribution)
        if expected_version is None:
            return [], (
                f"{distribution} has no exact resolved version in {lock_path.name}"
            )
        module = IMPORT_NAME_OVERRIDES.get(distribution, distribution.replace("-", "_"))
        requirements.append(
            BaseRequirement(distribution, module, expected_version)
        )

    if not requirements:
        return [], "no direct BASE requirements were found"
    return requirements, ""


def _run_base_environment_probe(
    python_path: Path,
    requirements: list[BaseRequirement],
    locked_distributions: list[str],
    repo_root: Path,
) -> tuple[dict[str, object] | None, str]:
    probe = f"""\
import importlib
import importlib.metadata
import json
import sys

request = json.loads(sys.argv[1])
requirements = request[\"imports\"]
results = []
for requirement in requirements:
    try:
        module = importlib.import_module(requirement[\"module\"])
        distribution_version = importlib.metadata.version(
            requirement[\"distribution\"]
        )
        module_version = getattr(module, \"__version__\", None)
        module_file = getattr(module, \"__file__\", None)
        results.append({{
            \"distribution\": requirement[\"distribution\"],
            \"distribution_version\": distribution_version,
            \"module_version\": (
                module_version if isinstance(module_version, str) else None
            ),
            \"module_file\": module_file if isinstance(module_file, str) else None,
            \"error\": None,
        }})
    except Exception as error:
        results.append({{
            \"distribution\": requirement[\"distribution\"],
            \"distribution_version\": None,
            \"module_version\": None,
            \"module_file\": None,
            \"error\": f\"{{type(error).__name__}}: {{error}}\",
        }})

installed_versions = {{}}
for distribution in request[\"locked_distributions\"]:
    try:
        installed_versions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass

payload = {{
    \"python_version\": list(sys.version_info[:3]),
    \"is_venv\": sys.prefix != sys.base_prefix,
    \"installed_versions\": installed_versions,
    \"results\": results,
}}
print({BASE_PROBE_MARKER!r} + json.dumps(payload, sort_keys=True))
"""
    probe_request = json.dumps(
        {
            "imports": [
                {
                    "distribution": requirement.distribution,
                    "module": requirement.module,
                }
                for requirement in requirements
            ],
            "locked_distributions": locked_distributions,
        }
    )
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    try:
        completed = subprocess.run(
            (str(python_path), "-I", "-c", probe, probe_request),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"could not run BASE import probe: {error}"

    marker_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(BASE_PROBE_MARKER)
        ),
        "",
    )
    if completed.returncode != 0 or not marker_line:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return None, f"BASE import probe failed: {detail or 'no diagnostic output'}"
    try:
        payload = json.loads(marker_line[len(BASE_PROBE_MARKER) :])
    except json.JSONDecodeError as error:
        return None, f"BASE import probe returned invalid JSON: {error}"
    if not isinstance(payload, dict):
        return None, "BASE import probe returned an invalid payload"
    return payload, ""


def _base_environment_checks(
    repo_root: Path = REPO_ROOT,
    input_path: Path = BASE_INPUT_PATH,
    lock_path: Path = BASE_LOCK_PATH,
    project_version_path: Path = PROJECT_VERSION_PATH,
    python_path: Path = BASE_VENV_PYTHON,
    strict: bool = False,
) -> list[Check]:
    requirements, requirement_error = _read_base_requirements(
        input_path, lock_path, project_version_path
    )
    if requirement_error:
        return [Check("base-environment", "fail", requirement_error)]

    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        venv_path = python_path.parent.parent
        venv_exists = venv_path.exists() or venv_path.is_symlink()
        status = "fail" if strict or venv_exists else "warn"
        return [
            Check(
                "base-environment",
                status,
                f"{python_path} is missing or not executable; "
                "run scripts/bootstrap_dev.sh",
            )
        ]

    locked_versions, lock_error = _read_locked_versions(lock_path)
    if lock_error:
        return [Check("base-environment", "fail", lock_error)]
    payload, probe_error = _run_base_environment_probe(
        python_path, requirements, sorted(locked_versions), repo_root
    )
    if probe_error or payload is None:
        return [Check("base-environment", "fail", probe_error)]

    python_version = payload.get("python_version")
    is_venv = payload.get("is_venv") is True
    python_ok = (
        isinstance(python_version, list)
        and len(python_version) == 3
        and python_version[:2] == [3, 11]
        and all(isinstance(part, int) for part in python_version)
    )
    version_detail = (
        ".".join(str(part) for part in python_version)
        if isinstance(python_version, list)
        else "unknown"
    )
    checks = [
        Check(
            "base-environment",
            "pass" if is_venv and python_ok else "fail",
            (
                f"{python_path.relative_to(repo_root)} uses Python {version_detail}"
                if is_venv and python_ok
                else (
                    "expected a Python 3.11 virtual environment, "
                    f"found {version_detail}"
                )
            ),
        )
    ]

    installed_versions = payload.get("installed_versions")
    lock_mismatches: list[str] = []
    if not isinstance(installed_versions, dict):
        lock_mismatches.append("probe returned no installed distribution versions")
    else:
        for distribution, expected_version in sorted(locked_versions.items()):
            installed_version = installed_versions.get(distribution)
            if installed_version is None:
                lock_mismatches.append(f"{distribution} is missing")
            elif installed_version != expected_version:
                lock_mismatches.append(
                    f"{distribution} expected {expected_version}, "
                    f"found {installed_version}"
                )
    if lock_mismatches:
        shown = "; ".join(lock_mismatches[:4])
        remaining = len(lock_mismatches) - 4
        if remaining > 0:
            shown += f"; and {remaining} more"
        lock_detail = shown
    else:
        lock_detail = (
            f"{len(locked_versions)} resolved distributions match the CPU lock"
        )
    checks.append(
        Check(
            "base-lock-sync",
            "fail" if lock_mismatches else "pass",
            lock_detail,
        )
    )

    raw_results = payload.get("results")
    results = (
        {
            result.get("distribution"): result
            for result in raw_results
            if isinstance(result, dict)
            and isinstance(result.get("distribution"), str)
        }
        if isinstance(raw_results, list)
        else {}
    )
    for requirement in requirements:
        result = results.get(requirement.distribution, {})
        error = result.get("error")
        installed_version = result.get("distribution_version")
        module_version = result.get("module_version")
        module_file = result.get("module_file")
        version_ok = installed_version == requirement.expected_version
        module_version_ok = module_version in (None, requirement.expected_version)
        source_ok = requirement.distribution != "funasr" or _is_repo_funasr_module(
            module_file, repo_root
        )
        import_ok = error is None and isinstance(installed_version, str)
        passed = import_ok and version_ok and module_version_ok and source_ok
        if error:
            detail = (
                f"import {requirement.module} failed: {error}; "
                f"expected {requirement.expected_version}"
            )
        elif not import_ok:
            detail = (
                f"import {requirement.module} returned no installed version; "
                f"expected {requirement.expected_version}"
            )
        elif not version_ok:
            detail = (
                f"expected {requirement.expected_version}, found {installed_version}"
            )
        elif not module_version_ok:
            detail = (
                f"metadata is {installed_version}, but module reports {module_version}"
            )
        elif not source_ok:
            detail = (
                f"funasr imports from {module_file or 'an unknown path'}; "
                f"expected {(repo_root / 'funasr').resolve()}"
            )
        elif requirement.distribution == "funasr":
            detail = f"{installed_version}; import funasr succeeded from {module_file}"
        else:
            detail = f"{installed_version}; import {requirement.module} succeeded"
        checks.append(
            Check(
                f"base-import-{requirement.distribution}",
                "pass" if passed else "fail",
                detail,
            )
        )
    return checks


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


def collect_checks(*, strict_base_env: bool = False) -> list[Check]:
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

    checks.append(
        Check(
            "environment-lock",
            "pass" if BASE_LOCK_PATH.is_file() else "fail",
            str(BASE_LOCK_PATH.relative_to(REPO_ROOT)),
        )
    )
    checks.extend(_base_environment_checks(strict=strict_base_env))
    return checks


def main() -> int:
    checks = collect_checks(strict_base_env="--strict-base-env" in sys.argv)
    if "--json" in sys.argv:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            print(f"[{check.status.upper():4}] {check.name}: {check.detail}")

    failures = [check for check in checks if check.status == "fail"]
    if failures:
        print(f"\nDoctor found {len(failures)} blocking issue(s).", file=sys.stderr)
        return 1
    print(
        "\nDoctor passed. Warnings are diagnostic; bootstrap enforces the "
        "strict BASE environment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
