#!/usr/bin/env python3
"""Measure fork drift and keep downstream work out of upstream core paths.

The command intentionally uses only the Python standard library.  Its stdout is
one JSON document so scheduled CI can archive or process the result directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = 1
DEFAULT_ORIGIN_REPOSITORY = "yydspanda/funasr-lab"
DEFAULT_UPSTREAM_REPOSITORY = "modelscope/FunASR"
DEFAULT_FORK_REF = "refs/remotes/origin/main"
DEFAULT_ACTIVE_REF = "refs/remotes/origin/develop"
DEFAULT_UPSTREAM_REF = "refs/remotes/upstream/main"
DEFAULT_LEDGER = ".notes/asr/upstream-core-patches.json"
DEFAULT_ROADMAP = ".notes/asr/delivery-roadmap.md"

# New files under these roots are the preferred downstream implementation
# surface.  A file that already existed at the vendor baseline is still
# reported as upstream-owned, even when it lives under one of these roots.
DOWNSTREAM_OWNED_PREFIXES = (
    ".agents/",
    ".github/workflows/",
    ".notes/",
    "asr_lab/",
    "designdocs/agents/",
    "eval/",
    "experiments/",
    "requirements/",
    "scripts/",
    "tests/",
)
DOWNSTREAM_OWNED_EXACT = {
    ".python-version",
    "AGENTS.md",
    "CLAUDE.md",
    "Makefile",
}

# Additions and edits here change the imported toolkit/runtime itself.  They
# therefore require an exact, non-wildcard ledger entry with focused tests.
CORE_PREFIXES = (
    "benchmarks/",
    "data/",
    "examples/",
    "funasr/",
    "fun_text_processing/",
    "integrations/",
    "model_zoo/",
    "runtime/",
)
CORE_EXACT = {
    "benchmark_vllm.py",
    "pyproject.toml",
    "setup.py",
}
TASK_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*-(?:\d{2}|SYNC)$")


class GuardError(RuntimeError):
    """An expected repository or policy validation failure."""


@dataclass(frozen=True)
class DiffChange:
    status: str
    paths: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "paths": list(self.paths)}


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise GuardError(f"git {' '.join(args)} failed: {detail}")
    return result


def github_repository_slug(url: str) -> str | None:
    """Return owner/repository for supported GitHub SSH/HTTPS URLs."""

    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]

    if value.startswith("git@github.com:"):
        path = value.split(":", 1)[1]
    else:
        parsed = urlparse(value)
        if parsed.hostname is None or parsed.hostname.lower() != "github.com":
            return None
        path = parsed.path.lstrip("/")

    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}".lower()


def remote_urls(repo: Path, remote: str, *, push: bool = False) -> list[str]:
    args = ["remote", "get-url", "--all"]
    if push:
        args.append("--push")
    args.append(remote)
    result = run_git(repo, *args, check=False)
    if result.returncode:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def inspect_remotes(
    repo: Path,
    expected_origin: str,
    expected_upstream: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    origin_fetch = remote_urls(repo, "origin")
    upstream_fetch = remote_urls(repo, "upstream")
    upstream_push = remote_urls(repo, "upstream", push=True)
    errors: list[dict[str, str]] = []

    expected_origin_slug = expected_origin.lower()
    expected_upstream_slug = expected_upstream.lower()
    origin_slugs = [github_repository_slug(url) for url in origin_fetch]
    upstream_slugs = [github_repository_slug(url) for url in upstream_fetch]

    if origin_slugs != [expected_origin_slug]:
        errors.append(
            {
                "code": "origin_remote_mismatch",
                "message": (
                    f"origin fetch URL must resolve only to {expected_origin}; "
                    f"found {origin_fetch or ['<missing>']}"
                ),
            }
        )
    if upstream_slugs != [expected_upstream_slug]:
        errors.append(
            {
                "code": "upstream_remote_mismatch",
                "message": (
                    f"upstream fetch URL must resolve only to {expected_upstream}; "
                    f"found {upstream_fetch or ['<missing>']}"
                ),
            }
        )
    if upstream_push != ["no_push"]:
        errors.append(
            {
                "code": "upstream_push_not_disabled",
                "message": (
                    "upstream must be fetch-only with push URL 'no_push'; "
                    f"found {upstream_push or ['<missing>']}"
                ),
            }
        )

    return (
        {
            "origin": {"fetch_urls": origin_fetch},
            "upstream": {
                "fetch_urls": upstream_fetch,
                "push_urls": upstream_push,
                "fetch_only": upstream_push == ["no_push"],
            },
        },
        errors,
    )


def fetch_tracking_refs(repo: Path) -> None:
    run_git(
        repo,
        "fetch",
        "--prune",
        "--no-tags",
        "origin",
        "+refs/heads/main:refs/remotes/origin/main",
        "+refs/heads/develop:refs/remotes/origin/develop",
    )
    run_git(
        repo,
        "fetch",
        "--force",
        "--prune",
        "--tags",
        "upstream",
        "+refs/heads/main:refs/remotes/upstream/main",
    )


def resolve_commit(repo: Path, ref: str) -> str:
    result = run_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    return result.stdout.strip()


def ahead_behind(repo: Path, fork_ref: str, upstream_ref: str) -> tuple[int, int]:
    result = run_git(
        repo,
        "rev-list",
        "--left-right",
        "--count",
        f"{fork_ref}...{upstream_ref}",
    )
    fields = result.stdout.split()
    if len(fields) != 2:
        raise GuardError(f"unexpected rev-list output: {result.stdout!r}")
    return int(fields[0]), int(fields[1])


def parse_name_status_z(raw: str) -> list[DiffChange]:
    fields = raw.split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    changes: list[DiffChange] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        if index + path_count > len(fields):
            raise GuardError("malformed NUL-delimited git diff output")
        paths = tuple(fields[index : index + path_count])
        index += path_count
        changes.append(DiffChange(status=status, paths=paths))
    return changes


def diff_changes(repo: Path, baseline_ref: str, downstream_ref: str) -> list[DiffChange]:
    result = run_git(
        repo,
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        f"{baseline_ref}...{downstream_ref}",
    )
    return parse_name_status_z(result.stdout)


def files_at_ref(repo: Path, ref: str) -> set[str]:
    result = run_git(repo, "ls-tree", "-r", "--name-only", "-z", ref)
    return {path for path in result.stdout.split("\0") if path}


def is_core_path(path: str) -> bool:
    return path in CORE_EXACT or path.startswith(CORE_PREFIXES)


def is_recommended_downstream_path(path: str) -> bool:
    return path in DOWNSTREAM_OWNED_EXACT or path.startswith(DOWNSTREAM_OWNED_PREFIXES)


def changed_paths(changes: Iterable[DiffChange]) -> set[str]:
    return {path for change in changes for path in change.paths}


def ref_object_type(repo: Path, ref: str, path: str) -> str | None:
    result = run_git(repo, "cat-file", "-t", f"{ref}:{path}", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = run_git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode not in (0, 1):
        detail = result.stderr.strip() or "unknown merge-base error"
        raise GuardError(f"cannot compare {ancestor} with {descendant}: {detail}")
    return result.returncode == 0


def valid_repo_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "*" not in value


def ledger_test_module(value: str) -> str | None:
    """Map a focused tests/test_*.py path to a unittest module safely."""

    path = PurePosixPath(value)
    if (
        not valid_repo_relative_path(value)
        or not value.startswith("tests/test_")
        or path.suffix != ".py"
    ):
        return None
    parts = path.with_suffix("").parts
    if not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def load_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GuardError(f"core patch ledger is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GuardError(f"core patch ledger is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GuardError("core patch ledger root must be a JSON object")
    return payload


def load_roadmap_task_ids(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GuardError(f"cannot read roadmap task registry {path}: {exc}") from exc
    task_ids = set(
        re.findall(
            r"^\|\s*`([A-Z][A-Z0-9]*-(?:\d{2}|SYNC))`\s*\|",
            text,
            re.MULTILINE,
        )
    )
    if not task_ids:
        raise GuardError(f"no roadmap task IDs found in {path}")
    return task_ids


def load_roadmap_baseline_commit(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GuardError(f"cannot read roadmap baseline {path}: {exc}") from exc
    values = re.findall(
        r"^- \*\*Baseline Commit:\*\* `([0-9a-f]{40})`$",
        text,
        re.MULTILINE,
    )
    if len(values) != 1:
        raise GuardError(
            f"expected exactly one full Baseline Commit field in {path}; "
            f"found {len(values)}"
        )
    return values[0]


def validate_ledger(
    repo: Path,
    ledger: dict[str, Any],
    baseline_commit: str,
    downstream_ref: str,
    core_paths: set[str],
    downstream_changed_paths: set[str],
    roadmap_task_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []

    if ledger.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            {
                "code": "invalid_core_patch_ledger",
                "message": f"ledger schema_version must be {SCHEMA_VERSION}",
            }
        )
    if ledger.get("baseline_commit") != baseline_commit:
        errors.append(
            {
                "code": "baseline_commit_mismatch",
                "message": (
                    "ledger baseline_commit must equal the resolved immutable vendor "
                    f"commit {baseline_commit}"
                ),
            }
        )
    if not isinstance(ledger.get("baseline_ref"), str) or not ledger["baseline_ref"].strip():
        errors.append(
            {
                "code": "invalid_core_patch_ledger",
                "message": "ledger baseline_ref must name the release or sync baseline",
            }
        )

    entries = ledger.get("core_patches")
    if not isinstance(entries, list):
        return [], errors + [
            {
                "code": "invalid_core_patch_ledger",
                "message": "ledger core_patches must be a JSON array",
            }
        ]

    valid_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"core_patches[{index}]"
        if not isinstance(entry, dict):
            errors.append(
                {"code": "invalid_core_patch_entry", "message": f"{label} must be an object"}
            )
            continue
        path = entry.get("path")
        task_id = entry.get("task_id")
        reason = entry.get("reason")
        tests = entry.get("tests")
        entry_ok = True
        if not isinstance(path, str) or not valid_repo_relative_path(path):
            errors.append(
                {
                    "code": "invalid_core_patch_entry",
                    "message": f"{label}.path must be one exact repository-relative path",
                }
            )
            entry_ok = False
        elif not is_core_path(path):
            errors.append(
                {
                    "code": "invalid_core_patch_entry",
                    "message": f"{label}.path is not an upstream core path: {path}",
                }
            )
            entry_ok = False
        elif path in seen:
            errors.append(
                {
                    "code": "duplicate_core_patch_entry",
                    "message": f"duplicate ledger path: {path}",
                }
            )
            entry_ok = False
        else:
            seen.add(path)

        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            errors.append(
                {
                    "code": "invalid_core_patch_entry",
                    "message": f"{label}.task_id must be a roadmap-style task ID",
                }
            )
            entry_ok = False
        elif task_id not in roadmap_task_ids:
            errors.append(
                {
                    "code": "unknown_core_patch_task",
                    "message": f"{label}.task_id does not exist in the roadmap: {task_id}",
                }
            )
            entry_ok = False
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append(
                {
                    "code": "invalid_core_patch_entry",
                    "message": (
                        f"{label}.reason must explain in at least 20 characters "
                        "why extension is insufficient"
                    ),
                }
            )
            entry_ok = False
        if not isinstance(tests, list) or not tests:
            errors.append(
                {
                    "code": "invalid_core_patch_entry",
                    "message": f"{label}.tests must name at least one focused versioned test",
                }
            )
            entry_ok = False
        else:
            for test_path in tests:
                valid_test_path = (
                    isinstance(test_path, str)
                    and ledger_test_module(test_path) is not None
                    and test_path in downstream_changed_paths
                    and ref_object_type(repo, downstream_ref, test_path) == "blob"
                )
                if not valid_test_path:
                    errors.append(
                        {
                            "code": "invalid_core_patch_test",
                            "message": (
                                f"{label}.tests must reference a changed, regular "
                                f"tests/test_*.py file: {test_path!r}"
                            ),
                        }
                    )
                    entry_ok = False
        if entry_ok:
            valid_entries.append(entry)

    ledger_paths = {entry["path"] for entry in valid_entries}
    for path in sorted(core_paths - ledger_paths):
        errors.append(
            {
                "code": "unregistered_core_patch",
                "message": f"upstream core change requires a ledger entry: {path}",
            }
        )
    for path in sorted(ledger_paths - core_paths):
        errors.append(
            {
                "code": "stale_core_patch_entry",
                "message": f"ledger entry does not correspond to a current core diff: {path}",
            }
        )
    return valid_entries, errors


def source_isolation_summary(
    repo: Path,
    baseline_ref: str | None,
    downstream_ref: str,
    ledger_path: Path,
    roadmap_path: Path | None = None,
    trusted_upstream_ref: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    ledger = load_ledger(ledger_path)
    declared_baseline = ledger.get("baseline_commit")
    if not isinstance(declared_baseline, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", declared_baseline
    ):
        raise GuardError("ledger baseline_commit must be one full 40-character Git SHA")
    baseline_commit = resolve_commit(repo, declared_baseline)
    declared_baseline_ref = ledger.get("baseline_ref")
    if not isinstance(declared_baseline_ref, str) or not declared_baseline_ref.strip():
        raise GuardError("ledger baseline_ref must name the release or sync baseline")
    baseline_ref_commit = resolve_commit(repo, declared_baseline_ref)
    if baseline_ref_commit != baseline_commit:
        raise GuardError(
            f"ledger baseline_ref resolves to {baseline_ref_commit}, but "
            f"baseline_commit records {baseline_commit}"
        )
    if baseline_ref is not None:
        override_commit = resolve_commit(repo, baseline_ref)
        if override_commit != baseline_commit:
            raise GuardError(
                f"--baseline-ref resolves to {override_commit}, but ledger records "
                f"{baseline_commit}"
            )
    downstream_commit = resolve_commit(repo, downstream_ref)
    if not is_ancestor(repo, baseline_commit, downstream_commit):
        raise GuardError(
            "ledger baseline_commit must be an ancestor of the downstream ref; "
            "advance it only as part of a controlled upstream integration"
        )
    if trusted_upstream_ref is not None and not is_ancestor(
        repo, baseline_commit, trusted_upstream_ref
    ):
        raise GuardError(
            "ledger baseline_commit is not part of trusted upstream history; "
            "a downstream commit cannot be used to hide core changes"
        )

    if roadmap_path is None:
        roadmap_path = repo / DEFAULT_ROADMAP
    roadmap_task_ids = load_roadmap_task_ids(roadmap_path)
    roadmap_baseline_commit = load_roadmap_baseline_commit(roadmap_path)
    if roadmap_baseline_commit != baseline_commit:
        raise GuardError(
            f"Roadmap Baseline Commit is {roadmap_baseline_commit}, but the "
            f"core patch ledger records {baseline_commit}"
        )
    baseline_files = files_at_ref(repo, baseline_commit)
    changes = diff_changes(repo, baseline_commit, downstream_ref)

    upstream_owned_paths = sorted(changed_paths(changes) & baseline_files)
    all_paths = changed_paths(changes)
    core_paths = {path for path in all_paths if is_core_path(path)}
    added_paths = {
        path
        for change in changes
        if change.status.startswith("A")
        for path in change.paths
    }
    outside_recommended = sorted(
        path
        for path in added_paths
        if path not in baseline_files
        and not is_recommended_downstream_path(path)
        and not is_core_path(path)
    )
    entries, errors = validate_ledger(
        repo,
        ledger,
        baseline_commit,
        downstream_ref,
        core_paths,
        all_paths,
        roadmap_task_ids,
    )
    registered = {entry["path"] for entry in entries}
    ledger_tests = sorted(
        {test_path for entry in entries for test_path in entry["tests"]}
    )

    return (
        {
            "baseline_ref": ledger.get("baseline_ref"),
            "baseline_commit": baseline_commit,
            "baseline_source": "ledger",
            "downstream_ref": downstream_ref,
            "downstream_commit": downstream_commit,
            "changed_file_count": len(all_paths),
            "changes": [change.as_dict() for change in changes],
            "upstream_owned_paths": upstream_owned_paths,
            "upstream_owned_path_count": len(upstream_owned_paths),
            "core_paths": sorted(core_paths),
            "registered_core_paths": sorted(core_paths & registered),
            "unregistered_core_paths": sorted(core_paths - registered),
            "ledger_tests": ledger_tests,
            "new_paths_outside_recommended_surfaces": outside_recommended,
            "recommended_downstream_prefixes": list(DOWNSTREAM_OWNED_PREFIXES),
            "ledger": str(ledger_path.relative_to(repo)),
        },
        errors,
    )


def error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def run_ledger_tests(
    repo: Path, test_paths: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Run each registered focused test with stdlib unittest and reject zero tests."""

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for test_path in test_paths:
        module = ledger_test_module(test_path)
        if module is None:
            errors.append(
                error(
                    "core_patch_test_failed",
                    f"cannot execute invalid focused test path {test_path!r}",
                )
            )
            continue
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", module, "-v"],
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        collected_match = re.search(r"\bRan (\d+) tests?\b", combined)
        skipped_match = re.search(r"\bskipped=(\d+)\b", combined)
        collected = int(collected_match.group(1)) if collected_match else 0
        skipped = int(skipped_match.group(1)) if skipped_match else 0
        executed = max(0, collected - skipped)
        passed = completed.returncode == 0 and executed > 0
        results.append(
            {
                "path": test_path,
                "module": module,
                "returncode": completed.returncode,
                "collected": collected,
                "skipped": skipped,
                "executed": executed,
                "passed": passed,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
        if not passed:
            if collected == 0:
                detail = "no unittest tests were collected"
            elif executed == 0:
                detail = "all collected unittest tests were skipped"
            else:
                detail = "test command failed"
            errors.append(
                error(
                    "core_patch_test_failed",
                    f"focused test {test_path!r} failed: {detail}",
                )
            )
    return results, errors


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--fork-ref", default=DEFAULT_FORK_REF)
    parser.add_argument("--active-ref", default=DEFAULT_ACTIVE_REF)
    parser.add_argument("--upstream-ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument(
        "--baseline-ref",
        help="optional ref that must resolve to the authoritative ledger baseline_commit",
    )
    parser.add_argument("--downstream-ref", default="HEAD")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--roadmap", default=DEFAULT_ROADMAP)
    parser.add_argument("--expected-origin", default=DEFAULT_ORIGIN_REPOSITORY)
    parser.add_argument("--expected-upstream", default=DEFAULT_UPSTREAM_REPOSITORY)
    parser.add_argument(
        "--max-behind",
        type=positive_int,
        default=positive_int(os.environ.get("ASR_UPSTREAM_MAX_BEHIND", "10")),
    )
    parser.add_argument(
        "--max-ahead",
        type=positive_int,
        default=positive_int(os.environ.get("ASR_UPSTREAM_MAX_AHEAD", "0")),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--run-ledger-tests",
        action="store_true",
        help="execute every focused unittest registered by a core patch entry",
    )
    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument("--fetch", dest="fetch", action="store_true")
    fetch_group.add_argument("--no-fetch", dest="fetch", action="store_false")
    parser.set_defaults(fetch=True)
    return parser


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repo = args.repo.resolve()
    errors: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "repository": str(repo),
        "fetch": {"attempted": bool(args.fetch), "succeeded": False},
    }

    try:
        remotes, remote_errors = inspect_remotes(
            repo,
            args.expected_origin,
            args.expected_upstream,
        )
        summary["remotes"] = remotes
        errors.extend(remote_errors)

        fetch_urls_are_safe = not any(
            item["code"] in {"origin_remote_mismatch", "upstream_remote_mismatch"}
            for item in remote_errors
        )
        if args.fetch and fetch_urls_are_safe:
            try:
                fetch_tracking_refs(repo)
                summary["fetch"]["succeeded"] = True
            except GuardError as exc:
                errors.append(error("fetch_failed", str(exc)))
        elif not args.fetch:
            summary["fetch"]["succeeded"] = None
        else:
            errors.append(
                error(
                    "fetch_skipped_unsafe_remote",
                    "fetch skipped because a remote URL is invalid",
                )
            )

        try:
            fork_commit = resolve_commit(repo, args.fork_ref)
            active_commit = resolve_commit(repo, args.active_ref)
            upstream_commit = resolve_commit(repo, args.upstream_ref)
            mirror_ahead, mirror_behind = ahead_behind(
                repo, args.fork_ref, args.upstream_ref
            )
            active_ahead, active_behind = ahead_behind(
                repo, args.active_ref, args.upstream_ref
            )
            summary["drift"] = {
                "upstream": {
                    "ref": args.upstream_ref,
                    "commit": upstream_commit,
                },
                "mirror_main": {
                    "ref": args.fork_ref,
                    "commit": fork_commit,
                    "ahead": mirror_ahead,
                    "behind": mirror_behind,
                },
                "active_downstream": {
                    "ref": args.active_ref,
                    "commit": active_commit,
                    "ahead": active_ahead,
                    "behind": active_behind,
                },
                "limits": {
                    "max_mirror_ahead": args.max_ahead,
                    "max_behind": args.max_behind,
                },
            }
            if mirror_ahead > args.max_ahead:
                errors.append(
                    error(
                        "fork_main_ahead",
                        f"fork main is ahead by {mirror_ahead}; maximum is {args.max_ahead}",
                    )
                )
            if mirror_behind > args.max_behind:
                errors.append(
                    error(
                        "fork_main_too_far_behind",
                        f"fork main is behind by {mirror_behind}; maximum is {args.max_behind}",
                    )
                )
            if active_behind > args.max_behind:
                errors.append(
                    error(
                        "active_downstream_too_far_behind",
                        f"active downstream is behind by {active_behind}; "
                        f"maximum is {args.max_behind}",
                    )
                )
        except GuardError as exc:
            errors.append(error("drift_measurement_failed", str(exc)))

        try:
            ledger_path = Path(args.ledger)
            if not ledger_path.is_absolute():
                ledger_path = repo / ledger_path
            roadmap_path = Path(args.roadmap)
            if not roadmap_path.is_absolute():
                roadmap_path = repo / roadmap_path
            isolation, isolation_errors = source_isolation_summary(
                repo,
                args.baseline_ref,
                args.downstream_ref,
                ledger_path,
                roadmap_path,
                args.upstream_ref,
            )
            summary["source_isolation"] = isolation
            errors.extend(isolation_errors)
            summary["ledger_test_execution"] = {
                "requested": bool(args.run_ledger_tests),
                "results": [],
            }
            if args.run_ledger_tests and not isolation_errors:
                test_results, test_errors = run_ledger_tests(
                    repo, isolation["ledger_tests"]
                )
                summary["ledger_test_execution"]["results"] = test_results
                errors.extend(test_errors)
            baseline_commit = isolation["baseline_commit"]
            baseline_ahead, baseline_behind = ahead_behind(
                repo, baseline_commit, args.upstream_ref
            )
            if "drift" in summary:
                summary["drift"]["accepted_baseline"] = {
                    "ref": isolation["baseline_ref"],
                    "commit": baseline_commit,
                    "ahead": baseline_ahead,
                    "behind": baseline_behind,
                }
            if baseline_behind > args.max_behind:
                errors.append(
                    error(
                        "accepted_baseline_too_far_behind",
                        f"accepted baseline is behind by {baseline_behind}; "
                        f"maximum is {args.max_behind}",
                    )
                )
        except (GuardError, ValueError) as exc:
            errors.append(error("source_isolation_failed", str(exc)))
    except (GuardError, OSError, ValueError) as exc:
        errors.append(error("guard_execution_failed", str(exc)))

    summary["errors"] = errors
    summary["status"] = "pass" if not errors else "fail"
    return summary, 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, exit_code = execute(args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    if args.json_output:
        output = args.json_output
        if not output.is_absolute():
            output = args.repo.resolve() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
