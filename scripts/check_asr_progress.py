#!/usr/bin/env python3
"""Validate the active ASR Roadmap/Progress governance contract.

The checker intentionally uses only the Python standard library so it can run
before project dependencies or model assets are installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROADMAP_PATH = Path(".notes/asr/delivery-roadmap.md")
PROGRESS_PATH = Path(".notes/asr/progress.md")
ARCHIVE_PATH = Path(".notes/archive/asr/progress")
ROADMAP_MAX_LINES = 240
PROGRESS_MAX_LINES = 120
EXPECTED_UPSTREAM = "modelscope/FunASR"
REQUIRED_TASK_IDS = {
    "BOOT-01",
    "BASE-01",
    "EVAL-01",
    "TRAIN-01",
    "EXP-01",
    "STREAM-01",
    "UP-SYNC",
}
ALLOWED_TASK_STATUSES = {
    "In Progress",
    "Pending",
    "Scheduled",
    "Done",
    "Cancelled",
    "Superseded",
    "Blocked",
}
TERMINAL_STATUSES = {"Done", "Cancelled", "Superseded"}
TASK_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
STAGE_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    status: str
    line_number: int


def _read_text(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing required file: {path}")
    except OSError as exc:
        errors.append(f"cannot read {path}: {exc}")
    return ""


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _plain_markdown(value: str) -> str:
    value = value.strip()
    value = value.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def _single_backtick_field(
    text: str,
    label: str,
    document: str,
    errors: list[str],
) -> str | None:
    line_re = re.compile(
        rf"^-\s+\*\*{re.escape(label)}:\*\*.*$",
        flags=re.MULTILINE,
    )
    lines = line_re.findall(text)
    if len(lines) != 1:
        errors.append(
            f"{document}: expected exactly one '{label}' field, found {len(lines)}"
        )
        return None

    value_re = re.compile(
        rf"^-\s+\*\*{re.escape(label)}:\*\*\s+`([^`\n]+)`\s*$"
    )
    match = value_re.fullmatch(lines[0])
    if not match:
        errors.append(f"{document}: '{label}' must contain one backtick value")
        return None
    return match.group(1)


def _single_text_field(
    text: str,
    label: str,
    document: str,
    errors: list[str],
) -> str | None:
    line_re = re.compile(
        rf"^-\s+\*\*{re.escape(label)}:\*\*\s+(.+?)\s*$",
        flags=re.MULTILINE,
    )
    values = line_re.findall(text)
    if len(values) != 1:
        errors.append(
            f"{document}: expected exactly one '{label}' field, found {len(values)}"
        )
        return None
    return values[0].strip()


def _parse_date(
    value: str | None,
    label: str,
    document: str,
    errors: list[str],
) -> dt.date | None:
    if value is None:
        return None
    if not DATE_RE.fullmatch(value):
        errors.append(f"{document}: '{label}' must use YYYY-MM-DD, got {value!r}")
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        errors.append(f"{document}: '{label}' is not a valid date: {value!r}")
        return None


def _table_rows(text: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        identifier = _plain_markdown(cells[0])
        status = _plain_markdown(cells[1])
        rows.append((identifier, status, line_number))
    return rows


def _parse_roadmap_rows(
    text: str, errors: list[str]
) -> tuple[dict[str, TaskRow], list[str]]:
    tasks: dict[str, TaskRow] = {}
    current_stages: list[str] = []

    for identifier, status, line_number in _table_rows(text):
        if TASK_ID_RE.fullmatch(identifier):
            if identifier in tasks:
                errors.append(
                    f"roadmap: duplicate task ID {identifier!r} at line {line_number}"
                )
                continue
            if status not in ALLOWED_TASK_STATUSES:
                errors.append(
                    f"roadmap: task {identifier} has unsupported status {status!r}"
                )
            tasks[identifier] = TaskRow(identifier, status, line_number)
        elif STAGE_ID_RE.fullmatch(identifier) and status == "Current":
            current_stages.append(identifier)

    if not tasks:
        errors.append("roadmap: no task rows found")
    missing = sorted(REQUIRED_TASK_IDS.difference(tasks))
    if missing:
        errors.append(f"roadmap: missing required task IDs: {', '.join(missing)}")
    return tasks, current_stages


def _validate_completion_records(
    text: str,
    tasks: dict[str, TaskRow],
    last_updated: dt.date | None,
    errors: list[str],
) -> None:
    header_re = re.compile(
        r"^###\s+(\d{4}-\d{2}-\d{2})\s+[—-]\s+.+$", re.MULTILINE
    )
    matches = list(header_re.finditer(text))
    if len(matches) > 10:
        errors.append(
            f"progress: {len(matches)} recent completion records exceed the budget of 10"
        )

    for index, match in enumerate(matches):
        record_date = _parse_date(
            match.group(1), "completion record date", "progress", errors
        )
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        task_id = _single_backtick_field(body, "Task", "completion record", errors)
        status = _single_backtick_field(body, "Status", "completion record", errors)
        _single_text_field(body, "Outcome", "completion record", errors)
        _single_text_field(body, "Verification", "completion record", errors)

        if task_id is not None and task_id not in tasks:
            errors.append(f"progress: completion record uses unknown task {task_id!r}")
        if status is not None and status not in TERMINAL_STATUSES:
            errors.append(
                f"progress: completion record status {status!r} is not terminal"
            )
        if (
            task_id is not None
            and task_id in tasks
            and task_id != "UP-SYNC"
            and status in TERMINAL_STATUSES
            and tasks[task_id].status != status
        ):
            errors.append(
                "progress: completion record status does not match Roadmap task "
                f"{task_id}: {status!r} != {tasks[task_id].status!r}"
            )
        if record_date is not None and last_updated is not None:
            if record_date > last_updated:
                errors.append(
                    "progress: completion record date is later than Last Updated: "
                    f"{record_date.isoformat()} > {last_updated.isoformat()}"
                )


def _run_git(root: Path, *args: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    return completed.returncode, output


def _validate_git_baseline(
    root: Path,
    tag: str | None,
    commit: str | None,
    baseline_date: dt.date | None,
    errors: list[str],
) -> None:
    if not (root / ".git").exists() or tag is None or commit is None:
        return

    status, resolved = _run_git(root, "rev-parse", f"{tag}^{{commit}}")
    if status != 0:
        errors.append(f"git: baseline tag {tag!r} cannot be resolved: {resolved}")
        return
    if resolved != commit:
        errors.append(
            f"git: baseline tag {tag} resolves to {resolved}, not recorded {commit}"
        )

    status, commit_date_text = _run_git(root, "show", "-s", "--format=%cs", commit)
    if status != 0:
        errors.append(f"git: baseline commit {commit} cannot be inspected: {commit_date_text}")
        return
    try:
        commit_date = dt.date.fromisoformat(commit_date_text)
    except ValueError:
        errors.append(f"git: invalid commit date returned for {commit}: {commit_date_text!r}")
        return
    if baseline_date is not None and commit_date != baseline_date:
        errors.append(
            "git: recorded Baseline Date does not match the baseline commit date: "
            f"{baseline_date.isoformat()} != {commit_date.isoformat()}"
        )


def _validate_archives(root: Path, errors: list[str]) -> None:
    archive_dir = root / ARCHIVE_PATH
    if not archive_dir.exists():
        return
    for path in sorted(archive_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        if not re.fullmatch(r"\d{4}-\d{2}\.md", path.name):
            errors.append(f"archive: unexpected progress filename {path.name!r}")
        text = _read_text(path, errors)
        if "**Current Stage:**" in text or "**In Progress Task:**" in text:
            errors.append(f"archive: {path} must not contain an active pointer")


def validate_repository(root: Path, *, verify_git: bool = True) -> list[str]:
    """Return all governance violations found below *root*."""

    root = root.resolve()
    errors: list[str] = []
    roadmap_text = _read_text(root / ROADMAP_PATH, errors)
    progress_text = _read_text(root / PROGRESS_PATH, errors)
    if not roadmap_text or not progress_text:
        return errors

    roadmap_lines = _line_count(roadmap_text)
    progress_lines = _line_count(progress_text)
    if roadmap_lines > ROADMAP_MAX_LINES:
        errors.append(
            f"roadmap: {roadmap_lines} lines exceed the budget of {ROADMAP_MAX_LINES}"
        )
    if progress_lines > PROGRESS_MAX_LINES:
        errors.append(
            f"progress: {progress_lines} lines exceed the budget of {PROGRESS_MAX_LINES}"
        )

    tasks, roadmap_current_stages = _parse_roadmap_rows(roadmap_text, errors)
    roadmap_stage = _single_backtick_field(
        roadmap_text, "Current Stage", "roadmap", errors
    )
    progress_stage = _single_backtick_field(
        progress_text, "Current Stage", "progress", errors
    )
    progress_task = _single_backtick_field(
        progress_text, "In Progress Task", "progress", errors
    )

    if len(roadmap_current_stages) != 1:
        errors.append(
            "roadmap: expected exactly one stage row with status Current, found "
            f"{len(roadmap_current_stages)}"
        )
    elif roadmap_stage is not None and roadmap_current_stages[0] != roadmap_stage:
        errors.append(
            "roadmap: Current Stage field does not match the Current stage row: "
            f"{roadmap_stage!r} != {roadmap_current_stages[0]!r}"
        )

    in_progress = [row.task_id for row in tasks.values() if row.status == "In Progress"]
    if len(in_progress) != 1:
        errors.append(
            "roadmap: expected exactly one In Progress task, found "
            f"{len(in_progress)}"
        )
    elif progress_task is not None and in_progress[0] != progress_task:
        errors.append(
            "progress: In Progress Task does not match Roadmap: "
            f"{progress_task!r} != {in_progress[0]!r}"
        )

    if roadmap_stage is not None and progress_stage is not None:
        if roadmap_stage != progress_stage:
            errors.append(
                "progress: Current Stage does not match Roadmap: "
                f"{progress_stage!r} != {roadmap_stage!r}"
            )
    if progress_task is not None:
        if progress_task not in tasks:
            errors.append(f"progress: In Progress Task {progress_task!r} is unknown")
        elif progress_stage is not None:
            expected_prefix = "UP" if progress_task == "UP-SYNC" else progress_stage
            if not progress_task.startswith(f"{expected_prefix}-"):
                errors.append(
                    f"progress: task {progress_task} does not belong to stage {progress_stage}"
                )

    _single_text_field(progress_text, "Current Objective", "progress", errors)
    _single_text_field(progress_text, "Next Gate", "progress", errors)
    if "[`delivery-roadmap.md`](delivery-roadmap.md)" not in progress_text:
        errors.append("progress: Roadmap must link to delivery-roadmap.md")

    shared_labels = (
        "Upstream Repository",
        "Baseline Tag",
        "Baseline Commit",
        "Baseline Date",
        "Last Updated",
    )
    roadmap_values: dict[str, str | None] = {}
    progress_values: dict[str, str | None] = {}
    for label in shared_labels:
        roadmap_values[label] = _single_backtick_field(
            roadmap_text, label, "roadmap", errors
        )
        progress_values[label] = _single_backtick_field(
            progress_text, label, "progress", errors
        )
        if (
            roadmap_values[label] is not None
            and progress_values[label] is not None
            and roadmap_values[label] != progress_values[label]
        ):
            errors.append(
                f"baseline: {label} differs between Roadmap and Progress: "
                f"{roadmap_values[label]!r} != {progress_values[label]!r}"
            )

    upstream = roadmap_values["Upstream Repository"]
    tag = roadmap_values["Baseline Tag"]
    commit = roadmap_values["Baseline Commit"]
    if upstream is not None and upstream != EXPECTED_UPSTREAM:
        errors.append(
            f"baseline: upstream must be {EXPECTED_UPSTREAM!r}, got {upstream!r}"
        )
    if tag is not None and not TAG_RE.fullmatch(tag):
        errors.append(f"baseline: invalid semantic version tag {tag!r}")
    if commit is not None and not SHA_RE.fullmatch(commit):
        errors.append(f"baseline: commit must be a full lowercase 40-character SHA")

    baseline_date = _parse_date(
        roadmap_values["Baseline Date"], "Baseline Date", "roadmap", errors
    )
    last_updated = _parse_date(
        roadmap_values["Last Updated"], "Last Updated", "roadmap", errors
    )
    today = dt.date.today()
    if baseline_date is not None and last_updated is not None:
        if baseline_date > last_updated:
            errors.append("baseline: Baseline Date cannot be later than Last Updated")
    if last_updated is not None and last_updated > today:
        errors.append(
            f"roadmap: Last Updated {last_updated.isoformat()} is in the future"
        )

    _validate_completion_records(progress_text, tasks, last_updated, errors)
    _validate_archives(root, errors)
    if verify_git:
        _validate_git_baseline(root, tag, commit, baseline_date, errors)
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the ASR Roadmap/Progress governance contract."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="skip resolving the recorded baseline against local Git objects",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate_repository(args.root, verify_git=not args.no_git)
    if errors:
        print("ASR governance check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    roadmap_text = (args.root / ROADMAP_PATH).read_text(encoding="utf-8")
    progress_text = (args.root / PROGRESS_PATH).read_text(encoding="utf-8")
    stage_match = re.search(r"^- \*\*Current Stage:\*\* `([^`]+)`$", progress_text, re.M)
    task_match = re.search(
        r"^- \*\*In Progress Task:\*\* `([^`]+)`$", progress_text, re.M
    )
    commit_match = re.search(
        r"^- \*\*Baseline Commit:\*\* `([0-9a-f]{40})`$", progress_text, re.M
    )
    stage = stage_match.group(1) if stage_match else "unknown"
    task = task_match.group(1) if task_match else "unknown"
    commit = commit_match.group(1)[:12] if commit_match else "unknown"
    print(
        "ASR governance OK: "
        f"stage={stage} task={task} baseline={commit} "
        f"roadmap={_line_count(roadmap_text)}/{ROADMAP_MAX_LINES} "
        f"progress={_line_count(progress_text)}/{PROGRESS_MAX_LINES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
