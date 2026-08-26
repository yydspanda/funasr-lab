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
from zoneinfo import ZoneInfo


ROADMAP_PATH = Path(".notes/asr/delivery-roadmap.md")
PROGRESS_PATH = Path(".notes/asr/progress.md")
ARCHIVE_PATH = Path(".notes/archive/asr/progress")
ROADMAP_MAX_LINES = 240
PROGRESS_MAX_LINES = 120
PROGRESS_MAX_RECORDS = 8
ARCHIVE_MAX_LINES = 600
PROJECT_TIMEZONE = ZoneInfo("Asia/Shanghai")
EXPECTED_UPSTREAM = "modelscope/FunASR"
REQUIRED_TASK_IDS = {
    "BOOT-01",
    "BASE-01",
    "EVAL-01",
    "TRAIN-01",
    "EXP-01",
    "STREAM-01",
    "SERVE-01",
    "UP-SYNC",
}
REQUIRED_STAGE_IDS = {
    "BOOT",
    "BASE",
    "EVAL",
    "TRAIN",
    "EXP",
    "STREAM",
    "SERVE",
    "MAINT",
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
ALLOWED_STAGE_STATUSES = {"Done", "Current", "Pending", "Recurring"}
TERMINAL_STATUSES = {"Done", "Cancelled", "Superseded"}
TASK_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*-(?:\d{2}|SYNC)$")
TASK_LIKE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
TASK_ID_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9-])"
    r"([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d{2}|[A-Z][A-Z0-9]*-SYNC)"
    r"(?![A-Z0-9-])"
)
STAGE_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BASELINE_REF_RE = re.compile(
    r"^(?:v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?|[0-9a-f]{40})$"
)
RECORD_HEADER_RE = re.compile(
    r"^###\s+(\d{4}-\d{2}-\d{2})\s+[—-]\s+(.+?)\s*$", re.MULTILINE
)


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    status: str
    line_number: int


@dataclass(frozen=True)
class CompletionRecord:
    date_text: str
    title: str
    task_id: str | None
    status: str | None
    start: int
    end: int
    text: str


def project_today(now: dt.datetime | None = None) -> dt.date:
    """Return the project calendar date independent of the runner timezone."""

    instant = now or dt.datetime.now(dt.timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("project_today requires an aware datetime")
    return instant.astimezone(PROJECT_TIMEZONE).date()


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
    stages: set[str] = set()
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
        elif TASK_LIKE_RE.fullmatch(identifier):
            errors.append(
                f"roadmap: malformed task ID {identifier!r} at line {line_number}; "
                "use STAGE-NN or UP-SYNC"
            )
        elif (
            STAGE_ID_RE.fullmatch(identifier)
            and status in ALLOWED_STAGE_STATUSES
        ):
            stages.add(identifier)
            if status == "Current":
                current_stages.append(identifier)

    if not tasks:
        errors.append("roadmap: no task rows found")
    missing = sorted(REQUIRED_TASK_IDS.difference(tasks))
    if missing:
        errors.append(f"roadmap: missing required task IDs: {', '.join(missing)}")
    missing_stages = sorted(REQUIRED_STAGE_IDS.difference(stages))
    if missing_stages:
        errors.append(
            "roadmap: missing required stage IDs: " + ", ".join(missing_stages)
        )
    return tasks, current_stages


def parse_completion_records(
    text: str, document: str, errors: list[str]
) -> list[CompletionRecord]:
    """Parse compact terminal records without depending on Markdown libraries."""

    matches = list(RECORD_HEADER_RE.finditer(text))
    records: list[CompletionRecord] = []
    for match in matches:
        next_heading = re.search(r"^##(?:#)?\s+", text[match.end() :], re.MULTILINE)
        end = (
            match.end() + next_heading.start()
            if next_heading is not None
            else len(text)
        )
        body = text[match.end() : end]
        record_label = f"{document} record {match.group(1)}"
        task_id = _single_backtick_field(body, "Task", record_label, errors)
        status = _single_backtick_field(body, "Status", record_label, errors)
        _single_text_field(body, "Outcome", record_label, errors)
        _single_text_field(body, "Verification", record_label, errors)
        records.append(
            CompletionRecord(
                date_text=match.group(1),
                title=match.group(2),
                task_id=task_id,
                status=status,
                start=match.start(),
                end=end,
                text=text[match.start() : end].strip(),
            )
        )
    return records


def _validate_progress_layout(
    text: str, records: list[CompletionRecord], errors: list[str]
) -> None:
    headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    expected = (
        "Current Pointer",
        "Current Constraints",
        "Recent Completion Records",
        "Update Contract",
    )
    names = [match.group(1) for match in headings]
    if not names or names[0] != "Current Pointer":
        errors.append("progress: Current Pointer must be the first level-two section")
    positions: list[int] = []
    for name in expected:
        found = [match for match in headings if match.group(1) == name]
        if len(found) != 1:
            errors.append(
                f"progress: expected exactly one '## {name}' section, found {len(found)}"
            )
        elif found:
            positions.append(found[0].start())
    if len(positions) == len(expected) and positions != sorted(positions):
        errors.append("progress: required sections are out of order")

    recent = [
        match for match in headings if match.group(1) == "Recent Completion Records"
    ]
    if len(recent) == 1:
        section_start = recent[0].end()
        following = [match.start() for match in headings if match.start() > section_start]
        section_end = min(following) if following else len(text)
        if any(
            record.start < section_start or record.end > section_end
            for record in records
        ):
            errors.append(
                "progress: every completion record must be inside "
                "'## Recent Completion Records'"
            )
        _validate_record_container(
            text,
            records,
            "progress",
            errors,
            start=section_start,
            end=section_end,
        )


def _validate_record_container(
    text: str,
    records: list[CompletionRecord],
    document: str,
    errors: list[str],
    *,
    start: int = 0,
    end: int | None = None,
) -> None:
    end = len(text) if end is None else end
    container = text[start:end]
    heading_count = len(re.findall(r"^###\s+", container, re.MULTILINE))
    contained_records = [
        record for record in records if record.start >= start and record.end <= end
    ]
    if heading_count != len(contained_records):
        errors.append(
            f"{document}: every level-three heading must be a valid dated "
            "completion record"
        )
    task_field_count = len(
        re.findall(r"^- \*\*Task:\*\*", container, re.MULTILINE)
    )
    if task_field_count != len(contained_records):
        errors.append(
            f"{document}: every Task field must belong to exactly one completion record"
        )


def _validate_task_id_mentions(
    text: str, document: str, tasks: dict[str, TaskRow], errors: list[str]
) -> None:
    # Free prose can legitimately contain tokens such as SHA-256 or UTF-8.
    # Numbered work IDs use the repository's STAGE-NN convention; named IDs
    # such as UP-SYNC are validated whenever they occupy a structured Task field.
    mentioned = set(TASK_ID_TOKEN_RE.findall(text))
    unknown = sorted(mentioned.difference(tasks))
    if unknown:
        errors.append(
            f"{document}: task IDs are not registered in Roadmap: {', '.join(unknown)}"
        )


def _validate_completion_records(
    records: list[CompletionRecord],
    document: str,
    tasks: dict[str, TaskRow],
    last_updated: dt.date | None,
    errors: list[str],
    *,
    expected_month: str | None = None,
    max_records: int | None = None,
) -> None:
    if max_records is not None and len(records) > max_records:
        errors.append(
            f"{document}: {len(records)} completion records exceed the budget of "
            f"{max_records}; run scripts/archive_asr_progress.py --apply"
        )

    previous_date: dt.date | None = None
    for record in records:
        record_date = _parse_date(
            record.date_text, "completion record date", document, errors
        )
        task_id = record.task_id
        status = record.status

        if task_id is not None and task_id not in tasks:
            errors.append(f"{document}: completion record uses unknown task {task_id!r}")
        if status is not None and status not in TERMINAL_STATUSES:
            errors.append(
                f"{document}: completion record status {status!r} is not terminal"
            )
        if (
            task_id is not None
            and task_id in tasks
            and task_id != "UP-SYNC"
            and status in TERMINAL_STATUSES
            and tasks[task_id].status != status
        ):
            errors.append(
                f"{document}: completion record status does not match Roadmap task "
                f"{task_id}: {status!r} != {tasks[task_id].status!r}"
            )
        if record_date is not None:
            if last_updated is not None and record_date > last_updated:
                errors.append(
                    f"{document}: completion record date is later than Last Updated: "
                    f"{record_date.isoformat()} > {last_updated.isoformat()}"
                )
            if expected_month is not None and record.date_text[:7] != expected_month:
                errors.append(
                    f"{document}: record {record.date_text} belongs in archive "
                    f"{record.date_text[:7]}.md; run "
                    "scripts/archive_asr_progress.py --apply"
                )
            if previous_date is not None and record_date > previous_date:
                errors.append(f"{document}: completion records must be newest first")
            previous_date = record_date


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
    baseline_ref: str | None,
    commit: str | None,
    baseline_date: dt.date | None,
    errors: list[str],
) -> None:
    if not (root / ".git").exists() or baseline_ref is None or commit is None:
        return

    status, resolved = _run_git(root, "rev-parse", f"{baseline_ref}^{{commit}}")
    if status != 0:
        errors.append(
            f"git: baseline ref {baseline_ref!r} cannot be resolved: {resolved}"
        )
        return
    if resolved != commit:
        errors.append(
            f"git: baseline ref {baseline_ref} resolves to {resolved}, "
            f"not recorded {commit}"
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


def _validate_archives(
    root: Path,
    tasks: dict[str, TaskRow],
    last_updated: dt.date | None,
    errors: list[str],
) -> list[tuple[str, CompletionRecord]]:
    archive_dir = root / ARCHIVE_PATH
    if not archive_dir.exists():
        return []
    archived_records: list[tuple[str, CompletionRecord]] = []
    for path in sorted(archive_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        filename_match = re.fullmatch(r"(\d{4}-\d{2})\.md", path.name)
        if filename_match is None:
            errors.append(f"archive: unexpected progress filename {path.name!r}")
        text = _read_text(path, errors)
        if "**Current Stage:**" in text or "**In Progress Task:**" in text:
            errors.append(f"archive: {path} must not contain an active pointer")
        line_count = _line_count(text)
        if line_count > ARCHIVE_MAX_LINES:
            errors.append(
                f"archive: {path.name} has {line_count} lines, exceeding the "
                f"monthly budget of {ARCHIVE_MAX_LINES}"
            )

        document = f"archive {path.name}"
        _validate_task_id_mentions(text, document, tasks, errors)
        records = parse_completion_records(text, document, errors)
        _validate_record_container(text, records, document, errors)
        if not records:
            errors.append(f"{document}: must contain at least one completion record")
        expected_month = filename_match.group(1) if filename_match is not None else None
        if expected_month is not None:
            expected_title = f"# ASR Progress Archive — {expected_month}"
            if not text.startswith(expected_title + "\n"):
                errors.append(
                    f"{document}: first heading must be {expected_title!r}"
                )
        _validate_completion_records(
            records,
            document,
            tasks,
            last_updated,
            errors,
            expected_month=expected_month,
        )
        archived_records.extend((document, record) for record in records)
    return archived_records


def _validate_unique_records(
    active_records: list[CompletionRecord],
    archived_records: list[tuple[str, CompletionRecord]],
    errors: list[str],
) -> None:
    seen: dict[tuple[str, str | None, str], str] = {}
    located = [("progress", record) for record in active_records]
    located.extend(archived_records)
    for document, record in located:
        identity = (record.date_text, record.task_id, record.title)
        if identity in seen:
            errors.append(
                f"history: duplicate completion record {record.date_text} "
                f"{record.task_id or '<missing task>'} {record.title!r} in "
                f"{seen[identity]} and {document}"
            )
        else:
            seen[identity] = document


def validate_repository(
    root: Path,
    *,
    verify_git: bool = True,
    today: dt.date | None = None,
) -> list[str]:
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
        "Baseline Ref",
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
    baseline_ref = roadmap_values["Baseline Ref"]
    commit = roadmap_values["Baseline Commit"]
    if upstream is not None and upstream != EXPECTED_UPSTREAM:
        errors.append(
            f"baseline: upstream must be {EXPECTED_UPSTREAM!r}, got {upstream!r}"
        )
    if baseline_ref is not None and not BASELINE_REF_RE.fullmatch(baseline_ref):
        errors.append(
            "baseline: Baseline Ref must be an immutable semantic release tag "
            f"or full lowercase commit SHA, got {baseline_ref!r}"
        )
    if commit is not None and not SHA_RE.fullmatch(commit):
        errors.append(f"baseline: commit must be a full lowercase 40-character SHA")

    baseline_date = _parse_date(
        roadmap_values["Baseline Date"], "Baseline Date", "roadmap", errors
    )
    last_updated = _parse_date(
        roadmap_values["Last Updated"], "Last Updated", "roadmap", errors
    )
    current_date = today or project_today()
    if baseline_date is not None and last_updated is not None:
        if baseline_date > last_updated:
            errors.append("baseline: Baseline Date cannot be later than Last Updated")
    if last_updated is not None and last_updated > current_date:
        errors.append(
            f"roadmap: Last Updated {last_updated.isoformat()} is in the future"
        )

    _validate_task_id_mentions(roadmap_text, "roadmap", tasks, errors)
    _validate_task_id_mentions(progress_text, "progress", tasks, errors)
    active_records = parse_completion_records(progress_text, "progress", errors)
    _validate_progress_layout(progress_text, active_records, errors)
    active_month = current_date.strftime("%Y-%m")
    _validate_completion_records(
        active_records,
        "progress",
        tasks,
        last_updated,
        errors,
        expected_month=active_month,
        max_records=PROGRESS_MAX_RECORDS,
    )
    archived_records = _validate_archives(root, tasks, last_updated, errors)
    _validate_unique_records(active_records, archived_records, errors)
    if verify_git:
        _validate_git_baseline(root, baseline_ref, commit, baseline_date, errors)
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
