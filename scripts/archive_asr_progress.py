#!/usr/bin/env python3
"""Move non-current ASR completion records into deterministic monthly archives."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from scripts import check_asr_progress as governance
else:
    import check_asr_progress as governance


class ArchiveError(RuntimeError):
    """Raised when automatic archiving cannot proceed without losing evidence."""


@dataclass(frozen=True)
class ArchivePlan:
    active_month: str
    today: dt.date
    selected: tuple[governance.CompletionRecord, ...]

    @property
    def counts_by_month(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for record in self.selected:
            counts[record.date_text[:7]] += 1
        return dict(sorted(counts.items()))


def _recent_section_bounds(text: str) -> tuple[int, int]:
    heading = re.search(r"^## Recent Completion Records\s*$", text, re.MULTILINE)
    if heading is None:
        raise ArchiveError("progress: missing '## Recent Completion Records' section")
    content_start = heading.end()
    next_section = re.search(r"^##\s+", text[content_start:], re.MULTILINE)
    content_end = (
        content_start + next_section.start()
        if next_section is not None
        else len(text)
    )
    return content_start, content_end


def build_plan(root: Path, *, today: dt.date | None = None) -> ArchivePlan:
    root = root.resolve()
    current_date = today or governance.project_today()
    progress_path = root / governance.PROGRESS_PATH
    try:
        text = progress_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArchiveError(f"cannot read {progress_path}: {exc}") from exc

    blocking_errors = []
    for error in governance.validate_repository(
        root, verify_git=False, today=current_date
    ):
        auto_fixable = (
            "belongs in archive" in error
            or (
                error.startswith("progress:")
                and "completion records exceed the budget" in error
            )
            or (
                error.startswith("progress:")
                and "lines exceed the budget" in error
            )
        )
        if not auto_fixable:
            blocking_errors.append(error)
    if blocking_errors:
        formatted = "\n- ".join(blocking_errors)
        raise ArchiveError(
            "governance errors must be fixed before archiving:\n- " + formatted
        )

    active_month = current_date.strftime("%Y-%m")
    parse_errors: list[str] = []
    records = governance.parse_completion_records(text, "progress", parse_errors)
    if parse_errors:
        raise ArchiveError("\n".join(parse_errors))

    section_start, section_end = _recent_section_bounds(text)
    outside = [
        record
        for record in records
        if record.start < section_start or record.end > section_end
    ]
    if outside:
        raise ArchiveError(
            "progress: every completion record must be inside "
            "'## Recent Completion Records'"
        )

    selected = [record for record in records if record.date_text[:7] != active_month]
    keep_in_month = [record for record in records if record.date_text[:7] == active_month]
    selected.extend(keep_in_month[governance.PROGRESS_MAX_RECORDS :])
    return ArchivePlan(
        active_month=active_month,
        today=current_date,
        selected=tuple(selected),
    )


def _render_progress(text: str, selected: set[tuple[str, str | None, str]]) -> str:
    errors: list[str] = []
    records = governance.parse_completion_records(text, "progress", errors)
    if errors:
        raise ArchiveError("\n".join(errors))
    kept = [
        record
        for record in records
        if (record.date_text, record.task_id, record.title) not in selected
    ]
    section_start, section_end = _recent_section_bounds(text)
    if kept:
        body = "\n\n".join(record.text for record in kept)
    else:
        body = "> No terminal completion records in the current month."
    return (
        text[:section_start].rstrip()
        + "\n\n"
        + body
        + "\n\n"
        + text[section_end:].lstrip("\n")
    )


def _render_archive(
    month: str,
    existing_text: str | None,
    additions: list[governance.CompletionRecord],
) -> str:
    existing_records: list[governance.CompletionRecord] = []
    if existing_text is not None:
        errors: list[str] = []
        existing_records = governance.parse_completion_records(
            existing_text, f"archive {month}.md", errors
        )
        if errors:
            raise ArchiveError("\n".join(errors))

    by_identity: dict[
        tuple[str, str | None, str], governance.CompletionRecord
    ] = {}
    for record in [*existing_records, *additions]:
        identity = (record.date_text, record.task_id, record.title)
        by_identity.setdefault(identity, record)
    records = sorted(
        by_identity.values(),
        key=lambda record: (record.date_text, record.task_id or "", record.title),
        reverse=True,
    )
    body = "\n\n".join(record.text for record in records)
    return (
        f"# ASR Progress Archive — {month}\n\n"
        "> Terminal completion records moved from `.notes/asr/progress.md`. "
        "This file contains no live execution pointer.\n\n"
        f"{body}\n"
    )


def apply_plan(root: Path, plan: ArchivePlan) -> list[Path]:
    root = root.resolve()
    if not plan.selected:
        return []

    progress_path = root / governance.PROGRESS_PATH
    archive_dir = root / governance.ARCHIVE_PATH
    progress_text = progress_path.read_text(encoding="utf-8")
    selected_identities = {
        (record.date_text, record.task_id, record.title) for record in plan.selected
    }
    writes: dict[Path, str] = {
        progress_path: _render_progress(progress_text, selected_identities)
    }
    grouped: dict[str, list[governance.CompletionRecord]] = defaultdict(list)
    for record in plan.selected:
        grouped[record.date_text[:7]].append(record)
    for month, additions in grouped.items():
        archive_path = archive_dir / f"{month}.md"
        existing_text = (
            archive_path.read_text(encoding="utf-8") if archive_path.exists() else None
        )
        writes[archive_path] = _render_archive(month, existing_text, additions)

    originals: dict[Path, str | None] = {
        path: path.read_text(encoding="utf-8") if path.exists() else None
        for path in writes
    }
    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path, content in writes.items():
            path.write_text(content, encoding="utf-8")
        errors = governance.validate_repository(
            root, verify_git=False, today=plan.today
        )
        if errors:
            raise ArchiveError(
                "archive result failed governance validation:\n- "
                + "\n- ".join(errors)
            )
    except Exception:
        for path, original in originals.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(original, encoding="utf-8")
        raise
    return sorted(writes)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check or apply monthly archiving for ASR progress completion records."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="report records requiring archiving without changing files (default)",
    )
    mode.add_argument(
        "--apply", action="store_true", help="move records and validate the result"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = build_plan(args.root)
        if not plan.selected:
            print(
                "ASR progress archive OK: "
                f"active_month={plan.active_month} nothing_to_move"
            )
            return 0
        summary = ", ".join(
            f"{month}={count}" for month, count in plan.counts_by_month.items()
        )
        if not args.apply:
            print(
                "ASR progress archive required: "
                f"{len(plan.selected)} record(s) ({summary}); "
                "run scripts/archive_asr_progress.py --apply",
                file=sys.stderr,
            )
            return 1
        changed = apply_plan(args.root, plan)
        print(
            "ASR progress archive applied: "
            f"{len(plan.selected)} record(s) ({summary}); "
            f"updated {len(changed)} file(s)"
        )
        return 0
    except (ArchiveError, OSError) as exc:
        print(f"ASR progress archive failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
