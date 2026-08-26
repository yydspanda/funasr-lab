from __future__ import annotations

import datetime as dt
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import archive_asr_progress as archiver
from scripts import check_asr_progress as governance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TODAY = dt.date(2026, 8, 26)


class AsrProgressGovernanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

        notes = self.root / ".notes/asr"
        notes.mkdir(parents=True)
        for relative_path in (
            governance.ROADMAP_PATH,
            governance.PROGRESS_PATH,
        ):
            destination = self.root / relative_path
            shutil.copy2(REPOSITORY_ROOT / relative_path, destination)

        archive = self.root / governance.ARCHIVE_PATH
        archive.mkdir(parents=True)
        shutil.copy2(
            REPOSITORY_ROOT / governance.ARCHIVE_PATH / "README.md",
            archive / "README.md",
        )

    def _read(self, relative_path: Path) -> str:
        return (self.root / relative_path).read_text(encoding="utf-8")

    def _write(self, relative_path: Path, text: str) -> None:
        (self.root / relative_path).write_text(text, encoding="utf-8")

    def _errors(self) -> list[str]:
        return governance.validate_repository(
            self.root, verify_git=False, today=FIXTURE_TODAY
        )

    @staticmethod
    def _record(date: str, title: str, task_id: str = "BOOT-01") -> str:
        return f"""\
### {date} — {title}

- **Task:** `{task_id}`
- **Status:** `Done`
- **Outcome:** A concise terminal outcome was recorded.
- **Verification:** The focused governance check passed.

"""

    def _insert_record(self, record: str) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress = progress.replace("## Update Contract", record + "## Update Contract", 1)
        self._write(governance.PROGRESS_PATH, progress)

    def test_repository_documents_are_valid(self) -> None:
        self.assertEqual([], self._errors())

    def test_service_stage_is_required(self) -> None:
        roadmap = self._read(governance.ROADMAP_PATH)
        roadmap = re.sub(
            r"^\| `SERVE` \| Pending \|.*\n",
            "",
            roadmap,
            count=1,
            flags=re.MULTILINE,
        )
        self._write(governance.ROADMAP_PATH, roadmap)

        errors = self._errors()

        self.assertTrue(
            any(
                "missing required stage IDs" in error and "SERVE" in error
                for error in errors
            ),
            errors,
        )

    def test_service_task_is_required(self) -> None:
        roadmap = self._read(governance.ROADMAP_PATH)
        roadmap = re.sub(
            r"^\| `SERVE-01` \| Pending \|.*\n",
            "",
            roadmap,
            count=1,
            flags=re.MULTILINE,
        )
        self._write(governance.ROADMAP_PATH, roadmap)

        errors = self._errors()

        self.assertTrue(
            any(
                "missing required task IDs" in error and "SERVE-01" in error
                for error in errors
            ),
            errors,
        )

    def test_project_calendar_uses_asia_shanghai_at_utc_month_boundary(self) -> None:
        utc_instant = dt.datetime(
            2026, 8, 31, 16, 30, tzinfo=dt.timezone.utc
        )

        self.assertEqual(dt.date(2026, 9, 1), governance.project_today(utc_instant))

    def test_duplicate_progress_pointer_is_rejected(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress += "\n- **Current Stage:** `BOOT`\n"
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(
            any("exactly one 'Current Stage'" in error for error in errors), errors
        )

    def test_second_in_progress_task_is_rejected(self) -> None:
        roadmap = self._read(governance.ROADMAP_PATH)
        pending = re.search(
            r"^\| `([A-Z][A-Z0-9]*-[A-Z0-9-]+)` \| Pending \|",
            roadmap,
            re.MULTILINE,
        )
        self.assertIsNotNone(pending)
        task_id = pending.group(1)
        roadmap = roadmap.replace(
            f"| `{task_id}` | Pending |",
            f"| `{task_id}` | **In Progress** |",
            1,
        )
        self._write(governance.ROADMAP_PATH, roadmap)

        errors = self._errors()

        self.assertTrue(
            any("exactly one In Progress task" in error for error in errors), errors
        )

    def test_unknown_current_task_is_rejected(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        current = re.search(
            r"^- \*\*In Progress Task:\*\* `([^`]+)`$", progress, re.MULTILINE
        )
        self.assertIsNotNone(current)
        current_task = current.group(1)
        unknown_task = f"{current_task.split('-', 1)[0]}-99"
        progress = progress.replace(
            f"- **In Progress Task:** `{current_task}`",
            f"- **In Progress Task:** `{unknown_task}`",
            1,
        )
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(
            any(f"{unknown_task}' is unknown" in error for error in errors), errors
        )

    def test_baseline_mismatch_is_rejected(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress = progress.replace(
            "eedd4e22d10dc2e81d9c2bb321edb3750253964b",
            "0000000000000000000000000000000000000000",
            1,
        )
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(
            any("Baseline Commit differs" in error for error in errors), errors
        )

    def test_full_commit_is_a_valid_immutable_baseline_ref(self) -> None:
        baseline = "eedd4e22d10dc2e81d9c2bb321edb3750253964b"
        for relative_path in (governance.ROADMAP_PATH, governance.PROGRESS_PATH):
            text = self._read(relative_path).replace(
                "- **Baseline Ref:** `v1.4.3`",
                f"- **Baseline Ref:** `{baseline}`",
            )
            self._write(relative_path, text)

        self.assertEqual([], self._errors())

    def test_roadmap_line_budget_is_enforced(self) -> None:
        roadmap = self._read(governance.ROADMAP_PATH)
        roadmap += "\n" + ("<!-- filler -->\n" * governance.ROADMAP_MAX_LINES)
        self._write(governance.ROADMAP_PATH, roadmap)

        errors = self._errors()

        self.assertTrue(any("exceed the budget" in error for error in errors), errors)

    def test_completion_record_must_use_registered_terminal_task(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        record = """\
### 2026-08-26 — Invalid record

- **Task:** `OTHER-01`
- **Status:** `In Progress`
- **Outcome:** Nothing was completed.
- **Verification:** No verification exists.

"""
        progress = progress.replace("## Update Contract", record + "## Update Contract", 1)
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(any("unknown task 'OTHER-01'" in error for error in errors), errors)
        self.assertTrue(any("is not terminal" in error for error in errors), errors)

    def test_archive_cannot_contain_active_pointer(self) -> None:
        archive = self.root / governance.ARCHIVE_PATH / "2026-08.md"
        archive.write_text("- **Current Stage:** `BOOT`\n", encoding="utf-8")

        errors = self._errors()

        self.assertTrue(
            any("must not contain an active pointer" in error for error in errors), errors
        )

    def test_completion_record_status_must_match_roadmap(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        record = """\
### 2026-08-26 — Premature completion

- **Task:** `BASE-01`
- **Status:** `Done`
- **Outcome:** The task was claimed complete.
- **Verification:** The Roadmap still says Pending.

"""
        progress = progress.replace("## Update Contract", record + "## Update Contract", 1)
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(
            any("does not match Roadmap task BASE-01" in error for error in errors),
            errors,
        )

    def test_any_task_id_mentioned_in_progress_must_be_registered(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress = progress.replace(
            "## Update Contract",
            "An undocumented follow-up references `MISSING-42`.\n\n## Update Contract",
            1,
        )
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(
            any(
                "task IDs are not registered" in error and "MISSING-42" in error
                for error in errors
            ),
            errors,
        )

    def test_any_task_id_mentioned_in_roadmap_must_be_registered(self) -> None:
        roadmap = self._read(governance.ROADMAP_PATH)
        roadmap += "\nA dependency typo points to `BASE-02`.\n"
        self._write(governance.ROADMAP_PATH, roadmap)

        errors = self._errors()

        self.assertTrue(
            any(
                "roadmap: task IDs are not registered" in error
                and "BASE-02" in error
                for error in errors
            ),
            errors,
        )

    def test_experiment_id_is_not_misclassified_as_a_roadmap_task(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress = progress.replace(
            "## Update Contract",
            "Evidence: `EXP-20260826-001-baseline`.\n\n## Update Contract",
            1,
        )
        self._write(governance.PROGRESS_PATH, progress)

        self.assertEqual([], self._errors())

    def test_multisegment_task_like_tokens_cannot_hide(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress = progress.replace(
            "## Update Contract",
            "Hidden references: `GHOST-BASE-01` and `EXP-BASE-01`.\n\n"
            "## Update Contract",
            1,
        )
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(any("GHOST-BASE-01" in error for error in errors), errors)
        self.assertTrue(any("EXP-BASE-01" in error for error in errors), errors)

    def test_common_version_tokens_are_not_task_ids(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress = progress.replace(
            "## Update Contract",
            "Formats: `UTF-8`, `ISO-8601`, and `SHA-256`.\n\n## Update Contract",
            1,
        )
        self._write(governance.PROGRESS_PATH, progress)

        self.assertEqual([], self._errors())

    def test_prior_month_record_requires_monthly_archive(self) -> None:
        self._insert_record(self._record("2026-07-31", "Older completion"))

        errors = self._errors()

        self.assertTrue(
            any("belongs in archive 2026-07.md" in error for error in errors), errors
        )

    def test_calendar_rollover_does_not_trust_stale_last_updated(self) -> None:
        september = dt.date(2026, 9, 1)
        active_count = len(
            governance.parse_completion_records(
                self._read(governance.PROGRESS_PATH), "progress", []
            )
        )

        errors = governance.validate_repository(
            self.root, verify_git=False, today=september
        )
        plan = archiver.build_plan(self.root, today=september)

        self.assertTrue(
            any("belongs in archive 2026-08.md" in error for error in errors), errors
        )
        self.assertEqual({"2026-08": active_count}, plan.counts_by_month)

    def test_archive_record_month_must_match_filename(self) -> None:
        archive = self.root / governance.ARCHIVE_PATH / "2026-06.md"
        archive.write_text(
            "# ASR Progress Archive — 2026-06\n\n"
            + self._record("2026-07-31", "Wrong archive month"),
            encoding="utf-8",
        )

        errors = self._errors()

        self.assertTrue(
            any(
                "archive 2026-06.md" in error
                and "belongs in archive 2026-07.md" in error
                for error in errors
            ),
            errors,
        )

    def test_duplicate_active_and_archived_record_is_rejected(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        records = governance.parse_completion_records(progress, "progress", [])
        self.assertGreaterEqual(len(records), 1)
        archive = self.root / governance.ARCHIVE_PATH / "2026-08.md"
        archive.write_text(
            "# ASR Progress Archive — 2026-08\n\n" + records[0].text + "\n",
            encoding="utf-8",
        )

        errors = self._errors()

        self.assertTrue(any("duplicate completion record" in error for error in errors), errors)

    def test_completion_record_outside_recent_section_is_rejected(self) -> None:
        progress = self._read(governance.PROGRESS_PATH)
        progress += "\n" + self._record("2026-08-26", "Misplaced completion")
        self._write(governance.PROGRESS_PATH, progress)

        errors = self._errors()

        self.assertTrue(
            any("must be inside '## Recent Completion Records'" in error for error in errors),
            errors,
        )

    def test_archiver_moves_prior_month_record_and_preserves_active_pointer(self) -> None:
        self._insert_record(self._record("2026-07-31", "Older completion"))

        plan = archiver.build_plan(self.root, today=FIXTURE_TODAY)
        changed = archiver.apply_plan(self.root, plan)

        self.assertEqual({"2026-07": 1}, plan.counts_by_month)
        self.assertIn(self.root / governance.PROGRESS_PATH, changed)
        progress = self._read(governance.PROGRESS_PATH)
        self.assertIn("- **Current Stage:** `BASE`", progress)
        self.assertNotIn("Older completion", progress)
        archive = self._read(governance.ARCHIVE_PATH / "2026-07.md")
        self.assertIn("# ASR Progress Archive — 2026-07", archive)
        self.assertIn("Older completion", archive)
        self.assertEqual([], self._errors())

    def test_archiver_keeps_only_eight_same_month_records(self) -> None:
        initial_count = len(
            governance.parse_completion_records(
                self._read(governance.PROGRESS_PATH), "progress", []
            )
        )
        additions = "".join(
            self._record("2026-08-25", f"Completion {index}")
            for index in range(1, 9)
        )
        self._insert_record(additions)

        plan = archiver.build_plan(self.root, today=FIXTURE_TODAY)
        archiver.apply_plan(self.root, plan)

        expected_overflow = initial_count + 8 - governance.PROGRESS_MAX_RECORDS
        self.assertEqual(expected_overflow, len(plan.selected))
        active_records = governance.parse_completion_records(
            self._read(governance.PROGRESS_PATH), "progress", []
        )
        self.assertEqual(governance.PROGRESS_MAX_RECORDS, len(active_records))
        archive_records = governance.parse_completion_records(
            self._read(governance.ARCHIVE_PATH / "2026-08.md"),
            "archive 2026-08.md",
            [],
        )
        self.assertEqual(expected_overflow, len(archive_records))
        self.assertEqual([], self._errors())


if __name__ == "__main__":
    unittest.main()
