from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import check_asr_progress as governance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
        return governance.validate_repository(self.root, verify_git=False)

    def test_repository_documents_are_valid(self) -> None:
        self.assertEqual([], self._errors())

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


if __name__ == "__main__":
    unittest.main()
