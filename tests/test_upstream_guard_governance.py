from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_upstream_guard as guard  # noqa: E402


class TemporaryFork:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._git("init", "-b", "main")
        self._git("config", "user.email", "guard@example.invalid")
        self._git("config", "user.name", "Guard Test")
        self.write("funasr/core.py", "VALUE = 1\n")
        self.write(
            "tests/test_existing_core.py",
            "import unittest\n\n"
            "class ExistingTest(unittest.TestCase):\n"
            "    def test_existing(self):\n"
            "        self.assertTrue(True)\n",
        )
        self.write(".gitignore", ".venv/\n")
        self._git("add", ".")
        self._git("commit", "-m", "vendor baseline")
        self.baseline = self._git("rev-parse", "HEAD").stdout.strip()

        self._git("remote", "add", "origin", "git@github.com:yydspanda/funasr-lab.git")
        self._git("remote", "add", "upstream", "https://github.com/modelscope/FunASR.git")
        self._git("config", "remote.upstream.pushurl", "no_push")
        self._git("update-ref", "refs/remotes/origin/main", self.baseline)
        self._git("update-ref", "refs/remotes/origin/develop", self.baseline)
        self._git(
            "update-ref",
            "refs/remotes/origin/vendor/funasr-v1.4.3",
            self.baseline,
        )
        self._git("update-ref", "refs/remotes/upstream/main", self.baseline)
        self.write(
            guard.DEFAULT_ROADMAP,
            "# Roadmap\n\n"
            f"- **Baseline Commit:** `{self.baseline}`\n\n"
            "| ID | Status |\n|---|---|\n| `EXP-01` | In Progress |\n",
        )
        self.write_ledger([])

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_ledger(self, entries: list[dict[str, object]]) -> None:
        payload = {
            "schema_version": guard.SCHEMA_VERSION,
            "baseline_ref": self.baseline,
            "baseline_commit": self.baseline,
            "core_patches": entries,
        }
        self.write(guard.DEFAULT_LEDGER, json.dumps(payload))

    def commit(self, message: str) -> str:
        self._git("add", ".")
        self._git("commit", "-m", message)
        return self._git("rev-parse", "HEAD").stdout.strip()

    def add_upstream_commit(self) -> str:
        current = self._git("branch", "--show-current").stdout.strip()
        self._git("switch", "-c", "upstream-work", self.baseline)
        self.write("upstream-change.txt", "new upstream work\n")
        self._git("add", "upstream-change.txt")
        self._git("commit", "-m", "upstream moves")
        commit = self._git("rev-parse", "HEAD").stdout.strip()
        self._git("update-ref", "refs/remotes/upstream/main", commit)
        self._git("switch", current)
        return commit


class UpstreamGuardGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = TemporaryFork(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def isolation(self) -> tuple[dict[str, object], list[dict[str, str]]]:
        return guard.source_isolation_summary(
            self.repo.root,
            "refs/remotes/origin/vendor/funasr-v1.4.3",
            "HEAD",
            self.repo.root / guard.DEFAULT_LEDGER,
        )

    def test_workflow_checks_candidate_head_without_weakening_recurring_audit(self) -> None:
        workflow = (
            REPO_ROOT / ".github/workflows/asr-upstream-guard.yml"
        ).read_text(encoding="utf-8")

        reject_main_pr = workflow.index(
            "github.event_name == 'pull_request' && github.base_ref == 'main'"
        )
        measure = workflow.index("- name: Measure drift and enforce fork boundary")
        self.assertLess(reject_main_pr, measure)
        self.assertIn("github.event_name == 'pull_request'", workflow)
        self.assertIn("github.event_name == 'push'", workflow)
        self.assertIn("github.ref_type == 'branch'", workflow)
        self.assertIn("github.ref_name != 'main'", workflow)
        self.assertIn("'HEAD' || 'refs/remotes/origin/develop'", workflow)
        self.assertIn('--active-ref "$ASR_ACTIVE_REF"', workflow)

    def test_github_repository_slug_accepts_https_and_scp_ssh(self) -> None:
        self.assertEqual(
            guard.github_repository_slug("https://github.com/modelscope/FunASR.git"),
            "modelscope/funasr",
        )
        self.assertEqual(
            guard.github_repository_slug("git@github.com:modelscope/FunASR.git"),
            "modelscope/funasr",
        )
        self.assertIsNone(guard.github_repository_slug("https://example.com/modelscope/FunASR"))

    def test_remote_guard_rejects_wrong_upstream_and_push_enabled(self) -> None:
        self.repo._git("remote", "set-url", "upstream", "https://github.com/other/project.git")
        self.repo._git("config", "--unset-all", "remote.upstream.pushurl")
        _, errors = guard.inspect_remotes(
            self.repo.root,
            guard.DEFAULT_ORIGIN_REPOSITORY,
            guard.DEFAULT_UPSTREAM_REPOSITORY,
        )
        codes = {item["code"] for item in errors}
        self.assertIn("upstream_remote_mismatch", codes)
        self.assertIn("upstream_push_not_disabled", codes)

    def test_drift_summary_fails_when_main_exceeds_behind_threshold(self) -> None:
        upstream_commit = self.repo.add_upstream_commit()
        args = guard.build_parser().parse_args(
            [
                "--repo",
                str(self.repo.root),
                "--no-fetch",
                "--max-behind",
                "0",
                "--max-ahead",
                "0",
            ]
        )
        summary, exit_code = guard.execute(args)
        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["drift"]["mirror_main"]["ahead"], 0)
        self.assertEqual(summary["drift"]["mirror_main"]["behind"], 1)
        self.assertEqual(summary["drift"]["active_downstream"]["behind"], 1)
        self.assertEqual(summary["drift"]["accepted_baseline"]["behind"], 1)
        self.assertEqual(summary["drift"]["upstream"]["commit"], upstream_commit)
        self.assertIn(
            "fork_main_too_far_behind",
            {item["code"] for item in summary["errors"]},
        )

    def test_active_downstream_drift_fails_even_when_mirror_main_is_current(self) -> None:
        upstream_commit = self.repo.add_upstream_commit()
        self.repo._git("update-ref", "refs/remotes/origin/main", upstream_commit)
        args = guard.build_parser().parse_args(
            [
                "--repo",
                str(self.repo.root),
                "--no-fetch",
                "--max-behind",
                "0",
            ]
        )

        summary, exit_code = guard.execute(args)

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["drift"]["mirror_main"]["behind"], 0)
        self.assertEqual(summary["drift"]["active_downstream"]["behind"], 1)
        self.assertIn(
            "active_downstream_too_far_behind",
            {item["code"] for item in summary["errors"]},
        )

    def test_upstream_owned_noncore_edit_is_reported_without_core_ledger(self) -> None:
        self.repo.write(".gitignore", ".venv/\ncache/\n")
        self.repo.write("scripts/downstream_check.py", "print('downstream')\n")
        self.repo.commit("downstream governance")

        summary, errors = self.isolation()

        self.assertEqual(errors, [])
        self.assertIn(".gitignore", summary["upstream_owned_paths"])
        self.assertEqual(summary["core_paths"], [])
        self.assertEqual(summary["new_paths_outside_recommended_surfaces"], [])

    def test_core_edit_requires_exact_ledger_reason_and_versioned_test(self) -> None:
        self.repo.write("funasr/core.py", "VALUE = 2\n")
        self.repo.write(
            "tests/test_focused_core_patch.py",
            "import unittest\n\n"
            "class FocusedCorePatchTest(unittest.TestCase):\n"
            "    def test_core_patch(self):\n"
            "        self.assertTrue(True)\n",
        )
        self.repo.commit("change core with focused test")

        summary, errors = self.isolation()
        self.assertEqual(summary["unregistered_core_paths"], ["funasr/core.py"])
        self.assertIn("unregistered_core_patch", {item["code"] for item in errors})

        self.repo.write_ledger(
            [
                {
                    "path": "funasr/core.py",
                    "task_id": "EXP-01",
                    "reason": (
                        "The existing registry cannot express this narrow state "
                        "transition safely."
                    ),
                    "tests": ["tests/test_focused_core_patch.py"],
                }
            ]
        )
        summary, errors = self.isolation()
        self.assertEqual(errors, [])
        self.assertEqual(summary["registered_core_paths"], ["funasr/core.py"])

    def test_ledger_test_must_be_changed_regular_unittest_module(self) -> None:
        self.repo.write("funasr/core.py", "VALUE = 2\n")
        self.repo.commit("change core without a focused test")
        entry = {
            "path": "funasr/core.py",
            "task_id": "EXP-01",
            "reason": (
                "The extension boundary cannot express this required core behavior "
                "safely."
            ),
            "tests": ["tests/test_existing_core.py"],
        }
        self.repo.write_ledger([entry])

        _, unchanged_errors = self.isolation()

        self.assertIn(
            "invalid_core_patch_test", {item["code"] for item in unchanged_errors}
        )

        entry["tests"] = ["tests/"]
        self.repo.write_ledger([entry])
        _, directory_errors = self.isolation()
        self.assertIn(
            "invalid_core_patch_test", {item["code"] for item in directory_errors}
        )

    def test_requested_ledger_test_execution_fails_the_guard(self) -> None:
        self.repo.write("funasr/core.py", "VALUE = 2\n")
        self.repo.write(
            "tests/test_failing_core_patch.py",
            "import unittest\n\n"
            "class FailingCorePatchTest(unittest.TestCase):\n"
            "    def test_core_patch(self):\n"
            "        self.fail('intentional fixture failure')\n",
        )
        self.repo.commit("change core with failing focused test")
        self.repo.write_ledger(
            [
                {
                    "path": "funasr/core.py",
                    "task_id": "EXP-01",
                    "reason": (
                        "The extension boundary cannot express this required core "
                        "behavior safely."
                    ),
                    "tests": ["tests/test_failing_core_patch.py"],
                }
            ]
        )
        args = guard.build_parser().parse_args(
            [
                "--repo",
                str(self.repo.root),
                "--no-fetch",
                "--run-ledger-tests",
            ]
        )

        summary, exit_code = guard.execute(args)

        self.assertEqual(exit_code, 1)
        self.assertFalse(
            summary["ledger_test_execution"]["results"][0]["passed"]
        )
        self.assertIn(
            "core_patch_test_failed", {item["code"] for item in summary["errors"]}
        )

    def test_ledger_test_execution_rejects_zero_collected_tests(self) -> None:
        self.repo.write("tests/test_empty_module.py", "VALUE = 1\n")

        results, errors = guard.run_ledger_tests(
            self.repo.root, ["tests/test_empty_module.py"]
        )

        self.assertFalse(results[0]["passed"])
        self.assertIn("no unittest tests were collected", errors[0]["message"])

    def test_ledger_test_execution_rejects_all_skipped_tests(self) -> None:
        self.repo.write(
            "tests/test_skipped_module.py",
            "import unittest\n\n"
            "class SkippedTest(unittest.TestCase):\n"
            "    @unittest.skip('platform dependency unavailable')\n"
            "    def test_core_patch(self):\n"
            "        self.fail('must not run')\n",
        )

        results, errors = guard.run_ledger_tests(
            self.repo.root, ["tests/test_skipped_module.py"]
        )

        self.assertEqual(results[0]["collected"], 1)
        self.assertEqual(results[0]["skipped"], 1)
        self.assertEqual(results[0]["executed"], 0)
        self.assertFalse(results[0]["passed"])
        self.assertIn("all collected unittest tests were skipped", errors[0]["message"])

    def test_new_file_inside_core_is_also_protected(self) -> None:
        self.repo.write("funasr/new_extension.py", "VALUE = 1\n")
        self.repo.commit("add code directly inside core")

        summary, errors = self.isolation()

        self.assertEqual(summary["core_paths"], ["funasr/new_extension.py"])
        self.assertIn("unregistered_core_patch", {item["code"] for item in errors})

    def test_all_upstream_implementation_surfaces_are_core(self) -> None:
        for path in (
            "benchmarks/speed.py",
            "data/config.yaml",
            "examples/train.py",
            "integrations/client.py",
            "model_zoo/model.py",
            "runtime/server.cpp",
            "benchmark_vllm.py",
            "pyproject.toml",
        ):
            with self.subTest(path=path):
                self.assertTrue(guard.is_core_path(path))

    def test_core_ledger_task_must_exist_in_roadmap(self) -> None:
        self.repo.write("funasr/core.py", "VALUE = 2\n")
        self.repo.write("tests/test_core_task.py", "def test_task():\n    assert True\n")
        self.repo.commit("change core")
        self.repo.write_ledger(
            [
                {
                    "path": "funasr/core.py",
                    "task_id": "MISSING-99",
                    "reason": "No downstream extension can safely expose this required behavior.",
                    "tests": ["tests/test_core_task.py"],
                }
            ]
        )

        _, errors = self.isolation()

        self.assertIn("unknown_core_patch_task", {item["code"] for item in errors})

    def test_ledger_baseline_must_match_roadmap_control_record(self) -> None:
        roadmap = self.repo.root / guard.DEFAULT_ROADMAP
        roadmap.write_text(
            "# Roadmap\n\n"
            f"- **Baseline Commit:** `{'f' * 40}`\n\n"
            "| ID | Status |\n|---|---|\n| `EXP-01` | In Progress |\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(guard.GuardError, "Roadmap Baseline Commit"):
            self.isolation()

    def test_stale_core_ledger_entry_fails(self) -> None:
        self.repo.write("tests/test_unused_patch.py", "def test_unused():\n    assert True\n")
        self.repo.commit("add only a test")
        self.repo.write_ledger(
            [
                {
                    "path": "funasr/core.py",
                    "task_id": "EXP-01",
                    "reason": (
                        "This sufficiently long explanation should not remain after "
                        "a patch is removed."
                    ),
                    "tests": ["tests/test_unused_patch.py"],
                }
            ]
        )

        _, errors = self.isolation()

        self.assertIn("stale_core_patch_entry", {item["code"] for item in errors})


if __name__ == "__main__":
    unittest.main()
