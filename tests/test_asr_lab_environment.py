from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import asr_lab_doctor as doctor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class BootstrapDevTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        (self.root / "scripts").mkdir()
        (self.root / "requirements").mkdir()
        shutil.copy2(
            REPOSITORY_ROOT / "scripts/bootstrap_dev.sh",
            self.root / "scripts/bootstrap_dev.sh",
        )
        (self.root / "requirements/lab-cpu.lock").write_text(
            "# test lock\n", encoding="utf-8"
        )

        self.log_path = self.root / "uv-calls.jsonl"
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        uv_path = fake_bin / "uv"
        uv_path.write_text(
            """#!/usr/bin/env python3
import json
import os
import stat
import sys
from pathlib import Path

with Path(os.environ["UV_TEST_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")

if len(sys.argv) > 1 and sys.argv[1] == "venv":
    venv = Path(sys.argv[-1])
    python = venv / "bin/python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
""",
            encoding="utf-8",
        )
        uv_path.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment["PATH"] = f"{fake_bin}{os.pathsep}{self.environment['PATH']}"
        self.environment["UV_TEST_LOG"] = str(self.log_path)

    def _fake_venv_python(self) -> Path:
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        return python

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("bash", "scripts/bootstrap_dev.sh"),
            cwd=self.root,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def _uv_calls(self) -> list[list[str]]:
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_existing_python_311_environment_is_reused_without_recreation(self) -> None:
        self._fake_venv_python()
        sentinel = self.root / ".venv/keep-me"
        sentinel.write_text("preserve", encoding="utf-8")

        completed = self._run()

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("Reusing existing Python 3.11 environment", completed.stdout)
        self.assertTrue(sentinel.is_file())
        calls = self._uv_calls()
        self.assertFalse(any(call and call[0] == "venv" for call in calls), calls)
        self.assertTrue(any(call[:2] == ["pip", "sync"] for call in calls), calls)
        bootstrap = (self.root / "scripts/bootstrap_dev.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("asr_lab_doctor.py --strict-base-env", bootstrap)

    def test_fresh_checkout_creates_environment_before_sync(self) -> None:
        completed = self._run()

        self.assertEqual(0, completed.returncode, completed.stderr)
        calls = self._uv_calls()
        self.assertEqual(1, sum(call[0] == "venv" for call in calls if call))
        self.assertTrue((self.root / ".venv/bin/python").is_file())
        self.assertTrue(any(call[:2] == ["pip", "sync"] for call in calls), calls)

    def test_invalid_existing_environment_is_preserved_and_not_synced(self) -> None:
        invalid_venv = self.root / ".venv"
        invalid_venv.mkdir()
        sentinel = invalid_venv / "keep-me"
        sentinel.write_text("preserve", encoding="utf-8")

        completed = self._run()

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("is not a usable virtual environment", completed.stderr)
        self.assertEqual("preserve", sentinel.read_text(encoding="utf-8"))
        calls = self._uv_calls()
        self.assertFalse(any(call[:2] == ["pip", "sync"] for call in calls), calls)


class BaseEnvironmentDoctorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.input_path = self.root / "requirements/lab-cpu.in"
        self.lock_path = self.root / "requirements/lab-cpu.lock"
        self.version_path = self.root / "funasr/version.txt"
        self.python_path = self.root / ".venv/bin/python"
        self.input_path.parent.mkdir(parents=True)
        self.version_path.parent.mkdir(parents=True)
        self.python_path.parent.mkdir(parents=True)
        self.input_path.write_text(
            "-e .\ntorch==2.11.0\ntorchaudio==2.11.0\n"
            "transformers==4.57.6\n",
            encoding="utf-8",
        )
        self.lock_path.write_text(
            "-e .\ntorch==2.11.0+cpu\ntorchaudio==2.11.0+cpu\n"
            "transformers==4.57.6\nfilelock==3.32.4\n",
            encoding="utf-8",
        )
        self.version_path.write_text("1.4.3\n", encoding="utf-8")
        self.python_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.python_path.chmod(0o755)

    def _checks(
        self, payload: dict[str, object], *, strict: bool = True
    ) -> list[doctor.Check]:
        with mock.patch.object(
            doctor, "_run_base_environment_probe", return_value=(payload, "")
        ):
            return doctor._base_environment_checks(
                self.root,
                self.input_path,
                self.lock_path,
                self.version_path,
                self.python_path,
                strict,
            )

    def _passing_payload(self) -> dict[str, object]:
        versions = {
            "funasr": "1.4.3",
            "torch": "2.11.0+cpu",
            "torchaudio": "2.11.0+cpu",
            "transformers": "4.57.6",
        }
        return {
            "python_version": [3, 11, 15],
            "is_venv": True,
            "installed_versions": {
                "torch": "2.11.0+cpu",
                "torchaudio": "2.11.0+cpu",
                "transformers": "4.57.6",
                "filelock": "3.32.4",
            },
            "results": [
                {
                    "distribution": name,
                    "distribution_version": version,
                    "module_version": version,
                    "module_file": (
                        str(self.root / "funasr/__init__.py")
                        if name == "funasr"
                        else f"/venv/site-packages/{name}/__init__.py"
                    ),
                    "error": None,
                }
                for name, version in versions.items()
            ],
        }

    def test_direct_import_versions_come_from_resolved_lock(self) -> None:
        requirements, error = doctor._read_base_requirements(
            self.input_path, self.lock_path, self.version_path
        )

        self.assertEqual("", error)
        self.assertEqual(
            ["1.4.3", "2.11.0+cpu", "2.11.0+cpu", "4.57.6"],
            [requirement.expected_version for requirement in requirements],
        )

    def test_ready_environment_passes_every_direct_import(self) -> None:
        checks = self._checks(self._passing_payload())

        self.assertEqual(
            [
                "base-environment",
                "base-lock-sync",
                "base-import-funasr",
                "base-import-torch",
                "base-import-torchaudio",
                "base-import-transformers",
            ],
            [check.name for check in checks],
        )
        self.assertTrue(all(check.status == "pass" for check in checks), checks)

    def test_import_error_and_version_mismatch_are_blocking(self) -> None:
        payload = self._passing_payload()
        results = payload["results"]
        self.assertIsInstance(results, list)
        results[1]["distribution_version"] = "2.10.0+cpu"
        results[1]["module_version"] = "2.10.0+cpu"
        results[2]["distribution_version"] = None
        results[2]["module_version"] = None
        results[2]["error"] = "OSError: libtorch mismatch"

        checks = {check.name: check for check in self._checks(payload)}

        self.assertEqual("fail", checks["base-import-torch"].status)
        self.assertIn("expected 2.11.0+cpu", checks["base-import-torch"].detail)
        self.assertEqual("fail", checks["base-import-torchaudio"].status)
        self.assertIn("libtorch mismatch", checks["base-import-torchaudio"].detail)

    def test_missing_transitive_distribution_fails_lock_sync(self) -> None:
        payload = self._passing_payload()
        installed_versions = payload["installed_versions"]
        self.assertIsInstance(installed_versions, dict)
        del installed_versions["filelock"]

        checks = {check.name: check for check in self._checks(payload)}

        self.assertEqual("fail", checks["base-lock-sync"].status)
        self.assertIn("filelock is missing", checks["base-lock-sync"].detail)

    def test_default_mode_blocks_drift_in_existing_environment(self) -> None:
        payload = self._passing_payload()
        installed_versions = payload["installed_versions"]
        self.assertIsInstance(installed_versions, dict)
        del installed_versions["filelock"]

        checks = {check.name: check for check in self._checks(payload, strict=False)}

        self.assertEqual("fail", checks["base-lock-sync"].status)

    def test_same_version_funasr_from_another_checkout_is_rejected(self) -> None:
        payload = self._passing_payload()
        results = payload["results"]
        self.assertIsInstance(results, list)
        results[0]["module_file"] = "/another/checkout/funasr/__init__.py"

        checks = {check.name: check for check in self._checks(payload)}

        self.assertEqual("fail", checks["base-import-funasr"].status)
        self.assertIn("another/checkout", checks["base-import-funasr"].detail)
        self.assertIn(str(self.root / "funasr"), checks["base-import-funasr"].detail)

    def test_missing_venv_fails_without_running_probe(self) -> None:
        self.python_path.unlink()

        with mock.patch.object(doctor, "_run_base_environment_probe") as probe:
            checks = doctor._base_environment_checks(
                self.root,
                self.input_path,
                self.lock_path,
                self.version_path,
                self.python_path,
                True,
            )

        probe.assert_not_called()
        self.assertEqual(["fail"], [check.status for check in checks])
        self.assertIn("run scripts/bootstrap_dev.sh", checks[0].detail)

    def test_missing_venv_is_non_blocking_in_default_diagnostic_mode(self) -> None:
        shutil.rmtree(self.root / ".venv")

        with mock.patch.object(doctor, "_run_base_environment_probe") as probe:
            checks = doctor._base_environment_checks(
                self.root,
                self.input_path,
                self.lock_path,
                self.version_path,
                self.python_path,
            )

        probe.assert_not_called()
        self.assertEqual(["warn"], [check.status for check in checks])

    def test_existing_broken_venv_is_blocking_in_default_mode(self) -> None:
        self.python_path.unlink()

        with mock.patch.object(doctor, "_run_base_environment_probe") as probe:
            checks = doctor._base_environment_checks(
                self.root,
                self.input_path,
                self.lock_path,
                self.version_path,
                self.python_path,
            )

        probe.assert_not_called()
        self.assertEqual(["fail"], [check.status for check in checks])

    def test_probe_queries_locked_versions_and_forces_offline_imports(self) -> None:
        requirement = doctor.BaseRequirement("torch", "torch", "2.11.0+cpu")
        payload = {
            "python_version": [3, 11, 15],
            "is_venv": True,
            "installed_versions": {},
            "results": [],
        }
        completed = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=doctor.BASE_PROBE_MARKER + json.dumps(payload),
            stderr="",
        )

        with mock.patch.object(subprocess, "run", return_value=completed) as run:
            result, error = doctor._run_base_environment_probe(
                self.python_path, [requirement], ["packaging", "torch"], self.root
            )

        self.assertEqual("", error)
        self.assertEqual(payload, result)
        arguments = run.call_args.args[0]
        self.assertIn("-I", arguments)
        probe_source = arguments[3]
        self.assertIn("importlib.metadata.version(distribution)", probe_source)
        self.assertNotIn("importlib.metadata.distributions()", probe_source)
        self.assertIn('getattr(module, "__file__", None)', probe_source)
        probe_request = json.loads(arguments[4])
        self.assertEqual(
            ["packaging", "torch"], probe_request["locked_distributions"]
        )
        environment = run.call_args.kwargs["env"]
        self.assertEqual("1", environment["HF_HUB_OFFLINE"])
        self.assertEqual("1", environment["TRANSFORMERS_OFFLINE"])

    def test_strict_cli_flag_is_forwarded_to_collection(self) -> None:
        with (
            mock.patch.object(sys, "argv", ["asr_lab_doctor.py", "--strict-base-env"]),
            mock.patch.object(doctor, "collect_checks", return_value=[]) as collect,
            mock.patch("builtins.print"),
        ):
            exit_code = doctor.main()

        self.assertEqual(0, exit_code)
        collect.assert_called_once_with(strict_base_env=True)


if __name__ == "__main__":
    unittest.main()
