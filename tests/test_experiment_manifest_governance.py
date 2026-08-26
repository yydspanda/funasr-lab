import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_experiment_manifests import validate_directory
from scripts.check_experiment_manifests import validate_manifest


def digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


class ExperimentManifestGovernanceTest(unittest.TestCase):
    def valid_manifest(self):
        return {
            "schema_version": 1,
            "experiment_id": "EXP-20260826-001-baseline",
            "task_id": "BASE-01",
            "hypothesis": "The pinned upstream baseline is reproducible on CPU.",
            "upstream_commit": "eedd4e22d10dc2e81d9c2bb321edb3750253964b",
            "code_commit": "cdea8f3ab3ed99e32ee7da4097dcf27f509b73af",
            "models": [
                {
                    "role": "asr",
                    "identifier": (
                        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-"
                        "common-vocab8404-pytorch"
                    ),
                    "revision": "v2.0.4",
                    "sha256": digest("resolved-paraformer-bundle"),
                },
                {
                    "role": "vad",
                    "identifier": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    "revision": "v2.0.4",
                    "sha256": digest("resolved-vad-bundle"),
                },
            ],
            "config_sha256": digest("effective-config"),
            "data_sha256": digest("ordered-dataset-manifest"),
            "eval_data_version": "smoke-v0.1",
            "normalizer_version": "zh-content-v0.1",
            "hardware": {
                "host_id": "cpu-lab-01",
                "os": "Ubuntu 24.04.3 LTS; Linux 6.8.0 x86_64",
                "cpu_model": "AMD Ryzen 9 7950X 16-Core Processor",
                "logical_cpu_count": 32,
                "memory_bytes": 68719476736,
                "device": "cpu",
                "accelerator": None,
            },
            "seed": 42,
            "command": {
                "working_directory": ".",
                "argv": [
                    ".venv/bin/python",
                    "scripts/run_baseline_smoke.py",
                    "--track",
                    "paraformer",
                    "--audio",
                    "eval/private/smoke.wav",
                    "--model-revision",
                    "v2.0.4",
                ],
                "environment": {"OMP_NUM_THREADS": "8"},
            },
            "metrics": {
                "content_cer": 0.1,
                "substitutions": 5,
                "deletions": 2,
                "insertions": 3,
                "reference_units": 100,
                "utterance_count": 12,
                "failed_count": 0,
                "rtf_p50": 0.2,
                "rtf_p95": 0.25,
                "peak_rss_mb": 512,
            },
            "artifacts": [
                {
                    "kind": "report",
                    "path": "artifacts/report.json",
                    "sha256": digest("report"),
                }
            ],
            "decision": "investigate",
        }

    def test_accepts_a_complete_manifest(self):
        self.assertEqual(validate_manifest(self.valid_manifest()), [])

    def test_rejects_missing_reproducibility_fields(self):
        manifest = self.valid_manifest()
        del manifest["data_sha256"]
        manifest["metrics"].pop("rtf_p50")
        errors = validate_manifest(manifest)
        self.assertTrue(any("data_sha256" in error for error in errors))
        self.assertTrue(any("metrics.rtf_p50" in error for error in errors))

    def test_rejects_short_or_synthetic_commits(self):
        manifest = self.valid_manifest()
        manifest["upstream_commit"] = "eedd4e22d10d"
        manifest["code_commit"] = "0123456789" * 4
        errors = validate_manifest(manifest)
        self.assertTrue(any("upstream_commit" in error for error in errors))
        self.assertTrue(
            any("code_commit looks like a placeholder" in error for error in errors)
        )

    def test_rejects_floating_model_revision_and_fake_component_hash(self):
        manifest = self.valid_manifest()
        manifest["models"][1]["revision"] = "refs/heads/main"
        manifest["models"][1]["sha256"] = "sha256:" + "0" * 64
        errors = validate_manifest(manifest)
        self.assertTrue(any("models[1].revision" in error for error in errors))
        self.assertTrue(any("models[1].sha256" in error for error in errors))

    def test_rejects_whitespace_disguised_floating_model_revision(self):
        manifest = self.valid_manifest()
        manifest["models"][0]["revision"] = " refs/heads/main "

        errors = validate_manifest(manifest)

        self.assertTrue(any("surrounding whitespace" in error for error in errors))
        self.assertTrue(any("floating revision" in error for error in errors))

    def test_rejects_placeholder_and_empty_content_hashes(self):
        manifest = self.valid_manifest()
        manifest["config_sha256"] = "sha256:" + "0123456789abcdef" * 4
        manifest["data_sha256"] = (
            "sha256:e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        )
        errors = validate_manifest(manifest)
        self.assertTrue(any("config_sha256" in error for error in errors))
        self.assertTrue(any("data_sha256" in error for error in errors))

    def test_rejects_placeholder_hardware_and_abridged_command(self):
        manifest = self.valid_manifest()
        manifest["hardware"]["host_id"] = "test-host"
        manifest["command"]["argv"] = ["python3", "..."]
        errors = validate_manifest(manifest)
        self.assertTrue(any("hardware.host_id" in error for error in errors))
        self.assertTrue(any("command.argv[1]" in error for error in errors))

    def test_rejects_non_finite_or_impossible_core_metrics(self):
        manifest = self.valid_manifest()
        manifest["metrics"]["content_cer"] = math.nan
        manifest["metrics"]["rtf_p50"] = 0
        manifest["metrics"]["rtf_p95"] = -1
        errors = validate_manifest(manifest)
        self.assertTrue(any("metrics.content_cer" in error for error in errors))
        self.assertTrue(any("metrics.rtf_p50" in error for error in errors))
        self.assertTrue(any("metrics.rtf_p95" in error for error in errors))

    def test_rejects_missing_or_inconsistent_cer_accounting(self):
        manifest = self.valid_manifest()
        manifest["metrics"].pop("deletions")
        manifest["metrics"]["failed_count"] = 13
        manifest["metrics"]["content_cer"] = 0.2

        errors = validate_manifest(manifest)

        self.assertTrue(any("metrics.deletions is required" in error for error in errors))
        self.assertTrue(any("failed_count cannot exceed" in error for error in errors))

        manifest["metrics"]["deletions"] = 2
        errors = validate_manifest(manifest)
        self.assertTrue(any("metrics.content_cer must equal" in error for error in errors))

    def test_rejects_unhashed_artifacts(self):
        manifest = self.valid_manifest()
        manifest["artifacts"] = ["artifacts/report.json"]
        errors = validate_manifest(manifest)
        self.assertTrue(
            any("artifacts[0] must be a JSON object" in error for error in errors)
        )

    def test_executed_experiment_requires_a_hashed_report_artifact(self):
        manifest = self.valid_manifest()
        manifest["artifacts"] = []
        errors = validate_manifest(manifest)
        self.assertTrue(any("kind 'report'" in error for error in errors))

    def test_non_report_artifact_does_not_satisfy_executed_report_gate(self):
        manifest = self.valid_manifest()
        manifest["artifacts"][0]["kind"] = "checkpoint"

        errors = validate_manifest(manifest)

        self.assertTrue(any("kind 'report'" in error for error in errors))

    def test_planned_experiment_may_not_have_a_report_yet(self):
        manifest = self.valid_manifest()
        manifest["decision"] = "planned"
        manifest["metrics"] = None
        manifest["artifacts"] = []
        self.assertEqual(validate_manifest(manifest), [])

    def test_planned_experiment_rejects_result_fields_before_execution(self):
        manifest = self.valid_manifest()
        manifest["decision"] = "planned"

        errors = validate_manifest(manifest)

        self.assertTrue(
            any("planned experiment metrics must be null" in error for error in errors)
        )
        self.assertTrue(
            any("planned experiment artifacts must be empty" in error for error in errors)
        )

    def test_empty_manifest_directory_remains_valid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.assertEqual(validate_directory(directory), [])

    def test_directory_validation_checks_each_json_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            experiment_id = self.valid_manifest()["experiment_id"]
            (directory / f"{experiment_id}.json").write_text(
                json.dumps(self.valid_manifest()), encoding="utf-8"
            )
            self.assertEqual(validate_directory(directory), [])

    def test_directory_rejects_filename_mismatch_and_duplicate_experiment_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = self.valid_manifest()
            experiment_id = manifest["experiment_id"]
            (directory / f"{experiment_id}.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (directory / "conflicting-copy.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            errors = validate_directory(directory)

            self.assertTrue(any("filename must be" in error for error in errors), errors)
            self.assertTrue(any("duplicate experiment_id" in error for error in errors), errors)

    def test_directory_rejects_duplicate_json_object_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = self.valid_manifest()
            serialized = json.dumps(manifest)
            serialized = serialized.replace(
                '"task_id": "BASE-01"',
                '"task_id": "BASE-01", "task_id": "GHOST-99"',
                1,
            )
            path = directory / f"{manifest['experiment_id']}.json"
            path.write_text(serialized, encoding="utf-8")

            errors = validate_directory(directory)

            self.assertTrue(any("duplicate JSON object key" in error for error in errors))

    def test_directory_rejects_task_not_registered_in_roadmap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = self.valid_manifest()
            manifest["task_id"] = "GHOST-99"
            (directory / "unknown-task.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            errors = validate_directory(directory)
            self.assertTrue(any("is not registered" in error for error in errors))

    def test_directory_rejects_realistic_but_missing_git_commits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = self.valid_manifest()
            manifest["upstream_commit"] = hashlib.sha1(
                b"missing upstream object"
            ).hexdigest()
            manifest["code_commit"] = hashlib.sha1(
                b"missing downstream object"
            ).hexdigest()
            path = directory / f"{manifest['experiment_id']}.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            errors = validate_directory(directory)

            self.assertTrue(
                any("upstream_commit does not resolve" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("code_commit does not resolve" in error for error in errors),
                errors,
            )

    def test_feature_commit_must_already_be_reachable_from_target_branch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def git(*args: str) -> str:
                completed = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return completed.stdout.strip()

            git("init", "-b", "develop")
            git("config", "user.email", "manifest@example.invalid")
            git("config", "user.name", "Manifest Test")
            (root / "baseline.py").write_text("BASELINE = True\n", encoding="utf-8")
            git("add", "baseline.py")
            git("commit", "-m", "accepted baseline")
            baseline = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/develop", baseline)

            (root / "feature.py").write_text("FEATURE = True\n", encoding="utf-8")
            git("add", "feature.py")
            git("commit", "-m", "unmerged feature code")
            feature_commit = git("rev-parse", "HEAD")

            notes = root / ".notes/asr"
            notes.mkdir(parents=True)
            roadmap = notes / "delivery-roadmap.md"
            roadmap.write_text(
                "# Roadmap\n\n"
                f"- **Baseline Commit:** `{baseline}`\n\n"
                "| ID | Status |\n|---|---|\n| `BASE-01` | In Progress |\n",
                encoding="utf-8",
            )
            directory = root / "experiments/manifests"
            directory.mkdir(parents=True)
            manifest = self.valid_manifest()
            manifest["upstream_commit"] = baseline
            manifest["code_commit"] = feature_commit
            path = directory / f"{manifest['experiment_id']}.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            errors = validate_directory(
                directory,
                roadmap,
                repo_root=root,
                code_ref="refs/remotes/origin/develop",
            )

            self.assertTrue(
                any("not reachable from target code ref" in error for error in errors),
                errors,
            )

    def test_rejects_a_missing_manifest_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            self.assertTrue(validate_directory(missing))


if __name__ == "__main__":
    unittest.main()
