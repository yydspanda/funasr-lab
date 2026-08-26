import tempfile
import unittest
from pathlib import Path

from scripts.check_experiment_manifests import validate_directory
from scripts.check_experiment_manifests import validate_manifest


class ExperimentManifestGovernanceTest(unittest.TestCase):
    def valid_manifest(self):
        return {
            "experiment_id": "EXP-20260826-001-baseline",
            "task_id": "BASE-01",
            "hypothesis": "The upstream baseline is reproducible.",
            "upstream_commit": "eedd4e22d10d",
            "code_commit": "1234567890ab",
            "model": "paraformer-zh",
            "model_revision": "pinned-revision",
            "config_sha256": "0" * 64,
            "data_sha256": "1" * 64,
            "eval_data_version": "smoke-v0.1",
            "normalizer_version": "zh-content-v0.1",
            "hardware": "test-host",
            "seed": 42,
            "command": "python eval/run.py",
            "metrics": {
                "content_cer": 0.1,
                "rtf_p50": 0.2,
                "peak_rss_mb": 512,
            },
            "artifacts": ["artifacts/report.json"],
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

    def test_rejects_a_missing_manifest_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            self.assertTrue(validate_directory(missing))


if __name__ == "__main__":
    unittest.main()
