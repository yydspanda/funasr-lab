import json
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class BaselineSmokeEntryTest(unittest.TestCase):
    def test_dry_run_resolves_plan_without_audio_or_model_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_baseline_smoke.py",
                "--track",
                "paraformer",
                "--audio",
                "eval/private/not-downloaded.wav",
                "--dry-run",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        document = json.loads(completed.stdout)
        self.assertEqual(document["kind"], "baseline-smoke-plan")
        self.assertEqual(document["plan"]["model"], "paraformer-zh")
        self.assertFalse(document["plan"]["audio_exists"])
        self.assertTrue(document["plan"]["disable_update"])


if __name__ == "__main__":
    unittest.main()
