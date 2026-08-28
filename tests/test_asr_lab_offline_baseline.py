import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr
from pathlib import Path

from eval.normalizers import NORMALIZER_VERSION
from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import BaselineError
from eval.offline_baseline import BaselineExecutionError
from eval.offline_baseline import DatasetValidationError
from eval.offline_baseline import TRACKS
from eval.offline_baseline import canonical_json_bytes
from eval.offline_baseline import cer_components
from eval.offline_baseline import command_environment
from eval.offline_baseline import effective_config
from eval.offline_baseline import load_frozen_dataset
from eval.offline_baseline import model_directory_sha256
from eval.offline_baseline import run_offline_baseline
from eval.offline_baseline import sha256_bytes
from eval.offline_baseline import sha256_file
from eval.offline_baseline import text_views
from eval.offline_baseline import validate_immutable_revision
from eval.offline_baseline import write_report
from scripts.run_offline_baseline import _expanded_argv
from scripts.run_offline_baseline import _validated_command_environment
from scripts.run_offline_baseline import _validated_working_directory
from scripts.run_offline_baseline import build_parser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MOCK_MODEL_SHA256 = sha256_bytes(b"mock model bundle")


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class OfflineBaselineTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo_root = Path(self.temporary_directory.name)
        (self.repo_root / "eval/private").mkdir(parents=True)
        (self.repo_root / "eval/manifests").mkdir(parents=True)

    def write_wav(
        self,
        name,
        *,
        duration_seconds=0.1,
        sample_rate=16_000,
        channels=1,
    ):
        relative_path = Path("eval/private") / name
        path = self.repo_root / relative_path
        frame_count = int(duration_seconds * sample_rate)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count * channels)
        return relative_path, path, frame_count / sample_rate

    def record(self, utterance_id, audio_relative, audio_path, duration, raw_text):
        return {
            "id": utterance_id,
            "audio": audio_relative.as_posix(),
            "audio_sha256": sha256_file(audio_path),
            "duration_seconds": duration,
            "sample_rate": 16_000,
            "channels": 1,
            "raw_text": raw_text,
            "reference_sha256": sha256_bytes(raw_text.encode("utf-8")),
            "normalizer_version": NORMALIZER_VERSION,
            "speaker_id": f"speaker-{utterance_id}",
            "session_id": f"session-{utterance_id}",
            "split": "smoke",
            "data_version": "smoke-v0.1",
        }

    def write_manifest(self, records, name="frozen.jsonl"):
        path = self.repo_root / "eval/manifests" / name
        payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ).encode("utf-8")
        path.write_bytes(payload)
        return path, payload

    def test_example_manifest_keeps_required_identity_fields(self):
        example_path = (
            REPOSITORY_ROOT / "eval/manifests/dataset-manifest.example.jsonl"
        )
        document = json.loads(example_path.read_text(encoding="utf-8"))

        self.assertTrue(
            {
                "id",
                "audio",
                "audio_sha256",
                "duration_seconds",
                "sample_rate",
                "channels",
                "raw_text",
                "reference_sha256",
                "normalizer_version",
                "speaker_id",
                "session_id",
                "split",
                "data_version",
            }.issubset(document)
        )

    def make_dataset(self, references):
        records = []
        for index, reference in enumerate(references, start=1):
            relative, path, duration = self.write_wav(f"utt-{index}.wav")
            records.append(
                self.record(f"utt-{index}", relative, path, duration, reference)
            )
        manifest_path, _ = self.write_manifest(records)
        return load_frozen_dataset(manifest_path, self.repo_root)

    def test_tracks_use_explicit_modelscope_identifiers_and_reject_floating_revisions(self):
        self.assertEqual(TRACKS["paraformer"].hub, "ms")
        self.assertEqual(
            TRACKS["paraformer"].model_identifier,
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        self.assertEqual(TRACKS["sensevoice"].model_identifier, "iic/SenseVoiceSmall")
        self.assertNotIn("vad_model", TRACKS["paraformer"].generate_options)
        self.assertNotIn("punc_model", TRACKS["paraformer"].generate_options)

        for revision in (
            "master",
            "MAIN",
            "latest",
            "HEAD",
            "refs/heads/main",
            "refs/remotes/origin/master",
            " master ",
        ):
            with self.subTest(revision=revision):
                with self.assertRaises(BaselineExecutionError):
                    validate_immutable_revision(revision)
        self.assertEqual(validate_immutable_revision("v2.0.4"), "v2.0.4")
        commit = "9f8d7c6b5a432109876543210fedcba987654321"
        self.assertEqual(validate_immutable_revision(commit), commit)
        with self.assertRaisesRegex(BaselineExecutionError, "fixes seed"):
            effective_config(
                BaselineConfig(
                    track="paraformer",
                    model_revision="v2.0.4",
                    seed=1,
                )
            )
        with self.assertRaisesRegex(BaselineExecutionError, "only permits"):
            effective_config(
                BaselineConfig(
                    track="paraformer",
                    model_revision="v2.0.4",
                    device="cuda",
                )
            )

    def test_cli_help_imports_downstream_evaluator_without_installed_package(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_offline_baseline.py", "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--model-revision", completed.stdout)
        self.assertIn("--seed", completed.stdout)

    def test_frozen_repository_smoke_manifest_passes_validation_cli(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/validate_offline_baseline_dataset.py",
                "--dataset-manifest",
                "eval/manifests/lab-base-smoke-001-v0.1.jsonl",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        document = json.loads(completed.stdout)
        self.assertEqual(document["data_version"], "LAB-BASE-SMOKE-001-v0.1")
        self.assertEqual(document["utterance_count"], 2)
        self.assertAlmostEqual(document["audio_seconds"], 11.5466875)
        self.assertEqual(
            document["data_sha256"],
            "sha256:775614f52d04f1b9aa320007af31e18e87c60c53a88f25625390c7a8389bcc10",
        )

        parser = build_parser()
        args = parser.parse_args(
            [
                "--track",
                "paraformer",
                "--dataset-manifest",
                "eval/private/smoke.jsonl",
                "--model-revision",
                "v2.0.4",
                "--output-report",
                "eval/reports/paraformer.json",
            ]
        )
        expanded = _expanded_argv(args)
        self.assertEqual(expanded[expanded.index("--seed") + 1], "0")
        self.assertEqual(expanded[expanded.index("--device") + 1], "cpu")
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--track",
                        "paraformer",
                        "--dataset-manifest",
                        "eval/private/smoke.jsonl",
                        "--model-revision",
                        "v2.0.4",
                        "--output-report",
                        "eval/reports/paraformer.json",
                        "--device",
                        "cuda",
                    ]
                )

    def test_cli_validates_working_directory_and_reproducibility_environment(self):
        self.assertEqual(_validated_working_directory(REPOSITORY_ROOT), ".")
        with self.assertRaisesRegex(BaselineError, "repository root"):
            _validated_working_directory(self.repo_root)

        expected_cache = str(REPOSITORY_ROOT / ".cache/modelscope")
        source = {
            "MODELSCOPE_CACHE": expected_cache,
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "4",
            "MODELSCOPE_API_TOKEN": "must-not-be-captured",
        }
        captured = _validated_command_environment(source)
        self.assertEqual(captured["MODELSCOPE_CACHE"], expected_cache)
        self.assertEqual(captured["PYTHONHASHSEED"], "0")
        self.assertNotIn("MODELSCOPE_API_TOKEN", captured)
        self.assertEqual(command_environment(source), captured)

        with self.assertRaisesRegex(BaselineError, "MODELSCOPE_CACHE"):
            _validated_command_environment({"PYTHONHASHSEED": "0"})
        with self.assertRaisesRegex(BaselineError, "PYTHONHASHSEED"):
            _validated_command_environment(
                {"MODELSCOPE_CACHE": expected_cache, "PYTHONHASHSEED": "random"}
            )

    def test_model_directory_hash_matches_sorted_inventory_contract(self):
        bundle = self.repo_root / "model-bundle"
        (bundle / "nested").mkdir(parents=True)
        (bundle / "z.bin").write_bytes(b"last")
        (bundle / "nested/a.txt").write_bytes(b"first")

        first_hash = hashlib.sha256(b"first").hexdigest()
        last_hash = hashlib.sha256(b"last").hexdigest()
        inventory = (
            f"{first_hash}  nested/a.txt\n"
            f"{last_hash}  z.bin\n"
        ).encode("utf-8")

        self.assertEqual(model_directory_sha256(bundle), sha256_bytes(inventory))

    def test_model_directory_hash_rejects_inventory_line_injection(self):
        bundle = self.repo_root / "ambiguous-model-bundle"
        bundle.mkdir()
        second_digest = hashlib.sha256(b"second").hexdigest()
        injected_name = f"a\n{second_digest}  b"
        (bundle / injected_name).write_bytes(b"first")

        with self.assertRaisesRegex(
            BaselineExecutionError, "ASCII control characters"
        ):
            model_directory_sha256(bundle)

    def test_loads_and_hashes_a_valid_frozen_manifest(self):
        relative, audio_path, duration = self.write_wav("valid.wav")
        record = self.record("utt-valid", relative, audio_path, duration, "测试文本")
        manifest_path, payload = self.write_manifest([record])

        dataset = load_frozen_dataset(manifest_path, self.repo_root)

        self.assertEqual(dataset.manifest_sha256, sha256_bytes(payload))
        self.assertEqual(dataset.data_version, "smoke-v0.1")
        self.assertEqual(dataset.items[0].audio_path, audio_path)
        self.assertEqual(dataset.items[0].duration_seconds, duration)

    def test_rejects_audio_hash_sample_rate_and_channel_mismatches(self):
        relative, audio_path, duration = self.write_wav("identity.wav")
        base_record = self.record("utt-identity", relative, audio_path, duration, "身份")

        bad_hash = dict(base_record)
        bad_hash["audio_sha256"] = "sha256:" + "0" * 64
        manifest_path, _ = self.write_manifest([bad_hash], "bad-hash.jsonl")
        with self.assertRaisesRegex(DatasetValidationError, "audio_sha256 mismatch"):
            load_frozen_dataset(manifest_path, self.repo_root)

        bad_rate = dict(base_record)
        bad_rate["sample_rate"] = 8_000
        manifest_path, _ = self.write_manifest([bad_rate], "bad-rate.jsonl")
        with self.assertRaisesRegex(DatasetValidationError, "must be 16 kHz"):
            load_frozen_dataset(manifest_path, self.repo_root)

        stereo_relative, stereo_path, stereo_duration = self.write_wav(
            "stereo.wav", channels=2
        )
        stereo = self.record(
            "utt-stereo", stereo_relative, stereo_path, stereo_duration, "双声道"
        )
        manifest_path, _ = self.write_manifest([stereo], "stereo.jsonl")
        with self.assertRaisesRegex(DatasetValidationError, "must be mono"):
            load_frozen_dataset(manifest_path, self.repo_root)

    def test_rejects_duplicate_ids_and_reference_hash_mismatch(self):
        relative, audio_path, duration = self.write_wav("duplicate.wav")
        record = self.record("same", relative, audio_path, duration, "原文")
        duplicate_path, _ = self.write_manifest([record, record], "duplicate.jsonl")
        with self.assertRaisesRegex(DatasetValidationError, "duplicate utterance id"):
            load_frozen_dataset(duplicate_path, self.repo_root)

        bad_reference = dict(record)
        bad_reference["raw_text"] = "被修改"
        mismatch_path, _ = self.write_manifest([bad_reference], "bad-reference.jsonl")
        with self.assertRaisesRegex(DatasetValidationError, "reference_sha256"):
            load_frozen_dataset(mismatch_path, self.repo_root)

    def test_cer_components_report_substitution_deletion_and_insertion(self):
        substitution = cer_components("测试", "测验")
        deletion = cer_components("你好世界", "你好世")
        insertion = cer_components("", "啊")

        self.assertEqual(substitution.substitutions, 1)
        self.assertEqual(substitution.total, 1)
        self.assertEqual(deletion.deletions, 1)
        self.assertEqual(deletion.total, 1)
        self.assertEqual(insertion.insertions, 1)
        self.assertEqual(insertion.total, 1)

    def test_sensevoice_views_remove_control_tags_without_display_emoji(self):
        raw = "<|zh|><|NEUTRAL|><|Speech|>你好，World！<|woitn|>"

        views = text_views("sensevoice", raw)

        self.assertEqual(views["raw"], raw)
        self.assertEqual(views["display"], "你好，World！")
        self.assertEqual(views["content"], "你好world")

    def test_runner_records_timing_cer_hashes_and_explicit_model_kwargs(self):
        dataset = self.make_dataset(["你好世界", "测试"])
        hypotheses = {"utt-1.wav": "你好世", "utt-2.wav": "测试啊"}
        captured_kwargs = {}

        class FakeModel:
            def __init__(self):
                self.calls = 0
                self.model_path = Path("/resolved/mock-paraformer")

            def generate(self, *, input, **kwargs):
                self.calls += 1
                self.last_generate_kwargs = kwargs
                return [{"text": hypotheses[Path(input).name]}]

        fake_model = FakeModel()
        hashed_paths = []

        def model_factory(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_model

        config = BaselineConfig(
            track="paraformer",
            model_revision="v2.0.4",
            ncpu=6,
            warmup_runs=1,
        )
        clock = SequenceClock([0.0, 2.0, 2.0, 2.5, 3.0, 3.02, 4.0, 4.04])

        report = run_offline_baseline(
            dataset,
            config,
            model_factory=model_factory,
            clock=clock,
            rss_reader=lambda: 512.0,
            model_bundle_hasher=lambda path: (
                hashed_paths.append(path) or MOCK_MODEL_SHA256
            ),
            command={"working_directory": ".", "argv": ["python", "runner"], "environment": {}},
        )

        self.assertEqual(captured_kwargs["hub"], "ms")
        self.assertEqual(
            captured_kwargs["model"],
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        self.assertEqual(captured_kwargs["model_revision"], "v2.0.4")
        self.assertFalse(captured_kwargs["check_latest"])
        self.assertEqual(captured_kwargs["seed"], 0)
        self.assertEqual(captured_kwargs["ngpu"], 0)
        self.assertEqual(captured_kwargs["batch_size"], 1)
        self.assertFalse(captured_kwargs["fp16"])
        self.assertFalse(captured_kwargs["bf16"])
        self.assertFalse(captured_kwargs["trust_remote_code"])
        self.assertIsNone(captured_kwargs["output_dir"])
        self.assertEqual(captured_kwargs["lm_weight"], 0.0)
        self.assertIsNone(captured_kwargs["lm_file"])
        self.assertEqual(captured_kwargs["token_lists"], [])
        self.assertEqual(captured_kwargs["seg_dicts"], [])
        self.assertIsNone(captured_kwargs["vad_model"])
        self.assertIsNone(captured_kwargs["punc_model"])
        self.assertIsNone(captured_kwargs["spk_model"])
        self.assertEqual(fake_model.calls, 4)
        self.assertEqual(hashed_paths, [Path("/resolved/mock-paraformer")])

        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["content_cer"], 2 / 6)
        self.assertEqual(metrics["substitutions"], 0)
        self.assertEqual(metrics["deletions"], 1)
        self.assertEqual(metrics["insertions"], 1)
        self.assertEqual(metrics["reference_units"], 6)
        self.assertEqual(metrics["failed_count"], 0)
        self.assertEqual(metrics["excluded_count"], 0)
        self.assertEqual(metrics["retried_count"], 0)
        self.assertEqual(metrics["rtf_attempted_count"], 2)
        self.assertEqual(metrics["rtf_successful_count"], 2)
        self.assertEqual(metrics["model_load_seconds"], 2.0)
        self.assertEqual(metrics["cold_inference_seconds"], 0.5)
        self.assertEqual(metrics["cold_start_seconds"], 2.5)
        self.assertEqual(metrics["warm_wall_seconds"], 0.06)
        self.assertEqual(metrics["warm_audio_seconds"], 0.2)
        self.assertEqual(metrics["rtf_p50"], 0.3)
        self.assertEqual(metrics["rtf_p95"], 0.39)
        self.assertEqual(metrics["successful_rtf_p50"], 0.3)
        self.assertEqual(metrics["successful_rtf_p95"], 0.39)
        self.assertEqual(metrics["peak_rss_mb"], 512.0)

        expected_config_hash = sha256_bytes(
            canonical_json_bytes(effective_config(config))
        )
        self.assertEqual(report["provenance"]["config_sha256"], expected_config_hash)
        self.assertEqual(
            report["provenance"]["data_sha256"], dataset.manifest_sha256
        )
        self.assertEqual(report["provenance"]["model"]["sha256"], MOCK_MODEL_SHA256)
        self.assertEqual(report["configuration"]["runtime"]["seed"], 0)
        self.assertEqual(
            report["configuration"]["runtime"]["rtf_population"],
            "all_warm_attempts",
        )

    def test_runner_preserves_failed_items_in_cer_denominator(self):
        dataset = self.make_dataset(["好", "坏"])

        class PartiallyFailingModel:
            model_path = Path("/resolved/mock-paraformer")

            def generate(self, *, input, **kwargs):
                if Path(input).name == "utt-2.wav":
                    raise RuntimeError("synthetic decoder failure")
                return [{"text": "好"}]

        report = run_offline_baseline(
            dataset,
            BaselineConfig(
                track="paraformer",
                model_revision="v2.0.4",
                warmup_runs=0,
            ),
            model_factory=lambda **kwargs: PartiallyFailingModel(),
            clock=SequenceClock([0.0, 1.0, 1.0, 1.1, 2.0, 2.01, 3.0, 3.02]),
            rss_reader=lambda: 256.0,
            model_bundle_hasher=lambda path: MOCK_MODEL_SHA256,
        )

        self.assertEqual(report["metrics"]["failed_count"], 1)
        self.assertEqual(report["metrics"]["deletions"], 1)
        self.assertEqual(report["metrics"]["reference_units"], 2)
        self.assertEqual(report["metrics"]["content_cer"], 0.5)
        self.assertEqual(report["metrics"]["rtf_attempted_count"], 2)
        self.assertEqual(report["metrics"]["rtf_successful_count"], 1)
        self.assertEqual(report["metrics"]["rtf_p50"], 0.15)
        self.assertEqual(report["metrics"]["rtf_p95"], 0.195)
        self.assertEqual(report["metrics"]["successful_rtf_p50"], 0.1)
        self.assertEqual(report["metrics"]["successful_rtf_p95"], 0.1)
        self.assertEqual(report["items"][1]["status"], "failed")
        self.assertIn("synthetic decoder failure", report["items"][1]["failure_reason"])

    def test_runner_scores_clean_sensevoice_content_separately_from_raw(self):
        dataset = self.make_dataset(["你好"])
        raw = "<|zh|><|NEUTRAL|><|Speech|>你好<|woitn|>"

        class FakeSenseVoice:
            model_path = Path("/resolved/mock-sensevoice")

            def generate(self, *, input, **kwargs):
                return [{"text": raw}]

        report = run_offline_baseline(
            dataset,
            BaselineConfig(
                track="sensevoice",
                model_revision="v1.0.0",
                warmup_runs=0,
            ),
            model_factory=lambda **kwargs: FakeSenseVoice(),
            clock=SequenceClock([0.0, 1.0, 1.0, 1.1, 2.0, 2.01]),
            rss_reader=lambda: 384.0,
            model_bundle_hasher=lambda path: MOCK_MODEL_SHA256,
        )

        hypothesis = report["items"][0]["hypothesis"]
        self.assertEqual(hypothesis["raw"], raw)
        self.assertEqual(hypothesis["display"], "你好")
        self.assertEqual(hypothesis["content"], "你好")
        self.assertEqual(report["metrics"]["content_cer"], 0.0)

    def test_canonical_report_hash_is_exact_and_existing_report_is_not_overwritten(self):
        report = {"z": [2, 1], "a": "中文"}
        reordered = {"a": "中文", "z": [2, 1]}
        self.assertEqual(canonical_json_bytes(report), canonical_json_bytes(reordered))

        output_path = self.repo_root / "eval/reports/baseline.json"
        digest = write_report(report, output_path)
        payload = output_path.read_bytes()

        self.assertEqual(digest, "sha256:" + hashlib.sha256(payload).hexdigest())
        self.assertEqual(payload, canonical_json_bytes(report))
        with self.assertRaisesRegex(BaselineExecutionError, "refusing to overwrite"):
            write_report(report, output_path)


if __name__ == "__main__":
    unittest.main()
