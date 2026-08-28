import copy
import argparse
import io
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import eval.sealed_decoder as sealed_decoder_module
from eval.collection import SEALED_INPUT_PROJECTION_KIND
from eval.core_report import IDENTITY_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
from eval.custodian_replay import CANDIDATE_LOCK_KIND
from eval.custodian_replay import CANDIDATE_LOCK_SCHEMA_VERSION
from eval.custodian_replay import INPUT_EXPORT_RECEIPT_KIND
from eval.custodian_replay import RECEIPT_SCHEMA_VERSION
from eval.custodian_replay import CustodianReplayError
from eval.custodian_replay import candidate_freeze_sha256
from eval.custodian_replay import canonical_candidate_lock_bytes
from eval.custodian_replay import canonical_custodian_receipt_bytes
from eval.custodian_replay import parse_sealed_input_projection
from eval.normalizers import NORMALIZER_VERSION
from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import TRACKS
from eval.offline_baseline import canonical_json_bytes
from eval.offline_baseline import effective_config
from eval.offline_baseline import sha256_bytes
from eval.offline_baseline import sha256_file
from eval.offline_baseline import model_directory_sha256
from eval.sealed_decoder import RUNNER_SOURCE_PATHS
from eval.sealed_decoder import SEALED_PYCACHE_PREFIX
from eval.sealed_decoder import SealedDecoderConfig
from eval.sealed_decoder import SealedDecoderError
from eval.sealed_decoder import describe_runtime
from eval.sealed_decoder import default_model_factory
from eval.sealed_decoder import execute_sealed_candidate
from eval.sealed_decoder import raw_prediction_jsonl_bytes
from eval.sealed_decoder import resolve_local_model_snapshot
from eval.sealed_decoder import runner_source_identity
from eval.sealed_candidate_contract import TRACK_COMPONENT_PROFILES
from eval.execution_envelope import load_execution_envelope
from eval.execution_envelope import build_execution_envelope
from scripts.run_sealed_asr_candidate import _publish_result
from scripts.run_sealed_asr_candidate import _run
from scripts.run_sealed_asr_candidate import _silence_decoder_streams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv/bin/python"


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def registration_fields(experiment_id: str) -> dict[str, str]:
    return {
        "candidate_registration_commit": "d" * 40,
        "candidate_manifest_path": f"experiments/manifests/{experiment_id}.json",
        "candidate_manifest_sha256": digest(
            f"registered-candidate:{experiment_id}"
        ),
    }


class SequenceClock:
    def __init__(self, values: list[int]):
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


class SealedDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio_directory = self.root / "audio"
        self.audio_directory.mkdir()
        self.environment = {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "CRC32C_SW_MODE": "auto",
            "HYDRA_FULL_ERROR": "1",
            "KMP_DUPLICATE_LIB_OK": "True",
            "KMP_INIT_AT_FORK": "FALSE",
            "MODELSCOPE_CACHE": str(self.root / ".cache/modelscope"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.root / ".cache/torchinductor"),
            "MODELSCOPE_API_TOKEN": "must-not-be-captured",
        }
        self.hardware = {
            "host_id": "sealed-runner-fixture",
            "os": "Linux fixture x86_64",
            "cpu_model": "Fixture CPU",
            "logical_cpu_count": 4,
            "memory_bytes": 8_589_934_592,
            "device": "cpu",
            "accelerator": None,
        }
        self.actual_argv = [
            ".venv/bin/python",
            "-P",
            "-S",
            "scripts/run_sealed_asr_candidate.py",
            "run",
            "--input-projection",
            str(self.root / "sealed-input.json"),
            "--candidate-lock",
            str(self.root / "candidate-lock.json"),
            "--input-receipt",
            str(self.root / "input-receipt.json"),
            "--audio-root",
            str(self.root),
            "--track",
            "paraformer",
            "--model-revision",
            "1" * 40,
            "--device",
            "cpu",
            "--ncpu",
            "4",
            "--warmup-runs",
            "1",
            "--seed",
            "0",
            "--hypothesis-adapter-version",
            IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
            "--output-raw-predictions",
            str(self.root / "raw-predictions.jsonl"),
            "--output-execution-envelope",
            str(self.root / "execution-envelope.json"),
        ]
        self.code_commit = "e3710de87620d9a7ecb4b89a5d87142b6c1a1d3f"
        self.runner_source_sha256 = digest("runner-source-inventory")
        self.model_sha256 = digest("model-bundle-inventory")
        self.model_revision = "1" * 40
        self.model_path = self.root / "model-snapshot"
        self.model_path.mkdir()
        (self.model_path / "model.bin").write_bytes(b"pinned model fixture")
        self.write_model_config("paraformer")
        (self.model_path / "model.pt").write_bytes(b"pinned weights fixture")
        self.runtime = {
            "python_implementation": "cpython",
            "python_version": "3.11.15",
            "python_cache_tag": "cpython-311",
            "dependency_lock_sha256": digest("lab-cpu-lock"),
            "installed_dependencies_sha256": digest("installed-dependencies"),
            "installed_dependency_count": 71,
            "unicode_version": "14.0.0",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bounded_reader_rejects_atomic_path_replacement(self):
        path = self.root / "runner-source.py"
        replacement = self.root / "replacement.py"
        path.write_bytes(b"committed runner source\n")
        replacement.write_bytes(b"replacement runner source\n")
        real_read = os.read
        replaced = False

        def replace_path(descriptor, maximum):
            nonlocal replaced
            if not replaced:
                replacement.replace(path)
                replaced = True
            return real_read(descriptor, maximum)

        with mock.patch(
            "eval.sealed_decoder.os.read",
            side_effect=replace_path,
        ):
            with self.assertRaisesRegex(
                SealedDecoderError, "path changed while it was read"
            ):
                sealed_decoder_module._read_bounded_regular_file(
                    path, 1024, "runner source fixture"
                )

    def test_verified_audio_rejects_leaf_rebound_during_first_read(self):
        config = self.config(warmup_runs=0)
        audio_path, duration_seconds = self.write_wav("leaf-rebound.wav")
        input_path, _, _, _ = self.write_contract(
            config,
            audio_specs=[(audio_path, duration_seconds)],
        )
        sealed_input = parse_sealed_input_projection(input_path.read_bytes())
        replacement = self.audio_directory / "leaf-replacement.wav"
        replacement.write_bytes(audio_path.read_bytes())
        real_read = os.read
        replaced = False

        def replace_leaf(descriptor, maximum):
            nonlocal replaced
            chunk = real_read(descriptor, maximum)
            if not replaced:
                replacement.replace(audio_path)
                replaced = True
            return chunk

        with mock.patch(
            "eval.sealed_decoder.os.read",
            side_effect=replace_leaf,
        ):
            with self.assertRaisesRegex(
                SealedDecoderError, "audio path changed while it was verified"
            ):
                sealed_decoder_module.load_verified_audio_items(
                    sealed_input,
                    self.root,
                )
        self.assertTrue(replaced)

    def test_verified_audio_rejects_parent_rebound_during_first_read(self):
        config = self.config(warmup_runs=0)
        audio_parent = self.audio_directory / "opened-parent"
        audio_parent.mkdir()
        audio_path, duration_seconds = self.write_wav(
            "opened-parent/parent-rebound.wav"
        )
        input_path, _, _, _ = self.write_contract(
            config,
            audio_specs=[(audio_path, duration_seconds)],
        )
        sealed_input = parse_sealed_input_projection(input_path.read_bytes())
        replacement_parent = self.audio_directory / "replacement-parent"
        replacement_parent.mkdir()
        (replacement_parent / audio_path.name).write_bytes(audio_path.read_bytes())
        displaced_parent = self.audio_directory / "displaced-parent"
        real_read = os.read
        replaced = False

        def replace_parent(descriptor, maximum):
            nonlocal replaced
            chunk = real_read(descriptor, maximum)
            if not replaced:
                audio_parent.rename(displaced_parent)
                replacement_parent.rename(audio_parent)
                replaced = True
            return chunk

        with mock.patch(
            "eval.sealed_decoder.os.read",
            side_effect=replace_parent,
        ):
            with self.assertRaisesRegex(
                SealedDecoderError, "audio path changed while it was verified"
            ):
                sealed_decoder_module.load_verified_audio_items(
                    sealed_input,
                    self.root,
                )
        self.assertTrue(replaced)

    def sealed_subprocess(
        self,
        script: str,
        *arguments: str,
        python_flags: tuple[str, ...] = ("-P", "-S"),
        environment_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/dev/null",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
        }
        if script == "scripts/run_sealed_asr_candidate.py":
            environment.update(
                {
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "NUMEXPR_NUM_THREADS": "2",
                    "CRC32C_SW_MODE": "auto",
                    "HYDRA_FULL_ERROR": "1",
                    "KMP_DUPLICATE_LIB_OK": "True",
                    "KMP_INIT_AT_FORK": "FALSE",
                    "MODELSCOPE_CACHE": str(
                        PROJECT_ROOT / ".cache/modelscope"
                    ),
                    "TORCHINDUCTOR_CACHE_DIR": str(
                        PROJECT_ROOT / ".cache/torchinductor"
                    ),
                }
            )
        if environment_overrides:
            environment.update(environment_overrides)
        return subprocess.run(
            [str(VENV_PYTHON), *python_flags, script, *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def write_wav(
        self,
        name: str,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        sample_width: int = 2,
        frames: int = 1_600,
    ) -> tuple[Path, float]:
        path = self.audio_directory / name
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00" * frames * channels * sample_width)
        return path, frames / sample_rate

    def write_model_config(self, track: str) -> None:
        profile = TRACK_COMPONENT_PROFILES[track]
        payload = "".join(
            f"{name}: {'null' if value is None else value}\n"
            for name, value in profile.items()
        )
        (self.model_path / "config.yaml").write_text(payload, encoding="utf-8")

    def config(
        self,
        *,
        track: str = "paraformer",
        revision: str | None = None,
        warmup_runs: int = 1,
    ) -> SealedDecoderConfig:
        adapter = (
            IDENTITY_HYPOTHESIS_ADAPTER_VERSION
            if track == "paraformer"
            else SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
        )
        return SealedDecoderConfig(
            track=track,
            model_revision=self.model_revision if revision is None else revision,
            device="cpu",
            ncpu=4,
            warmup_runs=warmup_runs,
            seed=0,
            hypothesis_adapter_version=adapter,
        )

    def command_argv(
        self,
        config: SealedDecoderConfig,
        base: list[str] | None = None,
    ) -> list[str]:
        argv = list(self.actual_argv if base is None else base)
        frozen_options = {
            "--track": config.track,
            "--model-revision": config.model_revision,
            "--device": config.device,
            "--ncpu": str(config.ncpu),
            "--warmup-runs": str(config.warmup_runs),
            "--seed": str(config.seed),
            "--hypothesis-adapter-version": config.hypothesis_adapter_version,
        }
        for option, value in frozen_options.items():
            positions = [index for index, argument in enumerate(argv) if argument == option]
            if len(positions) == 1 and positions[0] + 1 < len(argv):
                argv[positions[0] + 1] = value
        return argv

    def write_contract(
        self,
        config: SealedDecoderConfig,
        *,
        audio_specs: list[tuple[Path, float]] | None = None,
        actual_argv: list[str] | None = None,
        candidate_overrides: dict[str, object] | None = None,
        lock_overrides: dict[str, object] | None = None,
    ) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
        self.write_model_config(config.track)
        if audio_specs is None:
            audio_specs = [self.write_wav("one.wav"), self.write_wav("two.wav")]
        items = []
        for index, (path, duration) in enumerate(audio_specs, start=1):
            items.append(
                {
                    "id": f"sealed-{index}",
                    "split": "sealed-blind",
                    "audio": path.relative_to(self.root).as_posix(),
                    "audio_sha256": sha256_file(path),
                    "duration_seconds": duration,
                    "sample_rate": 16_000,
                    "channels": 1,
                    "sample_width_bits": 16,
                }
            )
        input_document: dict[str, object] = {
            "schema_version": 2,
            "kind": SEALED_INPUT_PROJECTION_KIND,
            "dataset_id": "LAB-SEED-TEST",
            "revision": "v0.1",
            "split": "sealed-blind",
            "manifest_sha256": digest("sealed-manifest"),
            "manifest_record_count": len(items),
            "item_count": len(items),
            "items": items,
        }
        input_payload = canonical_json_bytes(input_document)
        sealed_input = parse_sealed_input_projection(input_payload)
        input_path = self.root / "sealed-input.json"
        input_path.write_bytes(input_payload)

        baseline_config = BaselineConfig(
            track=config.track,
            model_revision=config.model_revision,
            device=config.device,
            ncpu=config.ncpu,
            warmup_runs=config.warmup_runs,
            seed=config.seed,
        )
        command_argv = self.command_argv(config, actual_argv)
        candidate: dict[str, object] = {
            "schema_version": 1,
            "experiment_id": "EXP-20260828-001-sealed-runner",
            "task_id": "EVAL-01",
            "hypothesis": (
                "The committed sealed runner produces measured predictions "
                "without opening references."
            ),
            "upstream_commit": "eedd4e22d10dc2e81d9c2bb321edb3750253964b",
            "code_commit": self.code_commit,
            "models": [
                {
                    "role": "asr",
                    "identifier": TRACKS[config.track].model_identifier,
                    "revision": config.model_revision,
                    "sha256": self.model_sha256,
                }
            ],
            "config_sha256": sha256_bytes(
                canonical_json_bytes(effective_config(baseline_config))
            ),
            "data_sha256": digest("collection-descriptor"),
            "eval_data_version": "LAB-SEED-TEST-v0.1",
            "normalizer_version": NORMALIZER_VERSION,
            "hardware": copy.deepcopy(self.hardware),
            "seed": 0,
            "command": {
                "working_directory": ".",
                "argv": command_argv,
                "environment": {
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                    "CRC32C_SW_MODE": "auto",
                    "HYDRA_FULL_ERROR": "1",
                    "KMP_DUPLICATE_LIB_OK": "True",
                    "KMP_INIT_AT_FORK": "FALSE",
                    "MODELSCOPE_CACHE": str(self.root / ".cache/modelscope"),
                    "PYTHONHASHSEED": "0",
                    "TORCHINDUCTOR_CACHE_DIR": str(
                        self.root / ".cache/torchinductor"
                    ),
                },
            },
        }
        if candidate_overrides:
            candidate.update(candidate_overrides)
        lock: dict[str, object] = {
            "schema_version": CANDIDATE_LOCK_SCHEMA_VERSION,
            "kind": CANDIDATE_LOCK_KIND,
            "state": "frozen",
            "access_class": "restricted",
            "dataset_id": input_document["dataset_id"],
            "revision": input_document["revision"],
            "split": "sealed-blind",
            "data_sha256": candidate["data_sha256"],
            "input_projection_sha256": sealed_input.sha256,
            "hypothesis_adapter_version": config.hypothesis_adapter_version,
            "record_identity_version": "eval-core-record-input-v1",
            "record_input_sha256": digest("restricted-record-input"),
            "decode_item_count": len(items),
            "decode_item_ids_sha256": sha256_bytes(
                canonical_json_bytes([item["id"] for item in items])
            ),
            "source_manifest_decision": "planned",
            **registration_fields(candidate["experiment_id"]),
            "candidate": candidate,
            "candidate_freeze_sha256": candidate_freeze_sha256(candidate),
        }
        if lock_overrides:
            lock.update(lock_overrides)
        lock_path = self.root / "candidate-lock.json"
        lock_path.write_bytes(canonical_candidate_lock_bytes(lock))
        return input_path, lock_path, input_document, lock

    def execute(
        self,
        input_path: Path,
        lock_path: Path,
        config: SealedDecoderConfig,
        *,
        model_factory,
        clock,
        environment: dict[str, str] | None = None,
        hardware: dict[str, object] | None = None,
        source_identity=None,
        model_sha256: str | None = None,
        model_hasher=None,
        model_snapshot_resolver=None,
        runtime_identity_reader=None,
        rss_reader=None,
    ):
        def bound_model_factory(**kwargs):
            model = model_factory(**kwargs)
            if model is not None:
                model.model_path = self.model_path
            return model

        return execute_sealed_candidate(
            input_path,
            lock_path,
            self.root,
            config,
            actual_argv=self.command_argv(config),
            environment_source=self.environment if environment is None else environment,
            current_directory=self.root,
            repository_root=self.root,
            hardware_reader=lambda device: (
                self.hardware if hardware is None else hardware
            ),
            source_identity_reader=(
                (
                    lambda code_commit, root: (
                        code_commit,
                        self.runner_source_sha256,
                    )
                )
                if source_identity is None
                else source_identity
            ),
            runtime_identity_reader=(
                (lambda root: self.runtime)
                if runtime_identity_reader is None
                else runtime_identity_reader
            ),
            model_snapshot_resolver=(
                (
                    lambda root, identifier, revision, expected_sha256, hasher: (
                        self.model_path
                    )
                )
                if model_snapshot_resolver is None
                else model_snapshot_resolver
            ),
            model_factory=bound_model_factory,
            model_bundle_hasher=(
                (
                    lambda path: (
                        self.model_sha256 if model_sha256 is None else model_sha256
                    )
                )
                if model_hasher is None
                else model_hasher
            ),
            clock=clock,
            rss_reader=(
                (lambda: 512 * 1024 * 1024)
                if rss_reader is None
                else rss_reader
            ),
        )

    def test_runs_cold_warmup_and_all_measured_attempts_without_references(self):
        config = self.config()
        input_path, lock_path, _, _ = self.write_contract(config)
        captured_kwargs: dict[str, object] = {}

        class FakeModel:
            model_path = Path("/fake/model")

            def __init__(self):
                self.calls = 0

            def generate(self, *, input, **kwargs):
                self.calls += 1
                if self.calls == 4:
                    raise RuntimeError("sensitive decoder detail")
                return [{"text": "你好"}]

        model = FakeModel()

        def factory(**kwargs):
            captured_kwargs.update(kwargs)
            return model

        result = self.execute(
            input_path,
            lock_path,
            config,
            model_factory=factory,
            clock=SequenceClock([0, 10, 20, 30, 40, 50, 60, 70, 80, 100]),
        )

        self.assertEqual(model.calls, 4)
        self.assertEqual(captured_kwargs["model"], str(self.model_path))
        self.assertEqual(captured_kwargs["model_revision"], self.model_revision)
        self.assertEqual(captured_kwargs["device"], "cpu")
        self.assertEqual(captured_kwargs["seed"], 0)
        self.assertFalse(captured_kwargs["check_latest"])
        self.assertEqual(captured_kwargs["model_path"], str(self.model_path))
        self.assertEqual(
            captured_kwargs["config"],
            str(self.model_path / "config.yaml"),
        )
        self.assertEqual(
            captured_kwargs["init_param"],
            str(self.model_path / "model.pt"),
        )
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
        self.assertEqual(captured_kwargs["tokenizer"], "CharTokenizer")
        self.assertEqual(captured_kwargs["frontend"], "WavFrontend")
        self.assertIsNone(captured_kwargs["vad_model"])
        self.assertIsNone(captured_kwargs["punc_model"])
        self.assertIsNone(captured_kwargs["spk_model"])

        self.assertEqual(
            [item["status"] for item in result.prediction_items],
            ["ok", "failed"],
        )
        self.assertEqual(result.prediction_items[1]["raw_text"], "")
        self.assertEqual(
            result.prediction_items[1]["reason_code"], "decoder_exception"
        )
        observation = result.observation
        self.assertEqual(observation["model_load_ns"], 10)
        self.assertEqual(observation["cold_attempt"]["elapsed_ns"], 10)
        self.assertEqual(len(observation["warmup_attempts"]), 1)
        self.assertEqual(
            [attempt["elapsed_ns"] for attempt in observation["decode_attempts"]],
            [10, 20],
        )
        self.assertEqual(
            [attempt["status"] for attempt in observation["decode_attempts"]],
            ["ok", "failed"],
        )
        self.assertEqual(observation["peak_rss_bytes"], 512 * 1024 * 1024)
        self.assertEqual(observation["runtime"], self.runtime)
        encoded_observation = json.dumps(observation, ensure_ascii=False)
        self.assertNotIn("你好", encoded_observation)
        self.assertNotIn("sensitive decoder detail", encoded_observation)
        self.assertNotIn("raw_text", encoded_observation)

        raw_payload = raw_prediction_jsonl_bytes(result.prediction_items)
        raw_documents = [json.loads(line) for line in raw_payload.splitlines()]
        self.assertTrue(raw_documents)
        for document in raw_documents:
            self.assertEqual(
                set(document), {"id", "raw_text", "status", "reason_code"}
            )
        self.assertEqual(
            observation["raw_predictions_sha256"], sha256_bytes(raw_payload)
        )
        self.assertEqual(
            observation["prediction_items_sha256"],
            sha256_bytes(canonical_json_bytes(list(result.prediction_items))),
        )

    def test_pinned_model_bundle_rejects_wrong_or_ambiguous_model_class(self):
        config = self.config(warmup_runs=0)
        wrong_encoder = (self.model_path / "config.yaml").read_text(
            encoding="utf-8"
        ).replace("encoder: SANMEncoder", "encoder: OpenAIWhisperEncoderWarp")
        external_normalizer = (self.model_path / "config.yaml").read_text(
            encoding="utf-8"
        ).replace("normalize: null", "normalize: GlobalMVN")
        cases = (
            ("model: SenseVoiceSmall\n", "does not match"),
            ("model: Paraformer\nmodel: SenseVoiceSmall\n", "duplicate"),
            (wrong_encoder, "selector 'encoder'"),
            (external_normalizer, "selector 'normalize'"),
            (
                "defaults: &defaults {model: Paraformer}\n<<: *defaults\n",
                "merge keys",
            ),
        )
        for payload, message in cases:
            with self.subTest(payload=payload):
                (self.model_path / "config.yaml").write_text(
                    payload,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SealedDecoderError, message):
                    sealed_decoder_module._validate_pinned_model_bundle_contract(
                        config,
                        self.model_path,
                    )

    def test_constructed_model_contract_rejects_aux_precision_and_external_resource(self):
        config = self.config(warmup_runs=0)

        class FakeTensor:
            def __init__(self, dtype: str = "torch.float32"):
                self.dtype = dtype

            def is_floating_point(self):
                return True

        Paraformer = type(
            "Paraformer",
            (),
            {
                "parameters": lambda model_self: iter([FakeTensor()]),
                "buffers": lambda model_self: iter(()),
            },
        )

        def component(name):
            return type(name, (), {})()

        def primary(dtype: str = "torch.float32"):
            model = Paraformer()
            model.parameters = lambda: iter([FakeTensor(dtype)])
            model.buffers = lambda: iter(())
            model.specaug = component("SpecAugLFR")
            model.normalize = None
            model.encoder = component("SANMEncoder")
            model.decoder = component("ParaformerSANMDecoder")
            model.predictor = component("CifPredictorV2")
            return model

        exact_kwargs = {
            "model": "Paraformer",
            "model_path": str(self.model_path),
            "config": str(self.model_path / "config.yaml"),
            "init_param": str(self.model_path / "model.pt"),
            "device": "cpu",
            "ngpu": 0,
            "batch_size": 1,
            "fp16": False,
            "bf16": False,
            "trust_remote_code": False,
            "output_dir": None,
            "lm_weight": 0.0,
            "lm_file": None,
            "token_lists": [],
            "seg_dicts": [],
            "vad_model": None,
            "punc_model": None,
            "spk_model": None,
            "tokenizer_conf": {
                "token_list": str(self.model_path / "config.yaml"),
                "non_linguistic_symbols": None,
            },
            "tokenizer": component("CharTokenizer"),
            "frontend": component("WavFrontend"),
        }

        def wrapped(**overrides):
            values = copy.deepcopy(exact_kwargs)
            values.update(overrides)
            return types.SimpleNamespace(
                kwargs=values,
                model=primary(),
                vad_model=None,
                punc_model=None,
                spk_model=None,
            )

        sealed_decoder_module._validate_constructed_model_contract(
            wrapped(),
            config,
            self.model_path,
        )
        cases = (
            (wrapped(fp16=True), "runtime contract"),
            (wrapped(output_dir=str(self.root)), "runtime contract"),
            (
                wrapped(token_lists=[str(self.root / "outside-token-list")]),
                "runtime contract",
            ),
            (wrapped(remote_code="outside.py"), "remote code"),
            (
                wrapped(
                    tokenizer_conf={
                        "init_param_path": str(self.root / "external-tokenizer")
                    }
                ),
                "unavailable",
            ),
            (
                wrapped(
                    tokenizer_conf={
                        "token_list": str(self.model_path / "config.yaml"),
                        "non_linguistic_symbols": str(
                            self.root / "external-symbols"
                        ),
                    }
                ),
                "tokenizer resolved outside",
            ),
            (
                wrapped(
                    tokenizer_conf={
                        "token_list": str(self.root / "outside-token-list")
                    }
                ),
                "unavailable",
            ),
        )
        aux = wrapped()
        aux.vad_model = object()
        cases += ((aux, "auxiliary"),)
        low_precision = wrapped()
        low_precision.model = primary("torch.float16")
        cases += ((low_precision, "FP32"),)
        for model, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SealedDecoderError, message):
                    sealed_decoder_module._validate_constructed_model_contract(
                        model,
                        config,
                        self.model_path,
                    )

    def test_peak_rss_is_sampled_before_post_measurement_model_inventory(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("rss-scope.wav")],
        )
        events: list[str] = []

        class FakeModel:
            model_path = self.model_path

            def generate(self, *, input, **kwargs):
                return [{"text": "范围"}]

        def model_hasher(path):
            events.append("model-hash")
            return self.model_sha256

        def rss_reader():
            events.append("rss")
            return 512 * 1024 * 1024

        self.execute(
            input_path,
            lock_path,
            config,
            model_factory=lambda **kwargs: FakeModel(),
            clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            model_hasher=model_hasher,
            rss_reader=rss_reader,
        )

        self.assertEqual(
            events,
            ["model-hash", "model-hash", "rss", "model-hash"],
        )

    def test_sensevoice_tag_only_output_is_frozen_as_empty(self):
        config = self.config(track="sensevoice", revision="2" * 40, warmup_runs=0)
        self.actual_argv[-1] = SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
        one_audio = [self.write_wav("sensevoice.wav")]
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=one_audio,
        )

        class FakeSenseVoice:
            model_path = Path("/fake/sensevoice")

            def generate(self, *, input, **kwargs):
                return [{"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>"}]

        result = self.execute(
            input_path,
            lock_path,
            config,
            model_factory=lambda **kwargs: FakeSenseVoice(),
            clock=SequenceClock([0, 5, 10, 20, 30, 45]),
        )
        self.assertEqual(result.prediction_items[0]["status"], "empty")
        self.assertEqual(
            result.prediction_items[0]["reason_code"], "empty_hypothesis"
        )
        self.assertEqual(result.observation["decode_attempts"][0]["elapsed_ns"], 15)

    def test_each_attempt_uses_same_verified_bytes_and_sanitized_environment(self):
        config = self.config(warmup_runs=0)
        audio_path, duration = self.write_wav("stable.wav")
        original_payload = audio_path.read_bytes()
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[(audio_path, duration)],
        )
        observed_input_hashes: list[str] = []
        observed_environments: list[dict[str, str]] = []
        observed_bytecode_flags: list[bool] = []
        observed_pycache_prefixes: list[str | None] = []
        original_pycache_prefix = sys.pycache_prefix

        class MutatingModel:
            model_path = self.model_path

            def generate(model_self, *, input, **kwargs):
                self.assertIsInstance(input, io.BytesIO)
                # Replace the path only after the runner read and verified this
                # attempt's descriptor. The model must still see those exact bytes.
                audio_path.write_bytes(b"unverified replacement")
                observed_input_hashes.append(sha256_bytes(input.read()))
                observed_environments.append(dict(os.environ))
                observed_bytecode_flags.append(sys.dont_write_bytecode)
                observed_pycache_prefixes.append(sys.pycache_prefix)
                audio_path.write_bytes(original_payload)
                return [{"text": "稳定"}]

        with mock.patch.dict(os.environ, {"HOST_SECRET": "must-be-restored"}):
            result = self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: MutatingModel(),
                clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            )
            self.assertEqual(os.environ["HOST_SECRET"], "must-be-restored")

        expected_hash = sha256_file(audio_path)
        self.assertEqual(observed_input_hashes, [expected_hash, expected_hash])
        expected_environment = {
            name: value
            for name, value in self.environment.items()
            if name != "MODELSCOPE_API_TOKEN"
        }
        self.assertEqual(observed_environments, [expected_environment] * 2)
        self.assertEqual(observed_bytecode_flags, [True, True])
        self.assertEqual(
            observed_pycache_prefixes,
            [SEALED_PYCACHE_PREFIX, SEALED_PYCACHE_PREFIX],
        )
        self.assertEqual(sys.pycache_prefix, original_pycache_prefix)
        self.assertEqual(result.prediction_items[0]["status"], "ok")

    def test_projection_duration_is_preserved_in_envelope_within_wav_tolerance(self):
        config = self.config(warmup_runs=0)
        audio_path, actual_duration = self.write_wav("duration.wav")
        frozen_duration = actual_duration + (0.5 / 16_000)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[(audio_path, frozen_duration)],
        )

        class FakeModel:
            model_path = self.model_path

            def generate(self, *, input, **kwargs):
                return [{"text": "时长"}]

        result = self.execute(
            input_path,
            lock_path,
            config,
            model_factory=lambda **kwargs: FakeModel(),
            clock=SequenceClock([0, 10, 20, 30, 40, 50]),
        )
        envelope = build_execution_envelope(
            result.observation,
            result.prediction_items,
            input_export_receipt_sha256=digest("input-export-receipt"),
        )
        self.assertEqual(
            result.observation["decode_attempts"][0]["audio_duration_seconds"],
            frozen_duration,
        )
        self.assertEqual(
            envelope["items"][0]["audio_duration_seconds"], frozen_duration
        )

    def test_cold_or_warmup_failure_refuses_performance_evidence(self):
        cases = ((0, 1, "cold inference failed"), (1, 2, "warmup inference failed"))
        for warmup_runs, fail_on_call, message in cases:
            with self.subTest(message=message):
                config = self.config(warmup_runs=warmup_runs)
                input_path, lock_path, _, _ = self.write_contract(
                    config,
                    audio_specs=[self.write_wav(f"failure-{warmup_runs}.wav")],
                )

                class FailingModel:
                    model_path = self.model_path

                    def __init__(model_self):
                        model_self.calls = 0

                    def generate(model_self, *, input, **kwargs):
                        model_self.calls += 1
                        if model_self.calls == fail_on_call:
                            raise RuntimeError("private decoder failure")
                        return [{"text": "预热"}]

                with self.assertRaisesRegex(SealedDecoderError, message):
                    self.execute(
                        input_path,
                        lock_path,
                        config,
                        model_factory=lambda **kwargs: FailingModel(),
                        clock=SequenceClock([0, 10, 20, 30, 40, 50]),
                    )

    def test_model_bundle_is_rechecked_after_measured_pass(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("model-recheck.wav")],
        )
        hashes = iter(
            [self.model_sha256, self.model_sha256, digest("mutated-model")]
        )

        class FakeModel:
            model_path = self.model_path

            def generate(self, *, input, **kwargs):
                return [{"text": "复核"}]

        with self.assertRaisesRegex(SealedDecoderError, "changed during"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: FakeModel(),
                model_hasher=lambda path: next(hashes),
                clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            )

    def test_runner_source_is_rechecked_after_measured_pass(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("source-recheck.wav")],
        )
        identities = iter(
            [
                (self.code_commit, self.runner_source_sha256),
                (self.code_commit, digest("changed-runner-source")),
            ]
        )

        class FakeModel:
            model_path = self.model_path

            def generate(self, *, input, **kwargs):
                return [{"text": "复核"}]

        with self.assertRaisesRegex(SealedDecoderError, "source identity changed"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: FakeModel(),
                source_identity=lambda commit, root: next(identities),
                clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            )

    def test_repository_model_shadow_is_rechecked_after_measured_pass(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("shadow-recheck.wav")],
        )

        class FakeModel:
            model_path = self.model_path

            def generate(self, *, input, **kwargs):
                return [{"text": "复核"}]

        def factory(**kwargs):
            (self.root / "numpy.py").write_text(
                "raise RuntimeError('late shadow')\n",
                encoding="utf-8",
            )
            return FakeModel()

        with self.assertRaisesRegex(SealedDecoderError, "model import shadow"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=factory,
                clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            )

    def test_runtime_and_sanitized_import_state_are_rechecked(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("runtime-recheck.wav")],
        )

        class FakeModel:
            model_path = self.model_path

            def generate(self, *, input, **kwargs):
                return [{"text": "复核"}]

        changed_runtime = {**self.runtime, "unicode_version": "15.0.0"}
        runtimes = iter([self.runtime, changed_runtime])
        with self.assertRaisesRegex(SealedDecoderError, "runtime identity changed"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: FakeModel(),
                runtime_identity_reader=lambda root: next(runtimes),
                clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            )

        original_environment = dict(os.environ)
        original_prefix = sys.pycache_prefix

        def mutating_factory(**kwargs):
            os.environ["OMP_NUM_THREADS"] = "999"
            sys.pycache_prefix = "/tmp/unsafe-sealed-prefix"
            return FakeModel()

        with self.assertRaisesRegex(SealedDecoderError, "environment changed"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=mutating_factory,
                clock=SequenceClock([0, 10, 20, 30, 40, 50]),
            )
        self.assertEqual(dict(os.environ), original_environment)
        self.assertEqual(sys.pycache_prefix, original_prefix)

    def test_model_prehash_mismatch_never_constructs_model(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("model-prehash.wav")],
        )
        calls: list[dict[str, object]] = []
        with self.assertRaisesRegex(SealedDecoderError, "pinned local model"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: calls.append(kwargs),
                model_sha256=digest("wrong-model"),
                clock=SequenceClock([]),
            )
        self.assertEqual(calls, [])

    def test_rejects_audio_hash_and_wav_identity_before_model_construction(self):
        config = self.config(warmup_runs=0)
        audio = self.write_wav("identity.wav")
        input_path, lock_path, _, _ = self.write_contract(config, audio_specs=[audio])
        audio[0].write_bytes(audio[0].read_bytes() + b"tamper")
        calls: list[dict[str, object]] = []

        with self.assertRaisesRegex(SealedDecoderError, "audio_sha256 mismatch"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: calls.append(kwargs),
                clock=SequenceClock([]),
            )
        self.assertEqual(calls, [])

        stereo = self.write_wav("stereo.wav", channels=2)
        input_path, lock_path, input_document, lock = self.write_contract(
            config,
            audio_specs=[stereo],
        )
        input_document["items"][0]["channels"] = 2
        input_payload = canonical_json_bytes(input_document)
        input_path.write_bytes(input_payload)
        lock["input_projection_sha256"] = sha256_bytes(input_payload)
        lock_path.write_bytes(canonical_candidate_lock_bytes(lock))
        with self.assertRaisesRegex(SealedDecoderError, "16 kHz mono"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: calls.append(kwargs),
                clock=SequenceClock([]),
            )
        self.assertEqual(calls, [])

    def test_rejects_lock_command_environment_hardware_config_and_source_mismatch(self):
        config = self.config(warmup_runs=0)
        one_audio = [self.write_wav("strict.wav")]
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=one_audio,
        )
        never_called = lambda **kwargs: self.fail("model must not be constructed")

        with self.assertRaisesRegex(SealedDecoderError, "complete command"):
            execute_sealed_candidate(
                input_path,
                lock_path,
                self.root,
                config,
                actual_argv=[*self.command_argv(config), "--unexpected"],
                environment_source=self.environment,
                current_directory=self.root,
                repository_root=self.root,
                hardware_reader=lambda device: self.hardware,
                source_identity_reader=lambda commit, root: (
                    commit,
                    self.runner_source_sha256,
                ),
                model_factory=never_called,
                clock=SequenceClock([]),
            )
        with self.assertRaisesRegex(SealedDecoderError, "OMP_NUM_THREADS"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=never_called,
                clock=SequenceClock([]),
                environment={
                    **self.environment,
                    "OMP_NUM_THREADS": "3",
                },
            )
        changed_hardware = {**self.hardware, "logical_cpu_count": 8}
        with self.assertRaisesRegex(SealedDecoderError, "hardware identity"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=never_called,
                clock=SequenceClock([]),
                hardware=changed_hardware,
            )
        changed_config = copy.copy(config)
        object.__setattr__(changed_config, "ncpu", 2)
        with self.assertRaisesRegex(SealedDecoderError, "decoder configuration"):
            self.execute(
                input_path,
                lock_path,
                changed_config,
                model_factory=never_called,
                clock=SequenceClock([]),
                environment={
                    **self.environment,
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "NUMEXPR_NUM_THREADS": "2",
                },
            )
        with self.assertRaisesRegex(SealedDecoderError, "code_commit"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=never_called,
                clock=SequenceClock([]),
                source_identity=lambda commit, root: ("2" * 40, digest("source")),
            )

    def test_cross_validates_input_and_lock_before_other_runtime_readers(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, lock = self.write_contract(
            config,
            audio_specs=[self.write_wav("handoff.wav")],
        )
        lock["input_projection_sha256"] = digest("detached-input")
        lock_path.write_bytes(canonical_candidate_lock_bytes(lock))
        hardware_calls: list[str] = []
        with self.assertRaisesRegex(SealedDecoderError, "input projection"):
            execute_sealed_candidate(
                input_path,
                lock_path,
                self.root,
                config,
                actual_argv=self.command_argv(config),
                environment_source=self.environment,
                current_directory=self.root,
                repository_root=self.root,
                hardware_reader=lambda device: hardware_calls.append(device),
                model_factory=lambda **kwargs: self.fail("model constructed"),
                clock=SequenceClock([]),
            )
        self.assertEqual(hardware_calls, [])

    def test_runner_source_inventory_rejects_dirty_bytes(self):
        self.assertIn("eval/__init__.py", RUNNER_SOURCE_PATHS)
        self.assertIn("eval/normalizers/__init__.py", RUNNER_SOURCE_PATHS)
        self.assertIn("eval/sealed_candidate_contract.py", RUNNER_SOURCE_PATHS)
        repository = self.root / "source-repository"
        repository.mkdir()
        source_path = repository / "runner.py"
        source_path.write_text("print('sealed')\n", encoding="utf-8")
        funasr_path = repository / "funasr" / "__init__.py"
        funasr_path.parent.mkdir()
        funasr_path.write_text("# committed upstream fixture\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=f@example.invalid",
                "add",
                "runner.py",
                "funasr/__init__.py",
            ],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=f@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            cwd=repository,
            check=True,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        with mock.patch("eval.sealed_decoder.RUNNER_SOURCE_PATHS", ("runner.py",)):
            resolved, inventory_sha256 = runner_source_identity(commit, repository)
            self.assertEqual(resolved, commit)
            self.assertRegex(inventory_sha256, r"^sha256:[0-9a-f]{64}$")
            source_path.write_text("print('dirty')\n", encoding="utf-8")
            with self.assertRaisesRegex(SealedDecoderError, "differs"):
                runner_source_identity(commit, repository)
            source_path.write_text("print('sealed')\n", encoding="utf-8")
            funasr_path.write_text("# dirty upstream fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(SealedDecoderError, "differs"):
                runner_source_identity(commit, repository)
            funasr_path.write_text("# committed upstream fixture\n", encoding="utf-8")
            (repository / "funasr" / "alternate.py").write_text(
                "# uncommitted alternate source\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SealedDecoderError, "uncommitted runtime"):
                runner_source_identity(commit, repository)
            (repository / "funasr" / "alternate.py").unlink()
            pycache = repository / "funasr" / "__pycache__"
            pycache.mkdir()
            (pycache / "__init__.cpython-311.pyc").write_bytes(b"alternate bytecode")
            self.assertEqual(
                runner_source_identity(commit, repository),
                (resolved, inventory_sha256),
            )
            source_path.write_text("print('replacement')\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "runner.py"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=f@example.invalid",
                    "commit",
                    "-qm",
                    "replacement fixture",
                ],
                cwd=repository,
                check=True,
            )
            replacement = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                text=True,
            ).strip()
            subprocess.run(
                ["git", "replace", commit, replacement],
                cwd=repository,
                check=True,
            )
            with self.assertRaisesRegex(SealedDecoderError, "replacement refs"):
                runner_source_identity(commit, repository)
            subprocess.run(
                ["git", "replace", "-d", commit],
                cwd=repository,
                check=True,
                capture_output=True,
            )
            (repository / ".git/info/grafts").write_text(
                f"{commit}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SealedDecoderError, "grafts"):
                runner_source_identity(commit, repository)

    def test_local_model_resolver_binds_identifier_revision_and_content(self):
        identifier = TRACKS["paraformer"].model_identifier
        snapshots = (
            self.root
            / ".cache"
            / "modelscope"
            / "models"
            / identifier.replace("/", "--")
            / "snapshots"
        )
        selected = snapshots / self.model_revision
        selected.mkdir(parents=True)
        (selected / "model.bin").write_bytes(b"exact revision")
        selected_hash = model_directory_sha256(selected)
        same_content_other_revision = snapshots / ("2" * 40)
        same_content_other_revision.mkdir()
        (same_content_other_revision / "model.bin").write_bytes(b"exact revision")

        self.assertEqual(
            resolve_local_model_snapshot(
                self.root,
                identifier,
                self.model_revision,
                selected_hash,
            ),
            selected,
        )
        with self.assertRaisesRegex(SealedDecoderError, "revision is absent"):
            resolve_local_model_snapshot(
                self.root,
                identifier,
                "3" * 40,
                model_directory_sha256(same_content_other_revision),
            )
        with self.assertRaisesRegex(SealedDecoderError, "full immutable"):
            resolve_local_model_snapshot(
                self.root,
                identifier,
                "v2.0.4",
                selected_hash,
            )

    def test_raw_prediction_jsonl_rejects_unknown_fields(self):
        with self.assertRaisesRegex(SealedDecoderError, "exactly the four"):
            raw_prediction_jsonl_bytes(
                [
                    {
                        "id": "sealed-1",
                        "raw_text": "text",
                        "status": "ok",
                        "reason_code": None,
                        "elapsed_ns": 1,
                    }
                ]
            )

    def test_describe_runtime_is_model_free_and_excludes_secret_environment(self):
        document = describe_runtime(
            4,
            environment_source=self.environment,
            current_directory=self.root,
            repository_root=self.root,
            hardware_reader=lambda device: self.hardware,
            runtime_identity_reader=lambda root: self.runtime,
        )
        self.assertEqual(document["working_directory"], ".")
        self.assertEqual(document["hardware"], self.hardware)
        self.assertEqual(document["runtime"], self.runtime)
        self.assertEqual(document["environment"]["PYTHONHASHSEED"], "0")
        self.assertNotIn("MODELSCOPE_API_TOKEN", document["environment"])

    def test_describe_runtime_rejects_cache_path_alias(self):
        environment = {
            **self.environment,
            "MODELSCOPE_CACHE": str(
                self.root / ".cache" / ".." / ".cache" / "modelscope"
            ),
        }
        with self.assertRaisesRegex(SealedDecoderError, "exact repository-local"):
            describe_runtime(
                4,
                environment_source=environment,
                current_directory=self.root,
                repository_root=self.root,
                hardware_reader=lambda device: self.hardware,
                runtime_identity_reader=lambda root: self.runtime,
            )

    def test_describe_runtime_rejects_symlinked_cache_component(self):
        external_cache = self.root / "external-cache"
        external_cache.mkdir()
        (self.root / ".cache").symlink_to(external_cache, target_is_directory=True)
        with self.assertRaisesRegex(SealedDecoderError, "only real directories"):
            describe_runtime(
                4,
                environment_source=self.environment,
                current_directory=self.root,
                repository_root=self.root,
                hardware_reader=lambda device: self.hardware,
                runtime_identity_reader=lambda root: self.runtime,
            )

    def test_direct_bootstrap_ignores_script_directory_stdlib_shadow(self):
        shadow = PROJECT_ROOT / "scripts/argparse.py"
        marker = self.root / "argparse-shadow-executed"
        if shadow.exists() or shadow.is_symlink():
            self.skipTest("scripts/argparse.py already exists")
        try:
            shadow.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "raise RuntimeError('stdlib shadow executed')\n",
                encoding="utf-8",
            )
            completed = self.sealed_subprocess(
                "scripts/run_sealed_asr_candidate.py",
                "describe-runtime",
                "--help",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("usage:", completed.stdout)
        finally:
            if shadow.exists() and not shadow.is_symlink():
                shadow.unlink()

    def test_direct_bootstrap_ignores_checkout_unchecked_hash_pyc(self):
        pycache = PROJECT_ROOT / "eval/__pycache__"
        pycache.mkdir(exist_ok=True)
        target = pycache / f"__init__.{sys.implementation.cache_tag}.pyc"
        marker = self.root / "unchecked-pyc-executed"
        malicious_source = self.root / "malicious_eval_init.py"
        malicious_source.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n",
            encoding="utf-8",
        )
        prior_payload = target.read_bytes() if target.is_file() else None
        prior_mode = target.stat().st_mode & 0o777 if target.is_file() else None
        try:
            py_compile.compile(
                str(malicious_source),
                cfile=str(target),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
            completed = self.sealed_subprocess(
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--help",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())
        finally:
            if prior_payload is None:
                if target.exists() and not target.is_symlink():
                    target.unlink()
            else:
                target.write_bytes(prior_payload)
                os.chmod(target, prior_mode)

    def test_direct_bootstrap_rejects_same_name_package_and_root_shadow(self):
        package_shadow = PROJECT_ROOT / "eval/custodian_replay"
        root_shadow = PROJECT_ROOT / "unicodedata.py"
        marker = self.root / "import-shadow-executed"
        if (
            package_shadow.exists()
            or package_shadow.is_symlink()
            or root_shadow.exists()
            or root_shadow.is_symlink()
        ):
            self.skipTest("bootstrap shadow fixture path already exists")
        try:
            package_shadow.mkdir()
            (package_shadow / "__init__.py").write_text(
                f"open({str(marker)!r}, 'w').write('package')\n",
                encoding="utf-8",
            )
            completed = self.sealed_subprocess(
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--help",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("import shadow", completed.stderr)
            self.assertFalse(marker.exists())
            (package_shadow / "__init__.py").unlink()
            package_shadow.rmdir()

            root_shadow.write_text(
                f"open({str(marker)!r}, 'w').write('root')\n",
                encoding="utf-8",
            )
            completed = self.sealed_subprocess(
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--help",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("repository import shadow", completed.stderr)
            self.assertFalse(marker.exists())
        finally:
            init_path = package_shadow / "__init__.py"
            if init_path.exists() and not init_path.is_symlink():
                init_path.unlink()
            package_pycache = package_shadow / "__pycache__"
            if package_pycache.is_dir() and not package_pycache.is_symlink():
                for cached_init in package_pycache.glob("__init__.*.pyc"):
                    if cached_init.is_file() and not cached_init.is_symlink():
                        cached_init.unlink()
                package_pycache.rmdir()
            if package_shadow.is_dir() and not package_shadow.is_symlink():
                package_shadow.rmdir()
            if root_shadow.exists() and not root_shadow.is_symlink():
                root_shadow.unlink()

    def test_direct_bootstrap_rejects_symlink_env_flags_and_runpy_wrapper(self):
        symlink_shadow = PROJECT_ROOT / "unicodedata.py"
        payload = self.root / "symlink_payload.py"
        payload.write_text("raise RuntimeError('symlink shadow executed')\n", encoding="utf-8")
        if symlink_shadow.exists() or symlink_shadow.is_symlink():
            self.skipTest("unicodedata.py already exists")
        try:
            symlink_shadow.symlink_to(payload)
            completed = self.sealed_subprocess(
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--help",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("import symlink", completed.stderr)
        finally:
            if symlink_shadow.is_symlink():
                symlink_shadow.unlink()

        for name, value in (
            ("PYTHONPYCACHEPREFIX", str(self.root / "foreign-pycache")),
            ("PYTHONOPTIMIZE", "1"),
            ("PYTHONWARNINGS", "error"),
            ("PYTHONUTF8", "1"),
            ("LD_PRELOAD", "/does/not/exist.so"),
            ("LANG", "C.UTF-8"),
        ):
            with self.subTest(environment=name):
                completed = self.sealed_subprocess(
                    "scripts/replay_asr_evaluation.py",
                    "export-input",
                    "--help",
                    environment_overrides={name: value},
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(name, completed.stderr)

        optimized = self.sealed_subprocess(
            "scripts/replay_asr_evaluation.py",
            "export-input",
            "--help",
            python_flags=("-O", "-P", "-S"),
        )
        self.assertNotEqual(optimized.returncode, 0)
        self.assertIn("optimization level 0", optimized.stderr)

        wrapper = subprocess.run(
            [
                str(VENV_PYTHON),
                "-P",
                "-S",
                "-c",
                (
                    "import runpy; "
                    "runpy.run_path('scripts/replay_asr_evaluation.py', "
                    "run_name='__main__')"
                ),
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--help",
            ],
            cwd=PROJECT_ROOT,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/dev/null",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertNotEqual(wrapper.returncode, 0)
        self.assertIn("exact direct process argv", wrapper.stderr)

    def test_rejects_startup_environment_that_cannot_be_undone(self):
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_PRELOAD",
            "FUNASR_HOME",
            "TORCH_LOGS",
            "OMP_DYNAMIC",
            "MKL_DYNAMIC",
            "KMP_AFFINITY",
            "GOMP_CPU_AFFINITY",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                SealedDecoderError, name
            ):
                describe_runtime(
                    4,
                    environment_source={**self.environment, name: "unsafe"},
                    current_directory=self.root,
                    repository_root=self.root,
                    hardware_reader=lambda device: self.hardware,
                    runtime_identity_reader=lambda root: self.runtime,
                )

    def test_rejects_model_stack_imported_before_environment_sanitization(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("preloaded-stack.wav")],
        )
        with mock.patch.dict(sys.modules, {"numpy": mock.Mock()}):
            with self.assertRaisesRegex(SealedDecoderError, "before environment"):
                execute_sealed_candidate(
                    input_path,
                    lock_path,
                    self.root,
                    config,
                    actual_argv=self.command_argv(config),
                    environment_source=self.environment,
                    current_directory=self.root,
                    repository_root=self.root,
                    hardware_reader=lambda device: self.hardware,
                    model_factory=default_model_factory,
                    clock=SequenceClock([]),
                )

    def test_rejects_repository_model_stack_shadow_before_model_construction(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("shadowed-stack.wav")],
        )
        shadow = self.root / "numpy"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("raise RuntimeError\n", encoding="utf-8")
        with self.assertRaisesRegex(SealedDecoderError, "model import shadow"):
            self.execute(
                input_path,
                lock_path,
                config,
                model_factory=lambda **kwargs: self.fail("model factory was called"),
                clock=SequenceClock([]),
            )

    def test_rejects_model_loaded_repository_module_outside_source_inventory(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("unverified-module.wav")],
        )
        rogue_path = self.root / "asr_lab/unverified_model_helper.py"
        rogue_path.parent.mkdir()
        rogue_path.write_text("VALUE = 'unverified'\n", encoding="utf-8")
        module_name = "asr_lab.unverified_model_helper"

        class FakeModel:
            model_path = self.model_path

        def importing_factory(**kwargs):
            sys.modules[module_name] = types.SimpleNamespace(__file__=str(rogue_path))
            return FakeModel()

        try:
            with self.assertRaisesRegex(
                SealedDecoderError, "outside the verified runner source inventory"
            ):
                self.execute(
                    input_path,
                    lock_path,
                    config,
                    model_factory=importing_factory,
                    clock=SequenceClock([0, 1]),
                )
        finally:
            sys.modules.pop(module_name, None)

    def test_bad_input_export_receipt_is_rejected_before_decoder_execution(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, input_document, lock = self.write_contract(
            config,
            audio_specs=[self.write_wav("receipt.wav")],
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": INPUT_EXPORT_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": lock["candidate"]["experiment_id"],
            "dataset_id": input_document["dataset_id"],
            "revision": input_document["revision"],
            "split": "sealed-blind",
            "decode_item_count": 1,
            "input_projection_sha256": sha256_file(input_path),
            "candidate_lock_sha256": digest("detached-lock"),
            "candidate_freeze_sha256": lock["candidate_freeze_sha256"],
            **registration_fields(lock["candidate"]["experiment_id"]),
        }
        receipt_path = self.root / "input-receipt.json"
        receipt_path.write_bytes(canonical_custodian_receipt_bytes(receipt))
        args = argparse.Namespace(
            input_projection=input_path,
            candidate_lock=lock_path,
            input_receipt=receipt_path,
            audio_root=self.root,
            track=config.track,
            model_revision=config.model_revision,
            device=config.device,
            ncpu=config.ncpu,
            warmup_runs=config.warmup_runs,
            seed=config.seed,
            hypothesis_adapter_version=config.hypothesis_adapter_version,
            output_raw_predictions=self.root / "raw.jsonl",
            output_execution_envelope=self.root / "envelope.json",
        )
        with mock.patch(
            "scripts.run_sealed_asr_candidate.execute_sealed_candidate"
        ) as decoder:
            with self.assertRaisesRegex(CustodianReplayError, "candidate_lock_sha256"):
                _run(args, actual_argv=self.command_argv(config))
        decoder.assert_not_called()

    def test_valid_handoff_reproves_registration_before_decode_or_publication(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, input_document, lock = self.write_contract(
            config,
            audio_specs=[self.write_wav("registered-handoff.wav")],
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": INPUT_EXPORT_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": lock["candidate"]["experiment_id"],
            "dataset_id": input_document["dataset_id"],
            "revision": input_document["revision"],
            "split": "sealed-blind",
            "decode_item_count": 1,
            "input_projection_sha256": sha256_file(input_path),
            "candidate_lock_sha256": sha256_file(lock_path),
            "candidate_freeze_sha256": lock["candidate_freeze_sha256"],
            "candidate_registration_commit": lock[
                "candidate_registration_commit"
            ],
            "candidate_manifest_path": lock["candidate_manifest_path"],
            "candidate_manifest_sha256": lock["candidate_manifest_sha256"],
        }
        receipt_path = self.root / "valid-input-receipt.json"
        receipt_path.write_bytes(canonical_custodian_receipt_bytes(receipt))
        args = argparse.Namespace(
            input_projection=input_path,
            candidate_lock=lock_path,
            input_receipt=receipt_path,
            audio_root=self.root,
            track=config.track,
            model_revision=config.model_revision,
            device=config.device,
            ncpu=config.ncpu,
            warmup_runs=config.warmup_runs,
            seed=config.seed,
            hypothesis_adapter_version=config.hypothesis_adapter_version,
            output_raw_predictions=self.root / "registered-raw.jsonl",
            output_execution_envelope=self.root / "registered-envelope.json",
        )
        registration_failure = CustodianReplayError(
            "registration re-proof sentinel"
        )
        with mock.patch(
            "scripts.run_sealed_asr_candidate.validate_registered_candidate_binding",
            side_effect=registration_failure,
        ) as validate_registration, mock.patch(
            "scripts.run_sealed_asr_candidate.execute_sealed_candidate"
        ) as decoder, mock.patch(
            "scripts.run_sealed_asr_candidate.write_atomic_outputs"
        ) as writer:
            with self.assertRaisesRegex(
                CustodianReplayError, "registration re-proof sentinel"
            ):
                _run(args, actual_argv=self.command_argv(config))

        validate_registration.assert_called_once_with(lock)
        decoder.assert_not_called()
        writer.assert_not_called()

    def test_decoder_process_streams_are_discarded(self):
        read_fd, write_fd = os.pipe()
        saved_stdout = os.dup(1)
        try:
            os.dup2(write_fd, 1)
            os.close(write_fd)
            with _silence_decoder_streams():
                os.write(1, b"prediction-must-not-escape")
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        self.assertEqual(os.read(read_fd, 1024), b"")
        os.close(read_fd)

    def test_publishes_raw_predictions_then_final_envelope_silently_without_overwrite(self):
        config = self.config(warmup_runs=0)
        input_path, lock_path, _, _ = self.write_contract(
            config,
            audio_specs=[self.write_wav("publish.wav")],
        )

        class FakeModel:
            model_path = Path("/fake/model")

            def generate(self, *, input, **kwargs):
                return [{"text": "发布"}]

        result = self.execute(
            input_path,
            lock_path,
            config,
            model_factory=lambda **kwargs: FakeModel(),
            clock=SequenceClock([0, 10, 20, 30, 40, 60]),
        )
        raw_path = self.root / "raw-predictions.jsonl"
        envelope_path = self.root / "execution-envelope.json"
        input_receipt_sha256 = digest("input-export-receipt")
        with mock.patch(
            "scripts.run_sealed_asr_candidate.write_atomic_outputs"
        ) as writer:
            _publish_result(
                result,
                raw_path,
                envelope_path,
                input_export_receipt_sha256=input_receipt_sha256,
            )
        writer.assert_called_once()
        ordered_outputs = writer.call_args.args[0]
        self.assertEqual(
            [target for target, _ in ordered_outputs],
            [raw_path, envelope_path],
        )
        self.assertEqual(ordered_outputs[-1][0], envelope_path)

        with redirect_stdout(io.StringIO()) as stdout:
            _publish_result(
                result,
                raw_path,
                envelope_path,
                input_export_receipt_sha256=input_receipt_sha256,
            )
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(raw_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(envelope_path.stat().st_mode & 0o777, 0o600)
        envelope = load_execution_envelope(envelope_path).document
        self.assertEqual(
            envelope["bindings"]["raw_predictions_sha256"],
            sha256_bytes(raw_path.read_bytes()),
        )
        self.assertEqual(
            envelope["bindings"]["input_export_receipt_sha256"],
            input_receipt_sha256,
        )
        self.assertNotIn("发布", envelope_path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(CustodianReplayError, "overwrite"):
            _publish_result(
                result,
                raw_path,
                envelope_path,
                input_export_receipt_sha256=input_receipt_sha256,
            )

        non_private_parent = self.root / "non-standard-private-mode"
        non_private_parent.mkdir(mode=0o500)
        with self.assertRaisesRegex(CustodianReplayError, "exactly 0700"):
            _publish_result(
                result,
                non_private_parent / "raw.jsonl",
                non_private_parent / "envelope.json",
                input_export_receipt_sha256=input_receipt_sha256,
            )


if __name__ == "__main__":
    unittest.main()
