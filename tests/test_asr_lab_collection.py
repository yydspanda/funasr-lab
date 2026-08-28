from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from eval.collection import ALLOWED_SPLITS
from eval.collection import CollectionValidationError
from eval.collection import ISOLATION_KEYS
from eval.collection import RECORD_FIELDS
from eval.collection import SCENARIO_TAXONOMY_VERSION
from eval.collection import build_sealed_input_projection
from eval.collection import canonical_json_bytes
from eval.collection import load_collection_descriptor
from eval.collection import load_validated_collection
from eval.collection import sha256_bytes
from eval.collection import sha256_file
from eval.collection import validate_collection
from eval.core_report import canonical_core_bytes
from eval.core_report import core_report_sha256
from eval.core_report import validate_core_report
from eval.custodian_replay import build_prediction_bundle
from eval.custodian_replay import PREDICTION_FREEZE_RECEIPT_KIND
from eval.custodian_replay import RECEIPT_SCHEMA_VERSION
from eval.custodian_replay import RegisteredCandidateManifest
from eval.custodian_replay import CustodianReplayError
from eval.custodian_replay import canonical_prediction_bundle_bytes
from eval.custodian_replay import canonical_custodian_receipt_bytes
from eval.custodian_replay import load_candidate_lock
from eval.custodian_replay import load_custodian_receipt
from eval.custodian_replay import load_sealed_input_projection
from eval.custodian_replay import validate_terminal_manifest_for_receipt
from eval.execution_envelope import build_execution_envelope
from eval.execution_envelope import canonical_execution_envelope_bytes
from eval.normalizers import NORMALIZER_VERSION
from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import TRACKS
from eval.offline_baseline import effective_config
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import record_input_sha256
from eval.scoring import MER_TOKENIZER_VERSION
from eval.sealed_decoder import SealedDecoderError
from eval.sealed_decoder import runtime_identity
from scripts import replay_asr_evaluation as replay_cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPOSITORY_ROOT / ".venv/bin/python"
SEALED_SUBPROCESS_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/dev/null",
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
}


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


class CollectionValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        (self.root / "manifests").mkdir()
        (self.root / "audio").mkdir()
        (self.root / "evidence").mkdir()
        self.transform_sha256 = digest("identity-transform-v1")
        self.dedup_report = self.root / "evidence/dedup.json"

    def write_wav(self, name: str, *, sample: int, frames: int) -> tuple[str, Path, float]:
        relative = Path("audio") / name
        path = self.root / relative
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(struct.pack("<h", sample) * frames)
        return relative.as_posix(), path, frames / 16_000

    def record(
        self,
        split: str,
        index: int,
        *,
        access_class: str,
        rights_group_id: str,
    ) -> dict[str, object]:
        audio, audio_path, duration = self.write_wav(
            f"{split}-{index}.wav",
            sample=100 + index * 37 + len(split),
            frames=1_600 + index,
        )
        raw_text = "" if split == "sealed-blind" and index == 99 else f"测试{index}"
        audio_sha256 = sha256_file(audio_path)
        environment = (
            "environment:meeting"
            if split == "sealed-blind"
            else "environment:near-field"
        )
        return {
            "schema_version": 1,
            "dataset_id": "LAB-SEED-TEST",
            "data_revision": "v0.1",
            "id": f"utt-{split}-{index}",
            "audio": audio,
            "audio_sha256": audio_sha256,
            "duration_seconds": duration,
            "sample_rate": 16_000,
            "channels": 1,
            "sample_width_bits": 16,
            "raw_text": raw_text,
            "reference_sha256": sha256_bytes(raw_text.encode("utf-8")),
            "normalizer_version": NORMALIZER_VERSION,
            "mer_tokenizer_version": MER_TOKENIZER_VERSION,
            "speaker_id": f"speaker-{split}-{index}",
            "session_id": f"session-{split}-{index}",
            "source_recording_id": f"recording-{split}-{index}",
            "lineage_group_id": f"lineage-{split}-{index}",
            "dedup_cluster_id": f"dedup-{split}-{index}",
            "lineage_kind": "source",
            "derived_from_id": None,
            "source_audio_sha256": audio_sha256,
            "transform_recipe_sha256": self.transform_sha256,
            "split": split,
            "provenance_group_id": "provenance-fixture",
            "rights_group_id": rights_group_id,
            "access_class": access_class,
            "scenario_taxonomy_version": SCENARIO_TAXONOMY_VERSION,
            "scenario_tags": ["language:zh", "noise:clean", environment],
            "evaluation_status": "included",
            "exclusion_reason": None,
        }

    def write_manifest(
        self, split: str, records: list[dict[str, object]]
    ) -> tuple[Path, bytes]:
        path = self.root / f"manifests/{split}.jsonl"
        payload = b"".join(canonical_json_bytes(record) for record in records)
        path.write_bytes(payload)
        return path, payload

    def ordered_records(
        self, records: dict[str, list[dict[str, object]]]
    ) -> list[dict[str, object]]:
        return [record for split in ALLOWED_SPLITS for record in records[split]]

    def manifest_inventory_sha256(self, manifests: list[dict[str, object]]) -> str:
        inventory = "".join(
            f"{manifest['sha256'].removeprefix('sha256:')}  {manifest['path']}\n"
            for manifest in manifests
        )
        return sha256_bytes(inventory.encode("utf-8"))

    def record_cluster_inventory_sha256(
        self, records: dict[str, list[dict[str, object]]]
    ) -> str:
        inventory = [
            {
                "id": record["id"],
                "dedup_cluster_id": record["dedup_cluster_id"],
            }
            for record in self.ordered_records(records)
        ]
        return sha256_bytes(canonical_json_bytes(inventory))

    def sealed_projection_document(
        self,
        descriptor: dict[str, object],
        records: dict[str, list[dict[str, object]]],
        *,
        references: bool,
    ) -> dict[str, object]:
        manifests = descriptor["manifests"]
        self.assertIsInstance(manifests, list)
        manifest = manifests[2]
        sealed_records = records["sealed-blind"]
        if references:
            items = [
                {
                    "id": record["id"],
                    "split": record["split"],
                    "raw_text": record["raw_text"],
                    "reference_sha256": record["reference_sha256"],
                    "normalizer_version": record["normalizer_version"],
                    "mer_tokenizer_version": record["mer_tokenizer_version"],
                    "scenario_taxonomy_version": record[
                        "scenario_taxonomy_version"
                    ],
                    "scenario_tags": list(record["scenario_tags"]),
                    "evaluation_status": record["evaluation_status"],
                    "exclusion_reason": record["exclusion_reason"],
                }
                for record in sealed_records
            ]
            kind = "asr-sealed-reference-input"
        else:
            items = [
                {
                    "id": record["id"],
                    "split": record["split"],
                    "audio": record["audio"],
                    "audio_sha256": record["audio_sha256"],
                    "duration_seconds": record["duration_seconds"],
                    "sample_rate": record["sample_rate"],
                    "channels": record["channels"],
                    "sample_width_bits": record["sample_width_bits"],
                }
                for record in sealed_records
                if record["evaluation_status"] == "included"
            ]
            kind = "asr-sealed-audio-input"
        return {
            "schema_version": 2,
            "kind": kind,
            "dataset_id": descriptor["dataset_id"],
            "revision": descriptor["revision"],
            "split": "sealed-blind",
            "manifest_sha256": manifest["sha256"],
            "manifest_record_count": manifest["record_count"],
            "item_count": len(items),
            "items": items,
        }

    def refresh_bound_evidence(
        self,
        descriptor: dict[str, object],
        records: dict[str, list[dict[str, object]]],
        *,
        rewrite_manifests: bool,
    ) -> None:
        manifests = descriptor["manifests"]
        self.assertIsInstance(manifests, list)
        if rewrite_manifests:
            for index, split in enumerate(ALLOWED_SPLITS):
                _, payload = self.write_manifest(split, records[split])
                manifests[index]["sha256"] = sha256_bytes(payload)
                manifests[index]["record_count"] = len(records[split])

        dedup = descriptor["dedup"]
        self.assertIsInstance(dedup, dict)
        report = {
            "schema_version": 1,
            "kind": "asr-dedup-review",
            "dataset_id": descriptor["dataset_id"],
            "revision": descriptor["revision"],
            "method": dedup["method"],
            "version": dedup["version"],
            "threshold": dedup["threshold"],
            "config_sha256": dedup["config_sha256"],
            "manifest_inventory_sha256": self.manifest_inventory_sha256(
                manifests
            ),
            "record_count": len(self.ordered_records(records)),
            "record_cluster_inventory_sha256": (
                self.record_cluster_inventory_sha256(records)
            ),
            "status": "reviewed",
        }
        self.dedup_report.write_bytes(canonical_json_bytes(report))
        dedup["report_sha256"] = sha256_file(self.dedup_report)

        blind = descriptor["blind"]
        self.assertIsInstance(blind, dict)
        input_projection = self.sealed_projection_document(
            descriptor, records, references=False
        )
        reference_projection = self.sealed_projection_document(
            descriptor, records, references=True
        )
        blind["input_projection_sha256"] = sha256_bytes(
            canonical_json_bytes(input_projection)
        )
        blind["reference_projection_sha256"] = sha256_bytes(
            canonical_json_bytes(reference_projection)
        )

    def descriptor(
        self, records: dict[str, list[dict[str, object]]]
    ) -> dict[str, object]:
        manifests = []
        for split in ALLOWED_SPLITS:
            _, payload = self.write_manifest(split, records[split])
            manifests.append(
                {
                    "split": split,
                    "path": f"manifests/{split}.jsonl",
                    "sha256": sha256_bytes(payload),
                    "record_count": len(records[split]),
                    "reference_access": (
                        "sealed" if split == "sealed-blind" else "restricted"
                    ),
                }
            )
        descriptor = {
            "schema_version": 1,
            "kind": "asr-evaluation-collection",
            "dataset_id": "LAB-SEED-TEST",
            "revision": "v0.1",
            "state": "frozen",
            "record_schema_version": 1,
            "normalizer_version": NORMALIZER_VERSION,
            "mer_tokenizer_version": MER_TOKENIZER_VERSION,
            "scenario_taxonomy_version": SCENARIO_TAXONOMY_VERSION,
            "manifests": manifests,
            "split_policy": {
                "allowed": list(ALLOWED_SPLITS),
                "isolation_keys": list(ISOLATION_KEYS),
            },
            "provenance_groups": [
                {
                    "id": "provenance-fixture",
                    "source_id": "source-fixture",
                    "source_kind": "synthetic",
                    "source_revision": "fixture-v1",
                    "collection_period": "2026-08",
                    "annotation_protocol_sha256": digest("annotation-v1"),
                    "transform_recipe_sha256": self.transform_sha256,
                }
            ],
            "rights_groups": [
                {
                    "id": "rights-restricted",
                    "basis": "synthetic",
                    "consent_status": "not-required",
                    "allowed_uses": ["asr-evaluation"],
                    "access_class": "restricted",
                    "evidence_sha256": digest("rights-restricted"),
                    "reviewed_at": "2026-08-27",
                },
                {
                    "id": "rights-sealed",
                    "basis": "synthetic",
                    "consent_status": "not-required",
                    "allowed_uses": ["asr-evaluation"],
                    "access_class": "sealed",
                    "evidence_sha256": digest("rights-sealed"),
                    "reviewed_at": "2026-08-27",
                },
            ],
            "dedup": {
                "method": "fixture-cluster",
                "version": "v1",
                "threshold": 0.95,
                "config_sha256": digest("dedup-config"),
                "report_path": "evidence/dedup.json",
                "report_sha256": digest("dedup-report-before-refresh"),
                "status": "reviewed",
            },
            "blind": {
                "split": "sealed-blind",
                "reference_access": "sealed",
                "custodian_role": "eval-custodian",
                "sealed_at": "2026-08-27T00:00:00Z",
                "unlock_policy": "candidate-config-command-hashes-frozen",
                "input_projection_sha256": digest(
                    "sealed-input-before-refresh"
                ),
                "reference_projection_sha256": digest(
                    "sealed-reference-before-refresh"
                ),
            },
        }
        self.refresh_bound_evidence(
            descriptor,
            records,
            rewrite_manifests=False,
        )
        return descriptor

    def write_bundle(
        self,
    ) -> tuple[Path, dict[str, object], dict[str, list[dict[str, object]]]]:
        records = {
            "smoke": [
                self.record(
                    "smoke", 1, access_class="restricted", rights_group_id="rights-restricted"
                )
            ],
            "dev": [
                self.record(
                    "dev", 2, access_class="restricted", rights_group_id="rights-restricted"
                )
            ],
            "sealed-blind": [
                self.record(
                    "sealed-blind",
                    3,
                    access_class="sealed",
                    rights_group_id="rights-sealed",
                )
            ],
        }
        descriptor = self.descriptor(records)
        path = self.root / "collection.json"
        path.write_bytes(canonical_json_bytes(descriptor))
        return path, descriptor, records

    def planned_candidate_manifest(self, data_sha256: str) -> dict[str, object]:
        model_revision = "28fe27c56aab3861cf77ae065b2bfc2aa3ab9692"
        config = BaselineConfig(
            track="paraformer",
            model_revision=model_revision,
            device="cpu",
            ncpu=4,
            warmup_runs=1,
            seed=0,
        )
        return {
            "schema_version": 1,
            "experiment_id": "EXP-20260827-001-eval-fixture",
            "task_id": "EVAL-01",
            "hypothesis": (
                "The frozen vanilla ASR candidate will produce a replayable "
                "sealed-blind prediction artifact without exposing references."
            ),
            "upstream_commit": "eedd4e22d10dc2e81d9c2bb321edb3750253964b",
            "code_commit": subprocess.check_output(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/dev/null",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "GIT_CONFIG_NOSYSTEM": "1",
                },
                text=True,
            ).strip(),
            "models": [
                {
                    "role": "asr",
                    "identifier": TRACKS["paraformer"].model_identifier,
                    "revision": model_revision,
                    "sha256": digest("fixture-model-inventory"),
                }
            ],
            "config_sha256": sha256_bytes(
                canonical_json_bytes(effective_config(config))
            ),
            "data_sha256": data_sha256,
            "eval_data_version": "LAB-SEED-TEST-v0.1",
            "normalizer_version": NORMALIZER_VERSION,
            "hardware": {
                "host_id": "fixture-cpu-host",
                "os": "Linux fixture x86_64",
                "cpu_model": "Fixture CPU Model",
                "logical_cpu_count": 4,
                "memory_bytes": 8_589_934_592,
                "device": "cpu",
                "accelerator": None,
            },
            "seed": 0,
            "command": {
                "working_directory": ".",
                "argv": [
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
                    str(self.root / "export-receipt.json"),
                    "--audio-root",
                    str(self.root),
                    "--track",
                    "paraformer",
                    "--model-revision",
                    "28fe27c56aab3861cf77ae065b2bfc2aa3ab9692",
                    "--device",
                    "cpu",
                    "--ncpu",
                    "4",
                    "--warmup-runs",
                    "1",
                    "--seed",
                    "0",
                    "--hypothesis-adapter-version",
                    "identity-v1",
                    "--output-raw-predictions",
                    str(self.root / "raw-predictions.jsonl"),
                    "--output-execution-envelope",
                    str(self.root / "execution-envelope.json"),
                ],
                "environment": {
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                    "CRC32C_SW_MODE": "auto",
                    "HYDRA_FULL_ERROR": "1",
                    "KMP_DUPLICATE_LIB_OK": "True",
                    "KMP_INIT_AT_FORK": "FALSE",
                    "MODELSCOPE_CACHE": str(
                        REPOSITORY_ROOT / ".cache/modelscope"
                    ),
                    "PYTHONHASHSEED": "0",
                    "TORCHINDUCTOR_CACHE_DIR": str(
                        REPOSITORY_ROOT / ".cache/torchinductor"
                    ),
                },
            },
            "metrics": None,
            "artifacts": [],
            "decision": "planned",
        }

    def write_planned_candidate_manifest(self, data_sha256: str) -> Path:
        candidate = self.planned_candidate_manifest(data_sha256)
        candidate_directory = self.root / "candidate-manifests"
        candidate_directory.mkdir()
        path = candidate_directory / f"{candidate['experiment_id']}.json"
        path.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def registered_candidate_manifest(
        self, path: Path
    ) -> RegisteredCandidateManifest:
        payload = path.read_bytes()
        document = json.loads(payload)
        return RegisteredCandidateManifest(
            document=document,
            payload=payload,
            sha256=sha256_bytes(payload),
            repository_path=(
                f"experiments/manifests/{document['experiment_id']}.json"
            ),
            registration_commit="d" * 40,
        )

    def export_input_direct(
        self,
        *,
        descriptor_path: Path,
        candidate_manifest_path: Path,
        input_path: Path,
        lock_path: Path,
        receipt_path: Path,
    ) -> dict[str, object]:
        registered = self.registered_candidate_manifest(candidate_manifest_path)
        scorer_identity = (
            str(registered.document["code_commit"]),
            digest("fixture-scorer-source"),
        )
        args = argparse.Namespace(
            descriptor=descriptor_path,
            collection_root=self.root,
            audio_root=self.root,
            candidate_manifest=candidate_manifest_path,
            candidate_registration_commit=registered.registration_commit,
            hypothesis_adapter_version="identity-v1",
            output_input=input_path,
            output_candidate_lock=lock_path,
            output_receipt=receipt_path,
        )
        with mock.patch.object(
            replay_cli,
            "load_planned_candidate_manifest",
            return_value=registered,
        ), mock.patch.object(
            replay_cli,
            "_scorer_identity_for_candidate",
            return_value=scorer_identity,
        ), mock.patch.object(
            replay_cli,
            "scorer_code_identity",
            return_value=scorer_identity,
        ):
            return replay_cli._export_input(args)

    def freeze_predictions_direct(
        self,
        *,
        input_path: Path,
        lock_path: Path,
        export_receipt_path: Path,
        raw_predictions_path: Path,
        execution_envelope_path: Path,
        predictions_path: Path,
        prediction_receipt_path: Path,
        registered_candidate_validator: mock.Mock | None = None,
    ) -> dict[str, object]:
        candidate = load_candidate_lock(lock_path).document["candidate"]
        envelope = json.loads(execution_envelope_path.read_bytes())
        scorer_identity = (
            str(candidate["code_commit"]),
            digest("fixture-scorer-source"),
        )
        runner_identity = (
            str(envelope["runner"]["code_commit"]),
            str(envelope["runner"]["source_sha256"]),
        )
        args = argparse.Namespace(
            input_projection=input_path,
            candidate_lock=lock_path,
            input_receipt=export_receipt_path,
            raw_predictions=raw_predictions_path,
            execution_envelope=execution_envelope_path,
            hypothesis_adapter_version="identity-v1",
            output_predictions=predictions_path,
            output_receipt=prediction_receipt_path,
        )
        binding_validator = (
            registered_candidate_validator
            if registered_candidate_validator is not None
            else mock.Mock()
        )
        with mock.patch.object(
            replay_cli,
            "validate_registered_candidate_binding",
            new=binding_validator,
        ), mock.patch.object(
            replay_cli,
            "_scorer_identity_for_candidate",
            return_value=scorer_identity,
        ), mock.patch.object(
            replay_cli,
            "scorer_code_identity",
            return_value=scorer_identity,
        ), mock.patch.object(
            replay_cli,
            "_validated_runner_source_identity",
            return_value=runner_identity,
        ):
            return replay_cli._freeze_predictions(args)

    def score_direct(
        self,
        *,
        descriptor_path: Path,
        input_path: Path,
        lock_path: Path,
        export_receipt_path: Path,
        predictions_path: Path,
        execution_envelope_path: Path,
        prediction_receipt_path: Path,
        core_path: Path,
        score_receipt_path: Path,
        registered_candidate_validator: mock.Mock | None = None,
    ) -> dict[str, object]:
        candidate = load_candidate_lock(lock_path).document["candidate"]
        envelope = json.loads(execution_envelope_path.read_bytes())
        scorer_identity = (
            str(candidate["code_commit"]),
            digest("fixture-scorer-source"),
        )
        runner_identity = (
            str(envelope["runner"]["code_commit"]),
            str(envelope["runner"]["source_sha256"]),
        )
        scorer_runtime = runtime_identity(REPOSITORY_ROOT)
        args = argparse.Namespace(
            descriptor=descriptor_path,
            collection_root=self.root,
            audio_root=self.root,
            input_projection=input_path,
            candidate_lock=lock_path,
            input_receipt=export_receipt_path,
            predictions=predictions_path,
            execution_envelope=execution_envelope_path,
            prediction_receipt=prediction_receipt_path,
            output_core=core_path,
            output_receipt=score_receipt_path,
        )
        binding_validator = (
            registered_candidate_validator
            if registered_candidate_validator is not None
            else mock.Mock()
        )
        with mock.patch.object(
            replay_cli,
            "validate_registered_candidate_binding",
            new=binding_validator,
        ), mock.patch.object(
            replay_cli,
            "_scorer_identity_for_candidate",
            return_value=scorer_identity,
        ), mock.patch.object(
            replay_cli,
            "scorer_code_identity",
            return_value=scorer_identity,
        ), mock.patch.object(
            replay_cli,
            "_validated_runner_source_identity",
            return_value=runner_identity,
        ), mock.patch.object(
            replay_cli,
            "_scorer_runtime_identity",
            return_value=scorer_runtime,
        ):
            return replay_cli._score(args)

    def write_execution_envelope(
        self,
        sealed_input,
        candidate_lock,
        prediction_items: list[dict[str, object]],
        raw_predictions_path: Path,
        input_export_receipt_path: Path,
        output_path: Path,
        *,
        warmup_runs: int = 1,
    ) -> dict[str, object]:
        """Build trusted-runner-shaped evidence for custodian-only fixtures."""

        prediction_by_id = {item["id"]: item for item in prediction_items}
        attempts = []
        for index, item in enumerate(sealed_input.document["items"]):
            prediction = prediction_by_id.get(item["id"])
            status = "failed" if prediction is None else prediction["status"]
            reason_code = (
                "missing_prediction"
                if prediction is None
                else prediction["reason_code"]
            )
            attempts.append(
                {
                    "id": item["id"],
                    "attempt_index": index,
                    "elapsed_ns": max(
                        1,
                        round(float(item["duration_seconds"]) * 100_000_000),
                    ),
                    "audio_duration_seconds": item["duration_seconds"],
                    "status": status,
                    "reason_code": reason_code,
                }
            )
        first = dict(attempts[0])
        first["attempt_index"] = 0
        candidate = candidate_lock.document["candidate"]
        runner_commit = str(candidate["code_commit"])
        runner_source_sha256 = digest("fixture-runner-source")
        observation = {
            "experiment_id": candidate["experiment_id"],
            "dataset_id": sealed_input.document["dataset_id"],
            "revision": sealed_input.document["revision"],
            "split": "sealed-blind",
            "candidate_freeze_sha256": candidate_lock.document[
                "candidate_freeze_sha256"
            ],
            "candidate_lock_sha256": candidate_lock.sha256,
            "input_projection_sha256": sealed_input.sha256,
            "hypothesis_adapter_version": candidate_lock.document[
                "hypothesis_adapter_version"
            ],
            "config_sha256": candidate["config_sha256"],
            "models": candidate["models"],
            "command": candidate["command"],
            "hardware": candidate["hardware"],
            "runner_code_commit": runner_commit,
            "runner_source_sha256": runner_source_sha256,
            "runtime": {
                "python_implementation": "cpython",
                "python_version": "3.11.15",
                "python_cache_tag": "cpython-311",
                "dependency_lock_sha256": digest("lab-cpu-lock"),
                "installed_dependencies_sha256": digest(
                    "installed-dependencies"
                ),
                "installed_dependency_count": 71,
                "unicode_version": "14.0.0",
            },
            "raw_predictions_sha256": sha256_file(raw_predictions_path),
            "prediction_items_sha256": sha256_bytes(
                canonical_json_bytes(prediction_items)
            ),
            "prediction_item_count": len(prediction_items),
            "started_at_utc": "2026-08-28T00:00:00Z",
            "finished_at_utc": "2026-08-28T00:00:01Z",
            "measurement_contract": {
                "clock_version": "python-perf-counter-ns-v1",
                "rss_version": "linux-rusage-self-maxrss-kib-v1",
                "rss_scope": "fresh-process-rusage-self",
                "rtf_population": "all-measured-attempts",
                "warmup_runs": warmup_runs,
            },
            "model_load_ns": 200_000_000,
            "cold_attempt": first,
            "warmup_attempts": [
                {**first, "attempt_index": index}
                for index in range(warmup_runs)
            ],
            "decode_attempts": attempts,
            "peak_rss_bytes": 1_048_576,
        }
        envelope = build_execution_envelope(
            observation,
            prediction_items,
            input_export_receipt_sha256=sha256_file(input_export_receipt_path),
        )
        output_path.write_bytes(canonical_execution_envelope_bytes(envelope))
        output_path.chmod(0o600)
        return envelope

    def write_valid_runner_handoff(
        self,
        *,
        input_path: Path,
        lock_path: Path,
        export_receipt_path: Path,
        sealed_record: dict[str, object],
    ) -> tuple[Path, Path]:
        sealed_input = load_sealed_input_projection(input_path)
        candidate_lock = load_candidate_lock(lock_path)
        prediction_items = [
            {
                "id": sealed_record["id"],
                "raw_text": sealed_record["raw_text"],
                "status": "ok",
                "reason_code": None,
            }
        ]
        raw_predictions_path = self.root / "raw-predictions.jsonl"
        raw_predictions_path.write_bytes(
            b"".join(canonical_json_bytes(item) for item in prediction_items)
        )
        raw_predictions_path.chmod(0o600)
        execution_envelope_path = self.root / "execution-envelope.json"
        self.write_execution_envelope(
            sealed_input,
            candidate_lock,
            prediction_items,
            raw_predictions_path,
            export_receipt_path,
            execution_envelope_path,
        )
        return raw_predictions_path, execution_envelope_path

    def rewrite_bundle(
        self,
        descriptor_path: Path,
        descriptor: dict[str, object],
        records: dict[str, list[dict[str, object]]],
    ) -> None:
        self.refresh_bound_evidence(
            descriptor,
            records,
            rewrite_manifests=True,
        )
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))

    def test_valid_collection_summary_and_cli_are_byte_stable(self) -> None:
        descriptor_path, _, _ = self.write_bundle()

        first = validate_collection(descriptor_path, self.root, self.root)
        second = validate_collection(descriptor_path, self.root, self.root)

        self.assertEqual(first, second)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(first["record_count"], 3)
        self.assertEqual(first["split_counts"], dict.fromkeys(ALLOWED_SPLITS, 1))
        self.assertEqual(first["normalizer_version"], NORMALIZER_VERSION)
        self.assertEqual(first["mer_tokenizer_version"], MER_TOKENIZER_VERSION)
        self.assertEqual(first["record_identity_version"], RECORD_IDENTITY_VERSION)
        self.assertEqual(
            first["dedup"]["manifest_inventory_sha256"],
            first["manifest_inventory_sha256"],
        )
        self.assertEqual(first["dedup"]["record_count"], first["record_count"])
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/validate_asr_collection.py",
                "--descriptor",
                str(descriptor_path),
                "--collection-root",
                str(self.root),
                "--audio-root",
                str(self.root),
            ],
            cwd=REPOSITORY_ROOT,
            env=SEALED_SUBPROCESS_ENVIRONMENT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stdout, canonical_json_bytes(first))

    def test_validated_records_and_audio_only_projection_share_frozen_identity(
        self,
    ) -> None:
        descriptor_path, descriptor, fixture_records = self.write_bundle()
        collection = load_validated_collection(
            descriptor_path,
            self.root,
            self.root,
        )

        expected_ids = [
            record["id"] for record in self.ordered_records(fixture_records)
        ]
        self.assertEqual([record["id"] for record in collection.records], expected_ids)
        self.assertTrue(
            all(set(record) == set(RECORD_FIELDS) for record in collection.records)
        )
        self.assertFalse(
            any(
                key.startswith("_")
                for record in collection.records
                for key in record
            )
        )
        self.assertEqual(
            collection.summary["record_input_sha256"],
            record_input_sha256(collection.records),
        )
        self.assertEqual(
            collection.summary,
            validate_collection(descriptor_path, self.root, self.root),
        )

        projection_bytes = build_sealed_input_projection(collection)
        self.assertEqual(
            projection_bytes,
            canonical_json_bytes(json.loads(projection_bytes)),
        )
        projection = json.loads(projection_bytes)
        manifests = descriptor["manifests"]
        self.assertIsInstance(manifests, list)
        blind = descriptor["blind"]
        self.assertIsInstance(blind, dict)
        self.assertEqual(projection["dataset_id"], descriptor["dataset_id"])
        self.assertEqual(projection["revision"], descriptor["revision"])
        self.assertEqual(projection["split"], "sealed-blind")
        self.assertEqual(projection["manifest_sha256"], manifests[2]["sha256"])
        self.assertEqual(projection["schema_version"], 2)
        self.assertEqual(projection["manifest_record_count"], 1)
        self.assertEqual(projection["item_count"], 1)
        self.assertNotIn("record_count", projection)
        self.assertEqual(
            sha256_bytes(projection_bytes),
            blind["input_projection_sha256"],
        )
        self.assertEqual(
            collection.summary["input_projection_sha256"],
            blind["input_projection_sha256"],
        )
        self.assertEqual(
            collection.summary["reference_projection_sha256"],
            blind["reference_projection_sha256"],
        )
        blind_reference = fixture_records["sealed-blind"][0]["raw_text"]
        self.assertIsInstance(blind_reference, str)
        self.assertNotIn(blind_reference.encode("utf-8"), projection_bytes)
        self.assertNotIn(b"raw_text", projection_bytes)
        self.assertNotIn(b"reference_sha256", projection_bytes)
        self.assertNotIn(b"rights_group_id", projection_bytes)
        self.assertNotIn(b"evidence_sha256", projection_bytes)
        self.assertNotIn(b"data_sha256", projection_bytes)
        self.assertNotIn(
            collection.summary["data_sha256"].encode("ascii"),
            projection_bytes,
        )

    def test_predeclared_excluded_audio_is_not_exported_for_blind_decode(
        self,
    ) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        excluded = self.record(
            "sealed-blind",
            4,
            access_class="sealed",
            rights_group_id="rights-sealed",
        )
        excluded["evaluation_status"] = "excluded"
        excluded["exclusion_reason"] = "predeclared_quality_gate"
        records["sealed-blind"].append(excluded)
        self.rewrite_bundle(descriptor_path, descriptor, records)

        collection = load_validated_collection(
            descriptor_path,
            self.root,
            self.root,
        )
        projection_bytes = build_sealed_input_projection(collection)
        projection = json.loads(projection_bytes)
        included = records["sealed-blind"][0]

        self.assertEqual(projection["schema_version"], 2)
        self.assertEqual(projection["manifest_record_count"], 2)
        self.assertEqual(projection["item_count"], 1)
        projected_ids = [item["id"] for item in projection["items"]]
        self.assertEqual(projected_ids, [included["id"]])
        self.assertNotIn(excluded["id"], projected_ids)
        self.assertNotIn(str(excluded["audio"]).encode("utf-8"), projection_bytes)
        self.assertNotIn(
            str(excluded["audio_sha256"]).encode("ascii"),
            projection_bytes,
        )

        reference_projection = self.sealed_projection_document(
            descriptor,
            records,
            references=True,
        )
        self.assertEqual(reference_projection["schema_version"], 2)
        self.assertEqual(reference_projection["manifest_record_count"], 2)
        self.assertEqual(reference_projection["item_count"], 2)
        self.assertNotIn("record_count", reference_projection)
        reference_items = reference_projection["items"]
        self.assertIsInstance(reference_items, list)
        excluded_reference = next(
            item for item in reference_items if item["id"] == excluded["id"]
        )
        self.assertEqual(excluded_reference["evaluation_status"], "excluded")
        self.assertEqual(
            excluded_reference["exclusion_reason"],
            "predeclared_quality_gate",
        )

    def test_custodian_replay_rejects_an_all_excluded_sealed_split(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        sealed_record = records["sealed-blind"][0]
        sealed_record["evaluation_status"] = "excluded"
        sealed_record["exclusion_reason"] = "predeclared_quality_gate"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        receipt_path = self.root / "export-receipt.json"

        with self.assertRaisesRegex(
            CustodianReplayError, "decode-eligible"
        ) as rejected:
            self.export_input_direct(
                descriptor_path=descriptor_path,
                candidate_manifest_path=candidate_manifest_path,
                input_path=input_path,
                lock_path=lock_path,
                receipt_path=receipt_path,
            )
        self.assertFalse(input_path.exists())
        self.assertFalse(lock_path.exists())
        self.assertFalse(receipt_path.exists())
        self.assertNotIn(
            str(sealed_record["raw_text"]), str(rejected.exception)
        )

    def test_custodian_cli_rejects_aliased_outputs_before_reading_inputs(self) -> None:
        shared_path = self.root / "shared-output.json"
        receipt_path = self.root / "receipt.json"
        completed = subprocess.run(
            [
                str(VENV_PYTHON),
                "-P",
                "-S",
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--descriptor",
                str(self.root / "missing-descriptor.json"),
                "--candidate-manifest",
                str(self.root / "missing-candidate.json"),
                "--candidate-registration-commit",
                "d" * 40,
                "--hypothesis-adapter-version",
                "identity-v1",
                "--output-input",
                str(shared_path),
                "--output-candidate-lock",
                str(shared_path),
                "--output-receipt",
                str(receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            env=SEALED_SUBPROCESS_ENVIRONMENT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"output paths must be distinct", completed.stderr)
        self.assertNotIn(b"missing-descriptor", completed.stderr)
        self.assertFalse(shared_path.exists())
        self.assertFalse(receipt_path.exists())

    def test_custodian_transitions_publish_completion_receipt_last(self) -> None:
        descriptor_path, _, records = self.write_bundle()
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        export_receipt_path = self.root / "export-receipt.json"
        real_writer = replay_cli.write_atomic_outputs

        with mock.patch.object(
            replay_cli,
            "write_atomic_outputs",
            wraps=real_writer,
        ) as export_writer:
            self.export_input_direct(
                descriptor_path=descriptor_path,
                candidate_manifest_path=candidate_manifest_path,
                input_path=input_path,
                lock_path=lock_path,
                receipt_path=export_receipt_path,
            )
        export_writer.assert_called_once()
        self.assertEqual(
            export_writer.call_args.args[0][-1][0],
            export_receipt_path,
        )

        raw_predictions_path, execution_envelope_path = (
            self.write_valid_runner_handoff(
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                sealed_record=records["sealed-blind"][0],
            )
        )
        predictions_path = self.root / "predictions.json"
        prediction_receipt_path = self.root / "prediction-receipt.json"
        with mock.patch.object(
            replay_cli,
            "write_atomic_outputs",
            wraps=real_writer,
        ) as freeze_writer:
            self.freeze_predictions_direct(
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                raw_predictions_path=raw_predictions_path,
                execution_envelope_path=execution_envelope_path,
                predictions_path=predictions_path,
                prediction_receipt_path=prediction_receipt_path,
            )
        freeze_writer.assert_called_once()
        self.assertEqual(
            freeze_writer.call_args.args[0][-1][0],
            prediction_receipt_path,
        )

        core_path = self.root / "core.json"
        score_receipt_path = self.root / "score-receipt.json"
        with mock.patch.object(
            replay_cli,
            "write_atomic_outputs",
            wraps=real_writer,
        ) as score_writer:
            self.score_direct(
                descriptor_path=descriptor_path,
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                predictions_path=predictions_path,
                execution_envelope_path=execution_envelope_path,
                prediction_receipt_path=prediction_receipt_path,
                core_path=core_path,
                score_receipt_path=score_receipt_path,
            )
        score_writer.assert_called_once()
        self.assertEqual(
            score_writer.call_args.args[0][-1][0],
            score_receipt_path,
        )

    def test_registered_candidate_binding_fails_before_later_replay_effects(
        self,
    ) -> None:
        descriptor_path, _, records = self.write_bundle()
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        export_receipt_path = self.root / "export-receipt.json"
        self.export_input_direct(
            descriptor_path=descriptor_path,
            candidate_manifest_path=candidate_manifest_path,
            input_path=input_path,
            lock_path=lock_path,
            receipt_path=export_receipt_path,
        )
        raw_predictions_path, execution_envelope_path = (
            self.write_valid_runner_handoff(
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                sealed_record=records["sealed-blind"][0],
            )
        )
        predictions_path = self.root / "predictions.json"
        prediction_receipt_path = self.root / "prediction-receipt.json"
        self.freeze_predictions_direct(
            input_path=input_path,
            lock_path=lock_path,
            export_receipt_path=export_receipt_path,
            raw_predictions_path=raw_predictions_path,
            execution_envelope_path=execution_envelope_path,
            predictions_path=predictions_path,
            prediction_receipt_path=prediction_receipt_path,
        )
        candidate_lock_document = load_candidate_lock(lock_path).document

        freeze_validator = mock.Mock(
            side_effect=CustodianReplayError("registered candidate sentinel")
        )
        failed_predictions_path = self.root / "must-not-freeze.json"
        failed_prediction_receipt_path = self.root / "must-not-freeze-receipt.json"
        with mock.patch.object(
            replay_cli,
            "write_atomic_outputs",
        ) as freeze_writer, mock.patch.object(
            replay_cli,
            "load_custodian_receipt",
        ) as freeze_receipt_loader:
            with self.assertRaisesRegex(
                CustodianReplayError,
                "registered candidate sentinel",
            ):
                self.freeze_predictions_direct(
                    input_path=input_path,
                    lock_path=lock_path,
                    export_receipt_path=export_receipt_path,
                    raw_predictions_path=raw_predictions_path,
                    execution_envelope_path=execution_envelope_path,
                    predictions_path=failed_predictions_path,
                    prediction_receipt_path=failed_prediction_receipt_path,
                    registered_candidate_validator=freeze_validator,
                )
        freeze_validator.assert_called_once_with(candidate_lock_document)
        freeze_receipt_loader.assert_not_called()
        freeze_writer.assert_not_called()
        self.assertFalse(failed_predictions_path.exists())
        self.assertFalse(failed_prediction_receipt_path.exists())

        score_validator = mock.Mock(
            side_effect=CustodianReplayError("registered candidate sentinel")
        )
        failed_core_path = self.root / "must-not-score.json"
        failed_score_receipt_path = self.root / "must-not-score-receipt.json"
        with mock.patch.object(
            replay_cli,
            "write_atomic_outputs",
        ) as score_writer, mock.patch.object(
            replay_cli,
            "load_custodian_receipt",
        ) as score_receipt_loader, mock.patch.object(
            replay_cli,
            "load_validated_collection",
        ) as sealed_collection_loader:
            with self.assertRaisesRegex(
                CustodianReplayError,
                "registered candidate sentinel",
            ):
                self.score_direct(
                    descriptor_path=descriptor_path,
                    input_path=input_path,
                    lock_path=lock_path,
                    export_receipt_path=export_receipt_path,
                    predictions_path=predictions_path,
                    execution_envelope_path=execution_envelope_path,
                    prediction_receipt_path=prediction_receipt_path,
                    core_path=failed_core_path,
                    score_receipt_path=failed_score_receipt_path,
                    registered_candidate_validator=score_validator,
                )
        score_validator.assert_called_once_with(candidate_lock_document)
        score_receipt_loader.assert_not_called()
        sealed_collection_loader.assert_not_called()
        score_writer.assert_not_called()
        self.assertFalse(failed_core_path.exists())
        self.assertFalse(failed_score_receipt_path.exists())

    def test_custodian_export_rejects_leaf_replacement_after_descriptor_read(
        self,
    ) -> None:
        descriptor_path, _, records = self.write_bundle()
        sealed_record = records["sealed-blind"][0]
        audio_path = self.root / str(sealed_record["audio"])
        original_payload = audio_path.read_bytes()
        displaced_path = audio_path.with_name("displaced-sealed-audio.wav")
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        outputs = (
            self.root / "sealed-input.json",
            self.root / "candidate-lock.json",
            self.root / "export-receipt.json",
        )
        real_audio_loader = replay_cli.load_verified_audio_items
        real_os_read = os.read
        leaf_replaced = False

        def replace_leaf_after_read(descriptor: int, count: int) -> bytes:
            nonlocal leaf_replaced
            chunk = real_os_read(descriptor, count)
            if chunk and not leaf_replaced:
                leaf_replaced = True
                audio_path.replace(displaced_path)
                audio_path.write_bytes(original_payload)
            return chunk

        def racing_audio_loader(*args, **kwargs):
            with mock.patch.object(
                os,
                "read",
                side_effect=replace_leaf_after_read,
            ):
                return real_audio_loader(*args, **kwargs)

        with mock.patch.object(
            replay_cli,
            "load_verified_audio_items",
            side_effect=racing_audio_loader,
        ):
            with self.assertRaisesRegex(
                SealedDecoderError,
                "path changed while it was verified",
            ):
                self.export_input_direct(
                    descriptor_path=descriptor_path,
                    candidate_manifest_path=candidate_manifest_path,
                    input_path=outputs[0],
                    lock_path=outputs[1],
                    receipt_path=outputs[2],
                )
        self.assertTrue(leaf_replaced)
        self.assertTrue(all(not path.exists() for path in outputs))

    def test_custodian_export_and_sealed_score_are_byte_stable(self) -> None:
        self.assertEqual(
            runtime_identity(REPOSITORY_ROOT)["installed_dependency_count"],
            71,
        )
        descriptor_path, _, records = self.write_bundle()
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        export_receipt_path = self.root / "export-receipt.json"
        self.export_input_direct(
            descriptor_path=descriptor_path,
            candidate_manifest_path=candidate_manifest_path,
            input_path=input_path,
            lock_path=lock_path,
            receipt_path=export_receipt_path,
        )
        export_receipt = load_custodian_receipt(export_receipt_path).document
        self.assertEqual(export_receipt["split"], "sealed-blind")
        self.assertEqual(export_receipt["decode_item_count"], 1)
        self.assertEqual(input_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(lock_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(export_receipt_path.stat().st_mode & 0o777, 0o600)

        sealed_input = load_sealed_input_projection(input_path)
        candidate_lock = load_candidate_lock(lock_path)
        sealed_records = records["sealed-blind"]
        prediction_items = [
            {
                "id": record["id"],
                "raw_text": record["raw_text"],
                "status": "ok",
                "reason_code": None,
            }
            for record in sealed_records
            if record["evaluation_status"] == "included"
        ]
        raw_predictions_path = self.root / "raw-predictions.jsonl"
        raw_predictions_path.write_bytes(
            b"".join(canonical_json_bytes(item) for item in prediction_items)
        )
        raw_predictions_path.chmod(0o600)
        execution_envelope_path = self.root / "execution-envelope.json"
        execution_envelope = self.write_execution_envelope(
            sealed_input,
            candidate_lock,
            prediction_items,
            raw_predictions_path,
            export_receipt_path,
            execution_envelope_path,
        )
        mismatched_envelope_path = self.root / "mismatched-warmup-envelope.json"
        self.write_execution_envelope(
            sealed_input,
            candidate_lock,
            prediction_items,
            raw_predictions_path,
            export_receipt_path,
            mismatched_envelope_path,
            warmup_runs=0,
        )
        with self.assertRaisesRegex(CustodianReplayError, "warmup count"):
            self.freeze_predictions_direct(
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                raw_predictions_path=raw_predictions_path,
                execution_envelope_path=mismatched_envelope_path,
                predictions_path=self.root / "mismatched-warmup-predictions.json",
                prediction_receipt_path=(
                    self.root / "mismatched-warmup-receipt.json"
                ),
            )
        predictions_path = self.root / "predictions.json"
        prediction_receipt_path = self.root / "prediction-receipt.json"
        self.freeze_predictions_direct(
            input_path=input_path,
            lock_path=lock_path,
            export_receipt_path=export_receipt_path,
            raw_predictions_path=raw_predictions_path,
            execution_envelope_path=execution_envelope_path,
            predictions_path=predictions_path,
            prediction_receipt_path=prediction_receipt_path,
        )
        prediction_receipt = load_custodian_receipt(
            prediction_receipt_path
        ).document
        self.assertEqual(prediction_receipt["missing_prediction_count"], 0)
        self.assertEqual(predictions_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(prediction_receipt_path.stat().st_mode & 0o777, 0o600)

        core_paths = [self.root / "first-core.json", self.root / "second-core.json"]
        receipts: list[dict[str, object]] = []
        for index, core_path in enumerate(core_paths):
            receipt_path = self.root / f"score-receipt-{index}.json"
            self.score_direct(
                descriptor_path=descriptor_path,
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                predictions_path=predictions_path,
                execution_envelope_path=execution_envelope_path,
                prediction_receipt_path=prediction_receipt_path,
                core_path=core_path,
                score_receipt_path=receipt_path,
            )
            receipts.append(load_custodian_receipt(receipt_path).document)
            self.assertEqual(core_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)

        self.assertEqual(core_paths[0].read_bytes(), core_paths[1].read_bytes())
        report = json.loads(core_paths[0].read_bytes())
        validate_core_report(report)
        self.assertEqual(
            report["configuration"]["evaluation_scope"],
            {"kind": "split", "split": "sealed-blind"},
        )
        self.assertEqual(
            [item["id"] for item in report["items"]],
            [record["id"] for record in sealed_records],
        )
        self.assertEqual(report["counts"]["failed_count"], 0)
        self.assertEqual(report["aggregate"]["cer"]["errors"], 0)
        self.assertEqual(
            core_paths[0].read_bytes(),
            canonical_core_bytes(report),
        )
        self.assertEqual(receipts[0]["core_sha256"], core_report_sha256(report))
        self.assertEqual(receipts[0], receipts[1])
        self.assertEqual(receipts[0]["public_release"]["state"], "withheld")
        self.assertEqual(receipts[0]["candidate_lock_sha256"], candidate_lock.sha256)
        self.assertEqual(
            receipts[0]["prediction_artifact_sha256"],
            prediction_receipt["prediction_artifact_sha256"],
        )
        self.assertEqual(
            receipts[0]["scorer_code_commit"],
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip(),
        )

        cer = report["aggregate"]["cer"]
        mer = report["aggregate"]["mer"]
        terminal = self.planned_candidate_manifest(sha256_file(descriptor_path))
        terminal["decision"] = "accept"
        terminal["metrics"] = {
            "content_cer": cer["errors"] / cer["reference_units"],
            "substitutions": cer["substitutions"],
            "deletions": cer["deletions"],
            "insertions": cer["insertions"],
            "reference_units": cer["reference_units"],
            "utterance_count": report["counts"]["utterance_count"],
            "failed_count": report["counts"]["failed_count"],
            "excluded_count": report["counts"]["excluded_count"],
            "mer": mer["errors"] / mer["reference_units"],
            "rtf_p50": 0.1,
            "rtf_p95": 0.1,
            "peak_rss_mb": 1.0,
            "rtf_attempted_count": 1,
            "retried_count": 0,
            "model_load_seconds": 0.2,
            "cold_inference_seconds": execution_envelope["measurement"][
                "cold_inference_ns"
            ]
            / 1_000_000_000,
            "cold_start_seconds": execution_envelope["measurement"][
                "cold_start_ns"
            ]
            / 1_000_000_000,
            "warm_wall_seconds": execution_envelope["measurement"][
                "measured_wall_ns"
            ]
            / 1_000_000_000,
            "warm_audio_seconds": execution_envelope["measurement"][
                "measured_audio_seconds"
            ],
        }
        terminal["artifacts"] = [
            {
                "kind": "other",
                "path": str(input_path),
                "sha256": receipts[0]["input_projection_sha256"],
            },
            {
                "kind": "other",
                "path": str(lock_path),
                "sha256": receipts[0]["candidate_lock_sha256"],
            },
            {
                "kind": "other",
                "path": str(export_receipt_path),
                "sha256": receipts[0]["input_export_receipt_sha256"],
            },
            {
                "kind": "prediction",
                "path": str(predictions_path),
                "sha256": receipts[0]["prediction_artifact_sha256"],
            },
            {
                "kind": "report",
                "path": str(execution_envelope_path),
                "sha256": receipts[0]["execution_envelope_sha256"],
            },
            {
                "kind": "other",
                "path": str(prediction_receipt_path),
                "sha256": receipts[0]["prediction_freeze_receipt_sha256"],
            },
            {
                "kind": "report",
                "path": str(core_paths[0]),
                "sha256": receipts[0]["core_sha256"],
            },
            {
                "kind": "other",
                "path": str(self.root / "score-receipt-0.json"),
                "sha256": sha256_bytes(
                    canonical_custodian_receipt_bytes(receipts[0])
                ),
            },
        ]
        validate_terminal_manifest_for_receipt(
            terminal,
            receipts[0],
            report,
            execution_envelope,
            export_receipt,
            prediction_receipt,
        )
        for receipt_path in (
            export_receipt_path,
            prediction_receipt_path,
            self.root / "score-receipt-0.json",
        ):
            receipt_bytes = receipt_path.read_bytes()
            for record in sealed_records:
                self.assertNotIn(
                    str(record["raw_text"]).encode("utf-8"), receipt_bytes
                )
        self.assertEqual(list(self.root.glob("*summary*")), [])

        missing_raw_path = self.root / "missing-raw-predictions.jsonl"
        missing_raw_path.write_bytes(b"")
        missing_raw_path.chmod(0o600)
        missing_execution_envelope_path = (
            self.root / "missing-execution-envelope.json"
        )
        self.write_execution_envelope(
            sealed_input,
            candidate_lock,
            [],
            missing_raw_path,
            export_receipt_path,
            missing_execution_envelope_path,
        )
        missing_predictions_path = self.root / "missing-predictions.json"
        missing_prediction_receipt_path = (
            self.root / "missing-prediction-receipt.json"
        )
        self.freeze_predictions_direct(
            input_path=input_path,
            lock_path=lock_path,
            export_receipt_path=export_receipt_path,
            raw_predictions_path=missing_raw_path,
            execution_envelope_path=missing_execution_envelope_path,
            predictions_path=missing_predictions_path,
            prediction_receipt_path=missing_prediction_receipt_path,
        )
        missing_prediction_receipt = load_custodian_receipt(
            missing_prediction_receipt_path
        ).document
        self.assertEqual(missing_prediction_receipt["prediction_item_count"], 0)
        self.assertEqual(missing_prediction_receipt["missing_prediction_count"], 1)

        missing_core_path = self.root / "missing-core.json"
        missing_score_receipt_path = self.root / "missing-score-receipt.json"
        self.score_direct(
            descriptor_path=descriptor_path,
            input_path=input_path,
            lock_path=lock_path,
            export_receipt_path=export_receipt_path,
            predictions_path=missing_predictions_path,
            execution_envelope_path=missing_execution_envelope_path,
            prediction_receipt_path=missing_prediction_receipt_path,
            core_path=missing_core_path,
            score_receipt_path=missing_score_receipt_path,
        )
        missing_report = json.loads(missing_core_path.read_bytes())
        self.assertEqual(missing_report["counts"]["failed_count"], 1)
        self.assertEqual(missing_report["items"][0]["status"], "failed")
        self.assertEqual(
            missing_report["items"][0]["reason_code"], "missing_prediction"
        )

    def test_custodian_export_rejects_symlinked_audio_leaf_without_outputs(self):
        descriptor_path, _, records = self.write_bundle()
        sealed_record = records["sealed-blind"][0]
        audio_path = self.root / str(sealed_record["audio"])
        target = audio_path.with_name("sealed-leaf-target.wav")
        audio_path.replace(target)
        audio_path.symlink_to(target.name)
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        outputs = (
            self.root / "sealed-input.json",
            self.root / "candidate-lock.json",
            self.root / "export-receipt.json",
        )
        with self.assertRaisesRegex(SealedDecoderError, "cannot safely open"):
            self.export_input_direct(
                descriptor_path=descriptor_path,
                candidate_manifest_path=candidate_manifest_path,
                input_path=outputs[0],
                lock_path=outputs[1],
                receipt_path=outputs[2],
            )
        self.assertTrue(all(not path.exists() for path in outputs))

    def test_custodian_export_rejects_symlinked_audio_parent_without_outputs(self):
        descriptor_path, _, _ = self.write_bundle()
        audio_directory = self.root / "audio"
        target_directory = self.root / "real-audio"
        audio_directory.replace(target_directory)
        audio_directory.symlink_to(target_directory.name, target_is_directory=True)
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        outputs = (
            self.root / "sealed-input.json",
            self.root / "candidate-lock.json",
            self.root / "export-receipt.json",
        )
        with self.assertRaisesRegex(SealedDecoderError, "cannot safely open"):
            self.export_input_direct(
                descriptor_path=descriptor_path,
                candidate_manifest_path=candidate_manifest_path,
                input_path=outputs[0],
                lock_path=outputs[1],
                receipt_path=outputs[2],
            )
        self.assertTrue(all(not path.exists() for path in outputs))

    def test_custodian_preflight_rejects_changed_prediction_lock(self) -> None:
        descriptor_path, _, records = self.write_bundle()
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        export_receipt_path = self.root / "export-receipt.json"
        self.export_input_direct(
            descriptor_path=descriptor_path,
            candidate_manifest_path=candidate_manifest_path,
            input_path=input_path,
            lock_path=lock_path,
            receipt_path=export_receipt_path,
        )

        sealed_input = load_sealed_input_projection(input_path)
        candidate_lock = load_candidate_lock(lock_path)
        sealed_record = records["sealed-blind"][0]
        raw_predictions_path = self.root / "raw-predictions.jsonl"
        prediction_items = [
            {
                "id": sealed_record["id"],
                "raw_text": sealed_record["raw_text"],
                "status": "ok",
                "reason_code": None,
            }
        ]
        raw_predictions_path.write_bytes(
            b"".join(canonical_json_bytes(item) for item in prediction_items)
        )
        raw_predictions_path.chmod(0o600)
        execution_envelope_path = self.root / "execution-envelope.json"
        execution_envelope = self.write_execution_envelope(
            sealed_input,
            candidate_lock,
            prediction_items,
            raw_predictions_path,
            export_receipt_path,
            execution_envelope_path,
        )
        prediction_bundle = build_prediction_bundle(
            sealed_input,
            candidate_lock.sha256,
            prediction_items,
            input_export_receipt_sha256=sha256_file(export_receipt_path),
            raw_predictions_sha256=sha256_file(raw_predictions_path),
            execution_envelope_sha256=sha256_file(execution_envelope_path),
            hypothesis_adapter_version="identity-v1",
        )
        prediction_bundle["candidate_lock_sha256"] = digest("different-lock")
        predictions_path = self.root / "predictions.json"
        predictions_path.write_bytes(
            canonical_prediction_bundle_bytes(prediction_bundle)
        )
        predictions_path.chmod(0o600)
        prediction_receipt_path = self.root / "prediction-receipt.json"
        prediction_receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": PREDICTION_FREEZE_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": candidate_lock.document["candidate"][
                "experiment_id"
            ],
            "dataset_id": sealed_input.document["dataset_id"],
            "revision": sealed_input.document["revision"],
            "split": "sealed-blind",
            "expected_decode_item_count": 1,
            "prediction_item_count": 1,
            "missing_prediction_count": 0,
            "input_projection_sha256": sealed_input.sha256,
            "candidate_lock_sha256": candidate_lock.sha256,
            "candidate_freeze_sha256": candidate_lock.document[
                "candidate_freeze_sha256"
            ],
            "candidate_registration_commit": candidate_lock.document[
                "candidate_registration_commit"
            ],
            "candidate_manifest_path": candidate_lock.document[
                "candidate_manifest_path"
            ],
            "candidate_manifest_sha256": candidate_lock.document[
                "candidate_manifest_sha256"
            ],
            "hypothesis_adapter_version": "identity-v1",
            "prediction_artifact_sha256": sha256_file(predictions_path),
            "prediction_items_sha256": prediction_bundle["items_sha256"],
            "input_export_receipt_sha256": sha256_file(export_receipt_path),
            "raw_predictions_sha256": sha256_file(raw_predictions_path),
            "execution_envelope_sha256": sha256_file(execution_envelope_path),
            "runner_code_commit": execution_envelope["runner"]["code_commit"],
            "runner_source_sha256": execution_envelope["runner"][
                "source_sha256"
            ],
        }
        prediction_receipt_path.write_bytes(
            canonical_custodian_receipt_bytes(prediction_receipt)
        )
        prediction_receipt_path.chmod(0o600)
        core_path = self.root / "must-not-exist.json"
        score_receipt_path = self.root / "must-not-exist-receipt.json"
        with self.assertRaisesRegex(
            CustodianReplayError,
            "prediction bundle does not bind the supplied candidate lock",
        ) as rejected:
            self.score_direct(
                descriptor_path=descriptor_path,
                input_path=input_path,
                lock_path=lock_path,
                export_receipt_path=export_receipt_path,
                predictions_path=predictions_path,
                execution_envelope_path=execution_envelope_path,
                prediction_receipt_path=prediction_receipt_path,
                core_path=core_path,
                score_receipt_path=score_receipt_path,
            )
        self.assertFalse(core_path.exists())
        self.assertFalse(score_receipt_path.exists())
        self.assertNotIn(str(sealed_record["raw_text"]), str(rejected.exception))
        self.assertNotIn(str(sealed_record["id"]), str(rejected.exception))

    def test_blind_projection_hashes_are_rebuilt_from_sealed_contract(self) -> None:
        for field in (
            "input_projection_sha256",
            "reference_projection_sha256",
        ):
            with self.subTest(field=field):
                descriptor_path, descriptor, _ = self.write_bundle()
                blind = descriptor["blind"]
                self.assertIsInstance(blind, dict)
                blind[field] = digest(f"wrong-{field}")
                descriptor_path.write_bytes(canonical_json_bytes(descriptor))
                with self.assertRaisesRegex(CollectionValidationError, field):
                    validate_collection(descriptor_path, self.root, self.root)

    def test_descriptor_must_be_canonical_and_reject_unknown_fields(self) -> None:
        descriptor_path, descriptor, _ = self.write_bundle()
        descriptor_path.write_text(
            json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaisesRegex(CollectionValidationError, "canonical JSON"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor["unexpected"] = True
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "unknown field"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_manifest_hash_count_and_record_identity_are_bound(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        manifests = descriptor["manifests"]
        self.assertIsInstance(manifests, list)
        manifests[0]["record_count"] = 2
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "record_count mismatch"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["smoke"][0]["raw_text"] = "被修改"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "reference_sha256"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_dedup_report_semantically_binds_collection_and_cluster_inventory(
        self,
    ) -> None:
        mutations = {
            "dataset_id": "OTHER-DATASET",
            "revision": "v9.9",
            "method": "different-method",
            "version": "v2",
            "threshold": 0.5,
            "config_sha256": digest("different-dedup-config"),
            "manifest_inventory_sha256": digest("different-manifests"),
            "record_count": 4,
            "record_cluster_inventory_sha256": digest("different-clusters"),
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                descriptor_path, descriptor, _ = self.write_bundle()
                report = json.loads(self.dedup_report.read_text(encoding="utf-8"))
                report[field] = value
                self.dedup_report.write_bytes(canonical_json_bytes(report))
                dedup = descriptor["dedup"]
                self.assertIsInstance(dedup, dict)
                dedup["report_sha256"] = sha256_file(self.dedup_report)
                descriptor_path.write_bytes(canonical_json_bytes(descriptor))
                with self.assertRaisesRegex(CollectionValidationError, field):
                    validate_collection(descriptor_path, self.root, self.root)

    def test_dedup_report_is_canonical_and_rejects_unknown_fields(self) -> None:
        descriptor_path, descriptor, _ = self.write_bundle()
        report = json.loads(self.dedup_report.read_text(encoding="utf-8"))
        self.dedup_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dedup = descriptor["dedup"]
        self.assertIsInstance(dedup, dict)
        dedup["report_sha256"] = sha256_file(self.dedup_report)
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "canonical JSON"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, _ = self.write_bundle()
        report = json.loads(self.dedup_report.read_text(encoding="utf-8"))
        report["unexpected"] = True
        self.dedup_report.write_bytes(canonical_json_bytes(report))
        dedup = descriptor["dedup"]
        self.assertIsInstance(dedup, dict)
        dedup["report_sha256"] = sha256_file(self.dedup_report)
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "unknown field"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_record_unknown_field_and_scenario_tag_are_rejected(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["unexpected"] = True
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "unknown field"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["scenario_tags"] = ["language:zh", "invented:condition"]
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "unknown value"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["scenario_tags"] = [
            "noise:clean",
            "environment:near-field",
        ]
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "language"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_exclusion_reason_is_a_stable_reason_code_not_free_text(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["evaluation_status"] = "excluded"
        records["dev"][0]["exclusion_reason"] = "manual review failed"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "snake_case"):
            validate_collection(descriptor_path, self.root, self.root)

        records["dev"][0]["exclusion_reason"] = "predeclared_quality_gate"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        summary = validate_collection(descriptor_path, self.root, self.root)
        self.assertEqual(summary["excluded_count"], 1)

    def test_speaker_session_source_lineage_and_dedup_cannot_cross_splits(self) -> None:
        for field in (
            "speaker_id",
            "session_id",
            "source_recording_id",
            "lineage_group_id",
            "dedup_cluster_id",
        ):
            with self.subTest(field=field):
                descriptor_path, descriptor, records = self.write_bundle()
                records["dev"][0][field] = records["smoke"][0][field]
                self.rewrite_bundle(descriptor_path, descriptor, records)
                with self.assertRaisesRegex(CollectionValidationError, "crosses splits"):
                    validate_collection(descriptor_path, self.root, self.root)

    def test_duplicate_audio_is_rejected_even_with_distinct_ids(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        smoke = records["smoke"][0]
        dev = records["dev"][0]
        for field in (
            "audio",
            "audio_sha256",
            "duration_seconds",
            "source_audio_sha256",
        ):
            dev[field] = smoke[field]
        self.rewrite_bundle(descriptor_path, descriptor, records)

        with self.assertRaisesRegex(CollectionValidationError, "duplicate audio_sha256"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_rights_provenance_and_frozen_scoring_versions_are_required(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["rights_group_id"] = "missing-rights"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "unknown rights_group_id"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["provenance_group_id"] = "missing-provenance"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(
            CollectionValidationError, "unknown provenance_group_id"
        ):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, _ = self.write_bundle()
        provenance = descriptor["provenance_groups"]
        self.assertIsInstance(provenance, list)
        provenance[0]["source_revision"] = "refs/heads/main"
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "floating"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, _ = self.write_bundle()
        descriptor["normalizer_version"] = "unknown-normalizer"
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "normalizer_version"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, _ = self.write_bundle()
        descriptor["mer_tokenizer_version"] = "unknown-tokenizer"
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "mer_tokenizer_version"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["normalizer_version"] = "unknown-normalizer"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "normalizer_version"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["mer_tokenizer_version"] = "unknown-tokenizer"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "mer_tokenizer_version"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_derived_lineage_requires_source_and_matching_recipe(self) -> None:
        descriptor_path, descriptor, records = self.write_bundle()
        dev = records["dev"][0]
        dev["lineage_kind"] = "derived"
        dev["derived_from_id"] = None
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "derived_from_id"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        records["dev"][0]["transform_recipe_sha256"] = digest("different-recipe")
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "transform_recipe"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        parent = records["dev"][0]
        derived = self.record(
            "dev",
            4,
            access_class="restricted",
            rights_group_id="rights-restricted",
        )
        derived.update(
            {
                "lineage_kind": "derived",
                "derived_from_id": parent["id"],
                "speaker_id": parent["speaker_id"],
                "session_id": parent["session_id"],
                "source_recording_id": parent["source_recording_id"],
                "lineage_group_id": parent["lineage_group_id"],
                "dedup_cluster_id": parent["dedup_cluster_id"],
                "source_audio_sha256": parent["source_audio_sha256"],
            }
        )
        records["dev"].append(derived)
        self.rewrite_bundle(descriptor_path, descriptor, records)
        validate_collection(descriptor_path, self.root, self.root)

        derived["source_audio_sha256"] = derived["audio_sha256"]
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "source_audio_sha256"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_sealed_blind_descriptor_and_record_boundaries_are_enforced(self) -> None:
        descriptor_path, descriptor, _ = self.write_bundle()
        manifests = descriptor["manifests"]
        self.assertIsInstance(manifests, list)
        manifests[2]["reference_access"] = "restricted"
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
        with self.assertRaisesRegex(CollectionValidationError, "must be 'sealed'"):
            validate_collection(descriptor_path, self.root, self.root)

        descriptor_path, descriptor, records = self.write_bundle()
        blind = records["sealed-blind"][0]
        blind["access_class"] = "restricted"
        blind["rights_group_id"] = "rights-restricted"
        self.rewrite_bundle(descriptor_path, descriptor, records)
        with self.assertRaisesRegex(CollectionValidationError, "access_class"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_schema_only_descriptor_cannot_be_used_as_frozen_evidence(self) -> None:
        descriptor_path, descriptor, _ = self.write_bundle()
        descriptor["state"] = "schema-only"
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))

        parsed = load_collection_descriptor(descriptor_path, require_frozen=False)
        self.assertEqual(parsed.dataset_id, "LAB-SEED-TEST")
        with self.assertRaisesRegex(CollectionValidationError, "not evidence"):
            validate_collection(descriptor_path, self.root, self.root)

    def test_tracked_examples_are_explicitly_schema_only(self) -> None:
        descriptor_path = (
            REPOSITORY_ROOT
            / "eval/manifests/collection-descriptor.schema-only.example.json"
        )
        parsed = load_collection_descriptor(descriptor_path, require_frozen=False)
        self.assertEqual(parsed.dataset_id, "LAB-SEED-SCHEMA-ONLY")
        self.assertEqual(parsed.normalizer_version, NORMALIZER_VERSION)
        self.assertEqual(parsed.mer_tokenizer_version, MER_TOKENIZER_VERSION)
        with self.assertRaisesRegex(CollectionValidationError, "not evidence"):
            load_collection_descriptor(descriptor_path)

        record_path = (
            REPOSITORY_ROOT
            / "eval/manifests/collection-record.schema-only.example.jsonl"
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(set(record), set(RECORD_FIELDS))
        self.assertEqual(record["dataset_id"], "LAB-SEED-SCHEMA-ONLY")
        self.assertEqual(record["normalizer_version"], NORMALIZER_VERSION)
        self.assertEqual(record["mer_tokenizer_version"], MER_TOKENIZER_VERSION)


if __name__ == "__main__":
    unittest.main()
