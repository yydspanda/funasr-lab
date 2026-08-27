from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

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
from eval.custodian_replay import canonical_prediction_bundle_bytes
from eval.custodian_replay import canonical_custodian_receipt_bytes
from eval.custodian_replay import load_candidate_lock
from eval.custodian_replay import load_custodian_receipt
from eval.custodian_replay import load_sealed_input_projection
from eval.custodian_replay import validate_terminal_manifest_for_receipt
from eval.normalizers import NORMALIZER_VERSION
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import record_input_sha256
from eval.scoring import MER_TOKENIZER_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
        return {
            "schema_version": 1,
            "experiment_id": "EXP-20260827-001-eval-fixture",
            "task_id": "EVAL-01",
            "hypothesis": (
                "The frozen vanilla ASR candidate will produce a replayable "
                "sealed-blind prediction artifact without exposing references."
            ),
            "upstream_commit": "eedd4e22d10dc2e81d9c2bb321edb3750253964b",
            "code_commit": "c1aa9d3ba29ac8e1a1791147a42e9b9920d97843",
            "models": [
                {
                    "role": "asr",
                    "identifier": "iic/fixture-asr-model",
                    "revision": "28fe27c56aab3861cf77ae065b2bfc2aa3ab9692",
                    "sha256": digest("fixture-model-inventory"),
                }
            ],
            "config_sha256": digest("fixture-effective-config"),
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
                    "scripts/replay_asr_evaluation.py",
                    "freeze-predictions",
                    "--input-projection",
                    "eval/private/fixture.input.json",
                    "--candidate-lock",
                    "eval/private/fixture.lock.json",
                    "--raw-predictions",
                    "eval/private/fixture.raw.jsonl",
                    "--hypothesis-adapter-version",
                    "identity-v1",
                    "--output-predictions",
                    "eval/private/fixture.predictions.json",
                    "--output-receipt",
                    "eval/private/fixture.prediction-receipt.json",
                ],
                "environment": {"PYTHONHASHSEED": "0"},
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
        input_path = self.root / "must-not-export-input.json"
        lock_path = self.root / "must-not-export-lock.json"
        receipt_path = self.root / "must-not-export-receipt.json"

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--descriptor",
                str(descriptor_path),
                "--collection-root",
                str(self.root),
                "--audio-root",
                str(self.root),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--hypothesis-adapter-version",
                "identity-v1",
                "--output-input",
                str(input_path),
                "--output-candidate-lock",
                str(lock_path),
                "--output-receipt",
                str(receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"decode-eligible", completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertFalse(input_path.exists())
        self.assertFalse(lock_path.exists())
        self.assertFalse(receipt_path.exists())
        self.assertNotIn(
            str(sealed_record["raw_text"]).encode("utf-8"), completed.stderr
        )

    def test_custodian_cli_rejects_aliased_outputs_before_reading_inputs(self) -> None:
        shared_path = self.root / "shared-output.json"
        receipt_path = self.root / "receipt.json"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--descriptor",
                str(self.root / "missing-descriptor.json"),
                "--candidate-manifest",
                str(self.root / "missing-candidate.json"),
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
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"output paths must be distinct", completed.stderr)
        self.assertNotIn(b"missing-descriptor", completed.stderr)
        self.assertFalse(shared_path.exists())
        self.assertFalse(receipt_path.exists())

    def test_custodian_export_and_sealed_score_are_byte_stable(self) -> None:
        descriptor_path, _, records = self.write_bundle()
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        export_receipt_path = self.root / "export-receipt.json"
        exported = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--descriptor",
                str(descriptor_path),
                "--collection-root",
                str(self.root),
                "--audio-root",
                str(self.root),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--hypothesis-adapter-version",
                "identity-v1",
                "--output-input",
                str(input_path),
                "--output-candidate-lock",
                str(lock_path),
                "--output-receipt",
                str(export_receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(exported.returncode, 0, exported.stderr.decode())
        self.assertEqual(exported.stdout, b"")
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
        predictions_path = self.root / "predictions.json"
        prediction_receipt_path = self.root / "prediction-receipt.json"
        frozen = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "freeze-predictions",
                "--input-projection",
                str(input_path),
                "--candidate-lock",
                str(lock_path),
                "--raw-predictions",
                str(raw_predictions_path),
                "--hypothesis-adapter-version",
                "identity-v1",
                "--output-predictions",
                str(predictions_path),
                "--output-receipt",
                str(prediction_receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(frozen.returncode, 0, frozen.stderr.decode())
        self.assertEqual(frozen.stdout, b"")
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
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/replay_asr_evaluation.py",
                    "score",
                    "--descriptor",
                    str(descriptor_path),
                    "--collection-root",
                    str(self.root),
                    "--audio-root",
                    str(self.root),
                    "--input-projection",
                    str(input_path),
                    "--candidate-lock",
                    str(lock_path),
                    "--predictions",
                    str(predictions_path),
                    "--output-core",
                    str(core_path),
                    "--output-receipt",
                    str(receipt_path),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(completed.stdout, b"")
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
            "mer": mer["errors"] / mer["reference_units"],
            "rtf_p50": 0.1,
            "rtf_p95": 0.1,
            "peak_rss_mb": 1.0,
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
                "kind": "prediction",
                "path": str(predictions_path),
                "sha256": receipts[0]["prediction_artifact_sha256"],
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
        validate_terminal_manifest_for_receipt(terminal, receipts[0], report)
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
        missing_predictions_path = self.root / "missing-predictions.json"
        missing_prediction_receipt_path = (
            self.root / "missing-prediction-receipt.json"
        )
        frozen_missing = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "freeze-predictions",
                "--input-projection",
                str(input_path),
                "--candidate-lock",
                str(lock_path),
                "--raw-predictions",
                str(missing_raw_path),
                "--hypothesis-adapter-version",
                "identity-v1",
                "--output-predictions",
                str(missing_predictions_path),
                "--output-receipt",
                str(missing_prediction_receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(
            frozen_missing.returncode, 0, frozen_missing.stderr.decode()
        )
        missing_prediction_receipt = load_custodian_receipt(
            missing_prediction_receipt_path
        ).document
        self.assertEqual(missing_prediction_receipt["prediction_item_count"], 0)
        self.assertEqual(missing_prediction_receipt["missing_prediction_count"], 1)

        missing_core_path = self.root / "missing-core.json"
        missing_score_receipt_path = self.root / "missing-score-receipt.json"
        scored_missing = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "score",
                "--descriptor",
                str(descriptor_path),
                "--collection-root",
                str(self.root),
                "--audio-root",
                str(self.root),
                "--input-projection",
                str(input_path),
                "--candidate-lock",
                str(lock_path),
                "--predictions",
                str(missing_predictions_path),
                "--output-core",
                str(missing_core_path),
                "--output-receipt",
                str(missing_score_receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(scored_missing.returncode, 0, scored_missing.stderr.decode())
        missing_report = json.loads(missing_core_path.read_bytes())
        self.assertEqual(missing_report["counts"]["failed_count"], 1)
        self.assertEqual(missing_report["items"][0]["status"], "failed")
        self.assertEqual(
            missing_report["items"][0]["reason_code"], "missing_prediction"
        )

    def test_custodian_preflight_rejects_changed_prediction_lock(self) -> None:
        descriptor_path, _, records = self.write_bundle()
        candidate_manifest_path = self.write_planned_candidate_manifest(
            sha256_file(descriptor_path)
        )
        input_path = self.root / "sealed-input.json"
        lock_path = self.root / "candidate-lock.json"
        export_receipt_path = self.root / "export-receipt.json"
        exported = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "export-input",
                "--descriptor",
                str(descriptor_path),
                "--collection-root",
                str(self.root),
                "--audio-root",
                str(self.root),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--hypothesis-adapter-version",
                "identity-v1",
                "--output-input",
                str(input_path),
                "--output-candidate-lock",
                str(lock_path),
                "--output-receipt",
                str(export_receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(exported.returncode, 0, exported.stderr.decode())

        sealed_input = load_sealed_input_projection(input_path)
        candidate_lock = load_candidate_lock(lock_path)
        sealed_record = records["sealed-blind"][0]
        prediction_bundle = build_prediction_bundle(
            sealed_input,
            candidate_lock.sha256,
            [
                {
                    "id": sealed_record["id"],
                    "raw_text": sealed_record["raw_text"],
                    "status": "ok",
                    "reason_code": None,
                }
            ],
            hypothesis_adapter_version="identity-v1",
        )
        prediction_bundle["candidate_lock_sha256"] = digest("different-lock")
        predictions_path = self.root / "predictions.json"
        predictions_path.write_bytes(
            canonical_prediction_bundle_bytes(prediction_bundle)
        )
        core_path = self.root / "must-not-exist.json"
        score_receipt_path = self.root / "must-not-exist-receipt.json"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/replay_asr_evaluation.py",
                "score",
                "--descriptor",
                str(descriptor_path),
                "--collection-root",
                str(self.root),
                "--audio-root",
                str(self.root),
                "--input-projection",
                str(input_path),
                "--candidate-lock",
                str(lock_path),
                "--predictions",
                str(predictions_path),
                "--output-core",
                str(core_path),
                "--output-receipt",
                str(score_receipt_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b"")
        self.assertFalse(core_path.exists())
        self.assertFalse(score_receipt_path.exists())
        self.assertNotIn(str(sealed_record["raw_text"]).encode("utf-8"), completed.stderr)
        self.assertNotIn(str(sealed_record["id"]).encode("utf-8"), completed.stderr)

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
