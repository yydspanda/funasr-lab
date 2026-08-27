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
                for record in records["sealed-blind"]
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
                for record in records["sealed-blind"]
            ]
            kind = "asr-sealed-audio-input"
        return {
            "schema_version": 1,
            "kind": kind,
            "dataset_id": descriptor["dataset_id"],
            "revision": descriptor["revision"],
            "split": "sealed-blind",
            "manifest_sha256": manifest["sha256"],
            "record_count": manifest["record_count"],
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
        self.assertEqual(projection["record_count"], 1)
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

    def test_blind_projection_hashes_are_rebuilt_from_full_sealed_records(self) -> None:
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
