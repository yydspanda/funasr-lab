from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.collection import CollectionDescriptor
from eval.collection import ValidatedCollection
from eval.collection import canonical_json_bytes
from eval.collection import sha256_bytes
from eval.core_report import build_split_core_report
from eval.core_report import canonical_core_bytes
from eval.custodian_replay import CANDIDATE_LOCK_KIND
from eval.custodian_replay import CANDIDATE_LOCK_SCHEMA_VERSION
from eval.custodian_replay import CUSTODIAN_SCORE_RECEIPT_KIND
from eval.custodian_replay import INPUT_EXPORT_RECEIPT_KIND
from eval.custodian_replay import RECEIPT_SCHEMA_VERSION
from eval.custodian_replay import CustodianReplayError
from eval.custodian_replay import LoadedArtifact
from eval.custodian_replay import SCORER_SOURCE_PATHS
from eval.custodian_replay import build_prediction_bundle
from eval.custodian_replay import candidate_freeze_projection
from eval.custodian_replay import candidate_freeze_sha256
from eval.custodian_replay import candidate_manifest_freeze_sha256
from eval.custodian_replay import canonical_candidate_lock_bytes
from eval.custodian_replay import canonical_custodian_receipt_bytes
from eval.custodian_replay import canonical_prediction_bundle_bytes
from eval.custodian_replay import load_custodian_receipt
from eval.custodian_replay import load_planned_candidate_manifest
from eval.custodian_replay import load_prediction_bundle
from eval.custodian_replay import load_prediction_items_jsonl
from eval.custodian_replay import parse_sealed_input_projection
from eval.custodian_replay import preflight_replay_artifacts
from eval.custodian_replay import scorer_code_identity
from eval.custodian_replay import validate_candidate_lock
from eval.custodian_replay import validate_candidate_request
from eval.custodian_replay import validate_custodian_receipt
from eval.custodian_replay import validate_prediction_bundle
from eval.custodian_replay import validate_terminal_manifest_for_receipt
from eval.custodian_replay import write_atomic_outputs
from eval.normalizers import NORMALIZER_VERSION
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import record_input_sha256


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


class CustodianReplayArtifactTest(unittest.TestCase):
    def projection(self) -> LoadedArtifact:
        document = {
            "schema_version": 2,
            "kind": "asr-sealed-audio-input",
            "dataset_id": "LAB-SEED-TEST",
            "revision": "v0.1",
            "split": "sealed-blind",
            "manifest_sha256": digest("sealed-manifest"),
            "manifest_record_count": 2,
            "item_count": 2,
            "items": [
                {
                    "id": "sealed-1",
                    "split": "sealed-blind",
                    "audio": "audio/sealed-1.wav",
                    "audio_sha256": digest("sealed-audio-1"),
                    "duration_seconds": 1.0,
                    "sample_rate": 16_000,
                    "channels": 1,
                    "sample_width_bits": 16,
                },
                {
                    "id": "sealed-2",
                    "split": "sealed-blind",
                    "audio": "audio/sealed-2.wav",
                    "audio_sha256": digest("sealed-audio-2"),
                    "duration_seconds": 2.0,
                    "sample_rate": 16_000,
                    "channels": 1,
                    "sample_width_bits": 16,
                },
            ],
        }
        return parse_sealed_input_projection(canonical_json_bytes(document))

    def predictions(self):
        return [
            {
                "id": "sealed-1",
                "raw_text": "第一条",
                "status": "ok",
                "reason_code": None,
            },
            {
                "id": "sealed-2",
                "raw_text": "",
                "status": "failed",
                "reason_code": "decoder_failure",
            },
        ]

    def planned_manifest(self):
        return {
            "schema_version": 1,
            "experiment_id": "EXP-20260827-002-lock-fixture",
            "task_id": "EVAL-01",
            "hypothesis": (
                "One pinned decoder candidate produces a complete sealed prediction "
                "bundle under the frozen evaluation contract."
            ),
            "upstream_commit": "eedd4e22d10dc2e81d9c2bb321edb3750253964b",
            "code_commit": "c1aa9d3ba29ac8e1a1791147a42e9b9920d97843",
            "models": [
                {
                    "role": "asr",
                    "identifier": "iic/fixture-asr-model",
                    "revision": "28fe27c56aab3861cf77ae065b2bfc2aa3ab9692",
                    "sha256": digest("candidate-model"),
                }
            ],
            "config_sha256": digest("candidate-config"),
            "data_sha256": digest("candidate-data"),
            "eval_data_version": "LAB-SEED-TEST-v0.1",
            "normalizer_version": NORMALIZER_VERSION,
            "hardware": {
                "host_id": "fixture-custodian-host",
                "os": "Linux fixture x86_64",
                "cpu_model": "Fixture CPU",
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

    def core_report(self):
        record = {
            "id": "sealed-1",
            "raw_text": "第一条",
            "split": "sealed-blind",
            "scenario_tags": ["language:zh", "environment:meeting"],
            "evaluation_status": "included",
            "exclusion_reason": None,
        }
        records = (record,)
        collection = ValidatedCollection(
            summary={
                "data_sha256": digest("candidate-data"),
                "record_identity_version": RECORD_IDENTITY_VERSION,
                "record_input_sha256": record_input_sha256(records),
            },
            records=records,
            _sealed_input_projection=b"test-only-projection\n",
        )
        return build_split_core_report(
            collection,
            [{"id": "sealed-1", "raw_text": "第一条"}],
            split="sealed-blind",
        )

    def score_receipt(self, core_report=None):
        planned = self.planned_manifest()
        core = self.core_report() if core_report is None else core_report
        return {
            "schema_version": 1,
            "kind": CUSTODIAN_SCORE_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": planned["experiment_id"],
            "dataset_id": "LAB-SEED-TEST",
            "revision": "v0.1",
            "evaluation_scope": {"kind": "split", "split": "sealed-blind"},
            "data_sha256": digest("candidate-data"),
            "input_projection_sha256": digest("sealed-input"),
            "record_identity_version": RECORD_IDENTITY_VERSION,
            "record_input_sha256": core["provenance"]["record_input_sha256"],
            "hypothesis_adapter_version": "identity-v1",
            "prediction_input_sha256": core["provenance"][
                "prediction_input_sha256"
            ],
            "candidate_lock_sha256": digest("candidate-lock"),
            "candidate_freeze_sha256": candidate_manifest_freeze_sha256(planned),
            "prediction_artifact_sha256": digest("prediction-artifact"),
            "prediction_items_sha256": digest("prediction-items"),
            "scorer_code_commit": "c1aa9d3ba29ac8e1a1791147a42e9b9920d97843",
            "scorer_source_sha256": digest("scorer-source-inventory"),
            "core_schema_version": 2,
            "core_sha256": sha256_bytes(canonical_core_bytes(core)),
            "public_release": {
                "state": "withheld",
                "summary_sha256": None,
                "reason_code": "release_policy_not_implemented",
            },
        }

    def test_prediction_bundle_binds_ordered_id_subsequence_adapter_and_items_hash(self):
        projection = self.projection()
        bundle = build_prediction_bundle(
            projection,
            digest("candidate-lock"),
            self.predictions(),
            hypothesis_adapter_version="identity-v1",
        )
        payload = canonical_prediction_bundle_bytes(bundle)

        self.assertEqual(bundle["item_count"], 2)
        self.assertEqual(
            bundle["items_sha256"],
            sha256_bytes(canonical_json_bytes(self.predictions())),
        )
        self.assertEqual(payload, canonical_json_bytes(json.loads(payload)))

        missing_first = build_prediction_bundle(
            projection,
            digest("candidate-lock"),
            self.predictions()[1:],
            hypothesis_adapter_version="identity-v1",
        )
        self.assertEqual(
            [item["id"] for item in missing_first["items"]],
            ["sealed-2"],
        )

        with self.assertRaisesRegex(CustodianReplayError, "preserve.*order"):
            build_prediction_bundle(
                projection,
                digest("candidate-lock"),
                list(reversed(self.predictions())),
                hypothesis_adapter_version="identity-v1",
            )
        unexpected = copy.deepcopy(self.predictions()[1])
        unexpected["id"] = "not-in-decode-input"
        with self.assertRaisesRegex(CustodianReplayError, "absent"):
            build_prediction_bundle(
                projection,
                digest("candidate-lock"),
                [unexpected],
                hypothesis_adapter_version="identity-v1",
            )

    def test_prediction_text_has_per_item_and_total_resource_limits(self):
        at_limit = copy.deepcopy(self.predictions())
        at_limit[0]["raw_text"] = "字" * 16_384
        build_prediction_bundle(
            self.projection(),
            digest("candidate-lock"),
            at_limit,
            hypothesis_adapter_version="identity-v1",
        )
        overlong = copy.deepcopy(at_limit)
        overlong[0]["raw_text"] += "字"
        with self.assertRaisesRegex(CustodianReplayError, "16384-character"):
            build_prediction_bundle(
                self.projection(),
                digest("candidate-lock"),
                overlong,
                hypothesis_adapter_version="identity-v1",
            )

        cumulative = copy.deepcopy(self.predictions())
        cumulative[0]["raw_text"] = "一二三"
        cumulative[1].update(
            {"raw_text": "四五六", "status": "ok", "reason_code": None}
        )
        with mock.patch(
            "eval.custodian_replay.MAX_TOTAL_HYPOTHESIS_CHARACTERS", 5
        ):
            with self.assertRaisesRegex(CustodianReplayError, "total limit"):
                build_prediction_bundle(
                    self.projection(),
                    digest("candidate-lock"),
                    cumulative,
                    hypothesis_adapter_version="identity-v1",
                )

    def test_prediction_schema_string_bounds_match_semantic_validation(self):
        bundle = build_prediction_bundle(
            self.projection(),
            digest("candidate-lock"),
            self.predictions(),
            hypothesis_adapter_version="identity-v1",
        )

        bounded_id = copy.deepcopy(bundle)
        bounded_id["items"][0]["id"] = "i" * 512
        bounded_id["items_sha256"] = sha256_bytes(
            canonical_json_bytes(bounded_id["items"])
        )
        validate_prediction_bundle(bounded_id)
        overlong_id = copy.deepcopy(bounded_id)
        overlong_id["items"][0]["id"] += "i"
        overlong_id["items_sha256"] = sha256_bytes(
            canonical_json_bytes(overlong_id["items"])
        )
        with self.assertRaisesRegex(CustodianReplayError, "512-character"):
            validate_prediction_bundle(overlong_id)

        bounded_reason = copy.deepcopy(bundle)
        bounded_reason["items"][1]["reason_code"] = "a" * 128
        bounded_reason["items_sha256"] = sha256_bytes(
            canonical_json_bytes(bounded_reason["items"])
        )
        validate_prediction_bundle(bounded_reason)
        overlong_reason = copy.deepcopy(bounded_reason)
        overlong_reason["items"][1]["reason_code"] += "a"
        overlong_reason["items_sha256"] = sha256_bytes(
            canonical_json_bytes(overlong_reason["items"])
        )
        with self.assertRaisesRegex(CustodianReplayError, "128-character"):
            validate_prediction_bundle(overlong_reason)

        bounded_dataset = copy.deepcopy(bundle)
        bounded_dataset["dataset_id"] = "d" * 256
        validate_prediction_bundle(bounded_dataset)
        overlong_dataset = copy.deepcopy(bounded_dataset)
        overlong_dataset["dataset_id"] += "d"
        with self.assertRaisesRegex(CustodianReplayError, "256-character"):
            validate_prediction_bundle(overlong_dataset)

        bounded_revision = copy.deepcopy(bundle)
        bounded_revision["revision"] = "r" * 256
        validate_prediction_bundle(bounded_revision)
        overlong_revision = copy.deepcopy(bounded_revision)
        overlong_revision["revision"] += "r"
        with self.assertRaisesRegex(CustodianReplayError, "256-character"):
            validate_prediction_bundle(overlong_revision)

    def test_prediction_status_matches_adapter_output_before_scoring(self):
        tag_only = {
            "id": "sealed-1",
            "raw_text": "<|zh|><|NEUTRAL|>",
            "status": "ok",
            "reason_code": None,
        }
        with self.assertRaisesRegex(CustodianReplayError, "adapt to empty"):
            build_prediction_bundle(
                self.projection(),
                digest("candidate-lock"),
                [tag_only],
                hypothesis_adapter_version="sensevoice-control-tags-v1",
            )

        tag_only["status"] = "empty"
        tag_only["reason_code"] = "empty_hypothesis"
        bundle = build_prediction_bundle(
            self.projection(),
            digest("candidate-lock"),
            [tag_only],
            hypothesis_adapter_version="sensevoice-control-tags-v1",
        )
        self.assertEqual(bundle["items"][0]["status"], "empty")

    def test_reference_free_preflight_accepts_missing_prediction_subsequence(self):
        sealed_input = self.projection()
        candidate = candidate_freeze_projection(self.planned_manifest())
        candidate_lock = {
            "schema_version": CANDIDATE_LOCK_SCHEMA_VERSION,
            "kind": CANDIDATE_LOCK_KIND,
            "state": "frozen",
            "access_class": "restricted",
            "dataset_id": sealed_input.document["dataset_id"],
            "revision": sealed_input.document["revision"],
            "split": "sealed-blind",
            "data_sha256": candidate["data_sha256"],
            "input_projection_sha256": sealed_input.sha256,
            "hypothesis_adapter_version": "identity-v1",
            "record_identity_version": RECORD_IDENTITY_VERSION,
            "record_input_sha256": digest("sealed-record-input"),
            "decode_item_count": 2,
            "decode_item_ids_sha256": sha256_bytes(
                canonical_json_bytes(["sealed-1", "sealed-2"])
            ),
            "source_manifest_decision": "planned",
            "candidate": candidate,
            "candidate_freeze_sha256": candidate_freeze_sha256(candidate),
        }
        lock_payload = canonical_candidate_lock_bytes(candidate_lock)
        loaded_lock = LoadedArtifact(
            candidate_lock,
            lock_payload,
            sha256_bytes(lock_payload),
        )
        prediction_bundle = build_prediction_bundle(
            sealed_input,
            loaded_lock.sha256,
            self.predictions()[1:],
            hypothesis_adapter_version="identity-v1",
        )
        prediction_payload = canonical_prediction_bundle_bytes(prediction_bundle)
        loaded_predictions = LoadedArtifact(
            prediction_bundle,
            prediction_payload,
            sha256_bytes(prediction_payload),
        )
        descriptor = CollectionDescriptor(
            path=Path("collection.json"),
            raw_sha256=candidate["data_sha256"],
            dataset_id=sealed_input.document["dataset_id"],
            revision=sealed_input.document["revision"],
            normalizer_version=NORMALIZER_VERSION,
            mer_tokenizer_version="zh-en-mixed-v0.1",
            manifests=(),
            provenance_groups={},
            rights_groups={},
            dedup={},
            blind={"input_projection_sha256": sealed_input.sha256},
        )

        preflight_replay_artifacts(
            descriptor,
            sealed_input,
            loaded_lock,
            loaded_predictions,
        )

        boundary_lock = copy.deepcopy(candidate_lock)
        boundary_lock["decode_item_count"] = 1_000_000
        validate_candidate_lock(boundary_lock)
        oversized_lock = copy.deepcopy(boundary_lock)
        oversized_lock["decode_item_count"] += 1
        with self.assertRaisesRegex(CustodianReplayError, "1000000 limit"):
            validate_candidate_lock(oversized_lock)

        mismatched_lock = copy.deepcopy(candidate_lock)
        mismatched_lock["hypothesis_adapter_version"] = (
            "sensevoice-control-tags-v1"
        )
        with self.assertRaisesRegex(CustodianReplayError, "does not match"):
            validate_candidate_lock(mismatched_lock)

    def test_prediction_parser_rejects_tampering_unknown_fields_and_duplicate_json(self):
        bundle = build_prediction_bundle(
            self.projection(),
            digest("candidate-lock"),
            self.predictions(),
            hypothesis_adapter_version="identity-v1",
        )
        tampered = copy.deepcopy(bundle)
        tampered["items"][0]["raw_text"] = "已篡改"
        with self.assertRaisesRegex(CustodianReplayError, "items_sha256"):
            validate_prediction_bundle(tampered)

        unknown = copy.deepcopy(bundle)
        unknown["display_text"] = "不允许"
        with self.assertRaisesRegex(CustodianReplayError, "unknown field"):
            validate_prediction_bundle(unknown)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate = root / "duplicate.json"
            duplicate.write_bytes(b'{"schema_version":1,"schema_version":1}\n')
            with self.assertRaisesRegex(CustodianReplayError, "duplicate JSON key"):
                load_prediction_bundle(duplicate)

            nonstandard = root / "nan.json"
            nonstandard.write_bytes(b'{"schema_version":NaN}\n')
            with self.assertRaisesRegex(
                CustodianReplayError, "non-standard JSON constant"
            ):
                load_prediction_bundle(nonstandard)

    def test_candidate_freeze_projection_excludes_mutable_result_fields(self):
        planned = self.planned_manifest()
        candidate = candidate_freeze_projection(planned)
        candidate_digest = candidate_freeze_sha256(candidate)

        executed = copy.deepcopy(planned)
        executed["decision"] = "accept"
        executed["metrics"] = {"result_fields_do_not_enter_lock": 1}
        executed["artifacts"] = [
            {"kind": "report", "path": "private/report.json", "sha256": digest("r")}
        ]
        executed_projection = {key: executed[key] for key in candidate}

        self.assertEqual(candidate, executed_projection)
        self.assertEqual(candidate_digest, candidate_freeze_sha256(executed_projection))
        self.assertNotIn("decision", candidate)
        self.assertNotIn("metrics", candidate)
        self.assertNotIn("artifacts", candidate)

    def test_candidate_schema_numeric_and_string_bounds_are_enforced(self):
        bounded = self.planned_manifest()
        bounded["hypothesis"] = "h" * 16_384
        bounded["seed"] = 4_294_967_295
        candidate_freeze_projection(bounded)

        overlong = copy.deepcopy(bounded)
        overlong["hypothesis"] += "h"
        with self.assertRaisesRegex(CustodianReplayError, "16384-character"):
            candidate_freeze_projection(overlong)

        oversized_seed = copy.deepcopy(bounded)
        oversized_seed["seed"] += 1
        with self.assertRaisesRegex(CustodianReplayError, "4294967295 limit"):
            candidate_freeze_projection(oversized_seed)

    def test_candidate_schema_collection_bounds_are_enforced(self):
        bounded_models = self.planned_manifest()
        model_template = bounded_models["models"][0]
        bounded_models["models"] = [
            {**copy.deepcopy(model_template), "role": f"model_{index}"}
            for index in range(32)
        ]
        candidate_freeze_projection(bounded_models)
        oversized_models = copy.deepcopy(bounded_models)
        oversized_models["models"].append(
            {**copy.deepcopy(model_template), "role": "model_32"}
        )
        with self.assertRaisesRegex(CustodianReplayError, "32-item"):
            candidate_freeze_projection(oversized_models)

        bounded_argv = self.planned_manifest()
        bounded_argv["command"]["argv"] = [
            f"argument-{index}" for index in range(1024)
        ]
        candidate_freeze_projection(bounded_argv)
        oversized_argv = copy.deepcopy(bounded_argv)
        oversized_argv["command"]["argv"].append("argument-1024")
        with self.assertRaisesRegex(CustodianReplayError, "1024-item"):
            candidate_freeze_projection(oversized_argv)

        bounded_environment = self.planned_manifest()
        bounded_environment["command"]["environment"] = {
            f"VAR_{index}": "value" for index in range(256)
        }
        candidate_freeze_projection(bounded_environment)
        oversized_environment = copy.deepcopy(bounded_environment)
        oversized_environment["command"]["environment"]["VAR_256"] = "value"
        with self.assertRaisesRegex(CustodianReplayError, "256-property"):
            candidate_freeze_projection(oversized_environment)

    def test_candidate_adapter_and_task_are_rejected_before_collection_load(self):
        descriptor = CollectionDescriptor(
            path=Path("collection.json"),
            raw_sha256=digest("candidate-data"),
            dataset_id="LAB-SEED-TEST",
            revision="v0.1",
            normalizer_version=NORMALIZER_VERSION,
            mer_tokenizer_version="zh-en-mixed-v0.1",
            manifests=(),
            provenance_groups={},
            rights_groups={},
            dedup={},
            blind={"input_projection_sha256": digest("sealed-input")},
        )
        with self.assertRaisesRegex(CustodianReplayError, "adapter"):
            validate_candidate_request(
                descriptor,
                self.planned_manifest(),
                hypothesis_adapter_version="sensevoice-control-tags-v1",
            )

        wrong_task = self.planned_manifest()
        wrong_task["task_id"] = "BASE-01"
        with self.assertRaisesRegex(CustodianReplayError, "task_id"):
            validate_candidate_request(
                descriptor,
                wrong_task,
                hypothesis_adapter_version="identity-v1",
            )

    def test_restricted_score_receipt_is_canonical_and_release_is_withheld(self):
        receipt = self.score_receipt()
        payload = canonical_custodian_receipt_bytes(receipt)
        self.assertEqual(payload, canonical_json_bytes(json.loads(payload)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "receipt.json"
            path.write_bytes(payload)
            loaded = load_custodian_receipt(path)
            self.assertEqual(loaded.document, receipt)
            self.assertEqual(loaded.sha256, sha256_bytes(payload))

        released = copy.deepcopy(receipt)
        released["public_release"]["state"] = "released"
        with self.assertRaisesRegex(CustodianReplayError, "remain withheld"):
            validate_custodian_receipt(released)

        unknown = copy.deepcopy(receipt)
        unknown["metrics"] = {"content_cer": 0}
        with self.assertRaisesRegex(CustodianReplayError, "unknown field"):
            validate_custodian_receipt(unknown)

        invalid_scorer = copy.deepcopy(receipt)
        invalid_scorer["scorer_code_commit"] = "short"
        with self.assertRaisesRegex(CustodianReplayError, "full Git commit"):
            validate_custodian_receipt(invalid_scorer)

        bounded_identity = copy.deepcopy(receipt)
        bounded_identity["experiment_id"] = "EXP-20260827-002-" + "a" * 111
        bounded_identity["dataset_id"] = "d" * 256
        bounded_identity["revision"] = "r" * 256
        validate_custodian_receipt(bounded_identity)
        overlong_identity = copy.deepcopy(bounded_identity)
        overlong_identity["experiment_id"] += "a"
        with self.assertRaisesRegex(CustodianReplayError, "128-character"):
            validate_custodian_receipt(overlong_identity)
        invalid_experiment_id = copy.deepcopy(receipt)
        invalid_experiment_id["experiment_id"] = "not-an-experiment-id"
        with self.assertRaisesRegex(CustodianReplayError, "invalid format"):
            validate_custodian_receipt(invalid_experiment_id)
        overlong_dataset = copy.deepcopy(bounded_identity)
        overlong_dataset["dataset_id"] += "d"
        with self.assertRaisesRegex(CustodianReplayError, "256-character"):
            validate_custodian_receipt(overlong_dataset)

        export_receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": INPUT_EXPORT_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": receipt["experiment_id"],
            "dataset_id": receipt["dataset_id"],
            "revision": receipt["revision"],
            "split": "sealed-blind",
            "decode_item_count": 1_000_000,
            "input_projection_sha256": receipt["input_projection_sha256"],
            "candidate_lock_sha256": receipt["candidate_lock_sha256"],
            "candidate_freeze_sha256": receipt["candidate_freeze_sha256"],
        }
        validate_custodian_receipt(export_receipt)
        export_receipt["decode_item_count"] += 1
        with self.assertRaisesRegex(CustodianReplayError, "1000000 limit"):
            validate_custodian_receipt(export_receipt)

    def test_scorer_code_identity_binds_head_and_exact_source_inventory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, relative_path in enumerate(SCORER_SOURCE_PATHS):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"scorer source {index}\n".encode("utf-8"))
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=ASR Test",
                    "-c",
                    "user.email=asr-test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture scorer",
                ],
                cwd=root,
                check=True,
            )

            with mock.patch.dict(
                os.environ,
                {"GIT_DIR": str(root / "does-not-exist")},
                clear=False,
            ):
                first = scorer_code_identity(root)
            second = scorer_code_identity(root)
            self.assertEqual(first, second)
            self.assertRegex(first[0], r"^[0-9a-f]{40}$")
            self.assertRegex(first[1], r"^sha256:[0-9a-f]{64}$")

            (root / SCORER_SOURCE_PATHS[0]).write_bytes(b"dirty scorer\n")
            with self.assertRaisesRegex(
                CustodianReplayError, "differs from scorer_code_commit"
            ):
                scorer_code_identity(root)

    def test_terminal_manifest_must_preserve_score_receipt_candidate_facts(self):
        core = self.core_report()
        cer = core["aggregate"]["cer"]
        mer = core["aggregate"]["mer"]
        terminal = self.planned_manifest()
        terminal["decision"] = "accept"
        terminal["metrics"] = {
            "content_cer": cer["errors"] / cer["reference_units"],
            "substitutions": cer["substitutions"],
            "deletions": cer["deletions"],
            "insertions": cer["insertions"],
            "reference_units": cer["reference_units"],
            "utterance_count": core["counts"]["utterance_count"],
            "failed_count": core["counts"]["failed_count"],
            "mer": mer["errors"] / mer["reference_units"],
            "rtf_p50": 0.1,
            "rtf_p95": 0.1,
            "peak_rss_mb": 1.0,
        }
        receipt = self.score_receipt(core)
        terminal["artifacts"] = [
            {
                "kind": "other",
                "path": "eval/private/sealed-input.json",
                "sha256": receipt["input_projection_sha256"],
            },
            {
                "kind": "other",
                "path": "eval/private/candidate-lock.json",
                "sha256": receipt["candidate_lock_sha256"],
            },
            {
                "kind": "prediction",
                "path": "eval/private/predictions.json",
                "sha256": receipt["prediction_artifact_sha256"],
            },
            {
                "kind": "report",
                "path": "eval/private/terminal-report.json",
                "sha256": receipt["core_sha256"],
            },
            {
                "kind": "other",
                "path": "eval/private/score-receipt.json",
                "sha256": sha256_bytes(
                    canonical_custodian_receipt_bytes(receipt)
                ),
            },
        ]
        validate_terminal_manifest_for_receipt(terminal, receipt, core)

        changed = copy.deepcopy(terminal)
        changed["seed"] = 1
        with self.assertRaisesRegex(CustodianReplayError, "candidate facts"):
            validate_terminal_manifest_for_receipt(changed, receipt, core)

        changed = copy.deepcopy(terminal)
        changed["metrics"]["failed_count"] = 1
        with self.assertRaisesRegex(CustodianReplayError, "failed_count"):
            validate_terminal_manifest_for_receipt(changed, receipt, core)

        changed = copy.deepcopy(terminal)
        changed["artifacts"] = changed["artifacts"][:-1]
        with self.assertRaisesRegex(CustodianReplayError, "score receipt"):
            validate_terminal_manifest_for_receipt(changed, receipt, core)

    def test_raw_prediction_jsonl_is_strict_and_can_freeze_a_partial_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "raw.jsonl"
            partial = self.predictions()[1:]
            path.write_bytes(
                b"".join(canonical_json_bytes(item) for item in partial)
            )
            self.assertEqual(load_prediction_items_jsonl(path), partial)

            path.write_bytes(
                canonical_json_bytes(partial[0]) + b"\n"
            )
            with self.assertRaisesRegex(CustodianReplayError, "must not be blank"):
                load_prediction_items_jsonl(path)

            path.write_bytes(
                b'{"id":"sealed-2","id":"duplicate",'
                b'"raw_text":"","reason_code":"decoder_failure",'
                b'"status":"failed"}\n'
            )
            with self.assertRaisesRegex(CustodianReplayError, "duplicate JSON key"):
                load_prediction_items_jsonl(path)

    def test_candidate_manifest_runs_full_directory_git_governance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            planned = self.planned_manifest()
            path = root / f"{planned['experiment_id']}.json"
            path.write_text(
                json.dumps(planned, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(load_planned_candidate_manifest(path), planned)

            invalid = copy.deepcopy(planned)
            invalid["code_commit"] = hashlib.sha1(
                b"nonexistent-candidate-code-commit"
            ).hexdigest()
            path.write_text(
                json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CustodianReplayError, "does not resolve to a Git commit"
            ):
                load_planned_candidate_manifest(path)

    def test_atomic_outputs_are_mode_0600_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.json"
            second = root / "second.json"
            write_atomic_outputs(
                [(first, b"first\n"), (second, b"second\n")]
            )

            self.assertEqual(first.read_bytes(), b"first\n")
            self.assertEqual(second.read_bytes(), b"second\n")
            self.assertEqual(first.stat().st_mode & 0o777, 0o600)
            self.assertEqual(second.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(root.glob(".*.tmp")), [])

            with self.assertRaisesRegex(CustodianReplayError, "overwrite"):
                write_atomic_outputs([(first, b"replacement\n")])
            self.assertEqual(first.read_bytes(), b"first\n")

    def test_atomic_outputs_reject_duplicate_and_resolved_alias_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate = root / "duplicate.json"
            with self.assertRaisesRegex(CustodianReplayError, "distinct"):
                write_atomic_outputs(
                    [(duplicate, b"first\n"), (duplicate, b"second\n")]
                )
            self.assertFalse(duplicate.exists())

            alias = root / "nested" / ".." / "alias.json"
            resolved = root / "alias.json"
            with self.assertRaisesRegex(CustodianReplayError, "distinct"):
                write_atomic_outputs(
                    [(alias, b"first\n"), (resolved, b"second\n")]
                )
            self.assertFalse(resolved.exists())

            open_parent = root / "open-parent"
            open_parent.mkdir(mode=0o755)
            open_parent.chmod(0o755)
            with self.assertRaisesRegex(CustodianReplayError, "group or other"):
                write_atomic_outputs(
                    [(open_parent / "restricted.json", b"restricted\n")]
                )

            first_parent = root / "first-private"
            second_parent = root / "second-private"
            first_parent.mkdir(mode=0o700)
            second_parent.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                CustodianReplayError, "share one private directory"
            ):
                write_atomic_outputs(
                    [
                        (first_parent / "artifact.json", b"artifact\n"),
                        (second_parent / "receipt.json", b"receipt\n"),
                    ]
                )
            self.assertFalse((first_parent / "artifact.json").exists())
            self.assertFalse((second_parent / "receipt.json").exists())

    def test_atomic_outputs_roll_back_on_base_exception(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.json"
            second = root / "second.json"
            real_link = os.link
            link_calls = 0

            def interrupt_second_link(source, destination):
                nonlocal link_calls
                link_calls += 1
                if link_calls == 2:
                    raise KeyboardInterrupt()
                return real_link(source, destination)

            with mock.patch(
                "eval.custodian_replay.os.link",
                side_effect=interrupt_second_link,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    write_atomic_outputs(
                        [(first, b"first\n"), (second, b"second\n")]
                    )

            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_artifact_loader_rejects_symlinks_and_oversized_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target.json"
            target.write_bytes(b"{}\n")
            symlink = root / "symlink.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(CustodianReplayError, "cannot read"):
                load_prediction_bundle(symlink)

            oversized = root / "oversized.json"
            oversized.write_bytes(b"123456789")
            with mock.patch("eval.custodian_replay.MAX_ARTIFACT_BYTES", 8):
                with self.assertRaisesRegex(CustodianReplayError, "safety limit"):
                    load_prediction_bundle(oversized)


if __name__ == "__main__":
    unittest.main()
