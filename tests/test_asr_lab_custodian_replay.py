from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import eval.custodian_replay as custodian_replay_module
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
from eval.custodian_replay import RegisteredCandidateManifest
from eval.custodian_replay import SCORER_SOURCE_PATHS
from eval.custodian_replay import build_prediction_bundle as _build_prediction_bundle
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
from eval.custodian_replay import validate_input_export_receipt_handoff
from eval.custodian_replay import validate_prediction_bundle
from eval.custodian_replay import validate_terminal_manifest_for_receipt
from eval.custodian_replay import validate_restricted_transition_paths
from eval.custodian_replay import write_atomic_outputs
from eval.execution_envelope import build_execution_envelope
from eval.execution_envelope import canonical_execution_envelope_bytes
from eval.normalizers import NORMALIZER_VERSION
from eval.offline_baseline import BaselineConfig
from eval.offline_baseline import TRACKS
from eval.offline_baseline import effective_config
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import record_input_sha256
from eval.sealed_decoder import SealedDecoderError
from eval.sealed_decoder import runner_source_identity
from eval.sealed_decoder import runtime_identity
from scripts import replay_asr_evaluation as replay_cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPOSITORY_ROOT / ".venv/bin/python"
ACTUAL_SCORER_RUNTIME = {
    "python_implementation": "cpython",
    "python_version": "3.11.15",
    "python_cache_tag": "cpython-311",
    "dependency_lock_sha256": "sha256:" + "a" * 64,
    "installed_dependencies_sha256": "sha256:" + "b" * 64,
    "installed_dependency_count": 71,
    "unicode_version": "14.0.0",
}


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


def registered_candidate(
    document: dict[str, object],
) -> RegisteredCandidateManifest:
    payload = canonical_json_bytes(document)
    fields = registration_fields(str(document["experiment_id"]))
    return RegisteredCandidateManifest(
        document=document,
        payload=payload,
        sha256=sha256_bytes(payload),
        repository_path=fields["candidate_manifest_path"],
        registration_commit=fields["candidate_registration_commit"],
    )


def build_prediction_bundle(*args, **kwargs):
    """Supply fixed execution bindings for prediction-only fixture tests."""

    kwargs.setdefault(
        "input_export_receipt_sha256", digest("input-export-receipt")
    )
    kwargs.setdefault("raw_predictions_sha256", digest("raw-predictions"))
    kwargs.setdefault("execution_envelope_sha256", digest("execution-envelope"))
    return _build_prediction_bundle(*args, **kwargs)


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
                    "identifier": TRACKS["paraformer"].model_identifier,
                    "revision": model_revision,
                    "sha256": digest("candidate-model"),
                }
            ],
            "config_sha256": sha256_bytes(
                canonical_json_bytes(effective_config(config))
            ),
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
                    "-P",
                    "-S",
                    "scripts/run_sealed_asr_candidate.py",
                    "run",
                    "--input-projection",
                    "eval/private/fixture.input.json",
                    "--candidate-lock",
                    "eval/private/fixture.lock.json",
                    "--input-receipt",
                    "eval/private/fixture.export-receipt.json",
                    "--audio-root",
                    ".",
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
                    "eval/private/fixture.raw.jsonl",
                    "--output-execution-envelope",
                    "eval/private/fixture.execution.json",
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
            "schema_version": RECEIPT_SCHEMA_VERSION,
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
            **registration_fields(planned["experiment_id"]),
            "prediction_artifact_sha256": digest("prediction-artifact"),
            "prediction_items_sha256": digest("prediction-items"),
            "input_export_receipt_sha256": digest("input-export-receipt"),
            "prediction_freeze_receipt_sha256": digest(
                "prediction-freeze-receipt"
            ),
            "execution_envelope_sha256": digest("execution-envelope"),
            "runner_code_commit": planned["code_commit"],
            "runner_source_sha256": digest("runner-source-inventory"),
            "scorer_code_commit": "c1aa9d3ba29ac8e1a1791147a42e9b9920d97843",
            "scorer_source_sha256": digest("scorer-source-inventory"),
            "scorer_runtime": copy.deepcopy(ACTUAL_SCORER_RUNTIME),
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
            **registration_fields(candidate["experiment_id"]),
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

    def test_input_export_receipt_is_bound_to_the_exact_projection_and_lock(self):
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
            **registration_fields(candidate["experiment_id"]),
            "candidate": candidate,
            "candidate_freeze_sha256": candidate_freeze_sha256(candidate),
        }
        lock_payload = canonical_candidate_lock_bytes(candidate_lock)
        loaded_lock = LoadedArtifact(
            candidate_lock,
            lock_payload,
            sha256_bytes(lock_payload),
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": INPUT_EXPORT_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": candidate["experiment_id"],
            "dataset_id": sealed_input.document["dataset_id"],
            "revision": sealed_input.document["revision"],
            "split": "sealed-blind",
            "decode_item_count": 2,
            "input_projection_sha256": sealed_input.sha256,
            "candidate_lock_sha256": loaded_lock.sha256,
            "candidate_freeze_sha256": candidate_lock[
                "candidate_freeze_sha256"
            ],
            **registration_fields(candidate["experiment_id"]),
        }
        receipt_payload = canonical_custodian_receipt_bytes(receipt)
        loaded_receipt = LoadedArtifact(
            receipt,
            receipt_payload,
            sha256_bytes(receipt_payload),
        )
        validate_input_export_receipt_handoff(
            sealed_input,
            loaded_lock,
            loaded_receipt,
        )

        tampered = copy.deepcopy(receipt)
        tampered["candidate_lock_sha256"] = digest("other-lock")
        tampered_payload = canonical_custodian_receipt_bytes(tampered)
        with self.assertRaisesRegex(CustodianReplayError, "candidate_lock_sha256"):
            validate_input_export_receipt_handoff(
                sealed_input,
                loaded_lock,
                LoadedArtifact(
                    tampered,
                    tampered_payload,
                    sha256_bytes(tampered_payload),
                ),
            )

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

    def test_invalid_sealed_execution_facts_are_rejected_before_collection_load(self):
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
        valid = self.planned_manifest()
        validate_candidate_request(
            descriptor,
            valid,
            hypothesis_adapter_version="identity-v1",
        )

        invalid_cases: list[tuple[str, dict[str, object], str]] = []

        wrong_model = copy.deepcopy(valid)
        wrong_model["models"][0]["identifier"] = "iic/wrong-model"
        invalid_cases.append(("model", wrong_model, "model component"))

        short_revision = copy.deepcopy(valid)
        revision_index = short_revision["command"]["argv"].index(
            "--model-revision"
        )
        short_revision["command"]["argv"][revision_index + 1] = "floating"
        invalid_cases.append(("revision", short_revision, "model revision"))

        wrong_seed = copy.deepcopy(valid)
        seed_index = wrong_seed["command"]["argv"].index("--seed")
        wrong_seed["command"]["argv"][seed_index + 1] = "1"
        wrong_seed["seed"] = 1
        invalid_cases.append(("seed", wrong_seed, "seed"))

        gpu_hardware = copy.deepcopy(valid)
        gpu_hardware["hardware"]["device"] = "cuda"
        gpu_hardware["hardware"]["accelerator"] = "fixture-gpu"
        invalid_cases.append(("hardware", gpu_hardware, "hardware"))

        missing_accelerator = copy.deepcopy(valid)
        del missing_accelerator["hardware"]["accelerator"]
        invalid_cases.append(
            ("hardware-fields", missing_accelerator, "exact sealed CPU fields")
        )

        unknown_option = copy.deepcopy(valid)
        warmup_index = unknown_option["command"]["argv"].index("--warmup-runs")
        unknown_option["command"]["argv"][warmup_index] = "--unknown"
        invalid_cases.append(("argv", unknown_option, "canonical sealed runner argv"))

        mismatched_threads = copy.deepcopy(valid)
        mismatched_threads["command"]["environment"]["OMP_NUM_THREADS"] = "3"
        invalid_cases.append(("environment", mismatched_threads, "OMP_NUM_THREADS"))

        wrong_config = copy.deepcopy(valid)
        wrong_config["config_sha256"] = digest("wrong-effective-config")
        invalid_cases.append(("config", wrong_config, "config_sha256"))

        excessive_warmups = copy.deepcopy(valid)
        warmup_value_index = (
            excessive_warmups["command"]["argv"].index("--warmup-runs") + 1
        )
        excessive_warmups["command"]["argv"][warmup_value_index] = "101"
        invalid_cases.append(
            ("warmup-bound", excessive_warmups, "warmup-runs must be at most 100")
        )

        enormous_ncpu = copy.deepcopy(valid)
        ncpu_value_index = enormous_ncpu["command"]["argv"].index("--ncpu") + 1
        enormous_ncpu["command"]["argv"][ncpu_value_index] = "9" * 16_384
        invalid_cases.append(
            ("ncpu-digit-bound", enormous_ncpu, "ncpu must be at most 4096")
        )

        for name, candidate, message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(CustodianReplayError, message):
                    validate_candidate_request(
                        descriptor,
                        candidate,
                        hypothesis_adapter_version="identity-v1",
                    )

                args = argparse.Namespace(
                    output_input=Path("sealed-input.json"),
                    output_candidate_lock=Path("candidate-lock.json"),
                    output_receipt=Path("input-receipt.json"),
                    descriptor=Path("collection.json"),
                    candidate_manifest=Path("candidate.json"),
                    candidate_registration_commit="d" * 40,
                    hypothesis_adapter_version="identity-v1",
                    collection_root=Path("."),
                    audio_root=Path("."),
                )
                with mock.patch.object(
                    replay_cli, "validate_output_paths"
                ), mock.patch.object(
                    replay_cli,
                    "load_collection_descriptor",
                    return_value=descriptor,
                ), mock.patch.object(
                    replay_cli,
                    "load_planned_candidate_manifest",
                    return_value=registered_candidate(candidate),
                ), mock.patch.object(
                    replay_cli,
                    "_scorer_identity_for_candidate",
                    return_value=("1" * 40, digest("scorer")),
                ), mock.patch.object(
                    replay_cli, "load_validated_collection"
                ) as collection_loader:
                    with self.assertRaises(CustodianReplayError):
                        replay_cli._export_input(args)
                    collection_loader.assert_not_called()

    def test_export_binds_handoff_paths_and_defaults_audio_root_before_collection_load(
        self,
    ):
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
        candidate = self.planned_manifest()
        argv = candidate["command"]["argv"]

        def option_path(option: str) -> Path:
            return Path(argv[argv.index(option) + 1])

        matching_args = argparse.Namespace(
            output_input=option_path("--input-projection"),
            output_candidate_lock=option_path("--candidate-lock"),
            output_receipt=option_path("--input-receipt"),
            descriptor=Path("collection.json"),
            candidate_manifest=Path("candidate.json"),
            candidate_registration_commit="d" * 40,
            hypothesis_adapter_version="identity-v1",
            collection_root=REPOSITORY_ROOT,
            audio_root=None,
        )
        with mock.patch.object(
            replay_cli, "validate_output_paths"
        ), mock.patch.object(
            replay_cli,
            "load_collection_descriptor",
            return_value=descriptor,
        ), mock.patch.object(
            replay_cli,
            "load_planned_candidate_manifest",
            return_value=registered_candidate(candidate),
        ), mock.patch.object(
            replay_cli,
            "_scorer_identity_for_candidate",
            return_value=("1" * 40, digest("scorer")),
        ), mock.patch.object(
            replay_cli,
            "load_validated_collection",
            side_effect=CustodianReplayError("stop after effective audio root"),
        ) as collection_loader:
            with self.assertRaisesRegex(
                CustodianReplayError, "effective audio root"
            ):
                replay_cli._export_input(matching_args)
            collection_loader.assert_called_once_with(
                matching_args.descriptor,
                REPOSITORY_ROOT,
                REPOSITORY_ROOT,
            )

        mismatched_args = copy.copy(matching_args)
        mismatched_args.output_input = Path("different-input.json")
        with mock.patch.object(
            replay_cli, "validate_output_paths"
        ), mock.patch.object(
            replay_cli,
            "load_collection_descriptor",
            return_value=descriptor,
        ), mock.patch.object(
            replay_cli,
            "load_planned_candidate_manifest",
            return_value=registered_candidate(candidate),
        ), mock.patch.object(
            replay_cli,
            "_scorer_identity_for_candidate",
            return_value=("1" * 40, digest("scorer")),
        ), mock.patch.object(
            replay_cli, "load_validated_collection"
        ) as collection_loader:
            with self.assertRaisesRegex(CustodianReplayError, "handoff paths"):
                replay_cli._export_input(mismatched_args)
            collection_loader.assert_not_called()

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

        invalid_runtime = copy.deepcopy(receipt)
        invalid_runtime["scorer_runtime"]["unicode_version"] = "unknown"
        with self.assertRaisesRegex(CustodianReplayError, "Unicode version"):
            validate_custodian_receipt(invalid_runtime)
        invalid_runtime = copy.deepcopy(receipt)
        invalid_runtime["scorer_runtime"]["installed_dependency_count"] = 71.0
        with self.assertRaisesRegex(CustodianReplayError, "dependency count"):
            validate_custodian_receipt(invalid_runtime)

        bounded_identity = copy.deepcopy(receipt)
        bounded_identity["experiment_id"] = "EXP-20260827-002-" + "a" * 111
        bounded_identity["candidate_manifest_path"] = (
            f"experiments/manifests/{bounded_identity['experiment_id']}.json"
        )
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
            **registration_fields(receipt["experiment_id"]),
        }
        validate_custodian_receipt(export_receipt)
        export_receipt["decode_item_count"] += 1
        with self.assertRaisesRegex(CustodianReplayError, "1000000 limit"):
            validate_custodian_receipt(export_receipt)

    def test_scorer_code_identity_binds_head_and_exact_source_inventory(self):
        self.assertIn("eval/__init__.py", SCORER_SOURCE_PATHS)
        self.assertIn("scripts/__init__.py", SCORER_SOURCE_PATHS)
        self.assertIn("eval/sealed_candidate_contract.py", SCORER_SOURCE_PATHS)
        self.assertIn("eval/sealed_decoder.py", SCORER_SOURCE_PATHS)
        self.assertIn("requirements/lab-cpu.lock", SCORER_SOURCE_PATHS)
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
            first_commit = subprocess.check_output(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
            ).strip()
            with self.assertRaises(TypeError):
                scorer_code_identity(root)

            with mock.patch.dict(
                os.environ,
                {"GIT_DIR": str(root / "does-not-exist")},
                clear=False,
            ):
                first = scorer_code_identity(root, code_commit=first_commit)
            second = scorer_code_identity(root, code_commit=first_commit)
            self.assertEqual(first, second)
            self.assertRegex(first[0], r"^[0-9a-f]{40}$")
            self.assertRegex(first[1], r"^sha256:[0-9a-f]{64}$")

            (root / "planned-manifest-note.txt").write_text(
                "registered after scorer source freeze\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", "planned-manifest-note.txt"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=ASR Test",
                    "-c",
                    "user.email=asr-test@example.invalid",
                    "commit",
                    "-qm",
                    "register fixture",
                ],
                cwd=root,
                check=True,
            )
            self.assertEqual(
                scorer_code_identity(root, code_commit=first[0]),
                first,
            )
            second_commit = subprocess.check_output(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
            ).strip()
            self.assertNotEqual(
                scorer_code_identity(root, code_commit=second_commit)[0],
                first[0],
            )

            (root / SCORER_SOURCE_PATHS[0]).write_bytes(b"dirty scorer\n")
            with self.assertRaisesRegex(
                CustodianReplayError, "differs from scorer_code_commit"
            ):
                scorer_code_identity(root, code_commit=first[0])

            subprocess.run(
                ["git", "add", SCORER_SOURCE_PATHS[0]],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=ASR Test",
                    "-c",
                    "user.email=asr-test@example.invalid",
                    "commit",
                    "-qm",
                    "mutate scorer fixture",
                ],
                cwd=root,
                check=True,
            )
            replacement = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
            ).strip()
            subprocess.run(
                ["git", "replace", first[0], replacement],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(CustodianReplayError, "replacement refs"):
                scorer_code_identity(root, code_commit=first[0])
            subprocess.run(
                ["git", "replace", "-d", first[0]],
                cwd=root,
                check=True,
                capture_output=True,
            )
            grafts = root / ".git/info/grafts"
            grafts.write_text(f"{first[0]}\n", encoding="utf-8")
            with self.assertRaisesRegex(CustodianReplayError, "grafts"):
                scorer_code_identity(root, code_commit=first[0])

    def test_custodian_recomputes_runner_source_instead_of_trusting_envelope(self):
        candidate = self.planned_manifest()
        expected = (candidate["code_commit"], digest("verified-runner-source"))
        envelope = {
            "runner": {
                "code_commit": expected[0],
                "source_sha256": expected[1],
            }
        }
        with mock.patch.object(
            replay_cli,
            "runner_source_identity",
            return_value=expected,
        ) as identity_reader:
            self.assertEqual(
                replay_cli._validated_runner_source_identity(candidate, envelope),
                expected,
            )
            identity_reader.assert_called_once_with(
                candidate["code_commit"], REPOSITORY_ROOT
            )

            envelope["runner"]["source_sha256"] = digest("self-reported-runner")
            with self.assertRaisesRegex(
                CustodianReplayError, "does not match candidate commit"
            ):
                replay_cli._validated_runner_source_identity(candidate, envelope)

    def assert_terminal_cli_rejects_unregistered_fixture(
        self,
        terminal,
        input_export_receipt,
        prediction_freeze_receipt,
        receipt,
        core,
        execution_envelope,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {
                "terminal": root / "terminal.json",
                "input": root / "input-receipt.json",
                "prediction": root / "prediction-receipt.json",
                "score": root / "score-receipt.json",
                "core": root / "core.json",
                "execution": root / "execution-envelope.json",
            }
            paths["terminal"].write_bytes(canonical_json_bytes(terminal))
            paths["input"].write_bytes(
                canonical_custodian_receipt_bytes(input_export_receipt)
            )
            paths["prediction"].write_bytes(
                canonical_custodian_receipt_bytes(prediction_freeze_receipt)
            )
            paths["score"].write_bytes(
                canonical_custodian_receipt_bytes(receipt)
            )
            paths["core"].write_bytes(canonical_core_bytes(core))
            paths["execution"].write_bytes(
                canonical_execution_envelope_bytes(execution_envelope)
            )
            for path in paths.values():
                path.chmod(0o600)
            environment = {
                "PATH": "/usr/bin:/bin",
                "HOME": "/dev/null",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
            }
            completed = subprocess.run(
                [
                    str(VENV_PYTHON),
                    "-P",
                    "-S",
                    "scripts/replay_asr_evaluation.py",
                    "validate-terminal",
                    "--input-receipt",
                    str(paths["input"]),
                    "--prediction-receipt",
                    str(paths["prediction"]),
                    "--score-receipt",
                    str(paths["score"]),
                    "--core",
                    str(paths["core"]),
                    "--execution-envelope",
                    str(paths["execution"]),
                    "--terminal-manifest",
                    str(paths["terminal"]),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
            )
            # The semantic happy path above is intentionally direct: this
            # synthetic fixture has no committed registration blob.  The real
            # process must fail closed instead of becoming a clean-CI-only
            # false success.
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, b"")
            self.assertTrue(completed.stderr)

            paths["terminal"].write_text(
                json.dumps(terminal, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            rejected = subprocess.run(
                completed.args,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertEqual(rejected.stdout, b"")
            self.assertIn(b"canonical JSON", rejected.stderr)

            paths["terminal"].write_bytes(canonical_json_bytes(terminal))
            paths["terminal"].chmod(0o644)
            rejected_mode = subprocess.run(
                completed.args,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
            )
            self.assertEqual(rejected_mode.returncode, 2)
            self.assertIn(b"exactly 0600", rejected_mode.stderr)

    def test_terminal_manifest_must_preserve_score_receipt_candidate_facts(self):
        core = self.core_report()
        cer = core["aggregate"]["cer"]
        mer = core["aggregate"]["mer"]
        terminal = self.planned_manifest()
        try:
            repository_commit = subprocess.check_output(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip()
            actual_scorer_identity = scorer_code_identity(
                code_commit=repository_commit
            )
            actual_scorer_runtime = runtime_identity(REPOSITORY_ROOT)
            actual_runner_identity = runner_source_identity(
                repository_commit,
                REPOSITORY_ROOT,
            )
        except (CustodianReplayError, SealedDecoderError):
            integration_identities = None
        else:
            integration_identities = (
                actual_scorer_identity,
                actual_scorer_runtime,
                actual_runner_identity,
            )
        if integration_identities is not None:
            actual_scorer_identity, actual_scorer_runtime, actual_runner_identity = (
                integration_identities
            )
            terminal["code_commit"] = actual_scorer_identity[0]
        terminal["decision"] = "accept"
        terminal["metrics"] = {
            "content_cer": cer["errors"] / cer["reference_units"],
            "substitutions": cer["substitutions"],
            "deletions": cer["deletions"],
            "insertions": cer["insertions"],
            "reference_units": cer["reference_units"],
            "utterance_count": core["counts"]["utterance_count"],
            "failed_count": core["counts"]["failed_count"],
            "excluded_count": core["counts"]["excluded_count"],
            "mer": mer["errors"] / mer["reference_units"],
            "rtf_p50": 0.1,
            "rtf_p95": 0.1,
            "peak_rss_mb": 1.0,
            "rtf_attempted_count": 1,
            "retried_count": 0,
            "model_load_seconds": 0.2,
            "cold_inference_seconds": 0.05,
            "cold_start_seconds": 0.25,
            "warm_wall_seconds": 0.1,
            "warm_audio_seconds": 1.0,
        }
        receipt = self.score_receipt(core)
        if integration_identities is not None:
            # Keep the executed terminal fixture schema-valid before deriving
            # its candidate freeze. Artifacts are intentionally outside that
            # immutable projection and are replaced by the full chain below.
            terminal["artifacts"] = [
                {
                    "kind": "report",
                    "path": "eval/private/preflight-report.json",
                    "sha256": digest("preflight-report"),
                }
            ]
            receipt["candidate_freeze_sha256"] = candidate_manifest_freeze_sha256(
                terminal
            )
            receipt["runner_code_commit"] = terminal["code_commit"]
            receipt["runner_source_sha256"] = actual_runner_identity[1]
            (
                receipt["scorer_code_commit"],
                receipt["scorer_source_sha256"],
            ) = actual_scorer_identity
            receipt["scorer_runtime"] = copy.deepcopy(actual_scorer_runtime)
        prediction_items = [
            {
                "id": "sealed-1",
                "raw_text": "第一条",
                "status": "ok",
                "reason_code": None,
            }
        ]
        raw_predictions_sha256 = sha256_bytes(
            b"".join(canonical_json_bytes(item) for item in prediction_items)
        )
        receipt["prediction_items_sha256"] = sha256_bytes(
            canonical_json_bytes(prediction_items)
        )
        input_export_receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": INPUT_EXPORT_RECEIPT_KIND,
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": terminal["experiment_id"],
            "dataset_id": receipt["dataset_id"],
            "revision": receipt["revision"],
            "split": "sealed-blind",
            "decode_item_count": 1,
            "input_projection_sha256": receipt["input_projection_sha256"],
            "candidate_lock_sha256": receipt["candidate_lock_sha256"],
            "candidate_freeze_sha256": receipt["candidate_freeze_sha256"],
            **registration_fields(terminal["experiment_id"]),
        }
        input_export_receipt_sha256 = sha256_bytes(
            canonical_custodian_receipt_bytes(input_export_receipt)
        )
        receipt["input_export_receipt_sha256"] = input_export_receipt_sha256
        observation = {
            "experiment_id": terminal["experiment_id"],
            "dataset_id": receipt["dataset_id"],
            "revision": receipt["revision"],
            "split": "sealed-blind",
            "candidate_freeze_sha256": receipt["candidate_freeze_sha256"],
            "candidate_lock_sha256": receipt["candidate_lock_sha256"],
            "input_projection_sha256": receipt["input_projection_sha256"],
            "hypothesis_adapter_version": "identity-v1",
            "config_sha256": terminal["config_sha256"],
            "models": terminal["models"],
            "command": terminal["command"],
            "hardware": terminal["hardware"],
            "runner_code_commit": terminal["code_commit"],
            "runner_source_sha256": receipt["runner_source_sha256"],
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
            "raw_predictions_sha256": raw_predictions_sha256,
            "prediction_items_sha256": sha256_bytes(
                canonical_json_bytes(prediction_items)
            ),
            "prediction_item_count": 1,
            "started_at_utc": "2026-08-28T00:00:00Z",
            "finished_at_utc": "2026-08-28T00:00:01Z",
            "measurement_contract": {
                "clock_version": "python-perf-counter-ns-v1",
                "rss_version": "linux-rusage-self-maxrss-kib-v1",
                "rss_scope": "fresh-process-rusage-self",
                "rtf_population": "all-measured-attempts",
                "warmup_runs": 1,
            },
            "model_load_ns": 200_000_000,
            "cold_attempt": {
                "id": "sealed-1",
                "attempt_index": 0,
                "elapsed_ns": 50_000_000,
                "audio_duration_seconds": 1.0,
                "status": "ok",
                "reason_code": None,
            },
            "warmup_attempts": [
                {
                    "id": "sealed-1",
                    "attempt_index": 0,
                    "elapsed_ns": 75_000_000,
                    "audio_duration_seconds": 1.0,
                    "status": "ok",
                    "reason_code": None,
                }
            ],
            "decode_attempts": [
                {
                    "id": "sealed-1",
                    "attempt_index": 0,
                    "elapsed_ns": 100_000_000,
                    "audio_duration_seconds": 1.0,
                    "status": "ok",
                    "reason_code": None,
                }
            ],
            "peak_rss_bytes": 1_048_576,
        }
        execution_envelope = build_execution_envelope(
            observation,
            prediction_items,
            input_export_receipt_sha256=input_export_receipt_sha256,
        )
        execution_sha256 = sha256_bytes(
            canonical_execution_envelope_bytes(execution_envelope)
        )
        prediction_freeze_receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": "asr-evaluation-prediction-freeze-receipt",
            "state": "complete",
            "access_class": "restricted",
            "experiment_id": terminal["experiment_id"],
            "dataset_id": receipt["dataset_id"],
            "revision": receipt["revision"],
            "split": "sealed-blind",
            "expected_decode_item_count": 1,
            "prediction_item_count": 1,
            "missing_prediction_count": 0,
            "input_projection_sha256": receipt["input_projection_sha256"],
            "candidate_lock_sha256": receipt["candidate_lock_sha256"],
            "candidate_freeze_sha256": receipt["candidate_freeze_sha256"],
            **registration_fields(terminal["experiment_id"]),
            "hypothesis_adapter_version": "identity-v1",
            "prediction_artifact_sha256": receipt[
                "prediction_artifact_sha256"
            ],
            "prediction_items_sha256": receipt["prediction_items_sha256"],
            "input_export_receipt_sha256": input_export_receipt_sha256,
            "raw_predictions_sha256": raw_predictions_sha256,
            "execution_envelope_sha256": execution_sha256,
            "runner_code_commit": terminal["code_commit"],
            "runner_source_sha256": receipt["runner_source_sha256"],
        }
        receipt["execution_envelope_sha256"] = execution_sha256
        receipt["prediction_freeze_receipt_sha256"] = sha256_bytes(
            canonical_custodian_receipt_bytes(prediction_freeze_receipt)
        )
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
                "kind": "other",
                "path": "eval/private/input-export-receipt.json",
                "sha256": input_export_receipt_sha256,
            },
            {
                "kind": "prediction",
                "path": "eval/private/predictions.json",
                "sha256": receipt["prediction_artifact_sha256"],
            },
            {
                "kind": "report",
                "path": "eval/private/execution-envelope.json",
                "sha256": receipt["execution_envelope_sha256"],
            },
            {
                "kind": "other",
                "path": "eval/private/prediction-receipt.json",
                "sha256": receipt["prediction_freeze_receipt_sha256"],
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
        validate_terminal_manifest_for_receipt(
            terminal,
            receipt,
            core,
            execution_envelope,
            input_export_receipt,
            prediction_freeze_receipt,
        )

        self.assert_terminal_cli_rejects_unregistered_fixture(
            terminal,
            input_export_receipt,
            prediction_freeze_receipt,
            receipt,
            core,
            execution_envelope,
        )

        changed = copy.deepcopy(terminal)
        changed["seed"] = 1
        with self.assertRaisesRegex(CustodianReplayError, "candidate facts"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        changed = copy.deepcopy(terminal)
        changed["metrics"]["failed_count"] = 1
        with self.assertRaisesRegex(CustodianReplayError, "failed_count"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        changed = copy.deepcopy(terminal)
        changed["metrics"]["rtf_p95"] = 0.2
        with self.assertRaisesRegex(CustodianReplayError, "rtf_p95"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        changed = copy.deepcopy(terminal)
        changed["metrics"]["rtf_attempted_count"] = 1.0
        with self.assertRaisesRegex(CustodianReplayError, "rtf_attempted_count"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        changed = copy.deepcopy(terminal)
        changed["metrics"]["unbound_metric"] = 1.0
        with self.assertRaisesRegex(CustodianReplayError, "exactly"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        changed = copy.deepcopy(terminal)
        changed["artifacts"] = changed["artifacts"][:-1]
        with self.assertRaisesRegex(CustodianReplayError, "score receipt"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        changed = copy.deepcopy(terminal)
        changed["artifacts"] = [
            artifact
            for artifact in changed["artifacts"]
            if artifact["sha256"] != input_export_receipt_sha256
        ]
        with self.assertRaisesRegex(CustodianReplayError, "input export receipt"):
            validate_terminal_manifest_for_receipt(
                changed,
                receipt,
                core,
                execution_envelope,
                input_export_receipt,
                prediction_freeze_receipt,
            )

        tampered_input_receipt = copy.deepcopy(input_export_receipt)
        tampered_input_receipt["decode_item_count"] = 2
        with self.assertRaisesRegex(CustodianReplayError, "input export receipt"):
            validate_terminal_manifest_for_receipt(
                terminal,
                receipt,
                core,
                execution_envelope,
                tampered_input_receipt,
                prediction_freeze_receipt,
            )

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
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Fixture"],
                cwd=root,
                check=True,
            )
            (root / "seed.txt").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "add", "seed.txt"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "seed"], cwd=root, check=True
            )
            code_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            planned = self.planned_manifest()
            planned["code_commit"] = code_commit
            manifest_directory = root / "experiments/manifests"
            manifest_directory.mkdir(parents=True)
            path = manifest_directory / f"{planned['experiment_id']}.json"
            path.write_text(
                json.dumps(planned, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", path.relative_to(root).as_posix()],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-q", "-m", "register candidate"],
                cwd=root,
                check=True,
            )
            registration_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                [
                    "git",
                    "update-ref",
                    "refs/remotes/origin/develop",
                    registration_commit,
                ],
                cwd=root,
                check=True,
            )

            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            fake_git = fake_bin / "git"
            marker = root / "fake-git-executed"
            fake_git.write_text(
                "#!/bin/sh\n"
                f"touch {str(marker)!r}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_git.chmod(0o755)
            with mock.patch.dict(
                os.environ,
                {"PATH": f"{fake_bin}:/usr/bin:/bin", "GIT_DIR": "/invalid"},
                clear=False,
            ), mock.patch.object(
                custodian_replay_module,
                "validate_directory",
                return_value=[],
            ) as directory_validator:
                loaded = load_planned_candidate_manifest(
                    path,
                    registration_commit,
                    repository_root=root,
                )
            self.assertEqual(loaded.document, planned)
            self.assertEqual(loaded.payload, path.read_bytes())
            self.assertEqual(loaded.sha256, sha256_bytes(path.read_bytes()))
            self.assertEqual(
                loaded.repository_path,
                path.relative_to(root).as_posix(),
            )
            self.assertEqual(loaded.registration_commit, registration_commit)
            self.assertEqual(
                directory_validator.call_args.kwargs["code_ref"], "HEAD"
            )
            self.assertFalse(marker.exists())

            path.write_text(
                json.dumps(planned, ensure_ascii=False, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CustodianReplayError, "bytes do not match"):
                load_planned_candidate_manifest(
                    path,
                    registration_commit,
                    repository_root=root,
                )

            path.write_bytes(loaded.payload)
            untracked = copy.deepcopy(planned)
            untracked["experiment_id"] = "EXP-20260828-999-untracked"
            untracked_path = manifest_directory / f"{untracked['experiment_id']}.json"
            untracked_path.write_text(
                json.dumps(untracked, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CustodianReplayError, "not tracked"):
                load_planned_candidate_manifest(
                    untracked_path,
                    registration_commit,
                    repository_root=root,
                )

            outside = root / f"{planned['experiment_id']}.json"
            outside.write_bytes(loaded.payload)
            path.write_text(
                loaded.payload.decode("utf-8"), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CustodianReplayError, "exact experiments/manifests path"
            ):
                load_planned_candidate_manifest(
                    outside,
                    registration_commit,
                    repository_root=root,
                )

            with self.assertRaisesRegex(CustodianReplayError, "full Git commit"):
                load_planned_candidate_manifest(
                    path,
                    registration_commit[:12],
                    repository_root=root,
                )

            subprocess.run(
                ["git", "update-ref", "-d", "refs/remotes/origin/develop"],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(CustodianReplayError, "origin/develop"):
                load_planned_candidate_manifest(
                    path,
                    registration_commit,
                    repository_root=root,
                )
            subprocess.run(
                [
                    "git",
                    "update-ref",
                    "refs/remotes/origin/develop",
                    code_commit,
                ],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(CustodianReplayError, "origin/develop"):
                load_planned_candidate_manifest(
                    path,
                    registration_commit,
                    repository_root=root,
                )
            subprocess.run(
                [
                    "git",
                    "update-ref",
                    "refs/remotes/origin/develop",
                    registration_commit,
                ],
                cwd=root,
                check=True,
            )

            subprocess.run(
                ["git", "update-ref", "HEAD", code_commit],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(CustodianReplayError, "checked-out HEAD"):
                load_planned_candidate_manifest(
                    path,
                    registration_commit,
                    repository_root=root,
                )
            subprocess.run(
                ["git", "update-ref", "HEAD", registration_commit],
                cwd=root,
                check=True,
            )

            tree = subprocess.run(
                ["git", "rev-parse", f"{code_commit}^{{tree}}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            foreign_commit = subprocess.run(
                ["git", "commit-tree", tree, "-p", code_commit, "-m", "foreign"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            invalid_registration = copy.deepcopy(planned)
            invalid_registration["code_commit"] = foreign_commit
            path.write_text(
                json.dumps(invalid_registration, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", path.relative_to(root).as_posix()],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-q", "-m", "invalid registration"],
                cwd=root,
                check=True,
            )
            invalid_registration_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                [
                    "git",
                    "update-ref",
                    "refs/remotes/origin/develop",
                    invalid_registration_commit,
                ],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(CustodianReplayError, "code_commit.*ancestor"):
                load_planned_candidate_manifest(
                    path,
                    invalid_registration_commit,
                    repository_root=root,
                )

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

    def test_restricted_transition_requires_one_private_evidence_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vault = root / "vault"
            other_vault = root / "other-vault"
            vault.mkdir(mode=0o700)
            other_vault.mkdir(mode=0o700)
            evidence = vault / "evidence.json"
            evidence.write_bytes(b"evidence\n")
            evidence.chmod(0o600)

            validated_inputs, validated_outputs = (
                validate_restricted_transition_paths(
                    [evidence],
                    [vault / "artifact.json", vault / "receipt.json"],
                )
            )
            self.assertEqual(validated_inputs, (evidence,))
            self.assertEqual(
                validated_outputs,
                (vault / "artifact.json", vault / "receipt.json"),
            )

            evidence.chmod(0o644)
            with self.assertRaisesRegex(CustodianReplayError, "exactly 0600"):
                validate_restricted_transition_paths(
                    [evidence],
                    [vault / "artifact.json", vault / "receipt.json"],
                )
            evidence.chmod(0o600)

            with self.assertRaisesRegex(CustodianReplayError, "share one private"):
                validate_restricted_transition_paths(
                    [evidence],
                    [other_vault / "artifact.json", other_vault / "receipt.json"],
                )

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
            with self.assertRaisesRegex(CustodianReplayError, "exactly 0700"):
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

    def test_atomic_outputs_persists_artifact_before_receipt_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact.json"
            receipt = root / "receipt.json"
            events: list[str] = []
            real_link = os.link

            def observed_link(source, destination):
                events.append(f"link:{Path(destination).name}")
                return real_link(source, destination)

            def observed_fsync(path):
                events.append("fsync-directory")

            with mock.patch(
                "eval.custodian_replay.os.link",
                side_effect=observed_link,
            ), mock.patch(
                "eval.custodian_replay._fsync_directory",
                side_effect=observed_fsync,
            ):
                write_atomic_outputs(
                    [(artifact, b"artifact\n"), (receipt, b"receipt\n")]
                )

            first_link = events.index("link:artifact.json")
            first_fsync = events.index("fsync-directory")
            receipt_link = events.index("link:receipt.json")
            self.assertLess(first_link, first_fsync)
            self.assertLess(first_fsync, receipt_link)

    def test_atomic_outputs_roll_back_receipt_before_predecessor_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact.json"
            receipt = root / "receipt.json"
            unlinked: list[str] = []
            fsync_calls = 0
            real_unlink = Path.unlink

            def fail_after_receipt_link(path):
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 2:
                    raise OSError("simulated receipt directory fsync failure")

            def observed_unlink(path, *args, **kwargs):
                unlinked.append(path.name)
                return real_unlink(path, *args, **kwargs)

            with mock.patch(
                "eval.custodian_replay._fsync_directory",
                side_effect=fail_after_receipt_link,
            ), mock.patch.object(Path, "unlink", autospec=True, side_effect=observed_unlink):
                with self.assertRaisesRegex(OSError, "receipt directory fsync"):
                    write_atomic_outputs(
                        [(artifact, b"artifact\n"), (receipt, b"receipt\n")]
                    )

            published_unlinks = [
                name for name in unlinked if name in {artifact.name, receipt.name}
            ]
            self.assertEqual(published_unlinks, [receipt.name, artifact.name])
            self.assertFalse(receipt.exists())
            self.assertFalse(artifact.exists())

    def test_atomic_outputs_preserves_predecessor_if_receipt_rollback_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact.json"
            receipt = root / "receipt.json"
            fsync_calls = 0
            real_unlink = Path.unlink

            def fail_after_receipt_link(path):
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 2:
                    raise OSError("simulated receipt directory fsync failure")

            def reject_receipt_unlink(path, *args, **kwargs):
                if path == receipt:
                    raise OSError("simulated receipt unlink failure")
                return real_unlink(path, *args, **kwargs)

            with mock.patch(
                "eval.custodian_replay._fsync_directory",
                side_effect=fail_after_receipt_link,
            ), mock.patch.object(
                Path,
                "unlink",
                autospec=True,
                side_effect=reject_receipt_unlink,
            ):
                with self.assertRaisesRegex(OSError, "receipt directory fsync"):
                    write_atomic_outputs(
                        [(artifact, b"artifact\n"), (receipt, b"receipt\n")]
                    )

            self.assertTrue(receipt.exists())
            self.assertTrue(artifact.exists())

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

    def test_regular_file_reader_rejects_identity_drift_during_read(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scorer-source.py"
            path.write_bytes(b"stable scorer source\n")
            real_fstat = os.fstat
            fstat_calls = 0

            def drifting_fstat(descriptor):
                nonlocal fstat_calls
                fstat_calls += 1
                metadata = real_fstat(descriptor)
                if fstat_calls == 2:
                    changed = mock.Mock()
                    changed.st_dev = metadata.st_dev
                    changed.st_ino = metadata.st_ino
                    changed.st_size = metadata.st_size
                    changed.st_mtime_ns = metadata.st_mtime_ns + 1
                    return changed
                return metadata

            with mock.patch(
                "eval.custodian_replay.os.fstat",
                side_effect=drifting_fstat,
            ):
                with self.assertRaisesRegex(
                    CustodianReplayError, "changed while it was read"
                ):
                    custodian_replay_module._regular_file_bytes(
                        path, 1024, "scorer source fixture"
                    )

    def test_regular_file_reader_rejects_atomic_path_replacement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "scorer-source.py"
            replacement = root / "replacement.py"
            path.write_bytes(b"committed scorer source\n")
            replacement.write_bytes(b"replacement scorer source\n")
            real_read = os.read
            replaced = False

            def replace_path(descriptor, maximum):
                nonlocal replaced
                if not replaced:
                    replacement.replace(path)
                    replaced = True
                return real_read(descriptor, maximum)

            with mock.patch(
                "eval.custodian_replay.os.read",
                side_effect=replace_path,
            ):
                with self.assertRaisesRegex(
                    CustodianReplayError, "path changed while it was read"
                ):
                    custodian_replay_module._regular_file_bytes(
                        path, 1024, "scorer source fixture"
                    )


if __name__ == "__main__":
    unittest.main()
