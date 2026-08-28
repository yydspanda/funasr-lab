import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.collection import canonical_json_bytes, sha256_bytes
from eval.execution_envelope import (
    CLOCK_VERSION,
    EXECUTION_ENVELOPE_KIND,
    ExecutionEnvelopeError,
    PERCENTILE_METHOD,
    RSS_SCOPE,
    RSS_VERSION,
    RTF_POPULATION,
    build_execution_envelope,
    canonical_execution_envelope_bytes,
    linear_percentile,
    load_execution_envelope,
    peak_rss_mib,
    validate_execution_envelope,
    validate_execution_envelope_for_predictions,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def predictions() -> list[dict[str, object]]:
    return [
        {"id": "utt-1", "raw_text": "你好", "status": "ok", "reason_code": None},
        {
            "id": "utt-2",
            "raw_text": "",
            "status": "failed",
            "reason_code": "decoder_failure",
        },
    ]


def observation(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "experiment_id": "EXP-20260828-001-envelope",
        "dataset_id": "LAB-SEALED-v1",
        "revision": "v0.1",
        "split": "sealed-blind",
        "candidate_freeze_sha256": digest("candidate-freeze"),
        "candidate_lock_sha256": digest("candidate-lock"),
        "input_projection_sha256": digest("input-projection"),
        "hypothesis_adapter_version": "identity-v1",
        "config_sha256": digest("effective-config"),
        "models": [
            {
                "role": "asr",
                "identifier": "fixture/asr",
                "revision": "v1.0.0",
                "sha256": digest("model-bundle"),
            }
        ],
        "command": {
            "working_directory": ".",
            "argv": [".venv/bin/python", "scripts/run_sealed_asr_candidate.py"],
            "environment": {"OMP_NUM_THREADS": "4", "PYTHONHASHSEED": "0"},
        },
        "hardware": {
            "host_id": "fixture-host",
            "os": "Linux fixture x86_64",
            "cpu_model": "Fixture CPU",
            "logical_cpu_count": 4,
            "memory_bytes": 8 * 1024**3,
            "device": "cpu",
            "accelerator": None,
        },
        "runtime": {
            "python_implementation": "cpython",
            "python_version": "3.11.15",
            "python_cache_tag": "cpython-311",
            "dependency_lock_sha256": digest("lab-cpu-lock"),
            "installed_dependencies_sha256": digest("installed-dependencies"),
            "installed_dependency_count": 71,
            "unicode_version": "14.0.0",
        },
        "runner_code_commit": "a" * 40,
        "runner_source_sha256": digest("runner-source"),
        "raw_predictions_sha256": digest("raw-prediction-jsonl"),
        "prediction_items_sha256": sha256_bytes(canonical_json_bytes(items)),
        "prediction_item_count": len(items),
        "started_at_utc": "2026-08-28T01:02:03.123456Z",
        "finished_at_utc": "2026-08-28T01:02:05.123456Z",
        "measurement_contract": {
            "clock_version": CLOCK_VERSION,
            "rss_version": RSS_VERSION,
            "rss_scope": RSS_SCOPE,
            "rtf_population": RTF_POPULATION,
            "warmup_runs": 1,
        },
        "model_load_ns": 1_000_000_000,
        "cold_attempt": {
            "id": "utt-1",
            "attempt_index": 0,
            "elapsed_ns": 200_000_000,
            "audio_duration_seconds": 1.0,
            "status": "ok",
            "reason_code": None,
        },
        "warmup_attempts": [
            {
                "id": "utt-1",
                "attempt_index": 0,
                "elapsed_ns": 100_000_000,
                "audio_duration_seconds": 1.0,
                "status": "ok",
                "reason_code": None,
            }
        ],
        "decode_attempts": [
            {
                "id": "utt-1",
                "attempt_index": 0,
                "elapsed_ns": 100_000_000,
                "audio_duration_seconds": 1.0,
                "status": "ok",
                "reason_code": None,
            },
            {
                "id": "utt-2",
                "attempt_index": 1,
                "elapsed_ns": 500_000_000,
                "audio_duration_seconds": 2.0,
                "status": "failed",
                "reason_code": "decoder_failure",
            },
        ],
        "peak_rss_bytes": 512 * 1024**2,
    }


class ExecutionEnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = predictions()
        self.observation = observation(self.predictions)
        self.input_export_receipt_sha256 = digest("input-export-receipt")
        self.envelope = build_execution_envelope(
            self.observation,
            self.predictions,
            input_export_receipt_sha256=self.input_export_receipt_sha256,
        )

    def assert_rejected(self, document: object) -> None:
        with self.assertRaises(ExecutionEnvelopeError):
            validate_execution_envelope(document)

    def test_build_recomputes_metrics_counts_and_text_free_bindings(self):
        envelope = self.envelope
        self.assertEqual(envelope["kind"], EXECUTION_ENVELOPE_KIND)
        self.assertEqual(envelope["access_class"], "restricted")
        self.assertEqual(
            envelope["bindings"]["raw_predictions_sha256"],
            self.observation["raw_predictions_sha256"],
        )
        self.assertEqual(
            envelope["bindings"]["prediction_items_sha256"],
            self.observation["prediction_items_sha256"],
        )
        self.assertEqual(
            envelope["bindings"]["input_export_receipt_sha256"],
            self.input_export_receipt_sha256,
        )
        self.assertEqual(envelope["runner"]["runtime"], self.observation["runtime"])
        measurement = envelope["measurement"]
        self.assertEqual(measurement["started_at_utc"], self.observation["started_at_utc"])
        self.assertEqual(measurement["finished_at_utc"], self.observation["finished_at_utc"])
        self.assertEqual(measurement["model_load_ns"], 1_000_000_000)
        self.assertEqual(measurement["cold_inference_ns"], 200_000_000)
        self.assertEqual(measurement["cold_start_ns"], 1_200_000_000)
        self.assertEqual(measurement["warmup_wall_ns"], 100_000_000)
        self.assertEqual(measurement["measured_wall_ns"], 600_000_000)
        self.assertEqual(measurement["measured_audio_seconds"], 3.0)
        self.assertEqual(measurement["rtf_p50"], 0.175)
        self.assertEqual(measurement["rtf_p95"], 0.2425)
        self.assertEqual(
            measurement["counts"],
            {
                "decode_item_count": 2,
                "prediction_item_count": 2,
                "missing_prediction_count": 0,
                "total_attempt_count": 2,
                "retried_item_count": 0,
                "ok_count": 1,
                "empty_count": 0,
                "failed_count": 1,
            },
        )
        payload = canonical_execution_envelope_bytes(envelope)
        self.assertNotIn("你好".encode(), payload)
        self.assertNotIn(b"raw_text", payload)
        self.assertNotIn(b"reference", payload)
        self.assertNotIn(b"exception", payload)
        self.assertNotIn(b"prediction_artifact", payload)

    def test_missing_prediction_is_explicit_and_cross_checked(self):
        partial = self.predictions[:1]
        facts = observation(partial)
        facts["decode_attempts"][1]["status"] = "failed"
        facts["decode_attempts"][1]["reason_code"] = "missing_prediction"
        envelope = build_execution_envelope(
            facts,
            partial,
            input_export_receipt_sha256=self.input_export_receipt_sha256,
        )
        self.assertEqual(envelope["items"][1]["reason_code"], "missing_prediction")
        counts = envelope["measurement"]["counts"]
        self.assertEqual(counts["missing_prediction_count"], 1)
        self.assertEqual(counts["prediction_item_count"], 1)
        validate_execution_envelope_for_predictions(
            envelope,
            partial,
            raw_predictions_sha256=facts["raw_predictions_sha256"],
        )

    def test_builder_rejects_order_hash_status_and_time_mismatches(self):
        cases: list[tuple[str, dict[str, object], list[dict[str, object]]]] = []

        reordered = copy.deepcopy(self.predictions)
        reordered.reverse()
        reordered_facts = observation(reordered)
        cases.append(("prediction order", reordered_facts, reordered))

        bad_hash = copy.deepcopy(self.observation)
        bad_hash["prediction_items_sha256"] = digest("wrong-items")
        cases.append(("prediction hash", bad_hash, self.predictions))

        bad_status = copy.deepcopy(self.observation)
        bad_status["decode_attempts"][1]["reason_code"] = "timeout"
        cases.append(("status", bad_status, self.predictions))

        bad_time = copy.deepcopy(self.observation)
        bad_time["finished_at_utc"] = "2026-08-28T01:02:02.123456Z"
        cases.append(("time order", bad_time, self.predictions))

        bad_format = copy.deepcopy(self.observation)
        bad_format["started_at_utc"] = "2026-08-28T01:02:03+00:00"
        cases.append(("time format", bad_format, self.predictions))

        for name, facts, items in cases:
            with self.subTest(name=name), self.assertRaises(ExecutionEnvelopeError):
                build_execution_envelope(
                    facts,
                    items,
                    input_export_receipt_sha256=self.input_export_receipt_sha256,
                )

    def test_validator_recomputes_all_derived_evidence(self):
        mutations = [
            ("measured_wall_ns", 1),
            ("rtf_p95", 1.0),
            ("cold_start_ns", 1),
        ]
        for field, value in mutations:
            document = copy.deepcopy(self.envelope)
            document["measurement"][field] = value
            with self.subTest(field=field):
                self.assert_rejected(document)

        document = copy.deepcopy(self.envelope)
        document["measurement"]["counts"]["failed_count"] = 0
        self.assert_rejected(document)

        document = copy.deepcopy(self.envelope)
        document["items"].reverse()
        self.assert_rejected(document)

        document = copy.deepcopy(self.envelope)
        document["schema_version"] = True
        self.assert_rejected(document)

        document = copy.deepcopy(self.envelope)
        document["runner"]["runtime"]["installed_dependency_count"] = True
        self.assert_rejected(document)

        document = copy.deepcopy(self.envelope)
        document["runner"]["runtime"]["unexpected"] = "unbound"
        self.assert_rejected(document)

    def test_prediction_cross_validation_rejects_bytes_text_and_status_changes(self):
        with self.assertRaises(ExecutionEnvelopeError):
            validate_execution_envelope_for_predictions(
                self.envelope,
                self.predictions,
                raw_predictions_sha256=digest("other-raw-bytes"),
            )

        changed = copy.deepcopy(self.predictions)
        changed[0]["raw_text"] = "您好"
        with self.assertRaises(ExecutionEnvelopeError):
            validate_execution_envelope_for_predictions(
                self.envelope,
                changed,
                raw_predictions_sha256=self.observation["raw_predictions_sha256"],
            )

        changed = copy.deepcopy(self.predictions)
        changed[1]["status"] = "empty"
        changed[1]["reason_code"] = "empty_hypothesis"
        with self.assertRaises(ExecutionEnvelopeError):
            validate_execution_envelope_for_predictions(
                self.envelope,
                changed,
                raw_predictions_sha256=self.observation["raw_predictions_sha256"],
            )

    def test_loader_requires_canonical_regular_non_symlink_json(self):
        payload = canonical_execution_envelope_bytes(self.envelope)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path = root / "execution-envelope.json"
            canonical_path.write_bytes(payload)
            loaded = load_execution_envelope(canonical_path)
            self.assertEqual(loaded.document, self.envelope)
            self.assertEqual(loaded.payload, payload)
            self.assertEqual(loaded.sha256, sha256_bytes(payload))

            pretty_path = root / "pretty.json"
            pretty_path.write_text(json.dumps(self.envelope), encoding="utf-8")
            with self.assertRaises(ExecutionEnvelopeError):
                load_execution_envelope(pretty_path)

            duplicate_path = root / "duplicate.json"
            duplicate_path.write_bytes(b'{"schema_version":1,"schema_version":1}\n')
            with self.assertRaises(ExecutionEnvelopeError):
                load_execution_envelope(duplicate_path)

            nan_path = root / "nan.json"
            nan_path.write_bytes(b'{"value":NaN}\n')
            with self.assertRaises(ExecutionEnvelopeError):
                load_execution_envelope(nan_path)

            bom_path = root / "bom.json"
            bom_path.write_bytes(b"\xef\xbb\xbf" + payload)
            with self.assertRaises(ExecutionEnvelopeError):
                load_execution_envelope(bom_path)

            if hasattr(os, "symlink"):
                link_path = root / "link.json"
                link_path.symlink_to(canonical_path)
                with self.assertRaises(ExecutionEnvelopeError):
                    load_execution_envelope(link_path)

    def test_loader_rejects_same_inode_mutation_during_read(self):
        payload = canonical_execution_envelope_bytes(self.envelope)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution-envelope.json"
            path.write_bytes(payload)
            real_read = os.read
            mutated = False

            def mutate_then_read(descriptor, maximum):
                nonlocal mutated
                if not mutated:
                    with path.open("ab") as destination:
                        destination.write(b" ")
                    mutated = True
                return real_read(descriptor, maximum)

            with mock.patch(
                "eval.execution_envelope.os.read",
                side_effect=mutate_then_read,
            ):
                with self.assertRaisesRegex(
                    ExecutionEnvelopeError,
                    "changed while it was read",
                ):
                    load_execution_envelope(path)

    def test_loader_rejects_path_replacement_during_read(self):
        payload = canonical_execution_envelope_bytes(self.envelope)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "execution-envelope.json"
            replacement = root / "replacement.json"
            path.write_bytes(payload)
            replacement.write_bytes(payload)
            real_read = os.read
            replaced = False

            def replace_then_read(descriptor, maximum):
                nonlocal replaced
                if not replaced:
                    replacement.replace(path)
                    replaced = True
                return real_read(descriptor, maximum)

            with mock.patch(
                "eval.execution_envelope.os.read",
                side_effect=replace_then_read,
            ):
                with self.assertRaisesRegex(
                    ExecutionEnvelopeError,
                    "path changed while it was read",
                ):
                    load_execution_envelope(path)

    def test_percentile_rss_and_schema_contract(self):
        self.assertEqual(linear_percentile([0.25, 0.10], 0.50), 0.175)
        self.assertAlmostEqual(linear_percentile([0.25, 0.10], 0.95), 0.2425)
        self.assertEqual(peak_rss_mib(512 * 1024**2), 512.0)
        with self.assertRaises(ExecutionEnvelopeError):
            linear_percentile([], 0.5)
        with self.assertRaises(ExecutionEnvelopeError):
            peak_rss_mib(0)

        schema_path = REPOSITORY_ROOT / "eval/execution-envelope.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["kind"]["const"], EXECUTION_ENVELOPE_KIND)
        self.assertIs(schema["additionalProperties"], False)
        measurement = schema["$defs"]["measurement"]
        self.assertIn("runtime", schema["$defs"]["runner"]["required"])
        self.assertIn(
            "installed_dependencies_sha256",
            schema["$defs"]["runtime"]["required"],
        )
        self.assertIn("started_at_utc", measurement["required"])
        self.assertIn("finished_at_utc", measurement["required"])
        self.assertNotIn("prediction_artifact_sha256", json.dumps(schema))
        self.assertEqual(
            measurement["properties"]["percentile_method"]["const"],
            PERCENTILE_METHOD,
        )


if __name__ == "__main__":
    unittest.main()
