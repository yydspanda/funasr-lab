import copy
import json
import unittest
from pathlib import Path

from eval.collection import ValidatedCollection
from eval.core_report import ALIGNMENT_VERSION
from eval.core_report import CORE_KIND
from eval.core_report import PUBLIC_SUMMARY_KIND
from eval.core_report import SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import CoreReportValidationError
from eval.core_report import build_core_report
from eval.core_report import build_public_summary
from eval.core_report import canonical_core_bytes
from eval.core_report import canonical_public_summary_bytes
from eval.core_report import core_report_sha256
from eval.core_report import validate_core_report
from eval.core_report import validate_core_report_for_collection
from eval.core_report import validate_public_summary
from eval.offline_baseline import strip_sensevoice_tags
from eval.record_identity import RECORD_IDENTITY_VERSION
from eval.record_identity import record_input_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "sha256:" + "1" * 64


class CoreReportTest(unittest.TestCase):
    def collection(
        self,
        records,
        *,
        data_sha256=DATA_SHA256,
        expected_record_input_sha256=None,
    ):
        frozen_records = []
        for source in records:
            record = copy.deepcopy(source)
            record.setdefault("scenario_tags", ["language:zh"])
            record.setdefault("evaluation_status", "included")
            record.setdefault("exclusion_reason", None)
            frozen_records.append(record)
        record_digest = (
            record_input_sha256(frozen_records)
            if expected_record_input_sha256 is None
            else expected_record_input_sha256
        )
        return ValidatedCollection(
            summary={
                "data_sha256": data_sha256,
                "record_identity_version": RECORD_IDENTITY_VERSION,
                "record_input_sha256": record_digest,
            },
            records=tuple(frozen_records),
            _sealed_input_projection=b"test-only-projection\n",
        )

    def records(self):
        return [
            {
                "id": "ok-mixed",
                "raw_text": "你好 world",
                "split": "dev",
                "scenario_tags": ["language:zh-en", "environment:meeting"],
            },
            {
                "id": "missing",
                "raw_text": "失败",
                "split": "dev",
                "scenario_tags": ["language:zh", "noise:clean"],
            },
            {
                "id": "empty",
                "raw_text": "空",
                "split": "sealed-blind",
                "scenario_tags": ["language:zh", "environment:far-field"],
            },
            {
                "id": "zero-reference",
                "raw_text": "",
                "split": "sealed-blind",
                "scenario_tags": ["language:zh", "signal:non-speech"],
            },
            {
                "id": "excluded",
                "raw_text": "排除",
                "split": "dev",
                "scenario_tags": ["language:zh", "noise:clean"],
                "evaluation_status": "excluded",
                "exclusion_reason": "consent_withdrawn",
            },
        ]

    def predictions(self, *, timing=9.25, detail="host-specific traceback"):
        return [
            {
                "id": "ok-mixed",
                "raw_text": "你号 word",
                "wall_seconds": timing,
                "raw_exception": detail,
            },
            {"id": "empty", "raw_text": ""},
            {"id": "zero-reference", "raw_text": "啊"},
        ]

    def build(self, records=None, predictions=None):
        selected_records = self.records() if records is None else records
        return build_core_report(
            self.collection(selected_records),
            self.predictions() if predictions is None else predictions,
        )

    def build_simple(self, records, predictions, **kwargs):
        return build_core_report(
            self.collection(records),
            predictions,
            **kwargs,
        )

    def test_builds_hand_calculated_cer_mer_failure_and_zero_reference_counts(self):
        report = self.build()

        self.assertEqual(report["kind"], CORE_KIND)
        self.assertEqual(report["access_class"], "restricted")
        self.assertEqual(
            report["provenance"]["record_identity_version"],
            "eval-core-record-input-v1",
        )
        self.assertEqual(
            report["scoring"]["alignment_version"],
            "levenshtein-diagonal-deletion-insertion-v1",
        )
        self.assertEqual(report["scoring"]["alignment_version"], ALIGNMENT_VERSION)
        self.assertEqual(
            [item["id"] for item in report["items"]],
            [record["id"] for record in self.records()],
        )
        self.assertEqual(
            report["configuration"]["slice_fields"],
            ["scenario_tags", "split"],
        )

        by_id = {item["id"]: item for item in report["items"]}
        mixed = by_id["ok-mixed"]
        self.assertEqual(mixed["reference"]["mer_units"], ["你", "好", "world"])
        self.assertEqual(mixed["hypothesis"]["mer_units"], ["你", "号", "word"])
        self.assertEqual(
            {
                "substitutions": mixed["mer"]["substitutions"],
                "deletions": mixed["mer"]["deletions"],
                "insertions": mixed["mer"]["insertions"],
                "reference_units": mixed["mer"]["reference_units"],
            },
            {
                "substitutions": 2,
                "deletions": 0,
                "insertions": 0,
                "reference_units": 3,
            },
        )

        missing = by_id["missing"]
        self.assertEqual(missing["status"], "failed")
        self.assertEqual(missing["reason_code"], "missing_prediction")
        self.assertEqual(missing["hypothesis"]["raw"], "")
        self.assertEqual(missing["cer"]["deletions"], 2)
        self.assertEqual(missing["cer"]["reference_units"], 2)

        empty = by_id["empty"]
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(empty["reason_code"], "empty_hypothesis")
        self.assertEqual(empty["cer"]["deletions"], 1)

        zero_reference = by_id["zero-reference"]
        self.assertEqual(zero_reference["cer"]["insertions"], 1)
        self.assertEqual(zero_reference["cer"]["reference_units"], 0)
        self.assertIsNone(zero_reference["cer"]["rate"])
        self.assertIsNone(zero_reference["cer"]["rate_decimal"])

        excluded = by_id["excluded"]
        self.assertEqual(excluded["status"], "excluded")
        self.assertEqual(excluded["reason_code"], "consent_withdrawn")
        self.assertIsNone(excluded["hypothesis"])
        self.assertIsNone(excluded["cer"])
        self.assertIsNone(excluded["mer"])

        self.assertEqual(
            report["counts"],
            {
                "utterance_count": 5,
                "scored_count": 4,
                "ok_count": 2,
                "failed_count": 1,
                "empty_count": 1,
                "excluded_count": 1,
                "zero_reference_count": 1,
            },
        )
        self.assertEqual(
            {
                "substitutions": report["aggregate"]["cer"]["substitutions"],
                "deletions": report["aggregate"]["cer"]["deletions"],
                "insertions": report["aggregate"]["cer"]["insertions"],
                "reference_units": report["aggregate"]["cer"]["reference_units"],
                "rate": report["aggregate"]["cer"]["rate"],
                "rate_decimal": report["aggregate"]["cer"]["rate_decimal"],
            },
            {
                "substitutions": 1,
                "deletions": 4,
                "insertions": 1,
                "reference_units": 10,
                "rate": {"numerator": 6, "denominator": 10},
                "rate_decimal": "0.6",
            },
        )
        self.assertEqual(
            {
                "substitutions": report["aggregate"]["mer"]["substitutions"],
                "deletions": report["aggregate"]["mer"]["deletions"],
                "insertions": report["aggregate"]["mer"]["insertions"],
                "reference_units": report["aggregate"]["mer"]["reference_units"],
                "rate": report["aggregate"]["mer"]["rate"],
            },
            {
                "substitutions": 2,
                "deletions": 3,
                "insertions": 1,
                "reference_units": 6,
                "rate": {"numerator": 6, "denominator": 6},
            },
        )

    def test_core_bytes_ignore_prediction_order_and_execution_only_metadata(self):
        first = self.build()
        changed_metadata = self.predictions(timing=1234.5, detail="another machine")
        changed_metadata.reverse()
        second = self.build(predictions=changed_metadata)

        first_bytes = canonical_core_bytes(first)
        self.assertEqual(first_bytes, canonical_core_bytes(first))
        self.assertEqual(first_bytes, canonical_core_bytes(second))
        self.assertEqual(core_report_sha256(first), core_report_sha256(second))
        self.assertNotIn(b"wall_seconds", first_bytes)
        self.assertNotIn(b"raw_exception", first_bytes)
        self.assertNotIn(b"traceback", first_bytes)

    def test_frozen_record_order_is_part_of_core_identity(self):
        records = list(reversed(self.records()))
        reordered = self.build(records=records)

        self.assertNotEqual(
            canonical_core_bytes(self.build()), canonical_core_bytes(reordered)
        )
        self.assertNotEqual(
            core_report_sha256(self.build()), core_report_sha256(reordered)
        )
        self.assertEqual(
            [item["id"] for item in reordered["items"]],
            [record["id"] for record in records],
        )

    def test_decoder_cleaned_display_is_scored_but_raw_prediction_is_preserved(self):
        records = [
            {
                "id": "sense",
                "raw_text": "你好",
                "display_text": "不得改写冻结参考",
                "split": "dev",
            }
        ]
        predictions = [
            {
                "id": "sense",
                "raw_text": "<|zh|><|Speech|>你好<|woitn|>",
            }
        ]

        report = self.build_simple(
            records,
            predictions,
            hypothesis_adapter_version=SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
        )
        item = report["items"][0]
        self.assertEqual(item["reference"]["raw"], "你好")
        self.assertEqual(item["reference"]["display"], "你好")
        self.assertEqual(
            item["hypothesis"]["raw"], "<|zh|><|Speech|>你好<|woitn|>"
        )
        self.assertEqual(item["hypothesis"]["display"], "你好")
        self.assertEqual(item["cer"]["errors"], 0)
        self.assertEqual(item["mer"]["errors"], 0)

    def test_caller_cannot_spoof_decoder_cleaned_display(self):
        records = [{"id": "spoof", "raw_text": "正确参考", "split": "dev"}]
        with self.assertRaisesRegex(CoreReportValidationError, "caller-controlled"):
            self.build_simple(
                records,
                [
                    {
                        "id": "spoof",
                        "raw_text": "完全错误",
                        "display_text": "正确参考",
                    }
                ],
            )

    def test_cleaned_empty_display_can_preserve_raw_decoder_control_tags(self):
        records = [{"id": "silence", "raw_text": "", "split": "dev"}]
        report = self.build_simple(
            records,
            [
                {
                    "id": "silence",
                    "raw_text": "<|zh|><|nospeech|>",
                }
            ],
            hypothesis_adapter_version=SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
        )

        item = report["items"][0]
        self.assertEqual(item["status"], "empty")
        self.assertEqual(item["hypothesis"]["raw"], "<|zh|><|nospeech|>")
        self.assertEqual(item["hypothesis"]["display"], "")
        self.assertEqual(item["cer"]["errors"], 0)
        self.assertIsNone(item["cer"]["rate"])

    def test_versioned_sensevoice_adapter_stays_compatible_with_base_cleanup(self):
        raw_values = (
            "<|zh|><|Speech|>你好，World！<|woitn|>",
            "  <|zh|><|nospeech|>  ",
            "没有控制标签",
        )
        for index, raw_text in enumerate(raw_values):
            with self.subTest(raw_text=raw_text):
                report = self.build_simple(
                    [{"id": f"sense-{index}", "raw_text": "", "split": "dev"}],
                    [{"id": f"sense-{index}", "raw_text": raw_text}],
                    hypothesis_adapter_version=SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
                )
                self.assertEqual(
                    report["items"][0]["hypothesis"]["display"],
                    strip_sensevoice_tags(raw_text),
                )

    def test_all_missing_predictions_still_produce_scored_failure_evidence(self):
        report = self.build_simple(
            [{"id": "missing-only", "raw_text": "失败", "split": "dev"}], []
        )

        self.assertEqual(report["counts"]["failed_count"], 1)
        self.assertEqual(report["counts"]["scored_count"], 1)
        self.assertEqual(report["aggregate"]["cer"]["deletions"], 2)
        self.assertEqual(report["aggregate"]["cer"]["reference_units"], 2)
        self.assertEqual(
            report["aggregate"]["cer"]["rate"],
            {"numerator": 2, "denominator": 2},
        )

    def test_rejects_duplicate_extra_and_dynamic_exclusion_predictions(self):
        records = [{"id": "one", "raw_text": "一", "split": "dev"}]
        duplicate = [
            {"id": "one", "raw_text": "一"},
            {"id": "one", "raw_text": "一"},
        ]
        with self.assertRaisesRegex(CoreReportValidationError, "duplicate prediction"):
            self.build_simple(records, duplicate)

        with self.assertRaisesRegex(CoreReportValidationError, "absent from frozen"):
            self.build_simple(
                records,
                [{"id": "extra", "raw_text": "一"}],
            )

        with self.assertRaisesRegex(CoreReportValidationError, "ok, failed, or empty"):
            self.build_simple(
                records,
                [{"id": "one", "raw_text": "", "status": "excluded"}],
            )

        excluded_records = [
            {
                "id": "one",
                "raw_text": "一",
                "split": "dev",
                "evaluation_status": "excluded",
                "exclusion_reason": "predeclared_quality_gate",
            }
        ]
        with self.assertRaisesRegex(
            CoreReportValidationError, "must not have a prediction"
        ):
            self.build_simple(
                excluded_records,
                [{"id": "one", "raw_text": "一"}],
            )

    def test_rejects_duplicate_records_and_unstable_reason_codes(self):
        duplicate_records = [
            {"id": "one", "raw_text": "一", "split": "dev"},
            {"id": "one", "raw_text": "一", "split": "blind"},
        ]
        with self.assertRaisesRegex(CoreReportValidationError, "duplicate record"):
            self.build_simple(duplicate_records, [])

        records = [{"id": "one", "raw_text": "一", "split": "dev"}]
        with self.assertRaisesRegex(CoreReportValidationError, "snake_case"):
            self.build_simple(
                records,
                [
                    {
                        "id": "one",
                        "raw_text": "",
                        "status": "failed",
                        "reason_code": "RuntimeError: device 7 failed",
                    }
                ],
            )

    def test_record_input_hash_and_nonempty_slices_are_enforced(self):
        records = [{"id": "one", "raw_text": "一", "split": "dev"}]
        with self.assertRaisesRegex(CoreReportValidationError, "does not match"):
            build_core_report(
                self.collection(
                    records,
                    expected_record_input_sha256="sha256:" + "2" * 64,
                ),
                [{"id": "one", "raw_text": "一"}],
            )

        empty_slice = [
            {"id": "one", "raw_text": "一", "split": "dev", "scenario_tags": []}
        ]
        with self.assertRaisesRegex(CoreReportValidationError, "must not be empty"):
            build_core_report(
                self.collection(
                    empty_slice,
                    expected_record_input_sha256="sha256:" + "2" * 64,
                ),
                [{"id": "one", "raw_text": "一"}],
            )

    def test_default_slices_match_the_strict_collection_record_contract(self):
        records = [
            {
                "id": "collection-item",
                "raw_text": "会议语音",
                "split": "dev",
                "scenario_tags": ["language:zh", "environment:meeting"],
                "evaluation_status": "included",
                "exclusion_reason": None,
            }
        ]
        collection = self.collection(records)
        expected_digest = collection.summary["record_input_sha256"]
        report = build_core_report(
            collection,
            [{"id": "collection-item", "raw_text": "会议语音"}],
        )

        self.assertEqual(
            report["configuration"]["slice_fields"],
            ["scenario_tags", "split"],
        )
        self.assertEqual(
            report["provenance"]["record_input_sha256"], expected_digest
        )

    def test_report_is_atomically_bound_to_its_validated_collection(self):
        records = [
            {
                "id": "bound",
                "raw_text": "原始参考",
                "split": "dev",
                "scenario_tags": ["language:zh", "noise:clean"],
            }
        ]
        collection = self.collection(records)
        report = build_core_report(
            collection, [{"id": "bound", "raw_text": "原始参考"}]
        )
        validate_core_report_for_collection(report, collection)

        unrelated_data = self.collection(
            records, data_sha256="sha256:" + "3" * 64
        )
        with self.assertRaisesRegex(CoreReportValidationError, "data_sha256"):
            validate_core_report_for_collection(report, unrelated_data)

        unrelated_records = self.collection(
            [
                {
                    "id": "bound",
                    "raw_text": "另一份参考",
                    "split": "dev",
                    "scenario_tags": ["language:zh", "noise:clean"],
                }
            ]
        )
        with self.assertRaisesRegex(CoreReportValidationError, "record_input_sha256"):
            validate_core_report_for_collection(report, unrelated_records)

    def test_semantic_validator_rejects_split_only_slice_configuration(self):
        report = self.build()
        report["configuration"]["slice_fields"] = ["split"]

        with self.assertRaisesRegex(CoreReportValidationError, "frozen collection"):
            validate_core_report(report)

    def test_semantic_validator_rejects_component_count_and_aggregate_tampering(self):
        report = self.build()
        validate_core_report(report)

        bad_component = copy.deepcopy(report)
        bad_component["items"][0]["cer"]["errors"] += 1
        with self.assertRaisesRegex(CoreReportValidationError, "errors must equal"):
            validate_core_report(bad_component)

        bad_count = copy.deepcopy(report)
        bad_count["counts"]["failed_count"] = 0
        with self.assertRaisesRegex(CoreReportValidationError, "counts do not match"):
            canonical_core_bytes(bad_count)

        bad_aggregate = copy.deepcopy(report)
        bad_aggregate["aggregate"]["cer"]["deletions"] += 1
        bad_aggregate["aggregate"]["cer"]["errors"] += 1
        bad_aggregate["aggregate"]["cer"]["rate"]["numerator"] += 1
        bad_aggregate["aggregate"]["cer"]["rate_decimal"] = "0.7"
        with self.assertRaisesRegex(CoreReportValidationError, "aggregate metrics"):
            validate_core_report(bad_aggregate)

    def test_versioned_json_schema_matches_emitted_core_contract(self):
        schema_path = REPOSITORY_ROOT / "eval/core-report.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(schema["properties"]["kind"]["const"], CORE_KIND)
        self.assertEqual(schema["properties"]["access_class"]["const"], "restricted")
        self.assertEqual(
            schema["properties"]["scoring"]["properties"]["alignment_version"]["const"],
            ALIGNMENT_VERSION,
        )
        self.assertEqual(
            schema["properties"]["configuration"]["properties"]["slice_fields"][
                "const"
            ],
            ["scenario_tags", "split"],
        )
        self.assertIn("rate", schema["$defs"]["metric"]["required"])
        self.assertIn("rate_decimal", schema["$defs"]["metric"]["required"])
        self.assertIn(
            "excluded",
            schema["$defs"]["item"]["properties"]["status"]["enum"],
        )

    def test_public_summary_is_bound_to_core_and_contains_no_blind_item_text(self):
        secret_id = "sealed-secret-utterance"
        secret_reference = "盲测绝密参考文本"
        secret_hypothesis = "盲测错误假设文本"
        core = self.build_simple(
            [
                {
                    "id": secret_id,
                    "raw_text": secret_reference,
                    "split": "sealed-blind",
                }
            ],
            [{"id": secret_id, "raw_text": secret_hypothesis}],
        )

        summary = build_public_summary(core)
        validate_public_summary(summary, source_core=core)
        payload = canonical_public_summary_bytes(summary)

        self.assertEqual(summary["kind"], PUBLIC_SUMMARY_KIND)
        self.assertEqual(summary["access_class"], "public")
        self.assertEqual(summary["core_sha256"], core_report_sha256(core))
        self.assertNotIn("items", summary)
        self.assertNotIn("provenance", summary)
        self.assertTrue(all("item_ids" not in item for item in summary["slices"]))
        for blind_value in (
            secret_id,
            secret_reference,
            secret_hypothesis,
            '"items"',
            '"item_ids"',
        ):
            self.assertNotIn(blind_value.encode("utf-8"), payload)
        self.assertEqual(payload, canonical_public_summary_bytes(summary))

        summary_schema = json.loads(
            (REPOSITORY_ROOT / "eval/core-summary.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary_schema["properties"]["kind"]["const"], PUBLIC_SUMMARY_KIND
        )
        self.assertEqual(
            summary_schema["properties"]["configuration"]["properties"][
                "slice_fields"
            ]["const"],
            ["scenario_tags", "split"],
        )
        self.assertNotIn("items", summary_schema["properties"])


if __name__ == "__main__":
    unittest.main()
