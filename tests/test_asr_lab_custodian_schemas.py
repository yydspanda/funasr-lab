import json
import unittest
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATHS = {
    "candidate": REPOSITORY_ROOT / "eval/candidate-lock.schema.json",
    "execution": REPOSITORY_ROOT / "eval/execution-envelope.schema.json",
    "predictions": REPOSITORY_ROOT / "eval/prediction-bundle.schema.json",
    "receipts": REPOSITORY_ROOT / "eval/custodian-receipt.schema.json",
}

SHA256_PATTERN = "^sha256:[0-9a-f]{64}$"

CANDIDATE_FREEZE_FIELDS = {
    "schema_version",
    "experiment_id",
    "task_id",
    "hypothesis",
    "upstream_commit",
    "code_commit",
    "models",
    "config_sha256",
    "data_sha256",
    "eval_data_version",
    "normalizer_version",
    "hardware",
    "seed",
    "command",
}

PREDICTION_BUNDLE_FIELDS = {
    "schema_version",
    "kind",
    "state",
    "access_class",
    "dataset_id",
    "revision",
    "split",
    "input_projection_sha256",
    "candidate_lock_sha256",
    "input_export_receipt_sha256",
    "raw_predictions_sha256",
    "execution_envelope_sha256",
    "hypothesis_adapter_version",
    "item_count",
    "items_sha256",
    "items",
}

RECEIPT_BRANCHES = {
    "inputExportReceipt": "asr-evaluation-input-export-receipt",
    "predictionFreezeReceipt": "asr-evaluation-prediction-freeze-receipt",
    "custodianScoreReceipt": "asr-evaluation-custodian-score-receipt",
}

SCORE_RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "state",
    "access_class",
    "experiment_id",
    "dataset_id",
    "revision",
    "evaluation_scope",
    "data_sha256",
    "input_projection_sha256",
    "record_identity_version",
    "hypothesis_adapter_version",
    "record_input_sha256",
    "prediction_input_sha256",
    "candidate_lock_sha256",
    "candidate_freeze_sha256",
    "candidate_registration_commit",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "prediction_artifact_sha256",
    "prediction_items_sha256",
    "input_export_receipt_sha256",
    "prediction_freeze_receipt_sha256",
    "execution_envelope_sha256",
    "runner_code_commit",
    "runner_source_sha256",
    "scorer_code_commit",
    "scorer_source_sha256",
    "scorer_runtime",
    "core_schema_version",
    "core_sha256",
    "public_release",
}


def _object_schemas(value: Any):
    if isinstance(value, dict):
        if value.get("type") == "object":
            yield value
        for child in value.values():
            yield from _object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from _object_schemas(child)


class CustodianArtifactSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in SCHEMA_PATHS.items()
        }

    def assert_exact_closed_object(self, schema, expected_fields):
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), expected_fields)
        self.assertEqual(set(schema["properties"]), expected_fields)

    def test_schemas_parse_as_draft_2020_12_and_close_every_object(self):
        self.assertEqual(set(self.schemas), set(SCHEMA_PATHS))
        for name, schema in self.schemas.items():
            with self.subTest(schema=name):
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertTrue(schema["$id"].endswith(SCHEMA_PATHS[name].name))
                object_schemas = list(_object_schemas(schema))
                self.assertTrue(object_schemas)
                for object_schema in object_schemas:
                    self.assertIs(object_schema.get("additionalProperties"), False)

    def test_candidate_lock_matches_the_strict_candidate_freeze_shape(self):
        schema = self.schemas["candidate"]
        expected_lock_fields = {
            "schema_version",
            "kind",
            "state",
            "access_class",
            "dataset_id",
            "revision",
            "split",
            "data_sha256",
            "input_projection_sha256",
            "hypothesis_adapter_version",
            "record_identity_version",
            "record_input_sha256",
            "decode_item_count",
            "decode_item_ids_sha256",
            "source_manifest_decision",
            "candidate_registration_commit",
            "candidate_manifest_path",
            "candidate_manifest_sha256",
            "candidate",
            "candidate_freeze_sha256",
        }
        self.assert_exact_closed_object(schema, expected_lock_fields)
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)
        self.assertEqual(
            schema["properties"]["kind"]["const"],
            "asr-evaluation-candidate-lock",
        )
        self.assertEqual(schema["properties"]["state"]["const"], "frozen")
        self.assertEqual(
            schema["properties"]["access_class"]["const"], "restricted"
        )
        self.assertEqual(schema["properties"]["split"]["const"], "sealed-blind")
        self.assertEqual(
            schema["properties"]["record_identity_version"]["const"],
            "eval-core-record-input-v1",
        )

        candidate = schema["$defs"]["candidateFreeze"]
        self.assert_exact_closed_object(candidate, CANDIDATE_FREEZE_FIELDS)
        self.assertNotIn("metrics", candidate["properties"])
        self.assertNotIn("artifacts", candidate["properties"])
        self.assertNotIn("decision", candidate["properties"])
        self.assertEqual(candidate["properties"]["models"]["maxItems"], 32)
        self.assertEqual(
            schema["$defs"]["command"]["properties"]["argv"]["maxItems"],
            1024,
        )
        self.assertEqual(
            schema["properties"]["decode_item_count"]["maximum"], 1000000
        )
        self.assertEqual(schema["properties"]["decode_item_count"]["minimum"], 1)
        self.assertEqual(
            candidate["properties"]["task_id"]["const"], "EVAL-01"
        )
        self.assertEqual(schema["$defs"]["sha256"]["pattern"], SHA256_PATTERN)

    def test_prediction_bundle_allows_a_bounded_ordered_subsequence_shape(self):
        schema = self.schemas["predictions"]
        self.assert_exact_closed_object(schema, PREDICTION_BUNDLE_FIELDS)
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)
        self.assertEqual(
            schema["properties"]["kind"]["const"],
            "asr-evaluation-predictions",
        )
        self.assertEqual(schema["properties"]["split"]["const"], "sealed-blind")

        item_count = schema["properties"]["item_count"]
        items = schema["properties"]["items"]
        self.assertEqual(item_count["minimum"], 0)
        self.assertEqual(item_count["maximum"], 1000000)
        self.assertEqual(items.get("minItems", 0), 0)
        self.assertEqual(items["maxItems"], 1000000)
        prediction_item = schema["$defs"]["predictionItem"]
        self.assert_exact_closed_object(
            prediction_item,
            {"id", "raw_text", "status", "reason_code"},
        )
        self.assertEqual(
            prediction_item["properties"]["raw_text"]["maxLength"], 16384
        )
        self.assertEqual(
            prediction_item["properties"]["id"]["maxLength"], 512
        )
        self.assertEqual(schema["$defs"]["sha256"]["pattern"], SHA256_PATTERN)

    def test_receipt_schema_has_three_disjoint_complete_restricted_branches(self):
        schema = self.schemas["receipts"]
        self.assertEqual(len(schema["oneOf"]), 3)
        self.assertEqual(
            {entry["$ref"] for entry in schema["oneOf"]},
            {f"#/$defs/{name}" for name in RECEIPT_BRANCHES},
        )
        for branch_name, kind in RECEIPT_BRANCHES.items():
            with self.subTest(branch=branch_name):
                branch = schema["$defs"][branch_name]
                self.assert_exact_closed_object(
                    branch,
                    set(branch["properties"]),
                )
                self.assertEqual(branch["properties"]["schema_version"]["const"], 2)
                self.assertEqual(branch["properties"]["kind"]["const"], kind)
                self.assertEqual(branch["properties"]["state"]["const"], "complete")
                self.assertEqual(
                    branch["properties"]["access_class"]["const"],
                    "restricted",
                )
        self.assertEqual(schema["$defs"]["boundedCount"]["maximum"], 1000000)
        self.assertEqual(
            schema["$defs"]["positiveBoundedCount"]["minimum"], 1
        )
        self.assertEqual(schema["$defs"]["sha256"]["pattern"], SHA256_PATTERN)

    def test_score_receipt_freezes_scope_lineage_and_public_withholding(self):
        schema = self.schemas["receipts"]
        score = schema["$defs"]["custodianScoreReceipt"]
        self.assert_exact_closed_object(score, SCORE_RECEIPT_FIELDS)
        self.assertEqual(score["properties"]["core_schema_version"]["const"], 2)
        self.assertEqual(
            score["properties"]["scorer_code_commit"]["$ref"],
            "#/$defs/gitCommit",
        )
        self.assertEqual(
            schema["$defs"]["gitCommit"]["pattern"],
            "^[0-9a-f]{40}$",
        )
        self.assertEqual(
            score["properties"]["record_identity_version"]["const"],
            "eval-core-record-input-v1",
        )

        scope = schema["$defs"]["sealedEvaluationScope"]
        self.assert_exact_closed_object(scope, {"kind", "split"})
        self.assertEqual(scope["properties"]["kind"]["const"], "split")
        self.assertEqual(scope["properties"]["split"]["const"], "sealed-blind")

        release = schema["$defs"]["withheldPublicRelease"]
        self.assert_exact_closed_object(
            release,
            {"state", "summary_sha256", "reason_code"},
        )
        self.assertEqual(release["properties"]["state"]["const"], "withheld")
        self.assertIsNone(release["properties"]["summary_sha256"]["const"])
        self.assertEqual(
            release["properties"]["reason_code"]["const"],
            "release_policy_not_implemented",
        )


if __name__ == "__main__":
    unittest.main()
