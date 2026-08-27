#!/usr/bin/env python3
"""Export or score one sealed EVAL-01 custodian replay.

The command is deliberately split into a reference-free decoder handoff,
prediction freezing, and restricted scoring. It never exposes sealed
references or metrics on stdout, and it never imports or starts a FunASR model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.collection import CollectionValidationError
from eval.collection import build_sealed_input_projection
from eval.collection import load_collection_descriptor
from eval.collection import load_validated_collection
from eval.collection import sha256_bytes
from eval.core_report import CORE_SCHEMA_VERSION
from eval.core_report import CoreReportValidationError
from eval.core_report import IDENTITY_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION
from eval.core_report import build_split_core_report
from eval.core_report import canonical_core_bytes
from eval.custodian_replay import CUSTODIAN_SCORE_RECEIPT_KIND
from eval.custodian_replay import INPUT_EXPORT_RECEIPT_KIND
from eval.custodian_replay import PREDICTION_FREEZE_RECEIPT_KIND
from eval.custodian_replay import RECEIPT_SCHEMA_VERSION
from eval.custodian_replay import SEALED_SPLIT
from eval.custodian_replay import CustodianReplayError
from eval.custodian_replay import LoadedArtifact
from eval.custodian_replay import build_candidate_lock
from eval.custodian_replay import build_prediction_bundle
from eval.custodian_replay import canonical_candidate_lock_bytes
from eval.custodian_replay import canonical_custodian_receipt_bytes
from eval.custodian_replay import canonical_prediction_bundle_bytes
from eval.custodian_replay import load_candidate_lock
from eval.custodian_replay import load_planned_candidate_manifest
from eval.custodian_replay import load_prediction_bundle
from eval.custodian_replay import load_prediction_items_jsonl
from eval.custodian_replay import load_sealed_input_projection
from eval.custodian_replay import parse_sealed_input_projection
from eval.custodian_replay import preflight_replay_artifacts
from eval.custodian_replay import scorer_code_identity
from eval.custodian_replay import validate_candidate_request
from eval.custodian_replay import validate_decode_handoff
from eval.custodian_replay import validate_output_paths
from eval.custodian_replay import validate_prediction_handoff
from eval.custodian_replay import validate_replay_collection
from eval.custodian_replay import write_atomic_outputs


def _collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Root for descriptor manifest and dedup-report paths.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Root for record audio paths; defaults to --collection-root.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline, sealed-reference ASR custodian workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-input",
        help="Bind a planned candidate and export reference-free decode input.",
    )
    _collection_arguments(export_parser)
    export_parser.add_argument("--candidate-manifest", type=Path, required=True)
    export_parser.add_argument(
        "--hypothesis-adapter-version",
        choices=(
            IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
            SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
        ),
        required=True,
    )
    export_parser.add_argument("--output-input", type=Path, required=True)
    export_parser.add_argument("--output-candidate-lock", type=Path, required=True)
    export_parser.add_argument("--output-receipt", type=Path, required=True)

    freeze_parser = subparsers.add_parser(
        "freeze-predictions",
        help="Freeze reference-free decoder JSONL into a canonical bundle.",
    )
    freeze_parser.add_argument("--input-projection", type=Path, required=True)
    freeze_parser.add_argument("--candidate-lock", type=Path, required=True)
    freeze_parser.add_argument("--raw-predictions", type=Path, required=True)
    freeze_parser.add_argument(
        "--hypothesis-adapter-version",
        choices=(
            IDENTITY_HYPOTHESIS_ADAPTER_VERSION,
            SENSEVOICE_HYPOTHESIS_ADAPTER_VERSION,
        ),
        required=True,
    )
    freeze_parser.add_argument("--output-predictions", type=Path, required=True)
    freeze_parser.add_argument("--output-receipt", type=Path, required=True)

    score_parser = subparsers.add_parser(
        "score",
        help="Score one frozen sealed prediction bundle inside custodian scope.",
    )
    _collection_arguments(score_parser)
    score_parser.add_argument("--input-projection", type=Path, required=True)
    score_parser.add_argument("--candidate-lock", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--output-core", type=Path, required=True)
    score_parser.add_argument("--output-receipt", type=Path, required=True)
    return parser


def _export_input(args: argparse.Namespace) -> dict[str, object]:
    # This preflight sees only public/restricted metadata.  It must pass before
    # load_validated_collection opens any sealed reference manifest.
    validate_output_paths(
        [args.output_input, args.output_candidate_lock, args.output_receipt]
    )
    descriptor = load_collection_descriptor(args.descriptor)
    candidate_manifest = load_planned_candidate_manifest(args.candidate_manifest)
    if candidate_manifest["data_sha256"] != descriptor.raw_sha256:
        raise CustodianReplayError(
            "candidate data_sha256 does not match collection descriptor"
        )
    if candidate_manifest["normalizer_version"] != descriptor.normalizer_version:
        raise CustodianReplayError(
            "candidate normalizer_version does not match collection descriptor"
        )
    validate_candidate_request(
        descriptor,
        candidate_manifest,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )

    collection = load_validated_collection(
        args.descriptor,
        args.collection_root,
        args.audio_root,
    )
    sealed_input = parse_sealed_input_projection(
        build_sealed_input_projection(collection)
    )
    candidate_lock = build_candidate_lock(
        descriptor,
        collection,
        sealed_input,
        candidate_manifest,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )
    candidate_lock_payload = canonical_candidate_lock_bytes(candidate_lock)
    candidate_lock_sha256 = sha256_bytes(candidate_lock_payload)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": INPUT_EXPORT_RECEIPT_KIND,
        "state": "complete",
        "access_class": "restricted",
        "experiment_id": candidate_lock["candidate"]["experiment_id"],
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "split": SEALED_SPLIT,
        "decode_item_count": candidate_lock["decode_item_count"],
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock_sha256,
        "candidate_freeze_sha256": candidate_lock["candidate_freeze_sha256"],
    }
    receipt_payload = canonical_custodian_receipt_bytes(receipt)
    write_atomic_outputs(
        [
            (args.output_input, sealed_input.payload),
            (args.output_candidate_lock, candidate_lock_payload),
            (args.output_receipt, receipt_payload),
        ]
    )
    return receipt


def _freeze_predictions(args: argparse.Namespace) -> dict[str, object]:
    validate_output_paths([args.output_predictions, args.output_receipt])
    sealed_input = load_sealed_input_projection(args.input_projection)
    candidate_lock = load_candidate_lock(args.candidate_lock)
    validate_decode_handoff(sealed_input, candidate_lock)
    if args.hypothesis_adapter_version != candidate_lock.document[
        "hypothesis_adapter_version"
    ]:
        raise CustodianReplayError(
            "requested hypothesis adapter does not match candidate lock"
        )
    prediction_items = load_prediction_items_jsonl(args.raw_predictions)
    bundle = build_prediction_bundle(
        sealed_input,
        candidate_lock.sha256,
        prediction_items,
        hypothesis_adapter_version=args.hypothesis_adapter_version,
    )
    bundle_payload = canonical_prediction_bundle_bytes(bundle)
    loaded_bundle = LoadedArtifact(
        document=bundle,
        payload=bundle_payload,
        sha256=sha256_bytes(bundle_payload),
    )
    validate_prediction_handoff(sealed_input, candidate_lock, loaded_bundle)
    expected_count = candidate_lock.document["decode_item_count"]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": PREDICTION_FREEZE_RECEIPT_KIND,
        "state": "complete",
        "access_class": "restricted",
        "experiment_id": candidate_lock.document["candidate"]["experiment_id"],
        "dataset_id": sealed_input.document["dataset_id"],
        "revision": sealed_input.document["revision"],
        "split": SEALED_SPLIT,
        "expected_decode_item_count": expected_count,
        "prediction_item_count": bundle["item_count"],
        "missing_prediction_count": expected_count - bundle["item_count"],
        "input_projection_sha256": sealed_input.sha256,
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": candidate_lock.document[
            "candidate_freeze_sha256"
        ],
        "hypothesis_adapter_version": bundle["hypothesis_adapter_version"],
        "prediction_artifact_sha256": loaded_bundle.sha256,
        "prediction_items_sha256": bundle["items_sha256"],
    }
    receipt_payload = canonical_custodian_receipt_bytes(receipt)
    write_atomic_outputs(
        [
            (args.output_predictions, bundle_payload),
            (args.output_receipt, receipt_payload),
        ]
    )
    return receipt


def _score(args: argparse.Namespace) -> dict[str, object]:
    # No sealed reference is opened until all externally supplied artifacts
    # agree with the descriptor and one frozen candidate.
    validate_output_paths([args.output_core, args.output_receipt])
    descriptor = load_collection_descriptor(args.descriptor)
    sealed_input = load_sealed_input_projection(args.input_projection)
    candidate_lock = load_candidate_lock(args.candidate_lock)
    predictions = load_prediction_bundle(args.predictions)
    preflight_replay_artifacts(
        descriptor,
        sealed_input,
        candidate_lock,
        predictions,
    )
    scorer_code_commit, scorer_source_sha256 = scorer_code_identity()

    collection = load_validated_collection(
        args.descriptor,
        args.collection_root,
        args.audio_root,
    )
    validate_replay_collection(collection, sealed_input, candidate_lock)
    report = build_split_core_report(
        collection,
        predictions.document["items"],
        split=SEALED_SPLIT,
        hypothesis_adapter_version=predictions.document[
            "hypothesis_adapter_version"
        ],
    )
    if report["provenance"]["record_input_sha256"] != candidate_lock.document[
        "record_input_sha256"
    ]:
        raise CustodianReplayError(
            "core record identity does not match candidate lock"
        )
    core_payload = canonical_core_bytes(report)
    core_sha256 = sha256_bytes(core_payload)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": CUSTODIAN_SCORE_RECEIPT_KIND,
        "state": "complete",
        "access_class": "restricted",
        "experiment_id": candidate_lock.document["candidate"]["experiment_id"],
        "dataset_id": descriptor.dataset_id,
        "revision": descriptor.revision,
        "evaluation_scope": report["configuration"]["evaluation_scope"],
        "data_sha256": report["provenance"]["data_sha256"],
        "input_projection_sha256": sealed_input.sha256,
        "record_identity_version": report["provenance"][
            "record_identity_version"
        ],
        "record_input_sha256": report["provenance"]["record_input_sha256"],
        "hypothesis_adapter_version": report["scoring"][
            "hypothesis_adapter_version"
        ],
        "prediction_input_sha256": report["provenance"][
            "prediction_input_sha256"
        ],
        "candidate_lock_sha256": candidate_lock.sha256,
        "candidate_freeze_sha256": candidate_lock.document[
            "candidate_freeze_sha256"
        ],
        "prediction_artifact_sha256": predictions.sha256,
        "prediction_items_sha256": predictions.document["items_sha256"],
        "scorer_code_commit": scorer_code_commit,
        "scorer_source_sha256": scorer_source_sha256,
        "core_schema_version": CORE_SCHEMA_VERSION,
        "core_sha256": core_sha256,
        "public_release": {
            "state": "withheld",
            "summary_sha256": None,
            "reason_code": "release_policy_not_implemented",
        },
    }
    receipt_payload = canonical_custodian_receipt_bytes(receipt)
    write_atomic_outputs(
        [
            (args.output_core, core_payload),
            (args.output_receipt, receipt_payload),
        ]
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export-input":
            _export_input(args)
        elif args.command == "freeze-predictions":
            _freeze_predictions(args)
        else:
            _score(args)
    except (
        CollectionValidationError,
        CoreReportValidationError,
        CustodianReplayError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
