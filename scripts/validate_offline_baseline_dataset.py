#!/usr/bin/env python3
"""Validate and identify a BASE-01 dataset without importing FunASR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.offline_baseline import BaselineError
from eval.offline_baseline import load_frozen_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a frozen BASE-01 offline dataset manifest."
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset = load_frozen_dataset(args.dataset_manifest, REPOSITORY_ROOT)
    except BaselineError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "kind": "offline-baseline-dataset",
                "manifest": str(args.dataset_manifest),
                "data_version": dataset.data_version,
                "data_sha256": dataset.manifest_sha256,
                "utterance_count": len(dataset.items),
                "audio_seconds": sum(
                    item.duration_seconds for item in dataset.items
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
