#!/usr/bin/env python3
"""Validate a frozen EVAL-01 collection without importing FunASR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.collection import CollectionValidationError
from eval.collection import canonical_json_bytes
from eval.collection import validate_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and identify one frozen ASR evaluation collection."
    )
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = validate_collection(
            args.descriptor,
            args.collection_root,
            args.audio_root,
        )
    except CollectionValidationError as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
