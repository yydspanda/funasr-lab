# Evaluation Workspace

This directory owns versioned, reviewable evaluation logic. Private audio and
generated reports remain local; only protocols, schemas, small intentional
fixtures, and code belong in Git.

## Layout

- `manifests/`: dataset metadata and split declarations.
- `normalizers/`: versioned text normalization code.
- `reports/`: generated reports; ignored except for this placeholder.
- `smoke/audio/`: optional tiny redistributable fixtures; ignored by default.

Dataset manifests must identify `speaker_id` and `session_id` so development
and blind-test splits cannot accidentally share the same speaker or recording
session. The authoritative metric rules live in
`.notes/asr/benchmark-protocol.md`.
