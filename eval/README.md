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

Every executed evaluation also needs a validated provenance record under
`experiments/manifests/`. That record binds the full upstream and downstream
commits, every loaded model component and content hash, the effective config
and ordered data-manifest hashes, structured hardware, the unabridged command,
metrics, and report hashes. See `experiments/manifests/README.md` for the format.

## BASE-01 Offline Runner

`scripts/run_offline_baseline.py` runs one pinned, ASR-only baseline track. It
uses explicit ModelScope identifiers instead of FunASR aliases, keeps VAD,
punctuation, and ITN outside the primary timing path, and emits a canonical JSON
report plus exact data, effective-config, and report SHA-256 values.
The resolved `AutoModel.model_path` is also content-hashed with the experiment
manifest contract: files sorted by POSIX relative path and inventory lines
encoded as `<file-sha256><two spaces><relative-path>\n`. The resulting bundle
hash is stored beside the model identifier and immutable revision. BASE-01 fixes
the inference seed to `0` and explicitly reports zero excluded and retried items.
It is CPU-only: accepting a GPU device here could let upstream silently fall
back to CPU while leaving incorrect device provenance. Primary `rtf_p50` and
`rtf_p95` cover every attempted warm item, including failed decodes; separate
`successful_rtf_p50` and `successful_rtf_p95` fields are diagnostic only.

The ordered JSONL input is frozen before inference. Every record requires:

- `id`, `speaker_id`, `session_id`, `split`, and one shared `data_version`;
- repository-relative `audio`, its `audio_sha256`, duration, 16 kHz sample rate,
  and mono channel count;
- verbatim `raw_text`, its `reference_sha256`, and the exact
  `normalizer_version`.

Both hashes use the `sha256:<64 lowercase hex characters>` form. Audio must be
an uncompressed PCM WAV for this first baseline slice. The runner validates the
complete corpus before importing FunASR, so an identity mismatch cannot trigger
a model download or produce a partial report.

`manifests/dataset-manifest.example.jsonl` is schema-only: its audio is not
tracked and its placeholder audio hash is not executable evidence. Replace the
path and both hashes when freezing a real corpus.

Validate a frozen manifest without importing FunASR or downloading a model:

```bash
.venv/bin/python scripts/validate_offline_baseline_dataset.py \
  --dataset-manifest eval/manifests/lab-base-smoke-001-v0.1.jsonl
```

After choosing and recording immutable model revisions, run each track
separately on the same manifest. Run from the repository root and fix both the
ModelScope cache and Python hash seed before the process starts:

```bash
export MODELSCOPE_CACHE="$PWD/.cache/modelscope"
export PYTHONHASHSEED=0

.venv/bin/python scripts/run_offline_baseline.py \
  --track paraformer \
  --dataset-manifest eval/manifests/lab-base-smoke-001-v0.1.jsonl \
  --model-revision <immutable-model-tag-or-commit> \
  --output-report eval/reports/paraformer-smoke-v0.1.json \
  --device cpu \
  --ncpu 6 \
  --warmup-runs 1 \
  --seed 0

.venv/bin/python scripts/run_offline_baseline.py \
  --track sensevoice \
  --dataset-manifest eval/manifests/lab-base-smoke-001-v0.1.jsonl \
  --model-revision <immutable-model-tag-or-commit> \
  --output-report eval/reports/sensevoice-smoke-v0.1.json \
  --device cpu \
  --ncpu 6 \
  --warmup-runs 1 \
  --seed 0
```

A real invocation may download the explicitly pinned model if it is absent from
the local cache. Do not execute it until the revision and planned experiment
manifest have been reviewed. A provisioning run that downloads model bytes is
not valid cold-start evidence; rerun from the verified local cache to a new
report path. Generated reports are never overwritten.
