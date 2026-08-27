# Evaluation Workspace

This directory owns versioned, reviewable evaluation logic. Private audio and
generated reports remain local; only protocols, schemas, small intentional
fixtures, and code belong in Git.

## Layout

- `manifests/`: dataset metadata and split declarations.
- `normalizers/`: versioned text normalization code.
- `reports/`: generated reports; ignored except for this placeholder.
- `smoke/audio/`: optional tiny redistributable fixtures; ignored by default.

Strict EVAL manifests identify speaker, recording session, source recording,
derived lineage, deduplication cluster, and exact audio identity so no such key
can cross development and blind-test splits. The authoritative metric rules
live in `.notes/asr/benchmark-protocol.md`.

In strict EVAL collection records, the compatibility field
`exclusion_reason` is a stable lowercase `snake_case` reason code, not free-form
prose. Human review notes belong outside the frozen record identity.

Every executed evaluation also needs a validated provenance record under
`experiments/manifests/`. That record binds the full upstream and downstream
commits, every loaded model component and content hash, the effective config
and ordered data-manifest hashes, structured hardware, the unabridged command,
metrics, and report hashes. See `experiments/manifests/README.md` for the format.

## EVAL-01 Scoring And Collection Contract

`scoring.py` freezes two independent content metrics without importing FunASR:

- CER uses `zh-content-v0.1` and scores the resulting Unicode code points;
- MER uses `zh-en-mixed-v0.1` directly on raw text, preserving Chinese
  characters and English word boundaries.

Both metrics share `levenshtein-diagonal-deletion-insertion-v1`, whose
equal-cost alignment order fixes the S/D/I breakdown. Zero-reference items have
a null item rate, while their insertions still enter the corpus numerator.

`core_report.py` produces the versioned, restricted `asr-evaluation-core`
document validated by `core-report.schema.json` and additional semantic checks.
Its canonical bytes bind the collection's ordered record-input hash, exact raw
predictions, a report-level `identity-v1` or `sensevoice-control-tags-v1`
adapter, frozen scoring versions, stable status codes, integer components,
exact rational rates, ordered items, and deterministic slices. Callers cannot
inject cleaned display text or separately pair records with an unrelated data
hash: the public builder accepts one `ValidatedCollection` atomically. Core and
schema always report both `scenario_tags` and `split` slices. Timing, RSS,
absolute paths, argv, timestamps, and raw exception text belong to a separate
execution envelope and cannot change the core hash. Missing predictions are
scored as empty failures; duplicate or extra prediction IDs fail validation.
Only a data exclusion declared before decoding is omitted from the denominator,
and it remains explicitly counted.

`core-summary.schema.json` defines the blind-safe public projection. It binds
the restricted core hash and keeps only scoring/config identity plus aggregate
and slice counts/metrics; it contains no item IDs, references, hypotheses, or
record/prediction projection hashes.

`collection.py` validates the EVAL collection as one unit rather than accepting
individual split files in isolation. The canonical descriptor binds all split
manifest hashes/counts, scoring and scenario versions, rights/provenance groups,
the reviewed dedup report, and the blind policy. The dedup report is itself
bound to this dataset/revision, ordered manifests, configuration, threshold,
record count, and ID-to-cluster inventory. Strict records bind audio and
reference hashes plus speaker, session, source, lineage, dedup cluster, and
scenario identity. Any isolation key crossing a split is rejected.

The files
`manifests/collection-descriptor.schema-only.example.json` and
`manifests/collection-record.schema-only.example.jsonl` document the exact
shape only. Their `schema-only` state and nonexistent private artifacts make
them deliberately invalid as frozen evidence. A custodian validates a real,
untracked collection with:

```bash
.venv/bin/python scripts/validate_asr_collection.py \
  --descriptor eval/private/LAB-SEED-001-v0.1.collection.json \
  --collection-root . \
  --audio-root .
```

The validator reads every full split manifest, including sealed references, so
that command belongs in the restricted custodian workflow. It rebuilds and
checks descriptor-bound audio-only and reference projection hashes. Iterative
model runs receive only `build_sealed_input_projection()` bytes; developers do
not receive the separately held reference projection.
The custodian uses `load_validated_collection` and exports only the canonical,
reference-free bytes returned by `build_sealed_input_projection` to decode
workers.

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
