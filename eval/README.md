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

`core-summary.schema.json` defines a restricted text-free projection. It binds
the restricted core hash and keeps only scoring/config identity plus aggregate
and slice counts/metrics; it contains no item IDs, references, hypotheses, or
record/prediction projection hashes. It remains `access_class: restricted` and
is not a release artifact. A future one-candidate and minimum-cell workflow must
produce a distinct public artifact and release receipt.

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

### Sealed custodian replay

Ordinary non-blind experiments use the manifest lifecycle described in
`experiments/manifests/README.md`. The workflow below is the EVAL-01 sealed-set
exception while public release is withheld.

`scripts/replay_asr_evaluation.py` implements four custodian commands
without importing FunASR or exposing a metric on stdout. Between export and
freeze, `scripts/run_sealed_asr_candidate.py` performs the only model-bearing,
reference-free transition:

1. `export-input` validates the whole planned-manifest directory, Git
   provenance, exact registered manifest blob, and candidate metadata before
   opening sealed references. The supplied registration commit must be a full
   commit containing exactly
   `experiments/manifests/<experiment_id>.json`; the candidate `code_commit`
   must be its ancestor, and that registration must be reachable from both the
   checked-out `HEAD` and fetched `origin/develop`. This proves durable Git
   reachability and exact bytes, not that CI passed; CI success remains a
   separate operator gate. The
   restricted custodian then validates the full collection and writes the
   reference-free sealed input, a custodian-owned candidate lock, and an export
   receipt. That receipt is required by the runner, freeze, and score
   transitions, so the input export remains part of every later evidence
   chain. The lock is therefore created inside the blind workflow, not supplied
   by the decoder.
2. The committed CPU runner reads only sealed input, candidate lock,
   input-export receipt, and audio. It rechecks WAV/audio, model bundle,
   effective config, actual argv,
   allowlisted environment, hardware, and its committed source inventory. One
   fresh process measures model-load, cold, warmup, every measured attempt, and
   Linux `RUSAGE_SELF` peak RSS. Per-attempt audio open/hash/WAV validation is
   mandatory but happens before that attempt's inference timer starts; the
   verified in-memory bytes are then passed to the model. It publishes each
   complete mode-`0600` file without overwrite: raw four-field prediction JSONL
   first and the hypothesis/reference-free `execution-envelope.schema.json`
   completion marker last. Successful real runs are silent. This runner
   transition has no separate receipt.
   `describe-runtime` prints diagnostic environment, exact dependency-runtime,
   and hardware facts for planning the manifest and later binding the execution
   envelope, without loading audio or downloading a model.
3. `freeze-predictions` verifies the input-export receipt, raw bytes, per-item
   timings/statuses, derived RTF/count/RSS facts, envelope, input, and lock
   before creating prediction bundle schema v2 and receipt schema v2. Missing
   decode IDs become `missing_prediction` failures; extra, duplicate, or
   out-of-order IDs are rejected. The receipt binds both prediction and
   execution artifacts and authenticates the preceding runner completion marker.
4. `score` validates the input-export receipt, input, lock, prediction bundle,
   execution envelope, and prediction-freeze receipt before opening sealed
   references. It writes the sealed-only core and authoritative score receipt,
   binding runner/scorer source identities and every preceding artifact.
   The custodian recomputes the runner inventory from the candidate commit
   rather than trusting the envelope's self-reported source hash.
   Runner, freeze, scorer, and terminal-validator source bytes must all match
   the candidate's frozen `code_commit`; a later manifest-only commit is usable
   only while those exact source inventories remain unchanged. The scorer also
   binds CPython, Unicode, the CPU lock, and installed-distribution inventory.
   Public release remains `withheld` until a separate authorization and
   minimum-cell policy exists.
5. `validate-terminal` silently validates the private terminal manifest against
   the input-export, prediction-freeze, and score receipts, restricted core,
   and execution envelope. While release is withheld, this result-bearing copy
   remains private and the tracked candidate manifest remains `planned`.

The candidate lock is custodian-owned; a decode worker needs the sealed input
and the expected lock identity, not authority to edit the lock. The planned
manifest's command must be the complete `run_sealed_asr_candidate.py run` argv
in the canonical option order, with exactly one value for every required option.
Option values are non-empty, cannot begin with `--`, and contain no C0 control
character or DEL. CPU v1 contains exactly one ASR model whose manifest and argv
revisions are identical.
The model revision is a full 40-character lowercase ModelScope snapshot commit;
tags and aliases are not accepted by this sealed CPU v1 contract. Runner v1 is
the already frozen CPU Paraformer/SenseVoice path only: it does not establish
GPU, subprocess-tree, batch-throughput, streaming, or service performance.
For both tracks the model is forced to CPU FP32 with batch size one, no VAD,
punctuation, speaker, LM, remote code, or secondary output directory. The
accepted registry component profile is exact: Paraformer uses
`CharTokenizer`, `WavFrontend`, `SpecAugLFR`, `SANMEncoder`,
`ParaformerSANMDecoder`, and `CifPredictorV2`; SenseVoiceSmall uses
`SentencepiecesTokenizer`, `WavFrontend`, `SpecAugLFR`, and
`SenseVoiceEncoderSmall`. Every configured resource path remains inside the
hashed snapshot. `ncpu` is 1..4096, warmup count is 0..100, and sealed seed is
exactly `0`.

For a real run, first commit the runner and scorer, pre-register the exact run
command in a later planned-manifest commit and let CI pass, then pre-create one
mode-`0700`, non-symlink output directory and use new paths in that same
directory for every transition. All outputs are canonical, mode `0600`, and
published without overwrite. Before references are opened, the planned
input/lock/input-receipt paths must equal this export's outputs, the planned
audio root must equal the actual lexical absolute audio root, and all five
handoff, raw, and envelope paths must be distinct new paths in that same
private directory. On a handled publication failure, rollback is attempted in
reverse publication
order, completion marker first; if the marker cannot be removed, its predecessor
artifacts are preserved so it never points at evidence that rollback deleted.
Any such residual set is indeterminate and must be quarantined rather than
reused. Each preceding artifact is directory-synced
before the receipt is published last in each custodian receipt-bearing
transition. The runner instead publishes raw JSONL first and its execution
envelope last; the later prediction-freeze receipt authenticates both. A core
without its matching receipt is incomplete evidence. Successful artifact
transitions and `validate-terminal` are silent; `describe-runtime` is the sole
diagnostic command that prints its pre-registration facts. Stdout is never an
authoritative receipt.

Before publishing the input/lock/receipt transition, export reopens every
decode-eligible WAV through the runner's same directory-descriptor policy.
The audio root, every parent component, and the leaf must be real directories
or a regular file as appropriate; aliases, `.`/`..`, symlinks, concurrent path
replacement, hash drift, and WAV-identity drift fail before any output is
published.

CPU evidence also requires the exact pinned model bundle to be fully present
and verified in `MODELSCOPE_CACHE` before the fresh runner starts. A cache miss
or download is a provisioning attempt: discard its outputs and rerun from a
fresh process after the cache is verified. `cold_start` covers only the timed
model-load plus cold-inference windows. The cold-inference window begins after
that attempt's audio read/hash/WAV validation; model-integrity inventory/hash
work before and after the timing windows is also excluded. Peak RSS is the
fresh runner process's Linux `RUSAGE_SELF` high-water mark sampled immediately
after the measured pass. It covers all in-process work from process start
through validation, model load, cold/warmup, and measured decode, but excludes
post-measurement model verification, child processes, and artifact publication;
it is not a serving-capacity claim.

The secured entrypoints require direct Linux CPython execution with `-P -S`,
an effective `PYTHONHASHSEED=0`, and no other non-empty `PYTHON*` startup
variable. They also reject every non-empty startup variable outside the fixed
custodian/runner allowlists. The clean `env -i` wrappers below make that startup
state explicit.
The planned manifest records the resulting Python argv beginning with
`.venv/bin/python -P -S`, plus the non-secret result-affecting environment; the
`env` utility itself is not present in `/proc/self/cmdline` after `exec`.

```bash
install -d -m 700 eval/private
install -d -m 700 eval/private/replay-001

CUSTODIAN_ENV=(/usr/bin/env -i PATH=/usr/bin:/bin HOME=/dev/null LANG=C \
  LC_ALL=C PYTHONHASHSEED=0)
RUNNER_ENV=("${CUSTODIAN_ENV[@]}" \
  MODELSCOPE_CACHE="$PWD/.cache/modelscope" \
  TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor" \
  CRC32C_SW_MODE=auto KMP_DUPLICATE_LIB_OK=True KMP_INIT_AT_FORK=FALSE \
  HYDRA_FULL_ERROR=1 \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
  NUMEXPR_NUM_THREADS=6)
MODEL_REVISION=replace-with-40-lowercase-hex-characters
CANDIDATE_REGISTRATION_COMMIT=replace-with-40-lowercase-git-commit

"${RUNNER_ENV[@]}" .venv/bin/python -P -S \
  scripts/run_sealed_asr_candidate.py describe-runtime \
  --device cpu --ncpu 6

"${CUSTODIAN_ENV[@]}" .venv/bin/python -P -S \
  scripts/replay_asr_evaluation.py export-input \
  --descriptor eval/private/LAB-SEED-001-v0.1.collection.json \
  --collection-root . --audio-root . \
  --candidate-manifest experiments/manifests/EXP-YYYYMMDD-NNN-candidate.json \
  --candidate-registration-commit "$CANDIDATE_REGISTRATION_COMMIT" \
  --hypothesis-adapter-version identity-v1 \
  --output-input eval/private/replay-001/sealed-input.json \
  --output-candidate-lock eval/private/replay-001/candidate-lock.json \
  --output-receipt eval/private/replay-001/export-receipt.json

"${RUNNER_ENV[@]}" .venv/bin/python -P -S \
  scripts/run_sealed_asr_candidate.py run \
  --input-projection eval/private/replay-001/sealed-input.json \
  --candidate-lock eval/private/replay-001/candidate-lock.json \
  --input-receipt eval/private/replay-001/export-receipt.json \
  --audio-root . --track paraformer \
  --model-revision "$MODEL_REVISION" \
  --device cpu --ncpu 6 --warmup-runs 1 --seed 0 \
  --hypothesis-adapter-version identity-v1 \
  --output-raw-predictions eval/private/replay-001/raw-predictions.jsonl \
  --output-execution-envelope eval/private/replay-001/execution-envelope.json

"${CUSTODIAN_ENV[@]}" .venv/bin/python -P -S \
  scripts/replay_asr_evaluation.py freeze-predictions \
  --input-projection eval/private/replay-001/sealed-input.json \
  --candidate-lock eval/private/replay-001/candidate-lock.json \
  --input-receipt eval/private/replay-001/export-receipt.json \
  --raw-predictions eval/private/replay-001/raw-predictions.jsonl \
  --execution-envelope eval/private/replay-001/execution-envelope.json \
  --hypothesis-adapter-version identity-v1 \
  --output-predictions eval/private/replay-001/predictions.json \
  --output-receipt eval/private/replay-001/prediction-receipt.json

"${CUSTODIAN_ENV[@]}" .venv/bin/python -P -S \
  scripts/replay_asr_evaluation.py score \
  --descriptor eval/private/LAB-SEED-001-v0.1.collection.json \
  --collection-root . --audio-root . \
  --input-projection eval/private/replay-001/sealed-input.json \
  --candidate-lock eval/private/replay-001/candidate-lock.json \
  --input-receipt eval/private/replay-001/export-receipt.json \
  --predictions eval/private/replay-001/predictions.json \
  --execution-envelope eval/private/replay-001/execution-envelope.json \
  --prediction-receipt eval/private/replay-001/prediction-receipt.json \
  --output-core eval/private/replay-001/core.json \
  --output-receipt eval/private/replay-001/score-receipt.json

"${CUSTODIAN_ENV[@]}" .venv/bin/python -P -S \
  scripts/replay_asr_evaluation.py validate-terminal \
  --input-receipt eval/private/replay-001/export-receipt.json \
  --prediction-receipt eval/private/replay-001/prediction-receipt.json \
  --score-receipt eval/private/replay-001/score-receipt.json \
  --core eval/private/replay-001/core.json \
  --execution-envelope eval/private/replay-001/execution-envelope.json \
  --terminal-manifest eval/private/replay-001/terminal-manifest.json
```

The terminal manifest in that last command is the mode-`0600`, result-bearing
private copy, not the tracked planned manifest. A successful validation emits
nothing and does not rewrite any input; the exit status is its only terminal
signal. Keep the manifest, core, envelope, and all three receipts together in
the restricted mode-`0700` directory, together with the sealed input,
candidate lock, raw predictions, and canonical prediction bundle. None of the
intermediate artifacts may be discarded merely because a later receipt exists.

The external contracts are reviewable in `candidate-lock.schema.json`,
`execution-envelope.schema.json`, `prediction-bundle.schema.json`, and
`custodian-receipt.schema.json`. JSON Schema fixes shape and resource bounds;
Python validation additionally checks hash parity, candidate/adapter identity,
ID order, time/count/percentile arithmetic, and cross-artifact ownership.
Candidate-lock schema v2 and all three receipt kinds repeat
`candidate_registration_commit`, `candidate_manifest_path`, and
`candidate_manifest_sha256`; every later transition re-proves those exact
registered bytes rather than trusting a copied hash.

## BASE-01 Offline Runner

`scripts/run_offline_baseline.py` runs one pinned, ASR-only baseline track. It
uses explicit ModelScope identifiers instead of FunASR aliases, keeps VAD,
punctuation, and ITN outside the primary timing path, and emits a canonical JSON
report plus exact data, effective-config, and report SHA-256 values.
The resolved `AutoModel.model_path` is also content-hashed with the experiment
manifest contract: files sorted by POSIX relative path and inventory lines
encoded as `<file-sha256><two spaces><relative-path>\n`. The resulting bundle
hash is stored beside the model identifier and immutable revision; relative
paths containing ASCII controls U+0000..U+001F or U+007F are rejected so the
line inventory is unambiguous. BASE-01 fixes the inference seed to `0` and
explicitly reports zero excluded and retried items.
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
