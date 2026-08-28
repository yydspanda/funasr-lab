# ASR Benchmark Protocol

> Status: **Draft to freeze under `EVAL-01`**
> Updated: `2026-08-28`

This protocol prevents data, normalization, decoding, and hardware changes from
being mistaken for model improvements. Until `EVAL-01` passes, results are
diagnostic and must not be used as promotion evidence.

## Evaluation Units

Each utterance record must bind:

- stable utterance, speaker, session, and split IDs;
- audio relative path, SHA-256, duration, sample rate, and channel count;
- verbatim reference plus the normalizer version;
- versioned scenario tags known before decoding.

The collection descriptor binds the ordered manifest hash and record count for
every split plus its provenance, rights, deduplication, and blind-access policy.
Splits are disjoint by speaker, recording session, source recording, derived
lineage, deduplication cluster, and exact audio identity. Missing, failed, and
empty predictions remain in the primary denominator with a stable reason code;
a missing or failed prediction is scored as empty content. A data-quality
exclusion must be frozen in the collection before decoding, stays visible in
counts and slices, and is the only case omitted from the scoring denominator;
predictions cannot dynamically mark an item excluded.
Observed error-taxonomy annotations are post-decode diagnostics and cannot
change the frozen input slices or primary denominator.

## Text Views

Keep three outputs instead of overwriting one:

1. `raw`: exact reference or decoder text;
2. `content`: frozen normalization and unitized scoring views;
3. `display`: punctuation, capitalization, ITN, and presentation processing.

Reference scoring always derives from the frozen raw reference. Hypothesis
display/scoring text is derived only by a report-level, versioned decoder
adapter; a prediction cannot supply its own cleaned display text.

The content normalizer and each scoring unitizer version and test every
operation. `zh-content-v0.1` applies NFKC, lowercases, and removes Unicode
whitespace and punctuation; its resulting Unicode code points are CER units.
It remains unchanged for BASE compatibility.

`zh-en-mixed-v0.1` operates on the raw text because English boundaries cannot
be recovered after the Chinese content normalizer. It applies NFKC and Unicode
case folding, scores each Han ideograph separately, groups each contiguous run
of Latin letters and digits as one unit, treats whitespace and Unicode
punctuation as boundaries, and preserves every other code point as a singleton
unit. Number expansion, English spelling expansion, apostrophe/hyphen joining,
and ITN are not implicit scoring transformations.

## Accuracy Metrics

For Chinese content, report character error rate:

```text
CER = (substitutions + deletions + insertions) / reference characters
```

Always report the numerator components, denominator, utterance count, failed
count, and frozen scenario/language slices in addition to aggregate CER. The
primary corpus rate is micro-aggregated from integer components; a slice or
per-utterance average must not replace it. For Chinese-English mixed speech,
report MER using `zh-en-mixed-v0.1` units. Also report domain-term/entity recall
when `EXP-01` studies contextual bias.

An item with zero reference units has a null item rate rather than infinity or
an invented zero. Its insertions still enter the corpus numerator. Equal-cost
Levenshtein alignments prefer diagonal, then deletion, then insertion so the
substitution/deletion/insertion breakdown is deterministic.

Punctuation F1, ITN accuracy, and display-text accuracy remain secondary and
must never replace content CER/MER.

## Performance Metrics

For each report, bind CPU/device identity, thread count, precision, batch, beam,
chunk, look-back, VAD/endpointing, software commit, model revision, command,
warm-up policy, and audio duration distribution.

Offline single-stream reports include:

- RTF P50/P95 and total wall/audio time;
- cold start and warm start separately;
- peak RSS;
- failed and retried inputs.

Streaming reports additionally include:

- first-partial and first-stable-token latency;
- endpoint-to-finalization latency;
- partial churn/revision rate;
- final CER/MER and boundary/reset failures.

Batch throughput is reported as throughput, never labeled as real-time
single-user latency.

## Controlled Comparison

Before decoding, freeze:

- baseline code/model/config/data/normalizer revisions;
- primary hypothesis and metric;
- regression metrics and allowed budgets;
- seeds and aggregation method;
- environment and exact command.

One comparison changes one declared variable. Report every planned seed and
failed run. Development data selects candidates; a sealed blind set decides
promotion. A normalization-only gain is reported as an evaluation change, not
an algorithm gain.

## Initial Baseline Matrix

| Track | Model | Required report |
|---|---|---|
| Offline algorithm | Paraformer | CER/MER, component counts, RTF P50/P95, cold/warm, RSS |
| Speed control | SenseVoiceSmall | Same frozen content metrics and performance fields |
| Native streaming | Paraformer-Streaming | Partial/final metrics, latency, churn, reset and long-stream stability |

External Qwen3-ASR, FireRedASR2, and Whisper comparisons must use the same audio,
normalizer, failure accounting, and available decoding controls; otherwise they
are labeled non-equivalent references.

## Required Artifacts

A promotable run produces an experiment manifest and a generated report outside
Git. Only the compact manifest, schema, evaluator, fixtures, and a human-readable
decision summary are versioned. The manifest must bind report hashes so the
summary cannot silently point at a different run.

The EVAL core report contains only deterministic data/prediction identities,
versioned scoring contracts, stable status/reason codes, integer components,
exact rational denominators, items, and frozen slices. Generated time,
performance measurements, absolute paths, argv, and raw exception text belong
to a separate execution envelope. They cannot perturb the core bytes or hash.
The full core is restricted because it contains references. Its text-free core
summary binds the core hash while removing item IDs, raw/reference/hypothesis
text, and record/prediction projection hashes, but remains restricted. It is
not a public artifact or authorization to publish sealed metrics.

The sealed-blind descriptor exposes only logical identities, hashes, counts,
aggregate coverage, and the seal policy. It binds independent hashes of an
audio/input projection and a restricted reference projection. Iterative runners
receive only the former. An isolated scoring process joins the frozen
hypothesis with the latter only after the candidate model, configuration,
command, and hashes are frozen.

The sealed audio projection uses schema version 2, omits every record whose
data exclusion was frozen before decoding, and distinguishes the full sealed
manifest count from the decode-eligible item count. The corresponding reference
projection retains all sealed records so exclusions remain auditable. A blind
core report uses the collection-owned `sealed-blind` scoring scope: its data
hash still binds the full descriptor, while its ordered record-input hash binds
only the sealed scoring records. Smoke or development items cannot be counted
as missing blind predictions.

For the initial export, the custodian validates planned-candidate metadata
before it opens the full collection. It then opens sealed references only
inside the restricted workflow to validate collection and scoring identity,
and creates the canonical candidate lock plus reference-free audio projection.
The export must receive a full candidate-registration commit. It authenticates
the exact tracked `experiments/manifests/<experiment_id>.json` blob, requires
`code_commit -> registration -> HEAD` ancestry and registration reachability
from fetched `origin/develop`, and repeats the registration commit, path, and
blob SHA-256 in candidate-lock schema v2 and all receipts. This does not attest
CI success; a passing registration CI run is an additional procedural gate.
The decoder receives neither references nor authority to edit that lock. The
committed, reference-free `run_sealed_asr_candidate.py` runner supports only the
pinned Paraformer and SenseVoice CPU profiles in v1. It rechecks every audio
hash and WAV identity, the actual model bundle, effective config, argv,
allowlisted environment, hardware, and runner-source inventory before it
publishes raw predictions followed by a canonical restricted execution
envelope. The planned argv uses the exact canonical option order and rejects
empty, flag-shaped, C0-control, or DEL option values. Its manifest/argv model
revision is the same full 40-character lowercase snapshot commit; tags and
aliases are invalid. Before opening the collection, export also reconciles the
planned handoff paths, audio root, and future runner outputs with this private
run directory. `perf_counter_ns` owns integer model-load, cold, warmup, and measured
attempt timing. Each attempt's timer starts only after its mandatory
O_NOFOLLOW audio read, SHA-256 check, and WAV validation; the model receives
those same verified in-memory bytes. Fresh-process Linux `RUSAGE_SELF` owns
peak RSS. Model output cannot supply either measurement. A later GPU or
hostile-candidate contract requires device synchronization and an external
process/cgroup supervisor; it cannot reuse the CPU v1 claim.

The CPU profile permits `ncpu` 1..4096, warmups 0..100, and seed `0`. It forces
FP32, batch size one, exact track-specific model/tokenizer/frontend and internal
registry components, snapshot-local resources, and no VAD, punctuation,
speaker, LM, remote code, or secondary output directory. Audio export and every
decode attempt traverse the root, parents, and leaf through no-follow directory
descriptors; non-canonical relative paths, symlinks, concurrent replacement,
hash drift, or WAV-identity drift fail closed.

Every sealed entrypoint uses direct Linux CPython argv with `-P -S`, effective
`PYTHONHASHSEED=0`, and no other non-empty `PYTHON*` startup variable.
Repository module/package/symlink shadows and unchecked checkout bytecode are
rejected. Source identity uses the fixed system Git executable with external
configuration and replacement objects disabled, and rejects replacement refs
or grafts.

CPU evidence runs require the exact pinned model bundle to be present and
verified in the planned cache before the fresh runner process starts. A cache
miss or any model download is provisioning, not an evidence run; discard its
outputs and restart from a fresh process after verification succeeds. Reported
`cold_start` is the timed model-load plus cold-inference windows. Cold inference
excludes that attempt's preceding audio read/hash/WAV validation. The metric
also excludes pre-measurement and post-measurement model-integrity
inventory/hash work, which remains mandatory lineage validation but is not
latency. Peak RSS is the fresh runner process's Linux `RUSAGE_SELF` high-water
mark sampled immediately after the measured pass. It covers all in-process
work from process start through validation, model load, cold/warmup, and
measured decode. It excludes post-measurement model verification, child
processes, and later artifact publication, and therefore is not a process-tree
or serving-capacity claim.

`freeze-predictions` verifies the raw bytes and complete execution envelope
against the sealed input and candidate lock, then creates prediction bundle
schema v2 and receipt schema v2. The envelope contains no hypothesis text,
reference, dedicated per-item audio-path or filename field, CER/MER, or raw
exception. It retains every decode ID, duration, stable status/reason, and integer wall time; counts, measured
wall/audio totals, RTF P50/P95, and byte-to-MiB RSS are rebuilt rather than
accepted as self-reported aggregates. The prediction bundle binds the exact raw
bytes, envelope, audio projection, ordered decode IDs, adapter, statuses,
reason codes, and prediction-item hash. Decoder failures are explicit empty
predictions; omitted decode IDs remain auditable as `missing_prediction`
failures. Extra, duplicate, or reordered IDs are invalid.

Each artifact transition writes all canonical restricted outputs into one
private directory with mode `0700`; every output file has mode `0600`, and
stdout is not evidence. In each custodian receipt-bearing transition, earlier
artifacts are directory-synced before the receipt is published last. The runner
publishes raw prediction JSONL first and its execution envelope last; the later
prediction-freeze receipt authenticates both. Before the score transition
reopens sealed references, it verifies the input-export receipt, input, lock,
prediction bundle, execution envelope, and prediction-freeze receipt as one chain. The score
receipt binds that freeze receipt and envelope,
the planned candidate freeze, candidate lock, exact prediction artifact/items,
sealed input and scoped record identity, derived scoring input, core hash,
committed runner/scorer revisions, and both exact source-inventory hashes. A
sealed score refuses scorer source bytes that differ from the candidate's
frozen code commit, independently recomputes the runner inventory from that
commit, and binds its exact CPython/Unicode, CPU-lock, and installed
distribution inventory. The runner, freeze, scorer, and terminal validator all
use that same candidate commit; a later manifest-only commit is permitted only
when the authenticated source inventories are byte-identical. A core without
its matching receipt is an incomplete replay. Before a result is
accepted, rejected, or marked for investigation, the terminal verifier must
preserve the same candidate facts, bind input/lock/prediction/envelope/core and
receipts, match core CER/MER components and counts, and match envelope RTF
P50/P95, all-attempt/retry counts, cold/warm timing, audio duration, and peak
RSS. Performance is inconclusive without this complete envelope; stdout or
manifest-only numbers are never evidence.

The offline custodian scorer emits only a restricted core report. A text-free
aggregate projection is not automatically safe to publish: exact metrics over
small cells or repeated candidate queries can act as a blind-reference oracle.
Public release therefore requires a separately frozen one-candidate
authorization and minimum-cell policy; until that contract is implemented, the
core summary remains inside the restricted custodian workflow.

Each ordinary executed experiment manifest must pass
`scripts/check_experiment_manifests.py` and bind:

- full 40-character upstream and downstream Git commits;
- every loaded model component's role, identifier, immutable revision, and
  content hash;
- SHA-256 hashes of the exact effective config and frozen collection descriptor
  (or legacy BASE ordered manifest);
- concrete OS, CPU, memory, device, and stable non-secret host identity;
- the complete argument vector and every non-secret environment variable that
  affects results;
- finite measured metrics, including CER numerator components and denominator,
  utterance/failure counts, RTF P50/P95, and peak RSS; once executed, at least
  one content-hashed generated report.

Pre-register the same manifest with `decision: planned`, `metrics: null`, and
no artifacts; every identity, hash, hardware, and command field is already
concrete. For an ordinary experiment, execution replaces the null with measured
metrics and adds hashed reports, so reviewers never need to accept fabricated
zero results. A sealed replay is the explicit exception while its score receipt
has `public_release.state: withheld`: its tracked manifest remains unchanged at
`planned`, while a restricted private terminal copy carries the terminal
decision, measured metrics, and artifact hashes. That copy must pass the sealed
terminal chain validator against the input-export, prediction-freeze, and score
receipts, restricted core, and execution envelope; it is neither committed nor
released until a separate release policy authorizes it.

Branch names, floating model revisions (`main`, `master`, `latest`, or `HEAD`),
abbreviated commits, placeholder values, and zero/repeated/empty digests are not
reproducible identities and fail governance checks. The canonical field formats
and copyable starting point live in `experiments/manifest.schema.json` and
`experiments/manifest.template.json`.
