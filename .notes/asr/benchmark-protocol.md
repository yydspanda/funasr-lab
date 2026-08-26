# ASR Benchmark Protocol

> Status: **Draft to freeze under `EVAL-01`**
> Updated: `2026-08-26`

This protocol prevents data, normalization, decoding, and hardware changes from
being mistaken for model improvements. Until `EVAL-01` passes, results are
diagnostic and must not be used as promotion evidence.

## Evaluation Units

Each utterance record must bind:

- stable utterance, speaker, session, and split IDs;
- audio relative path, SHA-256, duration, sample rate, and channel count;
- verbatim reference plus the normalizer version;
- scenario and error-taxonomy tags known before decoding.

Splits are speaker- and session-disjoint. Missing, failed, empty, and excluded
items remain counted in the report with a reason.

## Text Views

Keep three outputs instead of overwriting one:

1. `raw`: exact reference or decoder text;
2. `content`: frozen normalization for CER/MER;
3. `display`: punctuation, capitalization, ITN, and presentation processing.

The content normalizer must version and test every operation. Initial candidate
rules are Unicode normalization, approved whitespace removal, case folding for
Latin tokens, and an explicit punctuation policy. Number/English expansion and
ITN are separate metrics until fixtures prove semantic equivalence.

## Accuracy Metrics

For Chinese content, report character error rate:

```text
CER = (substitutions + deletions + insertions) / reference characters
```

Always report the numerator components, denominator, utterance count, failed
count, and macro slices in addition to aggregate CER. For Chinese-English mixed
speech, report MER using Chinese characters and English words as scoring units.
Also report domain-term/entity recall when `EXP-01` studies contextual bias.

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

Each executed experiment manifest must pass
`scripts/check_experiment_manifests.py` and bind:

- full 40-character upstream and downstream Git commits;
- every loaded model component's role, identifier, immutable revision, and
  content hash;
- SHA-256 hashes of the exact effective config and frozen ordered data manifest;
- concrete OS, CPU, memory, device, and stable non-secret host identity;
- the complete argument vector and every non-secret environment variable that
  affects results;
- finite measured metrics, including CER numerator components and denominator,
  utterance/failure counts, RTF P50/P95, and peak RSS; once executed, at least
  one content-hashed generated report.

Pre-register the same manifest with `decision: planned`, `metrics: null`, and
no artifacts; every identity, hash, hardware, and command field is already
concrete. Execution replaces the null with measured metrics and adds hashed
reports, so reviewers never need to accept fabricated zero results.

Branch names, floating model revisions (`main`, `master`, `latest`, or `HEAD`),
abbreviated commits, placeholder values, and zero/repeated/empty digests are not
reproducible identities and fail governance checks. The canonical field formats
and copyable starting point live in `experiments/manifest.schema.json` and
`experiments/manifest.template.json`.
