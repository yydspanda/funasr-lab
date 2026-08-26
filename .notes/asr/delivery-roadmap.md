# ASR Lab Delivery Roadmap

> Status: **Authoritative execution order**
> This document owns stage order, task IDs, and exit gates. The sole live
> execution pointer is mirrored and cross-checked in `progress.md`.

## Control Record

- **Current Stage:** `BASE`
- **Upstream Repository:** `modelscope/FunASR`
- **Baseline Ref:** `v1.4.3`
- **Baseline Commit:** `eedd4e22d10dc2e81d9c2bb321edb3750253964b`
- **Baseline Date:** `2026-08-21`
- **Last Updated:** `2026-08-26`

## Stage Order

| Stage | Status | Outcome | Exit gate |
|---|---|---|---|
| `BOOT` | Done | Reproducible fork, CPU-first environment, governance, and smoke command | Repository doctor and all lightweight governance checks pass from a clean environment |
| `BASE` | **Current** | Frozen upstream offline and speed-control results | Paraformer and SenseVoice produce versioned accuracy/performance reports on the same smoke set |
| `EVAL` | Pending | Evaluation protocol and seed corpus frozen | Normalization, CER/MER accounting, manifests, splits, and hashes pass independent replay |
| `TRAIN` | Pending | Minimal training loop proven | A tiny subset intentionally overfits; save, reload, and inference reproduce the expected result |
| `EXP` | Pending | First attributable quality experiment | One hypothesis beats the frozen baseline and survives blind verification without a declared regression |
| `STREAM` | Pending | Native-streaming baseline and regression gate | Partial/final accuracy, latency, churn, and state reset are reproducible on long streams |
| `SERVE` | Pending | Versioned ASR backend service for external Apps and clients | Pinned offline and streaming service profiles pass API, isolation, reliability, observability, and load gates |
| `MAINT` | Recurring | Downstream remains integrable with upstream | Drift is measured and each sync or compatibility decision has evidence |

Execution order:

```text
BOOT-01 -> BASE-01 -> EVAL-01 -> TRAIN-01 -> EXP-01 -> STREAM-01 -> SERVE-01
                         ^                                             |
                         +------------ UP-SYNC recurring ---------------+
```

## Task Registry

| ID | Status | Deliverable | Acceptance |
|---|---|---|---|
| `BOOT-01` | Done | Fork topology, pinned Python environment, doctor/smoke entry point, agent workflows, governance documents and CI | `origin`/fetch-only `upstream` are correct; baseline resolves to the recorded commit; doctor, governance, manifest, unit, and static checks pass without downloading a model |
| `BASE-01` | **In Progress** | Reproducible Paraformer offline and SenseVoiceSmall control runs | Same frozen audio/normalizer; exact model revisions and commands recorded; CER components, RTF P50/P95, peak RSS, cold/warm timing reported |
| `EVAL-01` | Pending | Frozen evaluator, seed manifest, normalizers, report schema, and dataset register | Hand-calculated fixtures match CER/MER; split and audio/text hashes are stable; repeated evaluation produces byte-stable core metrics |
| `TRAIN-01` | Pending | CPU-feasible tiny-training and checkpoint round trip | Forward/backward/update work; loss descends on a tiny subset; saved checkpoint reloads; resulting inference and config lineage are verified |
| `EXP-01` | Pending | First one-variable contextual-bias experiment | Hypothesis and regression budget declared before results; all seeds reported; dev gain is confirmed on the blind set and error taxonomy explains the change |
| `STREAM-01` | Pending | Paraformer-Streaming baseline, stream simulator, and long-stream tests | Chunk/look-back/VAD settings fixed; first partial, stable token, finalization, churn, RTF, reset, and boundary accuracy reported |
| `SERVE-01` | Pending | Downstream-qualified ASR service contract and reproducible deployment profile | External clients use only the versioned API; model/runtime/image revisions are pinned; offline and streaming integration, session isolation, bounded resources, health, recovery, and target-load behavior pass |
| `MAINT-01` | Done | Bounded progress history, strict experiment provenance, upstream drift monitoring, and source-isolation guard | Active progress is month/record/line bounded; all task references resolve; executed experiments bind reproducible hashes, hardware, command, metrics, and reports; scheduled CI fails on excessive drift or unregistered upstream-core changes |
| `UP-SYNC` | Scheduled | Recurring upstream drift measurement and controlled integration | Record full SHAs and ahead/behind for mirror main, active develop, and accepted baseline; use `sync/upstream-*`; resolve conflicts without erasing downstream evidence or broadening core patches |

## Gate Details

### BOOT Gate

- A clean checkout identifies the expected Python and system dependencies.
- A smoke inference entry point is explicit even when model files are absent.
- Governance and manifest validation use only the Python standard library.
- CI performs no model, dataset, or checkpoint download.

### BASE Gate

- Audio identity, model revision, inference configuration, command, environment,
  device, precision, and batch are captured before comparing results.
- Throughput and single-stream latency are labeled separately.
- Failed and excluded utterances are counted, not silently dropped.

### EVAL Gate

- Content CER is primary for Chinese; MER is added for mixed Chinese-English.
- Raw, content-normalized, and display-text results remain distinct.
- The blind set is sealed and unavailable for iterative tuning.

### TRAIN Gate

- A tiny deterministic test proves the complete optimizer/checkpoint loop.
- Resume behavior, seed handling, and config/data lineage are tested.
- A large training run cannot substitute for this diagnostic gate.

### EXP Gate

- Exactly one primary hypothesis and frozen baseline are declared.
- All planned seeds and regressions are reported, not only the best run.
- Blind-set evidence, not a normalization-only or dev-only gain, decides
  promotion.

### STREAM Gate

- Partial and final hypotheses are both evaluated.
- Chunking, look-back, caches, VAD, endpointing, and reset semantics are fixed.
- Long streams include silence, boundary speech, noise, and reconnect cases.

### SERVE Gate

- Offline and native-streaming endpoints publish versioned request, result,
  partial/final, and error contracts; an external-client integration test uses
  only those public contracts.
- Source, model, runtime, image, and effective configuration identities are
  pinned; health and readiness expose enough revision data to diagnose a run.
- Concurrent sessions isolate state; finalization, cancellation, disconnect,
  and reconnect reset caches correctly; queues, buffers, and request sizes are
  bounded and overload fails explicitly.
- End-to-end latency, throughput, saturation, failures, and recovery are
  measured on the target deployment without relabeling model-only benchmarks
  as service performance.
- TLS, authentication, rate and payload limits, and audit/retention policy are
  enforced at the gateway boundary before an App or other external client can
  reach the ASR service.

### MAINT Gate

- Active progress contains only its current-month record window; monthly
  archives preserve older verified history and cannot contain live pointers.
- Weekly CI fetches trusted upstream and records full SHAs plus ahead/behind for
  mirror `main`, active `develop`, and the accepted baseline. None may be more
  than ten commits behind; mirror `main` must also be zero commits ahead.
- The accepted baseline remains in trusted upstream history. Any downstream
  addition or edit under upstream implementation surfaces has an exact patch
  ledger entry, registered task, reason, and focused tests.

## Parking Lot

These directions require explicit rescheduling and may not interrupt the
current task:

- large Speech-LLM fine-tuning or training from scratch;
- diarization, punctuation, and inverse-text-normalization optimization;
- App/client UI, mobile/embedded packaging, and platform-specific acceleration;
- two-pass streaming plus LLM final correction;
- replacement of the FunASR training/runtime foundation.

## Anti-Drift Rules

1. Exactly one stage is `Current` and exactly one task is `In Progress`.
2. Every implementation slice carries a registered task ID.
3. `progress.md` points to the same stage/task and contains only the current
   Asia/Shanghai calendar month, newest first, with at most eight terminal records;
   older records live in matching monthly archives.
4. New ideas enter the Parking Lot or explicitly replace the current pointer;
   chat history is not project state.
5. Code, focused tests, experiment manifest, and owning documentation move in
   the same delivery slice.
6. Every executed experiment binds full upstream/downstream commits, every
   loaded model revision/hash, config/data hashes, structured hardware, the
   complete command, finite metrics, and hashed reports.
7. Fork `main` stays zero commits ahead; mirror `main`, active `develop`, and
   the accepted baseline stay no more than ten behind upstream. Upstream
   implementation changes require an exact checked ledger entry.
8. Roadmap/progress, manifest, archive, and fork-boundary edits must pass their
   standard-library governance scripts and CI.
9. Keep this active Roadmap within 240 lines and `progress.md` within 120 lines.
