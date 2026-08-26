# ASR Lab Delivery Roadmap

> Status: **Authoritative execution order**
> This document owns stage order, task IDs, and exit gates. The sole live
> execution pointer is mirrored and cross-checked in `progress.md`.

## Control Record

- **Current Stage:** `BASE`
- **Upstream Repository:** `modelscope/FunASR`
- **Baseline Tag:** `v1.4.3`
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
| `MAINT` | Recurring | Downstream remains integrable with upstream | Drift is measured and each sync or compatibility decision has evidence |

Execution order:

```text
BOOT-01 -> BASE-01 -> EVAL-01 -> TRAIN-01 -> EXP-01 -> STREAM-01
                         ^                                |
                         +------- UP-SYNC recurring ------+
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
| `UP-SYNC` | Scheduled | Recurring upstream drift measurement and controlled integration | Record upstream SHA and ahead/behind counts; use `sync/upstream-*`; resolve conflicts without erasing downstream evidence or broadening core patches |

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

## Parking Lot

These directions require explicit rescheduling and may not interrupt the
current task:

- large Speech-LLM fine-tuning or training from scratch;
- diarization, punctuation, and inverse-text-normalization optimization;
- mobile/embedded packaging and platform-specific acceleration;
- two-pass streaming plus LLM final correction;
- replacement of the FunASR training/runtime foundation.

## Anti-Drift Rules

1. Exactly one stage is `Current` and exactly one task is `In Progress`.
2. Every implementation slice carries a registered task ID.
3. `progress.md` points to the same stage/task and contains at most ten recent
   terminal records.
4. New ideas enter the Parking Lot or explicitly replace the current pointer;
   chat history is not project state.
5. Code, focused tests, experiment manifest, and owning documentation move in
   the same delivery slice.
6. Model, data, config, normalization, decoding, VAD, hardware, and seed changes
   are declared before interpreting a comparison.
7. Roadmap and progress edits must pass `python3 scripts/check_asr_progress.py`.
8. Keep this active Roadmap within 240 lines and `progress.md` within 120 lines.
