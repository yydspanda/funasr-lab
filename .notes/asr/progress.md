# ASR Lab Execution Progress

> This file contains only the sole live execution pointer, current constraints,
> and recent terminal records. Stage definitions and task gates belong to
> `delivery-roadmap.md`.

## Current Pointer

- **Current Stage:** `BASE`
- **In Progress Task:** `BASE-01`
- **Current Objective:** Install the locked CPU environment, freeze a small speaker/session-aware smoke corpus, pin every model revision, and produce comparable Paraformer and SenseVoiceSmall baseline reports.
- **Next Gate:** Run both offline tracks on the same frozen audio and normalizer, recording CER components, RTF P50/P95, cold/warm time, peak RSS, failures, exact commands, and artifact hashes in the first experiment manifest.
- **Roadmap:** [`delivery-roadmap.md`](delivery-roadmap.md)
- **Upstream Repository:** `modelscope/FunASR`
- **Baseline Tag:** `v1.4.3`
- **Baseline Commit:** `eedd4e22d10dc2e81d9c2bb321edb3750253964b`
- **Baseline Date:** `2026-08-21`
- **Last Updated:** `2026-08-26`

## Current Constraints

| Boundary | Current fact |
|---|---|
| Scope | Chinese microphone/meeting speech; offline quality first, native streaming second |
| Algorithm tracks | Paraformer offline, Paraformer-Streaming live, SenseVoiceSmall speed control |
| Fork boundary | Prefer registered downstream additions and narrow hooks over broad upstream-core edits |
| Environment | Bootstrap and CI are CPU-first; model downloads and training runs are explicit, separately recorded actions |
| Evidence | No quality or speed claim is promotable before the evaluator, data hashes, model/config revisions, command, and hardware are recorded |

## Recent Completion Records

### 2026-08-26 — Downstream fork and reproducible bootstrap

- **Task:** `BOOT-01`
- **Status:** `Done`
- **Outcome:** Forked `modelscope/FunASR` to `yydspanda/funasr-lab`, fixed the downstream baseline at `v1.4.3`, created the Python 3.11 CPU lock and `.venv`, added the explicit no-download smoke entry, authoritative project governance, CI, and two validated ASR project Skills.
- **Verification:** `make PYTHON=.venv/bin/python check` passed doctor, Roadmap/Progress, empty-manifest, static compile, four lab tests, and eleven governance tests; both Skills passed `quick_validate.py`; no ASR model was downloaded.

## Update Contract

1. Keep exactly one `Current Stage` and one `In Progress Task`, both present in
   the Roadmap and matching its current statuses.
2. A completion record uses a Roadmap task ID, terminal status, concrete
   outcome, and verification evidence; do not copy architecture or long logs.
3. Keep at most ten recent records. Move older records to the
   [monthly archive](../archive/asr/progress/README.md).
4. Update the baseline fields in both active documents when an upstream sync is
   accepted; a moving branch name is not a baseline.
5. After editing either active document, run:

   ```bash
   python3 scripts/check_asr_progress.py
   ```
