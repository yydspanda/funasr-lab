# ASR Lab Execution Progress

> This file contains only the sole live execution pointer, current constraints,
> and recent terminal records. Stage definitions and task gates belong to
> `delivery-roadmap.md`.

## Current Pointer

- **Current Stage:** `EVAL`
- **In Progress Task:** `UP-SYNC`
- **Current Objective:** Qualify trusted upstream `main` at `2d2d385545698a1c216cd421666695cefbc56c32` as an immutable FunASR 1.4.14 post-tag sync baseline while preserving every downstream EVAL and prior sync commit without rebase or squash.
- **Next Gate:** Regenerate and sync the CPU lock for upstream's `numpy<2` constraint, pass environment, manifest, source-isolation, affected-upstream, cached-model compatibility, and EVAL regression gates on `sync/upstream-v1.4.14`, then update mirror `main` and integrate the qualified candidate before restoring `EVAL-01`.
- **Roadmap:** [`delivery-roadmap.md`](delivery-roadmap.md)
- **Upstream Repository:** `modelscope/FunASR`
- **Baseline Ref:** `2d2d385545698a1c216cd421666695cefbc56c32`
- **Baseline Commit:** `2d2d385545698a1c216cd421666695cefbc56c32`
- **Baseline Date:** `2026-09-07`
- **Last Updated:** `2026-09-07`

## Current Constraints

| Boundary | Current fact |
|---|---|
| Scope | Chinese microphone/meeting speech; offline quality first, native streaming second |
| Algorithm tracks | Paraformer offline, Paraformer-Streaming live, SenseVoiceSmall speed control |
| Fork boundary | Upstream source is read-only by default; every unavoidable core path requires a checked ledger entry and focused tests |
| Environment | Bootstrap and CI are CPU-first; model downloads and training runs are explicit, separately recorded actions |
| Evidence | No quality or speed claim is promotable before full commits, every model/config/data hash, structured hardware, complete command, and report hash are recorded; sealed CPU RTF/RSS additionally requires the runner-owned execution envelope and complete receipt chain |
| EVAL-01 reality | The controlled CPU replay/evidence path is implemented, but real `LAB-SEED-001` consent/access evidence, dedup report, audio, references, descriptor, split manifests, and candidate are absent; no sealed CER/MER/RTF/RSS, GPU-capacity, service-capacity, or product-readiness result exists |
| Sync exclusions | The accepted snapshot reports package version 1.4.14 but is 23 commits after tag `v1.4.14`; Paraformer BPE timestamp output, MOSS/Nano accelerators, and upstream Nano realtime/service demos remain unqualified and are not substitutes for the planned Chinese CharTokenizer, native-streaming, or `SERVE-01` tracks |
| Remote enforcement | Both governance workflows run on push and weekly schedule; `develop` has no branch protection/ruleset, so direct writers can still bypass failed checks |

## Recent Completion Records

> No terminal completion records in the current month.

## Update Contract

1. Keep exactly one `Current Stage` and one `In Progress Task`, both present in
   the Roadmap and matching its current statuses.
2. A completion record uses a Roadmap task ID, terminal status, concrete
   outcome, and verification evidence; do not copy architecture or long logs.
3. Keep only records from the current Asia/Shanghai calendar month, newest first, and
   at most eight. Move prior-month and overflow records to the matching
   [`YYYY-MM.md` archive](../archive/asr/progress/README.md).
4. Update the baseline fields in both active documents when an upstream sync is
   accepted; a moving branch name is not a baseline.
5. Check archive readiness before editing, and apply a rollover when requested:

   ```bash
   python3 scripts/archive_asr_progress.py --check
   python3 scripts/archive_asr_progress.py --apply
   ```

6. After editing either active document, run:

   ```bash
   python3 scripts/check_asr_progress.py
   ```
