# ASR Lab Execution Progress

> This file contains only the sole live execution pointer, current constraints,
> and recent terminal records. Stage definitions and task gates belong to
> `delivery-roadmap.md`.

## Current Pointer

- **Current Stage:** `EVAL`
- **In Progress Task:** `EVAL-01`
- **Current Objective:** Replace the EVAL schema-only examples with the real microphone/meeting `LAB-SEED-001` descriptor and speaker/session/source/lineage-disjoint smoke, development, and sealed-blind manifests, then run the frozen evaluator through the custodian replay.
- **Next Gate:** Review consent/access evidence and the dedup report; bind real audio/reference and ordered split hashes; seal blind references; pre-register one committed CPU runner command; pass cross-split isolation plus byte-identical collection/core replay with a receipt-bound execution envelope without changing the pinned BASE models.
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
| EVAL-01 reality | The controlled CPU replay now derives a private terminal manifest from the complete nine-file vault and revalidates every file/path/hash, but real `LAB-SEED-001` consent/access evidence, dedup review, audio, references, descriptor, split manifests, and candidate are absent; the collection-preparation evidence index/finalizer is still pending, and no sealed CER/MER/RTF/RSS, GPU-capacity, service-capacity, or product-readiness result exists |
| Sync exclusions | The accepted snapshot reports package version 1.4.14 but is 23 commits after tag `v1.4.14`; cached Paraformer timestamp output was single/batch stable but exposed a `-90 ms` boundary and the BPE path remains unqualified; MOSS/Nano accelerators and upstream Nano realtime/service demos are not substitutes for the planned content, native-streaming, or `SERVE-01` tracks |
| Remote enforcement | Both governance workflows run on push and weekly schedule; `develop` has no branch protection/ruleset, so direct writers can still bypass failed checks |

## Recent Completion Records

### 2026-09-07 — FunASR 1.4.14 post-tag controlled upstream integration

- **Task:** `UP-SYNC`
- **Status:** `Done`
- **Outcome:** Fast-forwarded mirror `main` to trusted upstream `fe719c007f49bb285859a037d2b2ef73b0f96fde`; merged frozen upstream `2d2d385545698a1c216cd421666695cefbc56c32` into the downstream overlay without rebase or squash via `61871ea693a22d38aceb30b152600ead5a5858a3`; accepted that full SHA as the immutable FunASR 1.4.14 post-tag baseline; regenerated the CPU lock for upstream's `numpy<2` requirement; preserved prior EVAL head `f15191f458fdd982e6d900a35cfa52c2895452d5`; and retained zero downstream toolkit/runtime core patches.
- **Verification:** Candidate/develop `3eacfdfab7d4d5f18e5e4bb07972c405518ecde3` passed local doctor, lock, governance, manifest, compile, 174 lab/EVAL, 65 governance, 234 affected CPU-contract, 67 release/runtime, 18 MOSS mock, 190 documentation, 160 product-site, and one isolated N8N test; two 174-page builds were identical across 236 files; both content-hashed cached model replays remained 2/2 with content CER 0 and S/D/I 0; candidate runs `34081397900`/`34081397862` and integrated runs `34087695867`/`34087695894` passed. Paraformer timestamp single/batch output matched but exposed a `-90 ms` boundary, so timestamp/BPE, GPU, MOSS/Nano, streaming, and service readiness remain explicitly unqualified.

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
