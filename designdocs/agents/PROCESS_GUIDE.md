# Development Session Process

This guide defines how a coding or research session moves one roadmap slice
from intent to verified evidence. It is deliberately static: execution state
lives only in [progress.md](../../.notes/asr/progress.md), and stage order lives
only in [delivery-roadmap.md](../../.notes/asr/delivery-roadmap.md).

Do not create `TODO.md`, `process.md`, or another project tracker. Temporary
scratch notes are not project state and must not become an acceptance source.

## 1. Bind The Session

Before editing code:

1. Read the sole `Current Stage`, `In Progress Task`, objective, and next gate
   in `progress.md`.
2. Find the task ID and exit gate in `delivery-roadmap.md`.
3. Read the owning protocol or register for the files being changed.
4. State one delivery slice small enough to verify in the current branch.
5. If the slice touches an upstream implementation surface, run the fork guard
   and either move the change into the downstream overlay or register the exact
   unavoidable path with its reason and focused tests.

If the requested work does not belong to the current task, either record it in
the Roadmap parking lot or explicitly replace the current pointer. Never run a
second hidden workstream.

## 2. Declare The Evidence

Before implementation, write down what would prove the slice complete:

- the exact command or test that exercises it;
- the expected artifact, metric, or observable behavior;
- the baseline commit, model/config/data revisions, and hardware when results
  are empirical;
- the failure condition that prevents promotion.

An ASR experiment also needs one primary hypothesis and one declared baseline.
Changing data, normalization, decoding, VAD, hardware, and model structure in
one comparison invalidates attribution unless the task explicitly studies the
combined system.

## 3. Implement A Narrow Slice

- Search for the existing extension point before modifying upstream core.
- Keep generic upstream behavior and downstream lab additions separable.
- Do not add or edit an upstream implementation path without updating
  `.notes/asr/upstream-core-patches.json`; new files inside core are patches too.
- Add focused tests with the code; do not defer verification to a later task.
- Keep generated audio, checkpoints, transcripts, reports, and caches out of
  Git.
- Record reproducible run facts in `experiments/manifests/`; prose summaries
  are not substitutes for manifests.

## 4. Verify In Layers

Run the smallest relevant check first, then the lightweight project gates:

```bash
python3 scripts/check_asr_progress.py
python3 scripts/archive_asr_progress.py --check
python3 scripts/check_experiment_manifests.py
python3 scripts/check_upstream_guard.py --no-fetch --run-ledger-tests
python3 -m unittest discover -s tests -p 'test_*governance.py' -v
python3 -m compileall -q scripts eval tests/test_asr_progress_governance.py
```

Model-download or full training tests are never implicit. When needed, announce
them first and record the exact model revision, cache, command, and hardware in
the experiment manifest.

## 5. Close Or Hand Off

A slice closes only when its Roadmap gate has evidence. In the same change:

1. mark the completed Roadmap task terminal;
2. move exactly one next task to `In Progress` and update the current stage if
   its gate has passed;
3. update the sole pointer and next gate in `progress.md`;
4. add one concise completion record with verification evidence;
5. run `scripts/archive_asr_progress.py --apply` to move prior-month records and
   records beyond the eight-record window into their monthly archives.

If the gate fails, keep the pointer where it is and record the concrete blocker
in the current objective or experiment artifact. A promising dev-set number,
an unverified claim, or a partially working demo is not completion.

## 6. Handoff Format

Every handoff should answer four questions:

1. What changed, under which roadmap task ID?
2. What exact evidence passed or failed?
3. What assumptions, generated artifacts, or local-only state remain?
4. What is the single next gate shown by `progress.md`?
