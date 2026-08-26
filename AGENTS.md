# AGENTS.md

This file is the source of truth for coding agents working in this downstream
FunASR fork. `CLAUDE.md` imports it so project rules stay in one place.

## Project Goal

Build a reproducible Chinese ASR research and product lab on top of FunASR.
The first baseline is FunASR `v1.4.3`; Paraformer is the offline algorithm
track, Paraformer-Streaming is the native streaming track, and SenseVoiceSmall
is the speed-oriented control.

The project follows this order:

```text
reproduce upstream -> freeze evaluation -> close tiny-training loop
-> run one-variable experiments -> promote only verified improvements
```

## Start Here

Before changing code:

1. Read [`.notes/asr/progress.md`](.notes/asr/progress.md) for the sole current
   task and next gate.
2. Find that task in
   [`.notes/asr/delivery-roadmap.md`](.notes/asr/delivery-roadmap.md).
3. Read [`.notes/asr/benchmark-protocol.md`](.notes/asr/benchmark-protocol.md)
   before changing inference, training, data, normalization, or metrics.
4. Read the nearest module `AGENTS.md` if one is added later.

Project direction belongs in `.notes/asr/solution.md`; execution status does
not. Do not create a parallel roadmap or treat chat history as project state.

## Fork And Branch Boundaries

- `origin` is `yydspanda/funasr-lab`.
- `upstream` is `modelscope/FunASR` and is fetch-only.
- `main` mirrors the upstream default branch and is not the downstream work
  branch.
- `vendor/funasr-v1.4.3` is the immutable starting snapshot.
- `develop` is the downstream integration branch.
- Use `exp/EXP-<date>-<slug>` for controlled experiments and
  `sync/upstream-<version>` for upstream integrations.

Keep downstream changes isolated. Prefer new registered model components,
evaluation modules, and small generic extension points over broad edits to
`AutoModel` or existing Paraformer internals. Every unavoidable upstream-core
edit must state why an extension is insufficient and include focused tests.

## Repository Map

```text
.agents/skills/              Project-specific agent workflows
.notes/asr/                  Authoritative solution, roadmap, progress, data,
                             benchmark, and error-taxonomy documents
.notes/reference-index/      Adopt/reject notes for external projects
designdocs/agents/           Static development-session process
eval/                        Versioned evaluation code and public manifests;
                             generated reports and private audio stay untracked
experiments/manifests/       One reproducibility manifest per experiment
funasr/                      Upstream toolkit and model implementation
scripts/                     Bootstrap, doctor, and governance checks
```

## Development Contract

- Exactly one roadmap task may be `In Progress`.
- Each code slice carries a roadmap task ID such as `BASE-01` or `EXP-01`.
- Each experiment starts with one primary hypothesis and one declared baseline.
- Record code/model/data/config revisions before looking at the result.
- Do not report only the best seed or silently change normalization, VAD,
  decoding, chunk, or hardware settings.
- Code, focused tests, experiment manifest, and authoritative documentation
  change in the same delivery slice.
- Large models, datasets, audio, generated transcripts, and reports do not
  enter Git. Commit only small redistributable smoke fixtures when intentional.

## Evaluation Contract

The primary accuracy metrics are content CER and, for Chinese-English mixed
speech, MER. Keep punctuation, inverse text normalization, entity recall, and
raw/display accuracy separate. Always report substitutions, deletions, and
insertions rather than only a total score.

Performance reports must distinguish single-stream latency from batch
throughput and include the relevant subset of:

- RTF P50/P95;
- first partial and first stable-token latency;
- finalization latency and partial churn;
- cold start, peak RSS, device, precision, batch, beam, chunk, look-back, VAD;
- command, software revision, model revision, and hardware identity.

Blind-test results decide promotion. Dev-set gains or normalization-only gains
do not count as algorithm improvements.

## Commands

Run from the repository root:

```bash
python3 scripts/asr_lab_doctor.py
python3 scripts/check_asr_progress.py
python3 scripts/check_experiment_manifests.py
python3 scripts/run_baseline_smoke.py --track paraformer \
  --audio eval/private/smoke.wav --dry-run
python3 -m compileall funasr examples tests eval scripts
```

Model tests under `tests_models/` may download checkpoints and are integration
tests. Do not run them implicitly as part of a lightweight validation command.
When a task needs model downloads, state that before running it and record the
exact model revision/cache in the experiment manifest.

`requirements/lab-cpu.lock` is the BOOT/BASE environment lock for Linux x86_64
and Python 3.11. Regenerate it from `requirements/lab-cpu.in` with the command
recorded in its generated header; do not hand-edit resolved versions.

## Documentation Ownership

- `solution.md`: product and architecture decisions.
- `delivery-roadmap.md`: stage/task registry, order, and exit gates.
- `progress.md`: one current pointer and recent verified completions.
- `benchmark-protocol.md`: frozen metric and comparison rules.
- `dataset-register.md`: dataset identity, split, lineage, and hashes.
- `error-taxonomy.md`: stable failure categories used to select experiments.
- `experiments/manifests/*.json`: reproducible facts for an individual run.

Update the owning document instead of copying its content elsewhere. Keep
`progress.md` short; archive older completion records by month.
