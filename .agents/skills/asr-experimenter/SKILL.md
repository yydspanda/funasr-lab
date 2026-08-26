---
name: asr-experimenter
description: Plan, run, and judge controlled ASR experiments in this FunASR fork. Use when changing a model, loss, data recipe, decoder, normalization, streaming parameter, or runtime setting and claiming an effect on accuracy, latency, memory, or throughput; also use when deciding whether a result should be accepted or rejected. Do not trigger for environment setup or unmeasured refactors.
---

# ASR Experimenter

Produce a reproducible accept/reject decision, not merely a favorable metric.
Keep each experiment tied to one roadmap task and one tracked manifest.

## Read Before Acting

1. Read root and nearest `AGENTS.md` files.
2. Read `.notes/asr/progress.md` and locate the task in
   `.notes/asr/delivery-roadmap.md`.
3. Read `.notes/asr/benchmark-protocol.md`; read
   `.notes/asr/dataset-register.md` and `.notes/asr/error-taxonomy.md` when data
   identity or error selection matters.
4. Read `experiments/manifests/README.md`, then inspect the baseline config,
   code, command, prior manifest, and relevant tests.

If an authoritative file needed to define the comparison is absent or unfinished,
surface that as a prerequisite. Do not invent a local substitute or call the result
promotion-ready.

## Freeze The Comparison

Before looking at treatment results, write an experiment manifest under
`experiments/manifests/` that fixes:

- one primary, falsifiable hypothesis and one declared baseline;
- the single intended treatment variable and all controlled settings;
- code, model, data/split, config, normalization, and dependency revisions;
- seed policy, command, device, precision, batch, beam, VAD, chunk, and look-back;
- primary metric, guardrails, acceptance threshold, rejection condition, and run count.

Use the versioned schema and template. Full 40-character upstream/downstream
commits are mandatory. List every loaded ASR, VAD, punctuation, LM, or ITN
component separately with an immutable revision and `sha256:` content hash;
also bind the effective config and ordered data-manifest hashes. Hardware is a
structured record, and the command is the complete argv plus every non-secret
result-affecting environment variable. An executed result must contain finite
metrics and at least one hashed report artifact. Branch names, abbreviated
commits, floating model revisions, placeholders, and synthetic hashes are not
acceptable provenance. The manifest task must already exist in the Roadmap.
Land the frozen code/config first, update from the target branch, and create the
manifest in a later commit: `code_commit` must resolve and already be an ancestor
of the manifest commit, while `upstream_commit` must belong to the accepted
upstream baseline history. Never bind provenance to a feature-branch commit that
squash or rebase merge may rewrite. The manifest checker also requires
`code_commit` to be reachable from the fetched durable target branch.
For the pre-run `planned` state, use `metrics: null` and no artifacts; all other
provenance must already validate. After execution, add measured metrics and at
least one hashed report before choosing a terminal decision.

Changing more than one causal variable creates separate experiments. Exploratory runs
may discover candidates, but label them exploratory and do not use them for promotion.

## Measure Comparable Outcomes

Follow the frozen benchmark protocol and report enough components to explain movement:

- content CER, including substitutions, deletions, insertions, reference units,
  utterance count, and failed count;
- MER for Chinese-English mixed speech;
- punctuation, inverse-text-normalization, entity, and display metrics separately;
- RTF P50/P95 for single-stream latency, without presenting batch throughput as latency;
- for streaming work, first partial, first stable token, finalization latency, and
  partial churn;
- cold start, peak RSS, hardware, software/model revisions, and the decoding/runtime
  settings that materially affect the comparison.

Use identical audio, segmentation, normalization, VAD, decoding, and measurement paths
unless one of them is the declared treatment. A score created only by normalization or
test-set leakage is not an algorithm gain.

## Run And Decide

1. Validate the path on the smallest relevant smoke slice.
2. Run the frozen baseline and treatment under the same harness.
3. Repeat according to the declared seed/run policy; never select only the best seed.
4. Diagnose regressions with stable error categories, not anecdotes.
5. Use dev results to select candidates and the blind test to decide promotion.
6. Record failures and negative results; do not silently alter the hypothesis or gate.
7. Run `python3 scripts/check_experiment_manifests.py` before presenting any
   result; an invalid manifest makes the result `Inconclusive`.

Return exactly one decision:

- `Accept`: the primary gate and every guardrail pass on the required promotion set;
- `Reject`: the primary gate fails or a guardrail regresses beyond its limit;
- `Inconclusive`: evidence is invalid, underpowered, or not comparable; name the next
  bounded run needed to decide.

Keep large audio, checkpoints, transcripts, and generated reports out of Git. Commit the
small reproducibility manifest and update the authoritative project documents only when
their owned facts change.

## Handoff

Lead with the decision, then provide the hypothesis, baseline/treatment revisions,
dataset split, metric table with deltas, guardrails, notable error movement, exact run
commands, artifact locations, and remaining external or blind-test gates.
