# FunASR Lab Solution

> Status: **Initial direction**
> Updated: `2026-08-26`

## Outcome

Build a reproducible Chinese ASR research and product lab that can improve
recognition quality without losing inference speed or native-streaming
behavior. The first product-shaped target is microphone and meeting speech:
Mandarin, accents, far field, noise, domain terms, and limited Chinese-English
code switching.

## Technical Decision

Fork `modelscope/FunASR` and hold the first downstream baseline at:

- tag: `v1.4.3`;
- commit: `eedd4e22d10dc2e81d9c2bb321edb3750253964b`;
- offline algorithm track: Paraformer;
- native-streaming track: Paraformer-Streaming;
- speed-oriented control: SenseVoiceSmall.

Qwen3-ASR, FireRedASR2, and Whisper are comparison baselines, not the first
implementation surface. WeNet remains an architecture reference if an
experiment requires a different native-streaming formulation.

## System Shape

```text
audio + immutable manifest
        |
        v
decode / resample / VAD boundary
        |
        +--> offline Paraformer --------+
        |                               |
        +--> streaming Paraformer ------+--> normalized evaluation report
        |                               |
        +--> SenseVoice speed control --+
                                        |
                            error taxonomy + experiment manifest
                                        |
                                 promotion decision
```

Model output and display post-processing remain separate. The evaluator owns
reference/hypothesis normalization and reports raw component counts so a text
normalization change cannot masquerade as an acoustic-model improvement.

## Research Loop

The project advances through one-variable comparisons:

```text
reproduce -> freeze evaluation -> tiny overfit -> controlled experiment
-> blind verification -> promote or reject -> streaming verification
```

Every promoted result must be reproducible from a versioned manifest that
binds source, upstream baseline, model, config, data hashes, command, seeds,
hardware, and metrics. Best-seed-only reporting is prohibited.

## Extension Boundary

The fork is a downstream overlay on a trusted upstream commit, not a rewrite.
Keep algorithm adapters and registrations in the downstream-only `asr_lab/`
namespace, and other lab-owned code in new evaluation, experiment, script,
test, agent, and note surfaces. Prefer new registered models or predictors next,
then narrow generic hooks. Modify existing `AutoModel`, Paraformer, runtime, or
training internals only when an extension point cannot express the experiment.

The machine-checked exception ledger at
`.notes/asr/upstream-core-patches.json` binds every such path to a Roadmap task,
the reason extension is insufficient, and focused tests. The accepted upstream
baseline must be its ancestor and must itself belong to trusted upstream
history. The current downstream diff contains no toolkit/runtime core patch.

Fork maintenance is explicit: `main` mirrors upstream, `develop` carries the
overlay, and weekly CI measures the mirror, active downstream branch, and
accepted baseline against upstream. A controlled `UP-SYNC` updates the mirror,
reconciles the overlay, and advances the accepted baseline together. CI fails
if `main` contains a fork-only commit or if `main`, `develop`, or the accepted
baseline becomes more than ten upstream commits behind.

## Initial Non-Goals

- training a large foundation ASR model from scratch;
- optimizing punctuation, diarization, VAD, decoding, and the acoustic model in
  one un-attributable experiment;
- treating batch throughput as single-stream latency;
- making promotion decisions from a development set alone;
- committing models, datasets, generated transcripts, or benchmark reports.
