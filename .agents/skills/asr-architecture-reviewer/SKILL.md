---
name: asr-architecture-reviewer
description: Review ASR architecture and implementation boundaries in this downstream FunASR fork. Use when a proposal or change affects upstream/downstream isolation, registries, model or trainer interfaces, training/inference parity, checkpoint/config compatibility, streaming cache or state, serving, or cross-module contracts. Do not trigger for isolated local fixes with no boundary impact or for judging experiment results.
---

# ASR Architecture Reviewer

Find the smallest durable design that can survive upstream synchronization and preserve
training, inference, and streaming contracts. Ground claims in repository evidence rather
than generic architecture preferences.

## Read Before Reviewing

1. Read root and nearest `AGENTS.md` files.
2. Read `.notes/asr/solution.md` for target direction.
3. Read `.notes/asr/benchmark-protocol.md` when behavior or performance may change.
4. Read `.notes/asr/delivery-roadmap.md` and `.notes/asr/progress.md` only when delivery
   order or current status matters.
5. Trace actual registries, configs, call sites, checkpoint loading, tests, and callers.

Distinguish the current implementation from the proposed design. Cite paths and line
numbers for material findings.

## Establish Boundaries

Classify the request as `discussion`, `architecture_review`, or `implementation`, and do
not expand beyond the requested mode. Map the affected behavior through:

- immutable upstream behavior versus downstream extension;
- registry/configuration and automatic model construction;
- data/frontend/tokenizer and model forward path;
- trainer, loss, optimizer, serialization, and checkpoint path;
- offline inference, streaming inference, and serving/runtime adapters;
- evaluation and externally visible output contracts.

Identify current owners, callers, state transitions, and compatibility surfaces before
proposing a new abstraction.

## Apply Review Gates

Evaluate only relevant gates and mark the rest not applicable:

| Gate | Required questions |
| --- | --- |
| Upstream isolation | Can a registered downstream component or small generic extension avoid broad edits to `AutoModel` or Paraformer internals? If not, why? |
| Registration | Does construction use the existing registry/config mechanism without a parallel factory or special-case branch? |
| Training/inference parity | Do features, tokenizer, shapes, masks, decoding semantics, precision, and exported state agree across paths? |
| Compatibility | Are config defaults, checkpoint keys, versioning, migration, and old-model loading behavior explicit? |
| Streaming state | Who creates, owns, resets, and finalizes cache/state? Are utterances and concurrent sessions isolated and memory bounded? |
| Streaming semantics | Are chunk, look-back, endpoint, partial/final, and offline-equivalence expectations explicit and testable? |
| Reliability | Are invalid input, cancellation, retry, partial failure, fallback, and resource exhaustion handled at the correct boundary? |
| Observability | Can a result be traced to code/model/config/runtime revisions and can latency or state growth be diagnosed? |
| Verification | Are focused unit, checkpoint/config, offline/streaming parity, concurrency/state-reset, and benchmark gates identified? |

## Choose The Minimum Durable Design

Prefer, in order, an existing public extension point, a new registered leaf component,
a narrow reusable hook with a real second consumer, and only then an upstream-core edit.
Do not add a framework, factory, service, or abstraction for a single speculative use.

For a material choice, compare at least one credible alternative and return one verdict:

- `Accept`: boundaries and verification are complete;
- `Accept with conditions`: the design is sound after named contract or test changes;
- `Spike`: a bounded prototype must resolve a specific uncertainty;
- `Reject`: it duplicates ownership, breaks compatibility/state isolation, or creates
  unjustified upstream coupling.

## Implementation Mode

Implement only when requested. Keep downstream code isolated, use existing registration
and configuration paths, and add focused tests at each changed boundary. An unavoidable
upstream-core edit must document why extension is insufficient and must include a
sync-resistant focused test. Update only the authoritative documents whose owned facts
changed; do not create parallel architecture or progress records.

## Handoff

Lead with the verdict and severity-ranked findings. Then state the current-to-proposed
boundary map, recommended design and alternative, registry/config/checkpoint impacts,
streaming-state lifecycle, verification commands and missing gates, upstream-sync risk,
and authoritative documentation impact.
