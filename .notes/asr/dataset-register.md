# Dataset Register

> Status: **BASE diagnostic registered; product evaluation remains pending `EVAL-01`**
> Updated: `2026-08-26`

This register identifies data and split lineage. Audio and large generated
artifacts stay outside Git; versioned manifests and hashes are the evidence.

## Registered Collections

| Dataset ID | Purpose | State | Split isolation | Manifest/hash | Data location |
|---|---|---|---|---|---|
| `LAB-BASE-SMOKE-001` | Two-utterance upstream parity diagnostic for `BASE-01` | Frozen diagnostic; not promotion evidence | Fixture-scoped synthetic IDs; speaker provenance unknown; smoke only | `eval/manifests/lab-base-smoke-001-v0.1.jsonl`; `sha256:775614f52d04f1b9aa320007af31e18e87c60c53a88f25625390c7a8389bcc10` | Tracked upstream runtime fixtures |
| `LAB-SEED-001` | Initial microphone/meeting smoke, dev, and sealed-blind evaluation | Planned | Speaker and recording session | Pending `EVAL-01` | External, untracked |
| `LAB-TINY-001` | Tiny deterministic overfit and checkpoint round-trip diagnostic | Planned | Derived only from the training partition | Pending `TRAIN-01` | External, untracked |
| `LAB-LONG-001` | Long-stream silence, boundary, noise, and reconnect validation | Planned | Session-disjoint from iterative dev | Pending `STREAM-01` | External, untracked |

Planned rows are not evidence. Change a row to `Frozen` only after its manifest,
audio/text hashes, split policy, and provenance have been reviewed.

`LAB-BASE-SMOKE-001` reuses the versioned FunASR API example and llama.cpp
regression sample already present in the accepted upstream snapshot. Their
references are cross-checked by upstream inference tests and frozen golden
outputs. Validate the ordered manifest and its hash with:

```bash
.venv/bin/python scripts/validate_offline_baseline_dataset.py \
  --dataset-manifest eval/manifests/lab-base-smoke-001-v0.1.jsonl
```

This diagnostic set proves reproducible wiring only. It has no microphone,
meeting, accent, noise, domain-term, or blind-test coverage and therefore cannot
support a quality or product-readiness claim.

## Manifest Requirements

Each collection manifest records:

- dataset ID and immutable revision;
- source/provenance and collection date where available;
- utterance, speaker, session, and split IDs;
- relative audio identity, SHA-256, duration, sample rate, and channels;
- raw reference identity and SHA-256;
- language/scenario tags and known consent/access boundaries;
- manifest SHA-256 and the script/command that produced it.

## Split Rules

1. A speaker or recording session belongs to only one of train, dev, smoke, and
   sealed blind.
2. Near duplicates and derived clips follow the source item into the same split.
3. The blind reference is sealed before experiment selection and is not used for
   prompt, vocabulary, hotword, or decoding adjustment.
4. Every exclusion is explicit and counted; failed decoding does not remove an
   item from the denominator.
5. A changed manifest or normalizer creates a new dataset/evaluation revision;
   it does not overwrite an accepted baseline.

## Initial Coverage Target

`LAB-SEED-001` should eventually cover clean near-field, far-field meeting,
stationary and transient noise, accents, long utterances, domain terms,
Chinese-English code switching, silence/non-speech, and clipped or low-volume
audio. Coverage counts are reported from the manifest rather than estimated in
prose.
