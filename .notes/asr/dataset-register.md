# Dataset Register

> Status: **BASE diagnostic registered; product evaluation remains pending `EVAL-01`**
> Updated: `2026-08-28`

This register identifies data and split lineage. Audio and large generated
artifacts stay outside Git; versioned manifests and hashes are the evidence.

## Registered Collections

| Dataset ID | Purpose | State | Split isolation | Manifest/hash | Data location |
|---|---|---|---|---|---|
| `LAB-BASE-SMOKE-001` | Two-utterance upstream parity diagnostic for `BASE-01` | Frozen diagnostic; not promotion evidence | Fixture-scoped synthetic IDs; speaker provenance unknown; smoke only | `eval/manifests/lab-base-smoke-001-v0.1.jsonl`; `sha256:775614f52d04f1b9aa320007af31e18e87c60c53a88f25625390c7a8389bcc10` | Tracked upstream runtime fixtures |
| `LAB-SEED-001` | Initial microphone/meeting smoke, dev, and sealed-blind evaluation | Planned | Speaker, session, source recording, lineage, dedup cluster, and exact audio | Pending real collection descriptor and manifests under `EVAL-01` | External, untracked |
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

Each collection descriptor records:

- dataset ID, immutable revision, record schema, normalizer, scoring unitizer,
  and scenario taxonomy versions;
- every split's logical manifest name, exact ordered byte hash, record count,
  and reference-access class;
- pseudonymous provenance and rights groups with hashed evidence, permitted
  uses, access class, and review state;
- the near-duplicate method, version, threshold, reviewed report hash, and
  blind seal/unlock policy plus separate sealed input/reference projection
  hashes.

Each strict split record binds the utterance, speaker, session, source
recording, derived lineage, deduplication cluster, split, relative audio
identity, audio/reference hashes, audio format, provenance/rights groups,
scenario tags, and explicit inclusion status. Unknown fields and free-form
scenario categories are rejected rather than silently ignored. Collection and
record validators contain no model import and produce canonical replay output.

## Split Rules

1. A speaker, recording session, source recording, derived lineage,
   deduplication cluster, or exact audio identity belongs to only one of train,
   dev, smoke, and sealed blind.
2. Near duplicates and derived clips follow the source item into the same split;
   SHA-256 alone is not accepted as near-duplicate detection.
3. The blind reference is sealed before experiment selection and is not used for
   prompt, vocabulary, hotword, or decoding adjustment.
4. Every exclusion is explicit and counted; failed decoding does not remove an
   item from the denominator.
5. A changed manifest or normalizer creates a new dataset/evaluation revision;
   it does not overwrite an accepted baseline.

## Sealed-Blind Boundary

The tracked collection descriptor may publish the sealed split's logical name,
hash, count, aggregate coverage, and policy, but not its full records or raw
references. It binds the hashes of both projections. The iterative runner sees
only the audio projection; the reference projection remains in restricted
untracked storage and is joined with frozen hypotheses by the blind-set
custodian after candidate selection is closed. The full core report stays
restricted. Projection schema version 2 omits predeclared excluded audio from
the decoder handoff while retaining those records in the restricted reference
projection and core accounting. A text-free aggregate is not publishable until
the separate one-candidate authorization and minimum-cell release policy pass.
The input-export receipt binds the sealed audio projection and custodian-owned
candidate lock before decoding. Candidate-lock schema v2 and every receipt also
bind the exact registered planned-manifest Git commit, repository path, and blob
hash; registration ancestry/reachability does not replace the separate CI-pass
gate. Before those artifacts are published, every included audio path is
reopened without following root, parent, or leaf symlinks and its hash/WAV
identity is rechecked. The execution envelope then binds that receipt,
raw prediction bytes, prediction items, and committed runner identity; the
prediction-freeze receipt binds the canonical prediction bundle to the same
input, lock, runner, and envelope. The restricted score receipt is the durable
completion marker that binds both preceding receipts, the execution envelope,
prediction artifact, sealed scoring identity, core hash, and committed
runner/scorer identities and source inventories at the same frozen candidate
commit, plus the scorer's CPython/Unicode, CPU-lock, and installed-distribution
identity. Stdout, an unpaired core, or any artifact detached from this receipt
chain does not count as replay evidence.

## Initial Coverage Target

`LAB-SEED-001` should eventually cover clean near-field, far-field meeting,
stationary and transient noise, accents, long utterances, domain terms,
Chinese-English code switching, silence/non-speech, and clipped or low-volume
audio. Coverage counts are reported from the manifest rather than estimated in
prose.
