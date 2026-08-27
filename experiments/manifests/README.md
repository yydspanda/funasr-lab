# Experiment Manifests

Store one compact JSON provenance record per planned or executed experiment in this
directory. Its filename must be `<experiment_id>.json`, and an experiment ID may
appear in exactly one file. Start from
[`../manifest.template.json`](../manifest.template.json), replace every
angle-bracket value and zero hardware quantity, then run:

```bash
python3 scripts/check_experiment_manifests.py
```

The template is deliberately invalid until its placeholders are replaced. The
checker accepts this directory while it contains no `*.json` manifests, but it
rejects partial or duplicate-key JSON, floating model revisions (including
whitespace-disguised forms), abbreviated commits, empty or mechanically repeated
hashes, abridged commands, placeholder hardware, and non-finite metrics. It also
rejects a `task_id` absent from the authoritative roadmap and an executed
experiment without at least one hashed report artifact.

A pre-registered plan uses `decision: "planned"`, `metrics: null`, and an empty
`artifacts` list. All identities, hashes, hardware, and command fields must
already be concrete, so the frozen plan can pass CI before results are viewed.
After execution, replace `metrics` with measured values, including CER
substitutions/deletions/insertions and denominator, utterance/failure counts, RTF
P50/P95, and peak RSS. Attach at least one hashed report and change the decision
to `accept`, `reject`, or `investigate`.

## Identity Rules

- Resolve `upstream_commit` and `code_commit` to the full lowercase 40-character
  commit IDs. Tags, branch names, and 7/12-character display abbreviations are
  not provenance.
- List every loaded model under `models`, not only the main ASR checkpoint. VAD,
  punctuation, language-model, and ITN components each need a unique `role`, a
  stable identifier, an immutable revision, and a content hash.
- Prefix all SHA-256 values with `sha256:`. A model directory hash means the
  SHA-256 of a UTF-8 inventory sorted by POSIX relative path, with each line
  encoded as `<file-sha256><two spaces><relative-path>\n`; use the same inventory
  algorithm for all compared runs.
- `config_sha256` hashes the exact effective configuration after overrides are
  resolved. `data_sha256` hashes the frozen collection descriptor (or the
  ordered manifest for legacy BASE runs), not a mutable directory name.
- Use stable, non-secret `host_id` values. Record the real OS/kernel/architecture,
  CPU model, logical CPU count, installed memory in bytes, and actual device.
- `command.argv` is the unabridged process argument vector. Expand wrapper
  defaults and record every non-secret environment variable that changes the
  result. For a pipeline, record `bash`, `-lc`, and the complete pipeline as the
  third argument.
- Give each artifact a `kind` (`report`, `prediction`, `log`, `checkpoint`, or
  `other`) and a content hash. Every executed experiment needs at least one
  `kind: "report"`; keep large reports outside Git as required by the benchmark
  protocol.

The checker resolves both commits in Git, requires `code_commit` to be an
ancestor of both the checked-out revision and the durable target
`refs/remotes/origin/develop`, and requires `upstream_commit` to belong to the
accepted upstream baseline history. Fetch `origin/develop` before local
validation; CI does this explicitly. Content hashes still cannot prove which
bytes a process loaded at runtime; the experiment runner remains responsible
for capturing them before examining results.

Use a two-step branch flow so `code_commit` remains durable under any GitHub
merge strategy: first land the experiment code, then update from the target
branch and create the planned manifest in a later commit using that already
reachable target-branch SHA. Execute exactly that SHA. Record results by updating
the same manifest afterward. Do not point a manifest at a feature-branch commit
that squash or rebase merge can rewrite.

## Sealed EVAL Candidate Lifecycle

A sealed replay candidate uses `task_id: "EVAL-01"`, remains `planned`, and is
named exactly `<experiment_id>.json` when `export-input` creates its candidate
lock. The command runs this directory's complete format, task-registry, commit,
baseline-history, checked-out ancestry, and durable `origin/develop` ancestry
checks before any sealed reference manifest is opened. For real evidence, land
the planned manifest and let CI pass before the custodian starts the replay;
local validation alone does not prove that prior registration happened.

The immutable candidate projection contains source/code commits, all model
identities and hashes, config/data identity, dataset and normalizer versions,
hardware, seed, and the full command. It excludes `decision`, `metrics`, and
`artifacts`, so the same `candidate_freeze_sha256` must survive the later result
update. The candidate command must freeze exactly one
`--hypothesis-adapter-version` matching the custodian lock.

After scoring, keep the input-export, prediction-freeze, and score receipts with
the private run artifacts. The score receipt is the authoritative completion
marker and binds candidate-lock, prediction-bundle, restricted-core, and scorer
source identities; stdout is not evidence. When updating the manifest to
`accept`, `reject`, or `investigate`, pass the terminal manifest, score receipt,
and restricted core to
`eval.custodian_replay.validate_terminal_manifest_for_receipt()`. It verifies
the frozen candidate, dataset/normalizer/adapter lineage, core CER/MER values and
counts, plus hashed input/lock/prediction/core/receipt artifacts. Include
`metrics.mer`, using `null` only when the core MER denominator is zero. The
helper does not prove the origin of RTF/RSS values; those still require the
separately hashed execution envelope. No public summary may be released while
the receipt says `public_release.state` is `withheld`.
