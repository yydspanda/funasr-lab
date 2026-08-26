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
  resolved. `data_sha256` hashes the frozen, ordered evaluation manifest, not a
  mutable directory name.
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
