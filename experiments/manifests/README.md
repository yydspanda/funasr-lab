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
hashes, structurally invalid commands, placeholder hardware, and non-finite
metrics. It also rejects a `task_id` absent from the authoritative roadmap and
an executed experiment without at least one hashed report artifact. Generic
command validation checks structure and placeholders; the task runner and review
own task-specific completeness, with the sealed EVAL command enforced by its
shared runner contract before references are opened.

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
  algorithm for all compared runs. Reject relative paths containing ASCII
  controls U+0000..U+001F or U+007F so that one file cannot inject another
  inventory line.
- `config_sha256` hashes the exact effective configuration after overrides are
  resolved. `data_sha256` hashes the frozen collection descriptor (or the
  ordered manifest for legacy BASE runs), not a mutable directory name.
- Use stable, non-secret `host_id` values. Record the real OS/kernel/architecture,
  CPU model, logical CPU count, installed memory in bytes, and actual device.
- `command.argv` is the unabridged process argument vector. Expand wrapper
  defaults and record every non-secret environment variable that changes the
  result. For a pipeline, record `bash`, `-lc`, and the complete pipeline as the
  third argument. This is a recording requirement; the generic checker cannot
  infer the semantic completeness of an arbitrary program's CLI.
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
reachable target-branch SHA. Execute source bytes authenticated against that
SHA; the checkout may contain the later manifest-only commit only if every
runner/scorer source byte still matches. Ordinary experiments record results by
updating the same manifest afterward; sealed results follow the private-terminal
exception below while release is withheld. Do not point a manifest at a
feature-branch commit that squash or rebase merge can rewrite.

## Sealed EVAL Candidate Lifecycle

A sealed replay candidate uses `task_id: "EVAL-01"`, remains `planned`, and is
named exactly `<experiment_id>.json` when `export-input` creates its candidate
lock. Supply its full registration commit through
`--candidate-registration-commit`. The command loads the exact Git blob at
`<registration>:experiments/manifests/<experiment_id>.json`, requires the
candidate `code_commit` to be an ancestor of that registration, and requires
the registration to be reachable from both checked-out `HEAD` and fetched
`origin/develop`, all with the fixed Git executable, before any sealed reference
manifest is opened. Candidate-lock schema v2 and every receipt repeat the
registration commit, exact path, and blob SHA-256. These checks prove bytes and
reachability, not CI success. For real evidence, land the planned manifest and
let CI pass before the custodian starts the replay; local validation alone does
not attest a successful CI run.
Directory governance also rejects any result-bearing EVAL-01 copy under this
tracked directory; terminal metrics and chain hashes belong only in restricted
private storage while release is withheld.

The immutable candidate projection contains source/code commits, all model
identities and hashes, config/data identity, dataset and normalizer versions,
hardware, seed, and the full command. It excludes `decision`, `metrics`, and
`artifacts`, so the same `candidate_freeze_sha256` must survive the later result
update. The candidate command uses the complete canonical sealed-runner option
order, one value per required option, a CPU-only profile, and a full lowercase
40-character model snapshot commit. Its adapter must match both the requested
custodian lock and the fixed model track. The sealed seed is `0`, `ncpu` is
bounded to 1..4096, and warmups to 0..100. The model track also freezes the
registered tokenizer/frontend/model components, CPU FP32 precision, batch size
one, snapshot-local resources, and the absence of VAD, punctuation, speaker,
LM, remote-code, and secondary-output paths.

After scoring, keep the execution envelope plus input-export,
prediction-freeze, and score receipts with the private run artifacts. The
input-export receipt is supplied to the runner, freeze, and score transitions;
the score receipt is the authoritative completion marker and binds both earlier
receipts, candidate lock, prediction bundle, execution envelope, restricted
core, and committed runner/scorer identities; stdout is not evidence.
Both identities must bind the candidate's same `code_commit`; the score receipt
also freezes CPython, Unicode, the CPU lock, and installed-distribution
inventory.

Create a private terminal copy outside this tracked directory, in the
mode-`0700` restricted run directory, and change only that copy to `accept`,
`reject`, or `investigate`. Its file mode is `0600`. While the score receipt
says `public_release.state: withheld`, the tracked manifest remains `planned`;
do not commit exact blind metrics or hashes as a side channel. Pass the private
terminal manifest, input-export, prediction-freeze, and score receipts,
restricted core, and execution envelope to the terminal validator described in
`eval/README.md`. It verifies candidate/data/normalizer/adapter lineage, core
CER/MER values and counts, envelope-derived RTF P50/P95, all-attempt/retry
counts, cold/warm timing and RSS, plus every required artifact hash. Include
exactly these 19 metric fields: `content_cer`, `substitutions`, `deletions`,
`insertions`, `reference_units`, `utterance_count`, `failed_count`,
`excluded_count`, `mer`, `rtf_p50`, `rtf_p95`, `peak_rss_mb`,
`rtf_attempted_count`, `retried_count`, `model_load_seconds`,
`cold_inference_seconds`, `cold_start_seconds`, `warm_wall_seconds`, and
`warm_audio_seconds`. Use `null` for `mer` only when the core MER denominator is
zero; count fields remain integers.

The terminal `artifacts` list must bind at least these eight chain members with
their required kinds: sealed input (`other`), candidate lock (`other`),
input-export receipt (`other`), canonical prediction bundle (`prediction`),
execution envelope (`report`), prediction-freeze receipt (`other`), restricted
core (`report`), and score receipt (`other`). Keep the raw prediction JSONL as
well even though its hash is transitively bound rather than one of the eight
minimum terminal entries.

For this CPU v1 evidence, the exact model bundle must already be present in the
planned cache; a cache miss or download invalidates the run as evidence.
`cold_start` excludes the cold attempt's preceding audio read/hash/WAV
validation and mandatory model-integrity inventory/hash work before and after
the timing windows. Peak RSS is the fresh-process Linux `RUSAGE_SELF` high-water
mark sampled immediately after the measured pass; it covers in-process startup,
validation, model load, cold/warmup, and measured decode, but not
post-measurement model verification, child processes, publication, or service
capacity. No terminal manifest or summary may leave restricted storage until a
separate release policy authorizes it.
