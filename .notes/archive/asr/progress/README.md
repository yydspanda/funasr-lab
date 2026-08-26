# ASR Progress Archive

[`progress.md`](../../../asr/progress.md) keeps only the current execution
pointer and up to eight terminal records from the current Asia/Shanghai calendar month.
Prior-month records and any same-month overflow belong in `YYYY-MM.md`
files here, selected by each record's completion date.

Archive files are history, not active execution pointers. They must not contain
`Current Stage` or `In Progress Task` fields. Every archived record preserves
its date, Roadmap-registered task ID, terminal status, outcome, and verification
evidence; archive records remain newest first. Monthly files are limited to 600
lines so unusually verbose evidence must be moved to an experiment manifest or
artifact rather than accumulated here.

Use the deterministic helper instead of cutting blocks by hand:

```bash
python3 scripts/archive_asr_progress.py --check
python3 scripts/archive_asr_progress.py --apply
python3 scripts/check_asr_progress.py
```

The first command is read-only and exits nonzero when rollover is required. The
second updates `progress.md` plus the necessary monthly files, then rolls back
its changes if the complete governance check fails. CI runs the final checker,
which validates filenames, record months, registered task IDs, terminal
statuses, ordering, duplicate history, active-pointer absence, and line budgets.
