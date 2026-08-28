PYTHON ?= python3

.PHONY: bootstrap doctor governance archive-progress upstream-guard smoke-plan static test-lab check

bootstrap:
	bash scripts/bootstrap_dev.sh

doctor:
	$(PYTHON) scripts/asr_lab_doctor.py

governance:
	$(PYTHON) scripts/check_asr_progress.py
	$(PYTHON) scripts/archive_asr_progress.py --check
	$(PYTHON) scripts/check_experiment_manifests.py

archive-progress:
	$(PYTHON) scripts/archive_asr_progress.py --apply

upstream-guard:
	$(PYTHON) scripts/check_upstream_guard.py --run-ledger-tests

smoke-plan:
	$(PYTHON) scripts/run_baseline_smoke.py --track paraformer \
		--audio eval/private/smoke.wav --dry-run

static:
	$(PYTHON) -m compileall -q asr_lab funasr examples tests eval scripts runtime/python

test-lab:
	$(PYTHON) -m unittest discover -s tests -p 'test_asr_lab_*.py'
	$(PYTHON) -m unittest discover -s tests -p 'test_*_governance.py'

check: doctor governance static test-lab
