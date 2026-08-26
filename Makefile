PYTHON ?= python3

.PHONY: bootstrap doctor governance smoke-plan static test-lab check

bootstrap:
	bash scripts/bootstrap_dev.sh

doctor:
	$(PYTHON) scripts/asr_lab_doctor.py

governance:
	$(PYTHON) scripts/check_asr_progress.py
	$(PYTHON) scripts/check_experiment_manifests.py

smoke-plan:
	$(PYTHON) scripts/run_baseline_smoke.py --track paraformer \
		--audio eval/private/smoke.wav --dry-run

static:
	$(PYTHON) -m compileall -q funasr examples tests eval scripts

test-lab:
	$(PYTHON) -m unittest discover -s tests -p 'test_asr_lab_*.py'
	$(PYTHON) -m unittest discover -s tests -p 'test_*_governance.py'

check: doctor governance static test-lab
