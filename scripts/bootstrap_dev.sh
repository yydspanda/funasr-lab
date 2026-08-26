#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv python install 3.11
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --torch-backend cpu \
  requirements/lab-cpu.lock
.venv/bin/python scripts/asr_lab_doctor.py

echo
echo "Development environment is ready. Activate it with:"
echo "  source .venv/bin/activate"
echo
echo "This script installs dependencies only; it does not download ASR models."
