#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv python install 3.11

venv_python=".venv/bin/python"
if [[ -e .venv || -L .venv ]]; then
  if [[ ! -x "${venv_python}" ]]; then
    echo ".venv exists but is not a usable virtual environment;" \
      "move it aside and retry" >&2
    exit 1
  fi
  if ! "${venv_python}" -c '
import sys

valid = sys.prefix != sys.base_prefix and sys.version_info[:2] == (3, 11)
raise SystemExit(0 if valid else 1)
'; then
    echo ".venv must be a Python 3.11 virtual environment; move it aside and retry" >&2
    exit 1
  fi
  echo "Reusing existing Python 3.11 environment at .venv"
else
  uv venv --python 3.11 .venv
fi

uv pip sync --python .venv/bin/python --torch-backend cpu \
  requirements/lab-cpu.lock
.venv/bin/python scripts/asr_lab_doctor.py --strict-base-env

echo
echo "Development environment is ready. Activate it with:"
echo "  source .venv/bin/activate"
echo
echo "This script installs dependencies only; it does not download ASR models."
