#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${QWEN_ASR_RUNTIME:-cann911}" in
    cann911)
        export POETRY_CACHE_DIR="${QWEN_RUNTIME_ROOT:-/home/qwen-upgrade-20261015}/poetry-cache"
        export POETRY_VIRTUALENVS_PATH="${QWEN_RUNTIME_ROOT:-/home/qwen-upgrade-20261015}/virtualenvs"
        unset VIRTUAL_ENV
        ;;
    legacy)
        unset VIRTUAL_ENV
        export POETRY_CACHE_DIR=/root/.cache/pypoetry
        export POETRY_VIRTUALENVS_PATH=/root/.cache/pypoetry/virtualenvs
        ;;
    *) echo 'QWEN_ASR_RUNTIME must be cann911 or legacy' >&2; exit 2 ;;
esac
exec "${POETRY_BIN:-/root/.local/bin/poetry}" run python scripts/check_asr.py "$@"
