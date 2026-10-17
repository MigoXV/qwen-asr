#!/usr/bin/env bash
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
case "${QWEN_ASR_RUNTIME:-cann911}" in
    cann911)
        source "$PROJECT_ROOT/scripts/env_ascend_291.sh"
        ;;
    legacy)
        unset VIRTUAL_ENV
        export POETRY_CACHE_DIR=/root/.cache/pypoetry
        export POETRY_VIRTUALENVS_PATH=/root/.cache/pypoetry/virtualenvs
        unset PYTHONPATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ATB_HOME_PATH
        export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
        source "${QWEN_ASR_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
        source "${QWEN_ASR_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}" --cxx_abi=1
        ;;
    *) echo "Unknown QWEN_ASR_RUNTIME; expected cann911 or legacy" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
exec "${POETRY_BIN:-/root/.local/bin/poetry}" run python -m qwen_asr.commands.app vllm-ascend "$@"
