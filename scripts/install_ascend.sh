#!/usr/bin/env bash
# 在当前 Poetry 环境构建不带 CUDA 的 vLLM；不会修改 CANN 或驱动。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "${QWEN_ASR_RUNTIME:-cann911}" == cann911 ]]; then
    exec bash "$PROJECT_ROOT/../qwen-llm/scripts/install_ascend_291.sh" "$PROJECT_ROOT"
fi
if [[ "${QWEN_ASR_RUNTIME}" != legacy ]]; then
    echo 'QWEN_ASR_RUNTIME must be cann911 or legacy' >&2
    exit 2
fi
unset VIRTUAL_ENV
export POETRY_CACHE_DIR=/root/.cache/pypoetry
export POETRY_VIRTUALENVS_PATH=/root/.cache/pypoetry/virtualenvs
poetry install --with ascend-legacy
poetry run python -m pip install -c deploy/ascend-constraints.txt 'setuptools>=77,<81' setuptools-scm ninja
VLLM_SOURCE_DIR="$PROJECT_ROOT/tmp_workspace/vendor/vllm"
if [[ ! -d "$VLLM_SOURCE_DIR/.git" ]]; then
    mkdir -p "$(dirname "$VLLM_SOURCE_DIR")"
    git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git "$VLLM_SOURCE_DIR"
fi
VLLM_REVISION="$(git -C "$VLLM_SOURCE_DIR" rev-parse HEAD)"
if [[ "$VLLM_REVISION" != 72506c98349d6bcd32b4e33eec7b5513453c1502 ]]; then
    echo 'vLLM 源码 revision 不匹配，拒绝覆盖现有目录。' >&2
    exit 1
fi
VLLM_TARGET_DEVICE=empty poetry run python -m pip install -c deploy/ascend-constraints.txt --no-build-isolation -e "$VLLM_SOURCE_DIR" \
    'torch==2.8.0' 'torch-npu==2.8.0' 'torchvision==0.23.0' \
    'typer>=0.24,<0.25' 'opencv-python-headless==4.11.0.86' \
    'openai>=2,<3' 'mcp>=1,<2' 'mistral-common[image]==1.8.6' 'gguf==0.17.1' \
    fastapi-cloud-cli absl-py ml-dtypes tornado
