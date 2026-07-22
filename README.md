# Qwen3-ASR gRPC 服务

本项目通过两个独立的 Typer 命令启动 Qwen3-ASR 推理服务。运行参数均可通过命令行或环境变量传入，不使用额外的 YAML 配置文件。

## 安装

```bash
poetry install
```

## 使用 vLLM 后端

```bash
poetry run qwen-asr vllm \
  --model /model-bin/Qwen/Qwen3-ASR-0.6B \
  --device cuda \
  --gpu-memory-utilization 0.27 \
  --max-model-len 1536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1536
```

## 使用 Transformers 后端

```bash
poetry run qwen-asr transformers \
  --model /model-bin/Qwen/Qwen3-ASR-0.6B \
  --device cpu
```

## 使用环境变量

复制环境变量示例，并将其加载到当前 shell：

```bash
cp .env.example .env
set -a
source .env
set +a
poetry run qwen-asr vllm
```

命令行参数的优先级高于环境变量。两个后端共用以下变量：

| 环境变量 | 对应参数 | 默认值 |
| --- | --- | --- |
| `QWEN_ASR_MODEL` | `--model` | 必填 |
| `QWEN_ASR_CONTEXT` | `--context` | 空字符串 |
| `QWEN_ASR_SERVER_PORT` | `--server-port` | `50051` |
| `QWEN_ASR_MAX_NEW_TOKENS` | `--max-new-tokens` | `256` |
| `QWEN_ASR_DEVICE` | `--device` | `cpu` |

vLLM 命令还支持：

| 环境变量 | 对应参数 | 默认值 |
| --- | --- | --- |
| `QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | `0.27` |
| `QWEN_ASR_VLLM_MAX_MODEL_LEN` | `--max-model-len` | `1536` |
| `QWEN_ASR_VLLM_MAX_NUM_SEQS` | `--max-num-seqs` | `4` |
| `QWEN_ASR_VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `1536` |
| `QWEN_ASR_VLLM_ENFORCE_EAGER` | `--enforce-eager` | `false` |

这组 vLLM 默认值面向 A100 40 GiB，目标是让单实例总显存保持在 12 GiB 以内。
60 秒有效音频约占 795 个输入 token，最多生成 256 token；1536 token 上下文仍为
热词和提示保留约 485 token。4 路序列并发和 1536 token 调度上限同时限制活跃 KV
占用、CUDA Graph profile 与单轮预填充规模，避免按通用大语言模型默认值过度预留。

运行以下命令可查看完整帮助：

```bash
poetry run qwen-asr vllm --help
poetry run qwen-asr transformers --help
```

## 排查 vLLM 显存增长

vLLM 会在启动阶段预分配 KV Cache，并保留 CUDA allocator 的历史高水位，因此
`nvidia-smi` 中的占用不会在每次请求结束后回到模型权重大小。正常状态应当是在预热后
进入平台期，而不是继续随请求数线性增长。

先启动服务，再用一段有效语音循环填充为 60 秒，执行完整请求和客户端取消请求：

```bash
poetry run python scripts/soak_vllm_memory.py \
  --audio data-bin/benchmarks/baseline-icl-preset/request-000.wav \
  --gpu-index 0 \
  --duration 60 \
  --warmup 5 \
  --requests 50 \
  --cancellations 20
```

脚本会在每个请求结束后采集 GPU 显存，默认以 12,288 MiB 为总占用上限，并检查预热
后的总增长和单请求增长斜率。需要
进一步区分活跃张量和 allocator 保留内存时，可用 `VLLM_LOGGING_LEVEL=DEBUG` 启动
服务；音频编码器会记录编码前后的 allocated、reserved 和缓存输出大小。
