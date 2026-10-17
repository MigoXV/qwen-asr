# Qwen3-ASR gRPC 服务

本项目通过 Typer 命令启动 Qwen3-ASR 推理服务，支持 Transformers、vLLM 和 vLLM Ascend。运行参数均可通过命令行或环境变量传入，不使用额外的 YAML 配置文件。

## 源码结构

```text
src/qwen_asr/
├── commands/
│   ├── app.py            # Typer 命令、参数、环境变量与轻量校验
│   ├── options.py        # 命令行选项枚举
│   └── serving.py        # 后端构造、服务组装与启动流程
├── servicer/
│   ├── server.py         # gRPC 服务生命周期与引擎故障监督
│   └── servicer.py       # 协议校验、容量控制和响应封装
├── inferencers/
│   ├── base.py           # 流式 ASR 推理接口
│   ├── pipeline.py       # ASRPipeline：音频解码与流式转写编排
│   ├── language.py       # 语言名称与代码映射
│   ├── text/             # 模型文本输出解析
│   ├── transformers/     # Transformers 推理适配
│   └── vllm/             # vLLM ASR 输入适配
├── engines/
│   ├── errors.py         # 引擎可用性与容量错误
│   ├── transformers/
│   │   └── engine.py     # 模型加载、设备张量准备与生成执行
│   └── vllm/
│       ├── engine.py     # 引擎加载、生成、取消与关闭
│       ├── process.py    # 单引擎子进程、Unix socket 通信与生命周期
│       ├── plugin.py     # vLLM 模型注册入口
│       ├── utils.py      # vLLM 引擎参数兼容
│       └── ascend/       # Ascend Worker、引擎配置与显存预算
├── models/
│   ├── common/           # 共享音频长度计算与正弦位置编码
│   ├── transformers/     # 模型配置、模型定义与 Processor
│   ├── vllm/             # vLLM 模型实现
│   └── vllm_ascend/      # Ascend 模型变体与模型专用 RoPE
└── protos/               # 服务协议与生成代码
```

请求经 `ASRServicer → ASRPipeline` 进入推理后端。Transformers 和普通 vLLM
命令直接使用对应推理器；Ascend 命令通过 `VLLMEngineClient` 将请求发送到单个
引擎子进程，再复用 vLLM 推理器。推理器通过组合引擎执行生成：`inferencers`
负责 ASR 语言、提示词、音频输入及输出编排，`engines` 持有模型和执行资源，
`models` 定义模型结构、权重与模型输入语义。进程入口在子进程中组装推理器，
执行引擎本身不依赖 ASR 推理器、CLI 或 gRPC 服务层。
导入 CLI 和进程客户端时不加载模型后端，由所选后端按需加载。

模型实现按职责拆分：Transformers 的 `config_audio.py`、`config_text.py` 和
`config_model.py` 定义音频、文本与组合配置，`modeling/` 下的 `layers.py`、
`audio.py`、`text.py`、`rotary.py`、`base.py` 与 `model.py` 分别实现基础层、
音频编码、文本解码、位置编码、基类和生成模型组装。相同的 RMSNorm、MLP
与文本注意力计算共享实现，保留各命名类和框架元数据。

vLLM 的 `audio_layers.py`、`audio.py`、`processing.py` 与 `model.py` 分别负责
音频网络层、音频编码器、多模态输入处理与模型组装；权重加载仍由对应模型管理。
Ascend 模型直接复用这些组件，只覆盖 NPU 注意力和模型专用 RoPE。
两个框架使用不同的执行算子和权重加载机制，因此分别保留编码器实现；音频输出长度
和正弦位置编码统一使用 `common/`，避免多份公式随修改产生偏差。

原 `configuration_qwen3_asr.py`、`modeling_qwen3_asr.py` 和 vLLM 的
`qwen3_asr.py` 保留为薄导出入口，模型类名、配置字段、权重键名和插件路径保持兼容。
包级模型导出按需加载，单独使用配置或 Processor 不会导入模型网络或 vLLM/NPU。

原 `core.*_backend` 中的模型代码归入 `models.*`，Ascend Worker 与引擎配置归入
`engines.vllm.ascend`；
`inferencers.grpc_inferencer.GrpcInferencer` 调整为 `inferencers.pipeline.ASRPipeline`，
`inferencers.vllm.pool` 调整为 `engines.vllm.process`。不再保留顶层 `backends` 和
`engine` 包。推理器的 `LLM(...)` 工厂及 `transcribe_stream(...)` 用法保留，
直接构造推理器时改为传入引擎实例。历史实验脚本未随此次整理迁移，
其中引用旧内部路径的脚本需要在重新使用时更新。CLI 命令及环境变量保持兼容。

vLLM 插件入口已迁移到 `engines.vllm.plugin`，已有环境升级源码后需执行
`poetry install --only-root` 更新入口元数据。

运行自动化测试：

```bash
poetry run python -m unittest discover -s tests -v
```

Ascend 硬件测试和 VAD 补丁测试需要对应环境及显式开关，默认跳过。

## 安装

```bash
poetry install
```

## 使用 vLLM Ascend 后端

当前默认环境为 Python 3.10.22、Torch / torch-npu 2.9.1、源码 vLLM 0.18.0、
源码 vLLM Ascend 0.18.0（本地Torch 2.9.1构建补丁）、Triton-Ascend 3.2.2、CANN 9.1.1。
新CANN与Poetry环境安装于数据盘 `/home/qwen-upgrade-20261015/`，原CANN 8.3.RC1与Torch 2.8环境保留。
[升级验证报告](docs/cann911-torch291.md)记录24组负载、816次识别的同条件比较；未超过20%性能回退门槛。
这套本地源码组合已通过实机测试，安装器不会更换驱动或覆盖旧CANN。
性能优化继续基于这套新栈，保留的旧环境用于退化定位对照。

```bash
# 复用兄弟项目qwen-llm中固定源码、补丁和依赖约束；要求数据盘CANN已安装。
bash scripts/install_ascend.sh

# 要求已安装 CANN Toolkit、910B kernels 和 NNAL/ATB。
bash scripts/serve_ascend.sh \
  --model /root/repositories/qwen-asr/model-bin/Qwen/Qwen3-ASR-1.7B \
  --workers 8 --max-concurrent-rpcs 32 --server-port 50011 \
  --max-num-batched-tokens 1024 --memory-utilization 0.11 --triton-decode
```

新vLLM以editable模式安装在 `/home/qwen-upgrade-20261015/src/vllm018`，运行时需要保留源码目录。
显式使用旧环境：`QWEN_ASR_RUNTIME=legacy bash scripts/serve_ascend.sh`；重建旧环境时同样设置
`QWEN_ASR_RUNTIME=legacy bash scripts/install_ascend.sh`，对应Poetry可选组为 `ascend-legacy`。

Ascend core 替换音频编码器的注意力执行，并融合 ASR 解码器的 RoPE，复用原 vLLM 后端的模型结构、权重加载、
输入处理、文本解码和流式推理器。平台插件在每个 vLLM 子进程中自动注册对应 core。
始终只加载一份模型权重、启动一个 NPU Engine Core。`--workers` 与
`--max-num-seqs` 是同一参数的两个名称，表示该引擎的活跃序列上限（测试 4/8）。
gRPC 外层并发独立限制为 32，多出的引擎请求由 vLLM 排队。API 进程通过有界
Unix socket 流式提交请求，不加载模型。客户端取消会关闭 IPC 流并 abort 对应请求。

| 环境变量 | 参数 | 默认值 |
| --- | --- | --- |
| `QWEN_ASR_WORKERS` | `--workers` / `--max-num-seqs` | `4`，共享单模型的引擎并发 |
| `QWEN_ASR_MAX_CONCURRENT_RPCS` | `--max-concurrent-rpcs` | `32` |
| `QWEN_ASR_MEMORY_UTILIZATION` | `--memory-utilization` | `0.10`，单引擎 Torch 峰值及 KV Cache 预算比例 |
| `QWEN_ASR_KV_CACHE_MEMORY_MIB` | `--kv-cache-memory-mib` | 按 BF16、模型结构、并发数和上下文长度计算；当前 4/8 并发为 448/896 MiB |
| `QWEN_ASR_VLLM_ENFORCE_EAGER` | `--enforce-eager` / `--no-enforce-eager` | `false` |
| `QWEN_ASR_ASCEND_GRAPH_MODE` | `--graph-mode` | `FULL_DECODE_ONLY`；也支持 `PIECEWISE` |
| `QWEN_ASR_ASCEND_FUSED_ROPE` | 模型构造时读取 | `true`；设为 `false` 后重启可恢复原 RoPE |
| `QWEN_ASR_ASCEND_TRITON_DECODE` | `--triton-decode` / `--no-triton-decode` | `false`；本机已验证的 910B 1.7B 配置设为 `true` |
| `QWEN_ASR_ASCEND_NATIVE_FIA_UPDATE` | `--native-fia-update` / `--no-native-fia-update` | `false`；新版 torch-npu 的 C++ 融合 FIA 图参数更新，与 Triton 解码分开选择 |
| `QWEN_ASR_ASCEND_ENCODER_GRAPH` | `--encoder-graph` / `--no-encoder-graph` | `false`；在 Triton 解码上叠加音频编码图，要求 `max-num-batched-tokens=1024` |
| `QWEN_ASR_VLLM_ASYNC_SCHEDULING` | `--async-scheduling` / `--no-async-scheduling` | `true` |
| `QWEN_ASR_VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `4096` |
| `QWEN_ASR_RUNTIME` | 启动/安装环境 | `cann911`；`legacy`选择旧栈 |
| `QWEN_RUNTIME_ROOT` | 新CANN、Poetry与缓存根目录 | `/home/qwen-upgrade-20261015` |
| `QWEN_ASR_CANN_ENV` | 旧环境的Toolkit环境文件 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `QWEN_ASR_ATB_ENV` | 旧环境的ATB环境文件 | `/usr/local/Ascend/nnal/atb/set_env.sh` |

Ascend 默认使用 BF16、ACL 完整解码图、异步调度，1024 token 上下文和 4096 token 调度上限，关闭前缀缓存和
多模态处理结果缓存。每次 RPC 上传一段完整的单声道 PCM16 音频，最长 30 秒；
返回的流式片段是离线音频的增量转写，不是实时分块音频输入。

融合 RoPE 仅在本项目的 Qwen3-ASR NPU 模型内启用，利用文本及音频位置的三轴相等
性质调用 `npu_mrope`，不修改 vLLM 全局类或第三方包。替换在模型构造时完成，先于
编译和图捕获。当前本机 1.7B 服务使用引擎并发 8、token 调度上限 1024 和固定 KV
896 MiB；显存以实际 HBM 采样验收，允许的硬上限为 16 GiB。

本机 910B4-1 的 1.7B BF16 可开启 Triton 分页解码注意力，直接读取设备端长度和页表，
减少完整解码图每层的 Host 参数更新。该路径限定 vLLM 0.18.0、torch-npu 2.9.1、
Triton-Ascend 3.2.2、单卡单模型、不超过 8 槽、1024 上下文及 `FULL_DECODE_ONLY`。
encoder 和 prefill 复用原实现；图缓冲地址变化会退到 eager 并记录原因。
使用 `--no-triton-decode` 或设置上述环境变量为 `false` 可回退原注意力路径。

使用 `--no-triton-decode --native-fia-update` 可在供应商 FIA 注意力路径中启用原生融合图更新，
减少每层 Python 调用、算子分发和队列等待。该适配只作用于当前 ASR Worker 的 runner，
保持已安装的 Torch/torch-npu 2.9.1、vLLM/Ascend 0.18 和 CANN 9.1.1。
当前验证范围为 910B、1.7B BF16、8 槽以内、1024 上下文、FULL_DECODE_ONLY、单卡且无推测解码或 KV 传输。
启动时检查版本和配置，PA 图保留上游更新方法；日志记录实际命中，关闭参数后恢复原调用。
[新栈 FIA 更新验证](docs/ascend-native-fia-update-20261016.md)记录与 Triton 路径分开进行的正确性和性能测试。

可选的 `--triton-decode --encoder-graph --max-num-batched-tokens 1024` 使用新栈
`npu_fusion_attention_v3` 捕获音频编码器的 Transformer 部分。长度和窗口边界完全相同才复用图，
每个 Worker 最多保留 4 档图和 64 条形状观察记录；首次遇到的长度走 eager，重复出现后捕获。
首次捕获和缓存淘汰可能增加请求延迟，持续变化的音频长度不保证获益。超出 1024 个编码 token
的输入走 eager；输出复制到独立存储，避免后续回放覆盖其他请求的音频 embedding。
使用 `--no-encoder-graph` 可保留 Triton 解码并关闭编码图。该开关不修改第三方安装或正式服务环境文件。

[组合优化实测](docs/ascend-triton-encoder-graph-20261016.md)：新栈下30秒音频gRPC吞吐为
单并发61.33×、32并发224.83×，对复测较快基线提升3.9%/10.9%。
10秒高并发及部分混合长度延迟有退化，因此编码图保持默认关闭，应按实际负载选择。

[910B 优化验收报告](docs/ascend-910b-optimization-20261016.md)记录当前源码的 24 组负载，
816 条转写对齐，以及中英文混合、持续负载、取消和容量验证。30 秒单并发吞吐
34.52→58.74 倍速，32 并发 181.27→206.68 倍速；性能数字来自未开启 profiler 的 RPC 测试。

完整解码图捕获包含 attention 的解码 forward；音频编码和 prefill 不包含在该图中。
按实际引擎并发捕获 1 到 `max_num_seqs` 的档位，不创建额外模型副本。
启动日志记录实际图模式、捕获数和 Torch 显存；要求图执行却未捕获任何图时启动失败。
真实运行应能看到 `Replaying aclgraph`，不能只凭关闭 eager 参数判定图已生效。
`--enforce-eager` 仅供显式诊断或基线对照。
复现旧 eager 基线时还需指定 `--no-async-scheduling --max-num-batched-tokens 1024`。

`SharedNPUWorker` 补齐该 Ascend 版本对 `kv_cache_memory_bytes` 的支持，避免自动
显存探测按设备比例过度预留缓存。引擎仍执行完整 encoder profiling，检查自身
Torch 峰值加固定 KV Cache 是否超过分配的预算，并检查设备剩余
显存。预算不是进程级硬显存上限，CANN、通信库和 allocator 另有开销，实际 HBM
以 `npu-smi` 采样为准。

### 系统服务

`deploy/qwen-asr-ascend.service` 的默认项目路径为 `/root/repositories/qwen-asr`。
模型加载后，`ExecStartPost` 会运行真实音频识别，成功才完成服务启动。
服务会随系统启动；退出时 systemd 清理整个进程组，故障时有限重启。

```bash
sudo install -m 644 deploy/qwen-asr-ascend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-asr-ascend
sudo journalctl -u qwen-asr-ascend -f
```

可在 `/etc/qwen-asr-ascend.env` 覆盖服务环境，例如 `QWEN_ASR_WORKERS=8`，
修改后执行 `sudo systemctl restart qwen-asr-ascend`。本机当前部署为单模型、引擎并发 8、gRPC 并发 32，端口 50011。

### 本机会议工作流

本机 Audio Harness 复用现有 `audio-harness.service`，HTTPS 端口 10000，TLS gRPC
端口 10001，已发布模型名为 `ai-meeting`。AGC 后的双 VAD 分支分别提供临时转写和
正式转写；两个 ASR 节点都连接 `localhost:50011`，VAD 连接 `localhost:50022`。
正式片段及占位分离批次最多 30 秒。说话人分离使用 `dummy_diarize`（说话人 0），
聚类暂未实现，不在执行图中，也不连接远端特征服务。

取消联调时修复了现有 VAD 镜像中“取消已移除会话后重复关闭”的问题；本机 VAD
容器使用 `local/fireredvad-ncnn:cancel-fix-20261014`，VAD 源仓库保持原状。补丁、
Dockerfile、回滚信息与复测记录位于
`tmp-workspace/2026-10-14/fused-ai-meeting/vad-fix/`。

在本项目的 Poetry 环境运行 `scripts/setup_ai_meeting.py --output <新输出目录>`
可通过 w223 读取远端参考并配置本机工作流。会议协议及压测脚本依赖 Harness 的
客户端包，使用其独立 Poetry 环境执行：

```bash
cd /root/repositories/audio-harness
poetry run python /root/repositories/qwen-asr/scripts/check_ai_meeting.py \
  --output /root/repositories/qwen-asr/tmp-workspace/2026-10-14/meeting-smoke-new
poetry run python /root/repositories/qwen-asr/scripts/benchmark_ai_meeting.py \
  --output /root/repositories/qwen-asr/tmp-workspace/2026-10-14/meeting-benchmark-new
```

压测默认测试原音及 10/20/30 秒循环变体，客户端并发 1/2/4/8/16/32、每档三轮，
100 ms PCM 分块、不限速发送。`--speed 1` 为实时发送，`--speed 10` 为 10 倍发送。
吞吐包含建连、双分支 VAD/ASR 和排空；首包可能是 VAD 事件，首条临时与正式转写
延迟单独统计，从开始发送音频计时。TLS 使用 Harness 本机证书并校验主机名。

本机已放行 public 区域的 TCP 50011 端口，运行时及永久防火墙规则均已配置。

### 并发基准

先将服务的 `QWEN_ASR_WORKERS` 设为待测的 4 或 8，重启并等待 systemd 状态为
`active`。确认启动日志中的实际并发和图捕获数后再压测；脚本的 `--workers` 只记录配置，
不会修改正在运行的服务。输出目录必须是新目录，已有结果时脚本拒绝追加。

```bash
poetry run python scripts/check_asr.py
poetry run python scripts/benchmark_asr.py \
  --workers 4 --requests-per-client 1 --min-requests 4 \
  --execution-mode FULL_DECODE_ONLY-batched4096-async \
  --output tmp-workspace/2026-10-14/new-run/workers-4
```

本机图模式重测报告位于 `tmp-workspace/2026-10-14/graph-optimized/report.html`；
原 eager 报告保留在上一级目录供对照。

基准使用 `data-bin/rita.wav` 原音频及循环拼接到 10、20、30 秒的变体，客户端并发
为 1、2、4、8、16、32。每种输入先以 worker 数并发预热两轮，各并发档测三轮，
本次测试每客户端每轮至少一次请求、每轮至少四次请求。原始请求、转写、失败、预热
和设备资源记录在输出目录，另生成 CSV/JSON 汇总。整卡 HBM 约每秒采样，含驱动
占用，超过 12 GiB 时中止压测。只测试 NPU + vLLM BF16。

吞吐倍速 = 成功处理的音频总秒数 / 整轮墙钟时间。首包延迟从发起 RPC 到首个响应；
另记录首个非空转写延迟，汇总 P50/P95/P99。音频一次性上传，计时包含上传、排队、
预处理、推理和响应传输；失败不计入吞吐分子，但保留在分母和失败统计中。

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
