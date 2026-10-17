# Qwen3-ASR-1.7B / 910B 热点定位

本次主要瓶颈是 **完整解码图回放前，Host 对 28 层 FIA 注意力逐层更新任务参数**。图已经捕获和回放，图内仍有很大的 Host 更新代价。首轮 30 秒、32 并发 trace 中，设备计算区间并集约 2.636 秒，首个到最后一个计算 kernel 的窗口约 6.670 秒，忙时占比约 39.5%。

通过让解码注意力直接读取设备端长度和页表，移除逐层 Host 更新，正式 RPC 的 24 组负载均获得提升。实现、完整验收和路线取舍见[优化报告](ascend-910b-optimization-20261016.md)。

## 源码、环境和现场

证据根目录为 `/home/qwen-asr-hotspots/20261016-002458/`，下文简称 `TASK`。初始代码、index、未合并项和逐文件 SHA256 均保存在该目录；没有使用旧 HEAD 代替磁盘源码。

用户随后要求提交全部已有改动，因此完成旧 `core/vllm_backend/qwen3_asr.py` 的删除冲突，保留新目录中的兼容改动，提交为 `2943cb0`。新架构还需要适配 vLLM 0.18 的注意力构造和 ProcessingInfo parser 工厂，分别提交为 `fda311f`、`94992c9`。这些是建立可运行基线的兼容修复，不计作性能收益。

基线来自 `94992c9` 的模型源码；最终推理实现为 `a2a0609`，包含此前的文本清理提交 `2e6b9a9`。测量时未提交的源码另有 `baseline-source-sha256.txt`、`candidate-source-sha256.txt` 和代码归档，最终源码与候选 SHA256 清单一致。基准脚本内的 HEAD 字段不能替代这些工作区证据。

| 项目 | 实际配置 |
| --- | --- |
| 设备 | 物理 NPU 4，Ascend910B4-1，64 GiB；进程内 NPU 0 |
| 驱动 | 26.1.1，未更改 |
| 软件 | Python 3.10.22；Torch/torch-npu 2.9.1；vLLM 0.18.0+empty；vLLM-Ascend 0.18.0+torch291.cann911；Triton-Ascend 3.2.2；Transformers 4.57.6；CANN 9.1.1 |
| 软件性质 | 既有本机源码构建与兼容补丁组合，并非官方原版兼容矩阵 |
| 模型 | `model-bin/Qwen/Qwen3-ASR-1.7B`；相同 BF16 权重 |
| 容量 | 一个模型、一个 EngineCore；8 执行槽、32 RPC；上下文 1024；调度上限 1024；KV 896 MiB |
| 执行 | async scheduling；FULL_DECODE_ONLY；捕获 1..8；融合 RoPE；前缀和多模态结果缓存关闭 |
| 图证据 | 启动完成记录、8 个图和实际 `Replaying aclgraph`；不是仅依据配置判断 |

实际加载库和线程亲和性见 `baseline-live-runtime.json`、`candidate-post-soak-runtime.json`、诊断目录的 `worker-info.json`。Torch 的 CPU wheel 后缀不代表此处使用 CPU 推理，真实 NPU kernel 已在 trace 中采到。

开始时 ASR、LLM 均 active。用户明确要求停止 NPU 服务后，已停止两项正式服务；实验使用独立瞬态单元，结束后维持正式服务停止，不执行原交接文档中的自动恢复。

## 链路与测量

请求链路是 `gRPC → ASRPipeline → VLLMEngineClient/Unix socket → 推理适配器 → VLLMEngine/AsyncLLM → EngineCore/NPUModelRunner → 音频编码器与文本解码器`。

每次 RPC 上传整段 PCM，返回增量转写；这不是实时音频分块推理。音频为原始 5.1299375 秒 `rita.wav` 及循环到 10/20/30 秒的输入；并发 1/2/4/8/16/32，每格 3 轮，预热 2 轮，requests-per-client=1、min-requests=4。吞吐是成功音频秒数除以整轮墙钟，不是 token TPS。

原始逐请求数据位于 `baseline-full/raw.jsonl`、`candidate-full/raw.jsonl`；包含首包、首字、完成延迟、完成状态和转写。各自的 `summary.json` 给出 P50/P95/P99，`resources.jsonl` 记录 HBM、温度、功耗和进程。6 格重复 FIA 对照位于 `control-repeat/`。

设备未锁频；C32长音频 trace 的基线频率采样包含800/1650 MHz，候选为1650 MHz。温度、功耗范围及频率原始采样汇总在 `hardware-conditions.json`。端到端成绩是在这些实际条件下取得，不能把所有收益视为排除频率影响后的纯算子收益。

在真正的 NPU Worker 中使用 `torch_npu.profiler` Level1 采集 CPU/NPU trace，并另外采集阶段计时和 cProfile。首轮目录是 `profile/`；原型是 `triton-engine/`；最终实现补采在 `final-profile/`。API 父进程另用信号控制 cProfile，结果是 `api-control.pstats`、`api-candidate.pstats`，覆盖 gRPC、Unix socket、PCM 转换、文本解析及响应封装。

指标口径必须分开：Host 区间包含派发与等待；嵌套阶段不能相加；NPU event 跨度包含 stream 空闲；设备忙时使用跨 stream 计算任务区间的并集，排除 NOP、事件、通知和内存等待任务。算子表的比例以算子耗时总和为分母，不是墙钟占比。所有 profiler 轮次只用于定位，验收使用独立的未插桩 RPC 结果。

## 基线热点证据

### 解码 Host 路径

30 秒、单并发的阶段采样包含 81 次 decode，实际输出 81 token。

| 阶段 | 次数 | Host 总时间 ms | 单次中位 ms |
| --- | ---: | ---: | ---: |
| decode 输入准备 | 81 | 110.6 | 1.360 |
| 完整图注意力参数更新 | 81 | 449.0 | 5.533 |
| decode model forward，包含上述更新 | 81 | 495.5 | 6.111 |
| execute_model，包含输入准备和 forward | 81 | 867.2 | 10.681 |
| sample_tokens | 81 | 104.8 | 1.285 |

Worker cProfile 中 `graph_task_update_begin/end` 各调用 2268 次，等于 81×28。更新过程每层重入 FIA `.out`，由 Host 长度列表重新准备图任务。该部分在插桩窗口占数百毫秒，具有显著收益上界；其占比不能原样套到无插桩延迟。

仅对上述 Host `execute_model` 区间作上界估算：449.0/867.2≈51.8%，如果该段成本完全消失且其余工作不变，867.2/(867.2−449.0)≈2.07 倍。这不是客户端吞吐预测：自定义 kernel 自身有成本，sampling、首字前处理、IPC 和排队也不会随之消失。

32 并发阶段采样有 312 次 decode 更新，总 Host 时间约 1776.9 ms，中位数 5.686 ms。高并发同时增加 encoder/prefill 调度和排队，故单流与聚合吞吐的收益不会相同。

### NPU 时间线和算子

| 30 秒负载 | 计算 kernel 数 | 计算窗口 ms | 忙时并集 ms | 忙时比例 |
| --- | ---: | ---: | ---: | ---: |
| C1 | 29425 | 1163.4 | 450.9 | 38.8% |
| C8 | 40518 | 1678.5 | 676.1 | 40.3% |
| C32 | 157076 | 6670.4 | 2636.4 | 39.5% |

短音频 C1/C8/C32 忙时比例约 31.3%/32.2%/31.7%。图内部 kernel 可见，不能把空闲当成未知设备计算。长音频 C32 的算子累计时间主要为 MatMulV2 52.395%、FusedInferAttentionScore 19.274%、RmsNorm 6.953%、RoPE 4.547%、AddRmsNormBias 3.250%。优化已有 GEMM 或极小 elementwise 的收益上界，小于先移除每层 Host 更新。

最终生产实现的 C32 补采中，长音频计算窗口4885.0 ms、计算并集3098.9 ms，忙时比例63.4%；短音频窗口1427.1 ms、并集616.8 ms，忙时比例43.2%。计算总量没有凭空减少：Triton 解码 kernel 的设备时间高于 FIA，但消除了更多 Host 空隙。cProfile 中 8736 次逐层 task-update 调用降为0；312次 decode 的图参数处理总时间1776.9→136.8 ms、中位5.686→0.436 ms。该残留是图缓冲签名校验等 Host 成本。补采与基线使用同一套诊断口径，验收仍以未插桩 RPC 为准。

### 首字及 API 路径

单并发阶段采样中 audio encoder 的 Host forward 约 33.5 ms，prefill model forward 约 49.5 ms。32 请求的输入处理采样总计 660.6 ms、每条中位数 20.63 ms。它们来自独立采样窗口，不能直接相加重建客户端首字延迟。

API profile 的 48 请求窗口有 3648 次增量解析，原重复清理相关 self 时间约 1.30 秒。等价匹配实现降到约 0.40 秒，但同实例吞吐 A/B/A 为 205.86/205.44/205.39 倍速，未显示独立端到端收益。它减少 API CPU 工作，不能据此把它称为本次吞吐提升的根因。

## 因果实验与边界

| 实验 | 实测 | 判断 |
| --- | --- | --- |
| 上游 PA，1..8 解码档位 | 30 秒 C1 34.88→26.71x，C8 190.48→154.39x，C32 194.10→160.50x | 单独切换更慢，不采用 |
| PA 实例中本地 NUMA 亲和性 A/B/A | C32 160.50/158.97/155.29x | 没有稳定收益；结论限该实验，不改默认 |
| BF16 权重 ND/NZ 微基准 | QKV、gate/up 的实际 shape 基本相同或 NZ 略慢，数值一致 | 不接入默认；不能把有 NZ 开关当成收益 |
| 关闭 NPUGraphEx | 长音频 C1/C8/C32 33.96/173.66/178.23x | 未优于原路径，保留 NPUGraphEx |
| ATB PA 直接使用设备端长度 | 本机报 `tensor.hostData is null`，无法 setup | 不能直接省略其 Host 更新 |
| Triton 设备端分页注意力 | kernel 层数值和动态回放通过；正式 RPC 24 格全部提升 | 支持 Host 更新是主要瓶颈的判断 |

本机 NPU4 的拓扑亲和 CPU 是 0–31，而上游逻辑 NPU0 的绑定扩展到 2–61，ACL/release 线程在 62/63。该现象有现场证据，但已有实验没有证明把亲和性作为默认能提高本负载性能。

这里只归因当前工作区与当前栈的热点，没有补做同代码、同配置的旧 Torch/CANN 栈 trace；因此不能把历史升级回退单独归因于 Torch 2.9 或某个 CANN 算子。历史 20% 环境回退线也不是本次优化验收线。

后续已补齐原生隔离的新旧 Worker trace、CANN 交叉和同实例 C++ FIA 图更新 A/B/A，具体退化原因与因果证据见[旧软件栈为何更快](ascend-stack-regression-20261016.md)。以上“未补做”是本报告第一阶段的证据边界。

## 复现

```bash
source scripts/env_ascend_291.sh
poetry run python scripts/diagnose_asr.py engine \
  --model model-bin/Qwen/Qwen3-ASR-1.7B --slots 8 \
  --memory-limit-mib 16384 --output /home/qwen-asr-hotspots/new-profile
poetry run python scripts/summarize_ascend_hotspots.py \
  /home/qwen-asr-hotspots/new-profile --output /home/qwen-asr-hotspots/new-summary.json
```

运行前确保设备没有其他模型实例；正式服务和测试单元在本次任务结束时的最终状态见[优化报告](ascend-910b-optimization-20261016.md)。
