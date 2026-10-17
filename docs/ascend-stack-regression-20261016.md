# Qwen3-ASR-1.7B：旧软件栈为何更快

本次已把差距定位到 **FIA 解码注意力的 Host 图参数更新路径**，并做了真实模型的因果验证。新 Torch／torch-npu 组合下，逐层分开执行 Python 调用、算子分发和队列等待的成本升高；当前 vLLM-Ascend 仍走这条路径。CANN 9.1.1 的 FIA 设备执行和 Host API 也有退化。此外，新 runner 的元数据、上下文准备增加了部分成本。

旧栈也做相同的 28 层更新。决定差距的是相同工作的耗时，以及 Host 下发与 NPU 执行能否及时衔接。

原始证据：`/home/qwen-asr-root-cause/20261016-100947/`。机器可读汇总：[ascend-stack-regression-20261016.json](ascend-stack-regression-20261016.json)。此前完整 RPC 矩阵见[原生隔离对照](ascend-isolated-comparison-20261016.md)；本文新增的模型因果实验采用直接 Engine 口径，两种口径的绝对倍速不能混用。

## 对照条件

- 物理 Ascend910B4-1 卡 4，进程内逻辑卡 0，驱动 26.1.1。直接使用物理 NPU，没有容器或 vNPU。
- 使用 `/home/qwen-asr-isolated/20261016-093250/` 中独立复制的 Python、Poetry、环境、vLLM、CANN、ATB、模型和应用源码。测试进程为 UID 65534，不能写入原环境。
- 应用快照 `7409390`，与当前应用实现一致；同一份 1.7B BF16 权重、30 秒 Rita 音频。
- 一个模型、8 执行槽、上下文和调度 token 上限均为 1024、KV 896 MiB、async scheduling、FULL_DECODE_ONLY 捕获 1..8、融合 RoPE。前缀缓存关闭，处理结果缓存为 0。
- 旧组合：Torch／torch-npu 2.8.0、vLLM 0.13、Ascend 0.13.0rc1、CANN 8.3.RC1。新组合：2.9.1、vLLM／Ascend 0.18、CANN 9.1.1。它们是现场保留的安装组合。

## 1. 大部分增加的时间是 NPU 空闲

真实 Worker 的 CPU/NPU trace 覆盖相同请求。跨 stream 合并实际 kernel 区间后：

| 负载 | 栈 | 首尾 kernel 窗口 ms | kernel 忙时并集 ms | 窗口内空闲 ms |
| --- | --- | ---: | ---: | ---: |
| C1 | 旧 | 981.232 | 422.642 | 558.590 |
| C1 | 新 | 1241.534 | 451.571 | 789.964 |
| C32 | 旧 | 5542.550 | 2583.753 | 2958.797 |
| C32 | 新 | 6845.249 | 2681.266 | 4163.984 |

C1 窗口增加约 260 ms，其中约 231 ms 是空闲增加；C32 增加约 1303 ms，其中约 1205 ms 是空闲增加。上述是 profiler 窗口，不能当无插桩端到端成绩。

两栈均记录到 C1 81 次、C32 312 次 decode 迭代，每次更新 28 层。逐层路径为：

```text
graph_task_update_begin
    → npu_fused_infer_attention_score.out
    → graph_task_update_end
    → ExternalEvent.record
```

FIA 使用 Host 端长度列表。长度随解码变化，即使模型计算已经捕获成图，仍需更新图中注意力参数。

## 2. 具体增加在哪里

同口径 C1 cProfile 中，28 层图更新的 Host 中位耗时：

| 实际模型组合／调用方式 | 每次 decode 更新 ms |
| --- | ---: |
| 旧 Torch／旧 vLLM／CANN 8.3 | 3.929 |
| 旧 Torch／旧 vLLM／CANN 9.1.1 | 4.842 |
| 新 Torch／新 vLLM／CANN 9.1.1，分开调用 | 6.221 |
| 同一新栈实例，改用 C++ 融合更新 | 4.903 |

旧／新栈各有 2268 次 FIA 更新。分开调用时，FIA builtin 平均 Host 耗时约 73.2→103.6 μs，`graph_task_update_end` 约 18.5→52.9 μs。CANN trace 中，`aclnnInnerFusedInferAttentionScoreGetWorkspaceSize` 平均约 13.4→27.6 μs，执行接口约 28.9→45.6 μs。

这些计时包括提交和等待。关闭任务队列后，部分耗时从 `update_end` 移到 FIA 调用，总更新时间没有相应大幅下降。因此不能把 `update_end` 的全部时间解释为 NPU 计算或某一个 CANN 同步 API 本身变慢。

原生 CPU 采样还能看到 `op_api::npu_fused_infer_attention_score_out_symint`、`Repository::Dequeue/ReadQueue/MakeSureQueueEmpty`、CANN workspace/launch 链路。新外部事件使用 ValueWrite 路径；单独换成旧 ACL 外部事件，仅使最小复现的更新约 5.17/5.21→4.71/4.90 ms，不能独立解释全部退化。

## 3. 同实例只换调用路径，性能随之变化

本机新 torch-npu 已导出 `_npu_fused_infer_attention_score_out_graph`。上游[对应 C++ 接口](https://github.com/Ascend/pytorch/blob/40cf4622655278625ea7203bb6db8ff90d715c9e/torch_npu/csrc/npu/Graph.cpp#L614)将参数解析、原生 FIA、图更新和事件记录合在一次调用中，并在执行部分释放 GIL。这里引用的是上游实现；本机是否可用、性能如何，以实际导出接口和测试为准。

诊断 Worker 仅替换当前 `update_graph_params` 中上述四步，保留原循环、元数据、长度、页表、workspace、精度、图和输出。没有省掉 28 次注意力更新。替换源码保存在 `engine-current-native-fia/native-fia-update-source.py`。

关闭 profiler 后，同一实例依次运行 A 原调用、B 融合调用、A 恢复原调用，各 3 轮：

| 30 秒音频 | A | B | 恢复 A | B 相对较快的 A |
| --- | ---: | ---: | ---: | ---: |
| C1 吞吐倍速中位数 | 32.600 | 38.240 | 33.449 | +14.32% |
| C32 吞吐倍速中位数 | 186.015 | 208.794 | 191.429 | +9.07% |

B 阶段最慢的一轮仍快于两个 A 阶段最快的一轮；完整逐轮值保存在 JSON 中。

这证明分散的图更新调用路径造成了实质性的端到端损失。独立最小复现也验证了相同方向：真实 ASR 的 Q `[B,16,128]`、KV `[64,128,1024]`、int8 mask、BF16、28 层，Host 更新中位数为：

| 调用方式 | B1 ms | B8 ms |
| --- | ---: | ---: |
| 旧栈 | 4.076 | 4.576 |
| 新栈分开调用 | 5.085 | 5.264 |
| 新栈 C++ 融合调用 | 3.373 | 3.581 |

最小复现没有 vLLM、模型权重、调度、音频前端或 IPC，进一步将该差异限定在底层图更新调用路径。它不能替代真实模型的吞吐结果，也不用于逐项线性分摊整段延迟。

## 4. CANN 是另一部分，不能独自解释全部差距

固定旧 Torch 和旧 vLLM，只在测试进程切换独立 CANN 副本；实际加载库已核对：

| 直接 Engine、中位倍速 | C1 | C32 |
| --- | ---: | ---: |
| 旧组合、CANN 8.3，第一次 | 42.154 | 232.080 |
| 旧组合、只换 CANN 9.1.1，5 轮 | 39.582 | 216.050 |
| 切回 CANN 8.3，5 轮 | 42.231 | 226.092 |

相对切回后的对照，CANN 升级使 C1 吞吐下降约 6.27%、C32 下降约 4.44%；另一次 CANN 9.1.1 模型测试为 39.972/215.261x。其影响可逆，但小于完整新栈的差距。

实际 decode FIA kernel 中位耗时，C1 约 20.10→27.10 μs，C32 约 41.14→51.36 μs；其他 kernel 有快有慢，整段 kernel 忙时只小幅增加。单独的 CANN 交叉微基准也重现了 FIA 静态图设备跨度的增加。

新栈融合调用后仍低于旧栈。相同 CANN 9.1.1 下，融合后的图更新 4.903 ms 已接近旧框架的 4.842 ms。扣除图更新后，`execute_model` 的其余 Host 段仍有额外开销；新 runner 的 `_build_attention_metadata`、`_pad_query_start_loc_for_fia`、上下文准备等在 CPU profile 中可见。其子区间嵌套，不能直接相加，也没有将某一个准备函数单独替换后宣称全部归因于它。

新 Torch 配旧 CANN 的独立 FIA 测试通过，但完整 vLLM-Ascend 0.18 因原生库要求 `aclrtGetHardwareSyncAddr` 等新接口而无法启动。该失败已保留，不能把这个交叉组合当作可部署方案。

## 排除与验证范围

- 全线程 CPU affinity 改成 0..63 的 A/B/A 未消除差距：新栈 C1 32.530/32.769/33.478x，C32 182.970/186.967/181.548x。绑核不是主要原因。
- 新旧都实际捕获并回放 1..8，实际 decode 更新次数相同。不能用“只有新栈有 28 次更新”解释回退。
- 本轮 1667 个实际模型请求全部成功，均生成 81 个 token，token ID 和转写逐一完全一致。
- 微基准完成 1512 次逐层动态输出校验、54 次独立 CPU FP32 SDPA 对照；覆盖长度 127/129/383、页表翻转、query 更新。最大 FP32 绝对误差 0.001523，低于预设容差。
- 测试只支持本模型、硬件和已测负载的判断，不构成通用准确率或所有模型的兼容性结论。

## 复现与交付

新增 `scripts/benchmark_fia_graph_update.py`，支持 `--native-update`；增强诊断 Worker 和 `scripts/diagnose_asr.py`，支持旧 runner 分段、图参数元数据与 `--native-fia-ablation`。生产默认推理路径和服务配置没有修改。

以下入口内部通过独立 Poetry 执行。脚本副本与本次提交内容相同；结果目录必须尚不存在，测试需独占空闲 NPU：

```bash
ISO=/home/qwen-asr-isolated/20261016-093250
DIAG=/home/qwen-asr-root-cause/20261016-100947/repro-scripts

"$ISO/run.sh" current python "$DIAG/benchmark_fia_graph_update.py" \
  --output /home/qwen-asr-root-cause/repro-split --steps 300
"$ISO/run.sh" current python "$DIAG/benchmark_fia_graph_update.py" \
  --output /home/qwen-asr-root-cause/repro-fused --steps 300 --native-update
"$ISO/run.sh" current python "$DIAG/diagnose_asr.py" engine \
  --output /home/qwen-asr-root-cause/repro-model \
  --model "$ISO/model" --slots 8 --clients 1 --clients 32 --rounds 3 \
  --no-short-audio --measure-only --native-fia-ablation --memory-limit-mib 16384
```

原始目录中的 `legacy/`、`current/` 包含 Worker 分段、cProfile 和完整 NPU trace；`engine-legacy-cann911*` 与 `engine-legacy-return/` 为 CANN 交叉；`engine-current-native-fia/` 为同实例 A/B/A；`asr-shape-*/` 是最终微基准；`verified-*/` 和 `perf-*/` 保留其他对照及原生 CPU 调用栈。

整卡 HBM 采样峰值 10897 MiB，全程至多一个 NPU 进程，低于 16 GiB 限制。结束后测试进程已退出，NPU 无运行进程；ASR/LLM 正式服务保持 inactive，50011/8100/50021 均未监听。

原环境 287745 项文件／目录／链接的保护清单无变化。原应用及依赖声明 69 个文件、冻结副本 104 个文件、原模型及副本各 15 个文件的 SHA256 均未变化，见 `application-model-verification.json`；环境保护清单见 `protected-changes.json`。没有安装、升级或替换原环境中的依赖、CANN、ATB 或驱动。

后续已将融合 FIA 更新接入正常 ASR Worker，保持 Torch/torch-npu 2.9.1、vLLM/Ascend 0.18、CANN 9.1.1。源码接入、完整 RPC 对照和默认路径选择见[新栈 FIA 更新验证](ascend-native-fia-update-20261016.md)。
