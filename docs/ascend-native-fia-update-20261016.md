# Qwen3-ASR-1.7B：在新软件栈中修复 FIA 图更新开销

本轮固定使用 **Torch/torch-npu 2.9.1、vLLM/Ascend 0.18、CANN 9.1.1**，将已确认的 FIA Host 图更新优化接入正常 ASR Worker。依赖包、运行库和驱动保持现状，其他引擎依赖的新栈保持原样。旧栈仅复用此前保存的转写作为对齐参照，本轮没有启动旧环境。

原始证据：`/home/qwen-asr-newstack/20261016-105439`。完整指标、校验和环境保护结论见[JSON](ascend-native-fia-update-20261016.json)。退化原因及底层因果实验见[根因报告](ascend-stack-regression-20261016.md)。

## 实现与范围

`src/qwen_asr/engines/vllm/ascend/fia_update.py` 为当前 runner 派生局部 attention backend，仅在该 runner 的 `update_full_graph_params` 中调用 torch-npu 已有的 C++ FIA 图更新入口。每次仍更新 28 层，动态长度、query 长度、页表、workspace、输出、事件和注意力计算均保持原语义。上游[对应 C++ 实现](https://github.com/Ascend/pytorch/blob/40cf4622655278625ea7203bb6db8ff90d715c9e/torch_npu/csrc/npu/Graph.cpp#L614)将 FIA、图更新与事件记录合为一次原生调用；本机可用性及效果由本轮实际运行确认。

派生类绑定在 ASR runner 上，全局 backend 注册和原类方法保持原样。该适配进入正常 CLI、配置构造和 `SharedNPUWorker.load_model`，启动后才进行图捕获。无需安装包补丁或重新构建 Torch/CANN。

- 显式使用 `--no-triton-decode --native-fia-update` 或相应环境变量。默认关闭；本机已有的 Triton 解码配置保留。
- 验证范围：910B、1.7B BF16、Q16/KV8/D128、28 层、上下文 1024、<=8 执行槽、FULL_DECODE_ONLY、TP/PP/DP/CP1，无推测解码或 KV 传输。
- 启动前校验已测版本及 C++ 接口；每轮校验完整的 28 层 FIA 图参数格式，异常在修改图任务前报错。PA/draft 更新保留上游方法，单元测试覆盖此分支；本轮模型采用 FIA。
- 日志记录安装及首次实际命中；诊断接口导出更新次数。关闭 `--native-fia-update` 后使用上游原调用，Triton 与原生 FIA 更新分开选择。

## 同卡、同新栈的完整 RPC 比较

物理 NPU 4，Ascend910B4-1，驱动 26.1.1。直接运行独立的原生进程，无容器或 vNPU。一个 1.7B BF16 模型、8 执行槽、32 RPC 容量、1024 上下文和调度 token、KV 896 MiB、融合 RoPE、async scheduling、捕获 1..8；前缀与多模态结果缓存关闭。输入为同一 Rita PCM 循环扩展，最大 30 秒。

新代码基底 `8841ad5` 加本次增量，`source-files.json` 保存 130 个代码、测试及依赖声明文件的 SHA256，并验证测试副本与提交工作区逐文件相同。运行通过独立 Poetry，UID 65534；原始环境和正式配置不具备写权限。

顺序运行 A1 原 FIA、B 原生融合 FIA、A2 恢复原 FIA，再测试 Triton。每组 3 轮、每种输入预热 2 轮，关闭 profiler；低并发每轮至少 4 请求。下表为 30 秒音频的吞吐倍速中位数，定义为成功音频总秒数/整轮墙钟时间。收益对较快的 A 阶段计算。此处为完整 gRPC 口径，不能直接与根因报告的直接 Engine 数字比较。

| 负载 | A1 原 FIA | B 融合 FIA | A2 原 FIA | B 收益 | Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 32.578 | 39.078 | 32.288 | +19.95% | 58.539 |
| C32 | 174.002 | 195.003 | 174.969 | +11.45% | 204.808 |

两个负载中，B 阶段最慢的一轮仍快于两次 A 阶段的所有轮次；P95 首字和总延迟也下降，完整值见 JSON。

融合 FIA 路径的 24 组矩阵如下；每格是 3 轮中位倍速。A 阶段和 Triton 本轮只复测原长 5.1299375 秒、30 秒的 C1/C32，其他负载的完整比较仍见此前[优化验收报告](ascend-910b-optimization-20261016.md)。

| 音频秒数 | C1 | C2 | C4 | C8 | C16 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.1299 | 20.74 | 30.49 | 58.42 | 110.46 | 115.10 | 114.37 |
| 10 | 28.80 | 47.32 | 89.99 | 168.81 | 173.36 | 173.91 |
| 20 | 36.13 | 60.77 | 117.57 | 177.52 | 178.69 | 179.72 |
| 30 | 39.08 | 69.09 | 116.27 | 186.77 | 190.80 | 195.00 |

融合 FIA 路径的选定 P95，单位 ms；C1 为 12 个请求，C32 为 96 个请求，完整 P50/P95/P99 和逐轮值保存在 JSON/raw.jsonl 中。

| 负载 | P95 首字 | P95 总延迟 |
| --- | ---: | ---: |
| C1 | 163.41 | 806.07 |
| C32 | 4235.49 | 4928.00 |

本轮将两条优化路径分别测量。Triton 直接从设备元数据执行分页注意力，原生 FIA 路径减少供应商图更新的 Host 调用开销，两者的收益不能相加。Triton 在本轮复测的负载中吞吐更高，因此继续保留本机 1.7B 的现有 Triton 配置；原生 FIA 提供使用供应商注意力算子的可选优化路径。剩余 CANN kernel 与 runner 准备开销的证据保留在根因报告中，不能据此宣称所有新旧差距均已消除。

## 正确性、资源与环境保护

- 17 项相关单元测试通过：动态页表/长度采用当前元数据、runner 隔离、异常图提前报错、PA 回退、工具链与 CLI 参数约束、现有 Triton 图行为及 Worker 内存预算。
- 正常 Worker 配置经诊断入口完成 57 个模型请求，全部生成 81 个 token，token ID 和转写逐一完全一致。实际命中 636 轮 28 层 FIA 更新，上游回退为 0；实际捕获 8 个完整解码图，并在推理中回放。
- 完整 RPC 测试包含 1464 个测量请求及 160 个预热请求，错误为 0，转写与保存的旧栈基线逐字相同。原生融合 FIA 的完整矩阵包含 816 个测量请求。合计本轮 1681 个模型/RPC 请求。
- 整卡 HBM 峰值 9775 MiB，最多 1 个 NPU 模型进程，低于 16 GiB。动态库记录确认使用独立的新 CANN/torch-npu 副本。
- 原环境 287745 项文件、目录和链接的保护清单无变化，正式配置 SHA256 无变化。`protected-changes.json` 为空。
- 测试结束后 NPU 无运行进程；ASR/LLM 正式服务均 inactive/MainPID=0。测试端口 50021 已关闭，正式服务保持停止。

## 复现

本轮脚本、原始数据、独立源码快照、源码校验清单和完整命令均保留在原始目录。以下入口内部使用新栈的独立 Poetry 环境，结果目录须尚不存在，测试须独占空闲 NPU：

```bash
TASK=/home/qwen-asr-newstack/20261016-105439
ISO=/home/qwen-asr-isolated/20261016-093250

"$TASK/run.sh" python -m unittest tests.test_ascend_fia_update tests.test_ascend_worker \
  tests.test_ascend_triton_config tests.test_cli_configuration -v

"$TASK/run.sh" python scripts/diagnose_asr.py engine \
  --output /home/qwen-asr-newstack/repro-native-fia --model "$ISO/model" \
  --slots 8 --clients 1 --clients 8 --clients 32 --rounds 3 \
  --no-short-audio --measure-only --production-native-fia --memory-limit-mib 16384
```

完整 gRPC 启动和基准命令见 `*-command.json`、`*-benchmark-command.json`，由 `validate.py` 进行就绪等待、显存监视和进程组清理。原始输出目录包含本轮证据，复现时应复制脚本并使用新的结果目录。
