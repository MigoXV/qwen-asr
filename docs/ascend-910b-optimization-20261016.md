# Qwen3-ASR-1.7B / 910B 性能优化验收

**保持原 1.7B BF16、单模型、8 执行槽、32 RPC、1024 上下文及 896 MiB KV，正式 RPC 的 24 组负载全部提速，816/816 请求成功且转写与当前源码基线逐字一致。** 30 秒单并发吞吐 34.52→58.74 倍速，P95 总延迟 885.5→512.5 ms；30 秒、32 并发 181.27→206.68 倍速，P95 总延迟 5299.6→4655.0 ms。

热点和采样口径见[热点报告](ascend-910b-hotspots-20261016.md)。原始证据根目录为 `/home/qwen-asr-hotspots/20261016-002458/`，下文简称 `TASK`。逐请求数据、未插桩结果、插桩结果和失败尝试分别保存，没有把 profiler 数字当作稳态验收成绩。

后续以同一份ASR源码、原生独立软件副本完成的新旧栈对照见[隔离对照报告](ascend-isolated-comparison-20261016.md)。该轮未使用容器或vNPU，单独记录低权限进程、清空环境变量后的结果；不与本文较早的运行现场混算。

## 实现与启用范围

最终推理源码提交为 `a2a0609`，其父提交 `2e6b9a9` 包含重复清理优化。基线为兼容修复后的 `94992c9`；用户已有重构已提交为 `2943cb0`。`candidate-source-sha256.txt`、`candidate-code.tar.gz` 记录测量对应的完整工作区，不只依赖基准 metadata 中的 HEAD。

诊断与数值/稳态验证工具单独提交为 `97213fb`，大型 trace、模型和编译缓存保留在数据盘，不进入 Git。

`engines/vllm/ascend/paged_decode.py` 实现两个 Triton kernel：GQA 的两个 query head 共享 K/V 加载，以设备端长度分配四个分区，在 64 token tile 内计算 FP32 softmax/累加，合并后输出 BF16。页表和真实长度都从设备张量读取，因此解码图不再为 28 层逐层重做 Host tiling/task update。

`triton_decode.py` 在当前模型实例的 attention 实现上安装该路径，encoder 和 prefill 仍走原实现。图 profile 按 1..8 捕获；运行前校验页表、长度和 KV 缓冲签名，元数据地址变化时退到 eager 并记录原因。每层、每个 batch 的工作区由单一 Worker 持有。小的 query/页表/output 非连续视图可归一化；不复制整份 KV。未验证的 KV 布局会在编译前明确拒绝，可用原 attention 路径运行。

启用条件是 Qwen3-ASR-1.7B 的 Q16/KV8/D128、28 层、BF16、910B、TP/PP/DP=1、FULL_DECODE_ONLY、1024 上下文和不超过 8 槽；版本限定本机已验证的 vLLM 0.18.0、torch-npu 2.9.1、Triton-Ascend 3.2.2。包级默认不开启；本机 1.7B 服务环境已显式开启。

```bash
bash scripts/serve_ascend.sh \
  --model model-bin/Qwen/Qwen3-ASR-1.7B --workers 8 \
  --max-concurrent-rpcs 32 --max-model-len 1024 \
  --max-num-batched-tokens 1024 --memory-utilization 0.11 \
  --triton-decode
```

回退使用 `--no-triton-decode`，或将 `/etc/qwen-asr-ascend.env` 中 `QWEN_ASR_ASCEND_TRITON_DECODE=false` 后重启。原环境备份在 `TASK/original-qwen-asr-ascend.env`。本任务按用户后续要求保持正式服务停止，不会为了启用新配置重新启动它们。

## 路线调研与实际结论

访问日期为 2026-10-16。文档/仓库支持、编译成功、数值通过和本机端到端收益是四个不同层次；下表只对实际完成的层次作结论。本机源码组合不是官方原版兼容矩阵。

| 路线与一手来源 | 当前栈及图兼容性 | 实测与取舍 | 维护成本 |
| --- | --- | --- | --- |
| [vLLM-Ascend 图模式](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/feature_guide/graph_mode.html)、[NPUGraphEx](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/feature_guide/npugraph_ex.html) | 本机默认开启，实际捕获/回放 8 档 | 关闭后 C8/C32 更慢，保留现有编译与完整图；没有另换推理引擎 | 低，沿用上游 |
| [上游 PA 配置](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/configuration/additional_config.html) | FULL_DECODE_ONLY 支持 pa_shape_list | 实机开启 1..8 更慢；ATB PA 要求 Host 长度张量，不能直接跳过更新 | 低，已测试但不采用 |
| [上游 CPU binding](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/feature_guide/cpu_binding.html) | ARM/910B 适用；不改变数值 | PA 实例的线程亲和性 ABA 未显示稳定收益，保持原设置 | 低 |
| [BF16 NZ 开关及发布说明](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/release_notes.html) | 本机 0.18 的 maybe_trans_nz 支持显式 BF16 NZ；图内可运行 | 真实 QKV/gate-up 权重微基准无收益，不改变默认布局 | 低 |
| [Triton-Ascend 仓库](https://github.com/triton-lang/triton-ascend)、[3.2.2 的 910B 镜像说明](https://github.com/triton-lang/triton-ascend/blob/main/docker/OVERVIEW.md) | 官方有 A2/910B 支持；本机 3.2.2 已实际编译、数值及 graph replay 通过 | 实现匹配热点的设备端分页解码，正式端到端有效；没有重写供应商 GEMM | 中，保留源码、版本限制、shape 检查和回退 |
| [Ascend C 自定义算子工程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0060.html) | 已核查本机 CANN 9.1.1；可以建立 AOT 自定义算子工程 | 本次未手写新 Ascend C：Triton 已解决主热点并通过实际模型验证，增加另一份同算法内核缺少额外收益证据。CANN/ATB 现成路线已实测 | 新增实现成本高于当前已验证路径 |
| [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend) | 查证存在 Ascend 适配和 910B 路线；没有在本环境编译运行 | 不宣称本机支持或性能成绩；当前局部 Triton 路线已满足目标，没有另增工具链 | 未引入 |

调研时上游已公告 Triton-Ascend 3.6.0，推荐软件组合与本机 3.2.2 不同；没有覆盖现有环境升级它。本次也没有量化、缩短输出、降低槽位、缩小 KV 或换模型来获得提速。

## 原型与排除证据

`experimental_paged_attention.py`、`ascend_triton_attention.py` 和 `benchmark_triton_paged_attention.py` 保存原型。最终产品代码使用 `paged_decode.py`，另以 `--production` 重新验证，避免只证明原型可运行。

| 实验 | 结果 | 结论 |
| --- | --- | --- |
| 每个 query head 独立 split-KV | 288 次数值/动态回放通过；长度384，B1/B8 约36.47/217.31 μs | 可消除 Host 更新，但大 batch kernel 需要优化 |
| 两个 GQA head 共享 KV，tile128 | UB 需要 2117632 bits，设备可用 1572864 bits，编译失败 | 保留最小错误与 IR；改为 tile64 |
| GQA tile64、16 分区 | 288 次通过；长度384，B1/B8 约21.16/101.93 μs | 首个真实 Worker 原型：长音频 C1/C8/C32 为59.85/212.17/220.03x，token 序列对齐 |
| 固定四分区、分区内在线 softmax | 长度384 B8 约165.97 μs | 负载不平衡，未选用 |
| 四分区按设备长度分配 tile | 连续布局 288 次通过；长度384 B1/B8 约18.46/92.50 μs；长度64约12.59/54.71 μs | 最终方案 |
| 非连续 KV 压力测试 | 本机 3.2.2 产生 undefined symbol: malloc，链接失败 | 限定生产 KV 为连续 ND；小视图归一化后再验证 |
| ATB PA 设备长度原型 | tensor.hostData is null | 当前 API 路径需要 Host 长度，未接入默认 |

部分最早失败来自基准 reference 没有传入 FIA mask、没有使用 Worker 的 KV flatten 视图；修复后重新验证。它们是测试搭建错误，不作为 Triton 能力缺陷证据。

最终 kernel 与 FIA 比较使用 atol=0.03125、rtol=0.02；24 个额外 FP32 对照使用 atol=0.015625、rtol=0.02。288 次连续布局验证中，相对 FIA 最大绝对误差0.015625，相对 FP32 最大绝对误差约0.006381；每例最大/平均误差见 `triton-pa-production-v2/correctness.json`。另有12个非连续 query/页表/output 组合及零长度 padding 回放验证。

冷缓存首次编译与装载合计约32.54秒，稳态 kernel 数字不包含这项成本；服务启动还包含模型加载和图捕获。缓存及失败 IR 均写入数据盘，源码、模型、实际构建缓存的 SHA256 清单保存在 TASK。

## 正式 RPC 前后对比

两次完整矩阵各 816 请求、24 格各3轮，均预热2轮。下面逐格比较相同音频长度，不能直接用不同长度的吞吐倍速比较模型性能。首字指客户端第一次收到可见转写。

| 音频秒 | 并发 | 吞吐倍速 基线→候选 | 提升 | 首字P95 ms 基线→候选 | 总P95 ms 基线→候选 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.130 | 1 | 18.94→28.09 | +48.3% | 140.9→119.6 | 273.1→183.5 |
| 5.130 | 2 | 28.89→36.70 | +27.0% | 229.5→212.8 | 366.2→292.6 |
| 5.130 | 4 | 55.15→68.87 | +24.9% | 245.0→226.6 | 385.1→308.9 |
| 5.130 | 8 | 103.40→126.14 | +22.0% | 253.9→232.6 | 397.1→327.8 |
| 5.130 | 16 | 107.24→130.82 | +22.0% | 625.5→541.6 | 768.9→636.9 |
| 5.130 | 32 | 109.58→133.14 | +21.5% | 1379.3→1163.7 | 1522.7→1259.0 |
| 10.000 | 1 | 26.26→41.13 | +56.6% | 141.7→122.1 | 385.5→244.0 |
| 10.000 | 2 | 42.61→58.16 | +36.5% | 232.1→211.4 | 484.5→348.1 |
| 10.000 | 4 | 80.99→106.55 | +31.6% | 248.6→230.0 | 499.0→380.9 |
| 10.000 | 8 | 152.28→190.65 | +25.2% | 266.3→246.2 | 525.2→422.6 |
| 10.000 | 16 | 157.33→194.96 | +23.9% | 756.2→644.1 | 1016.3→820.7 |
| 10.000 | 32 | 158.22→197.76 | +25.0% | 2071.6→1461.5 | 2333.3→1639.0 |
| 20.000 | 1 | 31.77→53.00 | +66.8% | 155.2→133.7 | 639.8→379.3 |
| 20.000 | 2 | 55.43→79.06 | +42.6% | 249.9→228.9 | 749.3→514.0 |
| 20.000 | 4 | 105.24→139.19 | +32.3% | 273.1→245.7 | 769.4→574.8 |
| 20.000 | 8 | 161.32→186.93 | +15.9% | 481.8→455.4 | 990.3→855.3 |
| 20.000 | 16 | 165.43→189.14 | +14.3% | 1423.2→1289.1 | 1940.5→1691.8 |
| 20.000 | 32 | 166.47→191.35 | +14.9% | 3354.7→2951.2 | 3868.0→3357.0 |
| 30.000 | 1 | 34.52→58.74 | +70.2% | 167.0→143.1 | 885.5→512.5 |
| 30.000 | 2 | 61.17→89.38 | +46.1% | 266.7→240.2 | 995.4→682.4 |
| 30.000 | 4 | 105.84→139.69 | +32.0% | 382.9→356.6 | 1140.7→859.0 |
| 30.000 | 8 | 174.51→201.10 | +15.2% | 609.3→580.5 | 1374.6→1194.4 |
| 30.000 | 16 | 179.04→204.21 | +14.1% | 1904.1→1738.4 | 2680.0→2353.7 |
| 30.000 | 32 | 181.27→206.68 | +14.0% | 4527.8→4034.9 | 5299.6→4655.0 |

全部 P50/P95/P99、吞吐轮次范围和 HBM 汇总另见[机器可读结果](ascend-910b-results-20261016.json)。数据来自未开启 profiler 的正式 RPC 路径。源目录分别为 `baseline-full/` 和 `candidate-full/`，重复基线在 `control-repeat/`。

## 正确性、稳定性与资源

除 816 请求矩阵外，官方中英文样本与四种 rita 时长混合，通过1..8、16、32并发验证，并继续运行60轮32并发，共2172请求，全部成功，转写均属于原 FIA 对照集合。原 FIA 的混合负载中，30秒 rita 本身有两种转写变体；同条件候选没有增加新变体。

附加音频来自[Qwen 官方示例](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_forced_aligner.py)，分别4.204秒中文和15.051秒英文；原文件 URL、SHA256、采样率及格式在 `evaluation-audio/sources.json`。这只是针对性对齐，不构成完整语料的准确率结论。

服务层已经验证：32条占用中的 RPC 会拒绝第33条；首字后取消、1ms deadline、取消/超时后的新请求均符合预期，见 `candidate-lifecycle.json`。图地址保护测试同时验证了内容原地变化继续回放、地址变化退到 eager。

重复清理改为有界匹配，保留原阈值、最短模式优先和尾部扫描边界。穷举、随机 Unicode、换行、周期和流式前缀均与原函数差分一致。CPU 微基准约2.64倍，但同实例 A/B/A 吞吐基本相同；只报告 CPU 工作减少，不把它算作独立端到端收益。

完整矩阵的吞吐提升范围为14.0%–70.2%；6格 FIA 重复对照的最大吞吐漂移约4.98%，见 `control-repeat/`。整卡 HBM 采样峰值9772→9567 MiB，包含驱动占用，保持在16384 MiB硬上限内。

同一物理设备独占测试，没有人为锁频。完整矩阵的温度范围为基线70–80°C、候选69–85°C；功耗范围94.2–162.2 W、90.3–184.2 W。长音频C32诊断 trace 的频率采样分别包含800/1650 MHz和1650 MHz，见 `hardware-conditions.json`。这些采样不是逐请求固定时钟证明，因此不能把全部端到端提升定量归因于某一种因素；Host task-update 调用消失和 Host 阶段减少则有独立证据。重复基线用于检查漂移，不能替代锁频实验。

官方中文样本参考13字，CER为0；英文参考36词，WER为2/36≈5.56%。候选与 FIA 转写一致，因此相对这两条参考的错误率也相同，不能据此宣称整体识别准确率提升。

最终生产实现的真实 Worker 补采覆盖原音、10/20/30秒及官方中英文。输出 token 分别为18/30/55/81和11/49，均为 `stop` 结束；没有通过截断生成提速。8档图均已捕获，运行结束的 `eager_fallbacks=0`。长音频 C32 trace 的计算忙时占比39.5%→63.4%，逐层 graph-task-update 调用8736→0，签名保护仍保留。原始输出、结束原因、批次和最终 Worker 状态在 `final-profile/rounds.jsonl`、`final-profile/final-worker-info.json`；汇总为 `final-hotspot-summary.json`、`final-verification.json`。

全量单测运行58项：49通过、7跳过、2失败。失败是基线已有的模型模块导入断言：本机 Torch backend autoload 和 Transformers 的 NPU 集成会间接导入 `torch_npu`，详见 `import-side-effect.txt`、`final-unit-tests-v2.log`。没有改写断言以掩盖环境副作用。新增的配置、版本限制、图地址回退和重复清理差分测试均已通过；图保护3项的独立复验在 `graph-guard-tests.log`。

构建与追溯清单还包括 `model-sha256.txt`、`runtime-triton-build-sha256.txt`、`operator-cold-build-sha256.txt`、`ascend-critical-source-sha256.txt`、`vllm-source-head.txt` 和 `vllm-source-existing.diff`。后两者记录已有 editable vLLM 的源码与补丁，没有在本次任务中修改第三方安装目录。最终推理源码再次逐文件核对候选清单通过。

## 复现与运维

```bash
source scripts/env_ascend_291.sh
poetry run python scripts/benchmark_triton_paged_attention.py \
  --production --block 64 --output /home/qwen-asr-hotspots/new-kernel-check
poetry run python scripts/benchmark_asr.py --workers 8 \
  --requests-per-client 1 --min-requests 4 --memory-limit-mib 16384 \
  --execution-mode triton-gqa-online64-parser-regex \
  --output /home/qwen-asr-hotspots/new-rpc-benchmark
poetry run python scripts/validate_asr_service.py \
  --output /home/qwen-asr-hotspots/new-lifecycle.json
```

模型推理、benchmark 和 source 环境均使用 Poetry。`scripts/profile_asr_api.py` 只用于诊断，SIGUSR1/SIGUSR2 控制采样、SIGWINCH 控制重复清理的同实例消融；正常启动仍使用 `scripts/serve_ascend.sh`。

复现矩阵前需要自行启动上述服务命令；benchmark 的 `--execution-mode` 只是记录标签，不会更改服务配置。补采生产路径可以使用：

```bash
poetry run python scripts/diagnose_asr.py engine \
  --model model-bin/Qwen/Qwen3-ASR-1.7B --slots 8 \
  --production-triton --validate-lengths --clients 32 \
  --evaluation-audio /home/qwen-asr-hotspots/20261016-002458/evaluation-audio \
  --memory-limit-mib 16384 --output /home/qwen-asr-hotspots/new-final-profile
poetry run python scripts/validate_asr_optimization.py \
  --audio-root /home/qwen-asr-hotspots/20261016-002458/evaluation-audio \
  --baseline /home/qwen-asr-hotspots/20261016-002458/mixed-control/summary.json \
  --rounds 3 --soak-rounds 60 --output /home/qwen-asr-hotspots/new-mixed-check
```

本次没有恢复正式 ASR 或 LLM，这是用户后续要求停止全部 NPU 服务的结果。所有诊断单元、NPU 模型进程和50011/8100监听在结束时清理；两项正式服务均为inactive/dead、MainPID=0，NPU无运行进程，驱动驻留HBM约3416 MiB，证据为 `final-services.txt`、`final-test-units.txt`、`final-npu.txt`、`final-ports.txt`。后续用户启动 ASR 时会读取已验证的新开关。历史环境、驱动及第三方包没有被覆盖或无记录修改。
