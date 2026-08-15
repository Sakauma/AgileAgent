# Ascend 310B 工程记录

本文记录截至 2026-08-15 已完成的 Ascend 310B 实现、部署、精度复核和性能测量。

## 计分口径重新复核（2026-08-15）

仓库评分规则确认，候选的数值硬门禁只有 Base mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95` 和正式 20 图 `/api/batch` FPS `≥30`。逐框/业务 JSON 零差异、新类 precision、误激活率、单请求均值、P95/P99、Scene/Sensor accuracy 和 P0 阈值边界均为非评分诊断，不再阻断晋级。数据隔离、无标签预测冻结、模型/评分资产哈希和服务可运行性仍是结果真实性前提。

评分器、训练产物、实验 profile、代际晋级和发布校验已统一为上述口径：precision/FAR 继续输出 `diagnostic_warnings`，但不再决定 `accepted`。历史 P0–P9 章节中的旧门禁文字保留执行背景；与本节冲突时以本节和主计划的“计分门禁、真实性前提与诊断项”为准。

## 板端环境

| 项目 | 已确认值 |
| --- | --- |
| 开发套件 | Atlas 200I DK A2，`aarch64` |
| 芯片 | Ascend310B1，NPU Health `OK` |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1` |
| Conda | Miniconda `23.5.0` |
| Python 环境 | `/usr/local/miniconda3/envs/agileagent`，Python `3.9.2` |
| 正式 release | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` |
| 服务 | `127.0.0.1:8501`，health `ready` |
| 驻留快照 | NPU 内存 `9479 / 11577 MB`，温度 `61°C` |

## 已实现链路

- `fair_agent/backends/ascend_acl.py` 使用 PyACL/AscendCL 加载和执行三个 OM；
- `fair_agent/modules/web_inference.py` 编排 Base、Incremental 和 Scene 三模型；
- `fair_agent/web/app.py` 提供 health、单图检测和批量检测 API；
- `configs/agent_pipeline_ascend310b.yaml` 登记设备、CANN、执行模式、OM 路径和 SHA256；
- `scripts/start_agent_ascend310b.sh` 使用命名环境 `agileagent` 启动 Uvicorn；
- 模型输出经过全局类别映射、逐类阈值、场景软证据、冲突仲裁和 class-aware NMS。

## 正式模型

| 模型 | 输入 | 产物 |
| --- | --- | --- |
| 三类基础检测器 | `1×3×736×896` | `base_detector.om` |
| 增量检测器 | `1×3×512×640` | `incremental_detector.om` |
| Scene-SensorNet | `1×3×160×160` | `scene_sensor_net.om` |

三个 OM 使用 `mixed_float16` 编译，配置记录对应 SHA256。每张图像执行三个模型并生成统一检测结果。

## 正式 release 精度

正式 release 已在 89 张固定混合测试集上完成无标签推理、预测冻结和评分：

| 指标 | 结果 |
| --- | ---: |
| Base mAP50 | `0.819407` |
| New-mAP50 | `0.728761` |
| KRR | `1.000000` |
| 新类 precision | `0.933333` |
| 误激活率 | `0.014286` |

基础目标检测、New-mAP 和 KRR 三项合计取得 `50/50` 精度分档结果。

## 性能测量

| 测量对象 | 样本量 | 已记录结果 |
| --- | ---: | --- |
| 正式 release 完整 89 图 | 89 | 引擎均值 `57.849 ms`、墙钟均值 `71.491 ms`、引擎 `17.29 FPS`、墙钟 `13.99 FPS` |
| 已解码 Agent 核心 | 200 | 均值 `32.148 ms`、P95 `33.193 ms`、`31.11 FPS` |
| AIPP staging multipart PNG API | 1,068 | 均值 `51.203 ms`、P95 `63.9 ms`、`19.53 FPS` |
| DVPP 编码输入测量 | 240 | 均值 `37.124 ms`、P95 `38.154 ms`、`26.94 FPS` |
| P8 raw 正式 score gate | 3×20 图 batch | 中位 `921.3 ms`、`21.708 FPS`，效率 `7/10` |
| P9 decoded 正式 score gate | 3×20 图 batch | 中位 `919.5 ms`、`21.751 FPS`，效率 `7/10` |

单请求 FPS 只作诊断，效率分只读取 20 图 batch FPS。当前正式可复现满分缺口是把 P8 的 `921.3 ms` 降到 `≤666.7 ms`。

## Wave 0 / P0 执行记录（2026-08-15）

本轮按照 `ascend-310b-p0-p3-improvement-plan.md` 建立了独立候选，不覆盖正式 release，也未修改 `8501` 服务。候选目录为：

```text
/home/HwHiAiUser/agileagent/candidates/20260814-wave0-p0-7c61f2b
```

### 板端 Git 工作副本同步

板端正式 release 和旧候选最初都是不含 `.git` 的文件快照，不能证明与远程仓库一致。现已在不修改正式 release 的前提下建立独立工作副本：

```text
/home/HwHiAiUser/agileagent/repo
```

板端无法直连 GitHub `443`，因此先在已与远程核对一致的本地仓库生成完整 Git bundle，经 SHA256 校验后在板端 clone；后续提交通过带 prerequisite 的增量 bundle fetch，并以 `--ff-only` 更新。当前工作副本跟踪 `origin/agent/ascend-310b-wave0-p0`，P0 复测提交为 `eac65cc42ed77fa9f7de4468b64089bd8ceb4941`，`origin` 仍为 `https://github.com/Sakauma/AgileAgent.git`，工作树 clean。完整初始 bundle SHA256 为 `f6bffae8d76265de4d9c788febdba1c6b6f757d047d7704306635aad8526b388`。同步前后正式 `8501` 均保持 `ready`。

### 受控构建与门禁

- 三份 AIPP 配置已纳入 `configs/ascend310b/aipp/`；
- `scripts/build_ascend_aipp_oms.sh` 固定 ONNX、SoC、precision、输入 shape，拒绝覆盖，并记录 ATC 命令、日志和全部 SHA256；
- 完整 P0-r3 构建清单 SHA256 为 `a62131586d33ade4090dbf925fb1adca3ad9a852049d1780fe4b990097c3d1d4`；
- 早期 P0-r3 构建清单中的 `git_sha` 错记为不存在于当前仓库的 `7c61f2b4308bc146009df260984a1506a6274737`；实际起点和当时 `origin/main` 均为 `7c61f2b9ec0004aec5a0f3c2ab8858a2f229c5e3`。两者前 7 位碰巧相同，旧候选目录短名未暴露该错误。该清单不得作为可晋级 provenance；后续设备报告从板端 clean Git 工作副本直接记录完整 `HEAD`、分支、远程和工作树状态；
- Base、Specialist、Scene OM SHA256 分别为：
  - `2bc60b224ba3702232f6e35363199ae2b2f3b7382498340a719bf093f80a8851`；
  - `69957129b060295736e9812b459588147f7f0dee7d35b1e600196d077a431b7a`；
  - `c1902bc8f66e8036e70a9ba12a3b91dc5711837408eca819eb02b6afddc1f1a`；
- PyACL 输入契约已复核为 Base `[1,736,896,3] uint8`、Specialist `[1,512,640,3] uint8`、Scene `[1,160,160,3] uint8`；
- 未验收候选必须同时设置 `validation_candidate: true` 和进程级 `AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1`，正式 `validated: true` 仍要求构建清单及 golden、精度、性能报告哈希全部闭环。

### 89 图数值与语义结论

旧比较逻辑按响应顺序配对框，曾把最大坐标差误报为 `120.67 px`。新工具先按类别和 IoU 稳定匹配，再计算数值差。原始 P0-r3 在生产阈值 `0.5` 下的真实结果为：

| 项目 | 原始单级 Scene DVPP | 多级 Scene DVPP |
| --- | ---: | ---: |
| 逐图检测数量/类别不一致 | `1` | `1` |
| Scene 标签不一致 | `18` | `0` |
| Sensor 标签不一致 | `6` | `0` |
| 最大坐标绝对差 | `0.54 px` | `0.54 px` |
| 最大 confidence 绝对差 | `0.003418` | `0.003418` |

设备输入回读证明 Specialist PNGD 输出和 Base resize/letterbox 与 CPU 契约逐字节一致。Scene 单次 `640×512 → 176×176` 双线性缩小缺少 Pillow 下采样抗锯齿：代表性 SAR 图平均像素差 `11.19`、最大差 `88`，从而翻转场景标签。候选增加以下 DVPP 多级缩放后，89 图 Scene/Sensor 标签全部恢复一致：

```text
640×512 → 208×192 → 288×230 → 176×176 → center crop 160×160
```

旧 Scene AIPP V1 归一化在 89 图上造成 `55` 个 Scene 和 `16` 个 Sensor 标签变化，已明确拒绝。Base AIPP 将 `1/255` 下调一个 FP16 ULP 的实验又造成 `15` 张图检测数量变化、最大 confidence 差 `0.061584`，同样明确拒绝。

当前唯一剩余的生产阈值差异是 `ir_r1_base_urban_000149.png`：正式 Base 的 tank confidence 为 `0.500000`，AIPP 为 `0.501465`，导致 `2 → 3` 个框。不得通过修改阈值或后处理容差掩盖该差异。

### 同口径 890 请求性能

两次候选均按 30 次预热、10 轮固定 89 图、单并发、板端回环 HTTP keep-alive 和 `confidence=0.5` 执行：

| 候选 | 服务端均值 | P95 | Engine 均值 | Multipart 解析均值 | P0 `40/42 ms` |
| --- | ---: | ---: | ---: | ---: | --- |
| 原始单级 Scene DVPP | `48.76 ms` | `65.90 ms` | `38.00 ms` | `9.31 ms` | 未通过 |
| 多级 Scene DVPP | `49.27 ms` | `66.71 ms` | `38.13 ms` | `9.69 ms` | 未通过 |
| 多级 Scene DVPP + 有界 multipart 快速解析 (`eac65cc`) | `41.439 ms` | `47.200 ms` | `38.371 ms` | `1.640 ms` | 未通过 |

复测表明 Starlette/python-multipart 通用逐块解析是确定热点：原多级候选上传解析均值/P95 约 `9.69/22.71 ms`；对带明确 `Content-Length` 且不超过 `2 MiB + 64 KiB` 的单图请求启用有界快速解析后降至 `1.640/3.491 ms`，完整 API 均值和 P95 分别改善约 `15.9%` 和 `29.3%`。较大或非标准请求仍走原解析器，公共 multipart 字段保持不变。

细分计时显示 `routing_fusion_ms` 的常态均值约 `0.36 ms`，旧记录中的 `16.64 ms` 已不再是当前候选热点。P0 最终受 Engine P95 `42.709 ms`、完整 API `41.439/47.200 ms` 和 Base 阈值边界差异共同阻断，继续微调 HTTP 层不能同时闭合精度与 `40/42 ms` 门禁。

### 晋级结论

P0 当前状态为 **拒绝晋级**：多级 Scene DVPP 和 multipart 快速解析保留为候选实现，但 Base 阈值边界、构建 provenance 和完整 API 性能门禁仍未通过。候选配置保持 `validated: false`，`8502` 已停止，正式 `8501` 保持 `ready`。后续 P1/P2/P3 只能从上一已验收正式版本建立独立实验候选，不得把该 P0 候选视为已晋级基线。

关键板端证据均保存在候选的 `validation/` 目录：

| 报告 | SHA256 |
| --- | --- |
| `p0-0.5-iou-alignment.json` | `e30fcd2180c47b0a6c79922fb8fdc9f51034b49038ad24b09070810c8e537319` |
| `p0-dvpp-input-alignment-multistage.json` | `545eebe7c91550b426fc3ab8987de5a0f12a9fcdf7fb731ded7cfff01d28b89d` |
| `p0-multistage-0.5-alignment.json` | `7b12b0ded32a6d3940cef04723f2700563aab24521881c9a3dd9375552497ec7` |
| `p0-multistage-890-characterization.json` | `18ec053814007ecea59297f1a3c2b31f3087511365f670529d8aa62a6fd03e61` |
| `p0-fast-multipart-eac65cc-890-characterization.json` | `756a537c54576bf0e665cad46de8b00b164cf6e7871fef76709126034bd247c0` |

## P1 Python 后处理、路由与融合（2026-08-15）

P1 提交为 `5ee0f0884745f724f4e3373b11cbfda1bd937fd6`。板端 clean Git 工作副本已通过带 prerequisite 和 SHA256 校验的增量 bundle 从 `eac65cc` 快进到该提交，正式 `8501` 服务在同步和测量前后均保持 `ready`。本轮只在隔离的 `8502` 候选执行真实 API 端到端测量；按操作者要求，板端未运行 Web pytest。

本轮实现包括：

- `AscendResult` 直接暴露后处理 records，Agent 路由不再通过 `boxes.xyxy/conf/cls` 生成三份中间 list 后反向重建记录；
- 阈值过滤、跨类冲突矩阵和 class-aware NMS 在候选规模足以抵消建表开销时使用 NumPy `float64` 批量计算，小集合保留原标量顺序；
- 继续使用 Python 稳定排序，保留同置信度的 Specialist 优先 tie-break、输入顺序、完整 rejected/conflict 审计和 fusion summary；
- 明确拒绝 `early_incremental_threshold`：请求阈值 `0.5` 继续传给 Specialist 后端，正式激活阈值 `0.63` 只在 Specialist NMS 后执行。回归测试固定了该约束。

本机 WSL 既有 `.venv` 全量回归为 `226 passed, 1 skipped`。此外，用固定随机种子生成 1,000 组 NMS 和 1,000 组跨类冲突输入，与 `f06968d` 的旧实现逐字段差分，`2,000/2,000` 完全一致。两次板端 890 请求中，每张图、每一轮的检测数量签名也与 P0 报告一致；遵循只测板端端到端时长的要求，没有另跑板端业务回归套件。

### 同口径端到端结果

两轮均使用 30 次预热、10×89 请求、板端回环 HTTP keep-alive、单并发和 `confidence=0.5`：

| 版本 | 服务端均值 | 服务端 P95 | 服务端 P99 | Engine 均值 | 路由融合均值 | 转换均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 `eac65cc` | `41.439 ms` | `47.200 ms` | `48.411 ms` | `38.371 ms` | `0.464 ms` | `0.149 ms` |
| P1 `5ee0f08` 首轮 | `41.214 ms` | `47.055 ms` | `48.200 ms` | `38.129 ms` | `0.371 ms` | `0.136 ms` |
| P1 `5ee0f08` 复轮 | `41.302 ms` | `47.355 ms` | `48.311 ms` | `38.178 ms` | `0.368 ms` | `0.138 ms` |

首轮相对 P0 的服务端均值/P95/P99 分别改善约 `0.54%/0.31%/0.44%`，Engine 均值改善约 `0.63%`，转换均值改善约 `8.47%`。复轮重复了相同量级。`routing_fusion_ms` 稳定在约 `0.37 ms`，显著低于 P1 `≤8 ms` 门禁；P0 的均值包含一次 `88.835 ms` 系统离群点，因此不能把路由融合均值的全部变化归因于代码。

客户端墙钟受与服务端耗时不对应的连接/调度抖动影响：首轮 P95/P99 为 `99.915/113.037 ms`，复轮为 `58.840/101.486 ms`，而对应服务端 P95/P99 保持约 `47.1/48.3 ms`。门禁继续采用报告中受控的完整服务端 API 口径，不用客户端离群点宣称代码回退或收益。

P1 结论为：**路由目标通过，保留实现作为后续 P2/P3 候选起点，但不晋级正式服务**。它没有让服务端 P95/P99 劣于 P0，且两轮均无请求失败；但完整 API 仍未达到 `40/42 ms`，P0 的数值和 provenance 阻断也未消失。`8502` 已停止，`8501` 保持 `ready`。

| 报告 | SHA256 |
| --- | --- |
| `p1-native-records-5ee0f08-890-characterization.json` | `83ba1bc0b6c57581fb771a14420b4ea2d75c3ac8feec1c9cb812f4514226fe73` |
| `p1-native-records-5ee0f08-repeat-890-characterization.json` | `3fb42739904191a9736acdcd69745650575539494206c238eafc3e49abeec5f8` |

## P2 msprof 定位与 AOE 调优（2026-08-15）

P2 在 clean 板端 Git 工作副本中建立了受控采集入口 `scripts/profile_ascend_api.sh`、固定 encoded 请求应用 `tools/100_profile_ascend_request.py`、机器摘要器 `tools/101_summarize_ascend_profile.py` 和 AOE 契约门禁 `tools/102_check_ascend_aoe.py`。最终门禁提交为 `d496945a9b7313544355fa9cefae7cb945629232`。所有实现提交均已推送，板端通过增量 bundle 校验后 `--ff-only` 同步；正式 `8501` 始终保持 `ready`。

### msprof 原始采集

使用板端 CANN `7.0.RC1` 官方 `msprof` 启用了 Runtime API、model execution、task time、AI Core、DVPP、CPU/内存和 DDR/LLC 数据。采集应用从同一候选配置加载三个原 OM，执行无标签 encoded PNG 生产路径；没有运行 Web pytest。

| 采集 | 请求数 | 原始文件 | 原始逻辑字节 | 关键设备迭代 |
| --- | ---: | ---: | ---: | --- |
| `p2-single-0f610a3` | 1 | `2019` | `8,551,049` | Base `21.138 ms`、Specialist `26.263 ms`、Scene `3.007 ms` |
| `p2-fixed-0f610a3` | 89 | `2017` | `25,412,042` | Base 均值 `25.292 ms`、Specialist `13.384 ms`、Scene `17.145 ms` |

固定请求集在 profiling 插桩下三条 stream 并行竞争，不能拿设备迭代均值替代未插桩 API 基准；其用途是定位相对关键路径。single trace 的最长模型占插桩 Engine `71.4%`，固定集 Base 占插桩 Engine `65.0%`，而 `routing_fusion_ms` 仅约 `0.40-0.46 ms`，因此模型执行明确高于 Host 路由，满足进入 AOE 兼容评估的前提。

固定采集的全进程 Runtime 数据（包含模型加载和两次引擎 warmup）记录：

- stream wait/synchronize `632` 次、累计 `6,991.549 ms`；三个模型并发，因此累计等待不能与单请求墙钟相加；
- model execute enqueue `273` 次、Host API 累计 `14.368 ms`；
- 各类复制 API `767` 次、Host API 累计 `34.887 ms`；
- `MemcpyInfo` 记录 H2D `310` 次/`45,184,872` 字节、D2H `364` 次/`46,691,736` 字节、D2D `89` 次/`171,529,344` 字节；其中 D2D 是 encoded DVPP staging 到 Base 输入的生产关键路径。

该 CANN 的 DVPP 汇总 CSV 只枚举 VDEC/VENC，PNGD/VPC 不出现在该表；摘要明确保留这一限制，并以应用 `dvpp_enqueue_ms` 和原始 trace 为准，不把零值误写成无 DVPP 开销。

### AOE 契约结论

P0 固定构建使用 `--precision_mode_v2=mixed_float16`。目标板 `aoe` 支持 `--insert_op_conf` 和 `job_type=1/2`，但命令：

```text
aoe --precision_mode_v2=mixed_float16 -h
```

返回码为 `2`，明确报告 `--precision_mode_v2` 不受支持；它只公开旧的 `--precision_mode=allow_mix_precision`。P2 计划要求 ONNX、AIPP、SoC、precision 和输入 shape 全部与 P0 一致，因此禁止把 `allow_mix_precision` 当作近似替代。AOE 门禁报告已重新校验三个 ONNX/AIPP 的实际 SHA256，并记录 AOE 二进制哈希、完整 help 和失败探针。未执行 `job_type=1/2`，未创建 tuning repository、`aoe_result_opat` 或候选 OM。

### 未换 OM 的端到端结束口径

P2 结束时用未修改的 P1/P0 OM 再执行 30 次预热 + 10×89 真实 API 请求：服务端均值/P95/P99 为 `42.043/47.955/49.400 ms`，Engine 均值 `38.723 ms`，客户端墙钟均值/P95/P99 为 `54.024/60.137/61.707 ms`，路由融合均值 `0.375 ms`，无请求失败。该轮没有 tuned OM 或运行时代码变化，且比 P1 两轮稍慢，不能声称 P2 性能收益。

P2 结论为：**profiling 已完成，拒绝 tuned OM**。不存在满足“模型均值下降至少 `5%`、完整 API 均值下降至少 `1%`、P95/P99 不恶化”的候选；P3 从未修改的 P1/P0 OM 链路继续。`8502` 已停止，正式 `8501` 保持 `ready`。

| 报告 | SHA256 |
| --- | --- |
| `p2-single-0f610a3/summary-7e62647.json` | `1e0bcf185141aa385f6a993b391bdf98905b3e4a0f142b4f50317acb59d0865c` |
| `p2-fixed-0f610a3/summary-7e62647.json` | `85eee35061d941e603aa8a674ad3328850852fe788a911b3fd1dc87f2058cc0b` |
| `p2-aoe-compatibility-d496945.json` | `e247b08261f8b2b7e1da1cc180b058b281ed3e1e36b294db2dfdc5e981dd94c5` |
| `p2-no-tuned-7e62647-890-characterization.json` | `a4e10d829f3b223a03dd618bf0384e6b02029e31028d593799b91678067a4d6c` |

## P3 锁页内存、异步拷贝与统一等待（2026-08-15）

P3 实现提交为 `b8a9b2db2efdcbc2d8d654d272737ace98a3ce61`，板端 DVPP 时间戳可见性修复提交为 `43795ebec88804dd5506de6d2ca61ea16688db37`。两次提交均已推送；板端 clean Git 工作副本通过带 prerequisite 和 SHA256 校验的增量 bundle 依次 `--ff-only` 快进至 `43795eb`。开始 P3 修改时，板端已经位于当时最新的代码提交 `d496945`，仅缺随后推送的纯文档提交 `eaf4795`，因此 P3 没有基于过期代码开发。正式 `8501` 在同步、启动候选、测量和停止候选期间始终保持 `ready`。

### 目标能力探针与实现

目标 PyACL `7.0.RC1` 已确认存在 `malloc_host`、`free_host`、`memcpy_async`、stream/event 创建、等待、同步和耗时查询 API。能力探针还发现：直接用 `acl.util.ptr_to_numpy` 映射 `malloc_host` 地址后再调用 `free_host`，进程退出时会触发 glibc `double free or corruption`。P3 因此禁止该 API，统一使用：

```python
owner = (ctypes.c_uint8 * size).from_address(pointer)
array = np.ctypeslib.as_array(owner)
```

该非 owning 视图在板端完成了锁页 H2D→D2H 异步 roundtrip，NumPy data pointer 与 ACL host pointer 完全相同，逐字节结果一致且可正常释放。实现内容包括：

- `memory_mode` 从配置传入每个 `AscendAclModel`；`pinned` 只允许与 `async_stream` 组合；
- 每个模型常驻一个锁页输入 staging 和每个输出的锁页缓冲，encoded PNG 使用受 `2 MiB` 上限约束、按需增长的可复用锁页 staging；
- `submit(array)` 入队 H2D→model→D2H，`submit_preloaded(event)` 入队 event wait→model→D2H；句柄 `result()` 对模型 completion event 只执行一次最终等待并缓存结果；
- 每模型限制一个 in-flight，enqueue 或最终 event 等待失败时先排空 stream，再传播原始错误；关闭时先拒绝新提交并收敛 outstanding handle，随后按 event/stream、dataset/device、host、model 的顺序释放；
- encoded 单图路径按 Scene→Base→Specialist 的顺序先完成三个 submit，再按相同顺序 collect，不再在每个 `execute_async` 后立即同步；
- API 向后兼容增加 `dvpp_device_ms`、`ascend_submit_ms`、`ascend_wait_ms`、`ascend_input_copy_max_ms` 和 `ascend_output_copy_max_ms`；原字段未删除或改名；
- 板端首次真实请求暴露 CANN `107006`：下游 model completion 已完成仍不足以直接读取上游 DVPP event 时间戳。`43795eb` 在读取时间戳前显式确认 producing event 完成；此时三个模型句柄均已完成，不延长设备关键路径。修复后 920 个真实请求全部成功。

本机只使用仓库既有 WSL `.venv`，没有创建环境、安装依赖或下载 CPU PyTorch。全量回归为 `231 passed, 1 skipped`；新增 fake ACL 覆盖异步调用顺序、唯一最终等待、单 in-flight、preloaded 无 H2D、enqueue 失败恢复、最终等待失败恢复和锁页缓冲释放顺序。按照操作者要求，板端没有运行 Web pytest，只执行真实端到端时长。

### 独立候选与端到端结果

P3 使用独立 `8502` 候选配置 `p3-candidate-pinned.yaml`，相对 P2/P0 多级配置的唯一差异是 `memory_mode: pageable → pinned`；配置 SHA256 为 `5950fb2b4b39d5ae8c094ad83bf6e7a2d3bf5612abd181fe1132cb267c97b827`。三个 OM 和构建清单均未改变。基准严格使用 30 次预热、10×89 请求、板端回环 HTTP keep-alive、单并发和 `confidence=0.5`：

| 指标 | P2 基线 | P3 pinned | 相对变化 | P3 门禁 |
| --- | ---: | ---: | ---: | --- |
| 服务端均值 | `42.043 ms` | `45.916 ms` | 恶化 `9.21%` | 要求改善 `≥3%` 且 `≤33.33 ms`，失败 |
| 服务端 P95 | `47.955 ms` | `51.700 ms` | 恶化 `7.81%` | 要求 `≤35 ms`，失败 |
| 服务端 P99 | `49.400 ms` | `53.800 ms` | 恶化 `8.91%` | 允许恶化不超过 `2%`，失败 |
| Engine 均值 | `38.723 ms` | `42.551 ms` | 恶化 `9.88%` | 无独立晋级门禁 |

P3 的 890 请求细分均值为：DVPP enqueue `1.040 ms`、DVPP device `9.128 ms`、Ascend submit `0.735 ms`、统一 collect 等待 `36.635 ms`、最大输入复制 `0.000 ms`（三个输入均由 DVPP 预加载）、最大输出复制 `2.981 ms`、路由融合 `0.374 ms`。全部请求成功；一次 `144.1 ms` 服务端离群点只约增加总体均值 `0.11 ms`，移除它也不可能接近任何最终门禁，因此没有重复测量或用筛选样本掩盖结论。

P3 结论为：**实现与异常安全门禁完成，但 pinned 候选拒绝晋级**。锁页输出复制和统一异步编排没有在当前 CANN/OM/310B 组合上产生收益，完整 API 反而稳定慢于 P2。候选保持 `validated: false`，未替换正式配置；`8502` 已停止，正式 `8501` 保持 `ready`。

| 报告 | SHA256 |
| --- | --- |
| `p3-pinned-43795eb-890-e2e.json` | `9d7e4511215a2a8189880db9a52675248792ffd977f2367e9862573be24575da` |

## P0–P3 最终结论

| 阶段 | 最终服务端均值 | P95 | P99 | 阶段结论 |
| --- | ---: | ---: | ---: | --- |
| P0 | `41.439 ms` | `47.200 ms` | `48.411 ms` | multipart 热点已优化，但精度边界、provenance 和 `40/42 ms` 失败，拒绝晋级 |
| P1 | `41.302 ms` | `47.355 ms` | `48.311 ms` | 路由约 `0.37 ms`，语义差分通过；完整 API 仍失败，不晋级 |
| P2 | `42.043 ms` | `47.955 ms` | `49.400 ms` | profiling 完成；AOE 不兼容固定 precision，拒绝 tuned OM |
| P3 | `45.916 ms` | `51.700 ms` | `53.800 ms` | pinned/异步实现完成但性能回退，拒绝晋级 |

最终结论为：**本轮 P0–P3 已全部执行并留下可复核报告，但工程未达到最终 `≤33.33 ms` 均值和 `≤35 ms` P95 目标。** 没有任何候选同时满足性能、精度和发布门禁，因此正式 release、正式 OM 和 `8501` 均保持原状。P0 的有界 multipart、P1 的 records/路由优化和 P2/P3 的诊断工具保留为后续工作的代码基础，但不得据此宣称 Ascend 310B 端到端指标已经达标。

## P4 P3 运行时消融（2026-08-15）

P4 实现提交为 `e8bb1072dd7a673015a6fc9515594027daeaf915`；首次板端启动发现运行时代码已经消费 `schedule_mode` 和 `detailed_event_timing`，但配置白名单遗漏这两个字段。修复提交 `1ffa6699a7022af3b35f291d4560bea07be0bdb1` 增加字段枚举、类型校验和完整配置回归。两个提交均已推送，板端 clean Git 工作副本通过带 prerequisite 和 SHA256 校验的增量 bundle `--ff-only` 快进到 `1ffa669`。本地只使用既有 WSL `.venv`，全量回归为 `234 passed, 1 skipped`，没有安装依赖或下载 CPU PyTorch。

实现将 P3 的两项变化拆为独立开关：

- `threaded_execute` 恢复每个模型在线程池内执行、等待自身 stream 并同步 D2H 的 P2 调度；
- `unified_enqueue` 保留 Scene→Base→Specialist 统一提交、异步 D2H 和最终收集；
- `memory_mode` 继续独立选择 `pageable` 或 `pinned`；
- `detailed_event_timing: false` 在正式轮保留模型 inference event，但不查询 copy/DVPP 细分 event；另跑开启详细 event 的诊断轮评估插桩扰动。

### 复现门禁与环境漂移

最初两轮 A 的均值为 `41.403 ms` 和 `42.056 ms`。审计发现板载 `xscreensaver` 在两轮之间启动 `/usr/libexec/xscreensaver/m6502 -root`，持续占用约 `38%` Host CPU，导致两轮负载条件不一致。正式矩阵没有停止桌面或修改系统服务，只在每轮开始前使用标准 `xscreensaver-command -deactivate` 重置空闲计时，并确认除常驻 `xscreensaver-systemd` 外没有屏保子进程；最初两份报告只保留为漂移诊断证据，不参与候选比较。

受控 A 两轮为 `41.173 / 46.955 / 48.200 ms` 和 `41.276 / 47.400 / 48.300 ms`。相对 P2 的均值一轮略低于 `-2%` 下界，P99 两轮快 `2.2%–2.4%`；开启 `detailed_event_timing` 的独立 890 请求诊断轮为 `41.689 / 47.500 / 48.800 ms`，Mean/P95/P99 全部落在 P2 `±2%` 复现带内。关闭详细 event 查询使两轮 A 平均均值相对诊断轮减少 `1.11%`，因此复现偏差可由按计划关闭插桩解释，P4 矩阵继续使用受控、关闭插桩的 A 作为公平对照。

### 四候选端到端结果

每个候选均使用独立 `8502` 配置执行两轮 30 次预热加 10×89 单并发回环 HTTP keep-alive 请求，`confidence=0.5`。三个 OM、AIPP、输入集和构建清单 SHA256 `a62131586d33ade4090dbf925fb1adca3ad9a852049d1780fe4b990097c3d1d4` 均未改变；板端未运行 Web pytest。

| 候选 | 组合 | Run 1 Mean / P95 / P99 | Run 2 Mean / P95 / P99 | 相对 P2 均值 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| A | pageable + threaded | `41.173 / 46.955 / 48.200 ms` | `41.276 / 47.400 / 48.300 ms` | `-2.07% / -1.82%` | 公平对照；Run 2 未达 `2%`，但按规则在无胜者时保留 |
| B | pageable + unified | `45.170 / 50.800 / 52.211 ms` | `45.093 / 50.900 / 52.300 ms` | `+7.44% / +7.25%` | 两轮均显著回退，淘汰 |
| C | pinned + threaded | `41.299 / 47.100 / 48.422 ms` | `41.224 / 47.255 / 48.300 ms` | `-1.77% / -1.95%` | 两轮改善均不足 `2%`；相对 A 平均慢 `0.09%`，淘汰 |
| D | pinned + unified | `44.353 / 50.000 / 51.500 ms` | `44.313 / 50.000 / 51.200 ms` | `+5.50% / +5.40%` | 两轮均显著回退，淘汰 |

8 份正式报告均为 890 个成功请求，Git head、配置哈希和构建清单一致；按 round、文件名和检测数量逐请求对齐，相对 A Run 1 的不一致数均为 0。B、D 说明统一提交是 P3 回退的主要来源；C 说明锁页内存单独没有稳定收益。没有候选在两轮中都满足“均值改善至少 `2%` 且 P95/P99 不恶化超过 `2%`”，P4 因此按规则保留 **pageable + threaded_execute**，作为 P5 关键路径调度的起点。该结果仍继承 P0 的阈值边界和 provenance 阻断，不晋级正式版。

| 配置 | SHA256 |
| --- | --- |
| `p4-a-pageable-threaded.yaml` | `bab2215c5d4c3c9886e33bd277b47fc18cab28d3fc270ab8ee38da14df5d9ebf` |
| `p4-b-pageable-unified.yaml` | `d50c6605473623ca1fc4ab81ed4314df34967f3b4c88efbbe85b17f408f14d91` |
| `p4-c-pinned-threaded.yaml` | `c5a36443e81a819ca5a7e9d455eaf40058a61327f8d82aa799281f50d57e976c` |
| `p4-d-pinned-unified.yaml` | `ab85f43a6701991e3c6d5bcb66efc5efdd5559b22bde37c17d8fb448c98afd82` |
| `p4-a-pageable-threaded-events.yaml` | `d993be155d2bb083e75853a069bb356399209a4f70c6fd9e468c4de6ebe620a7` |

| 报告 | SHA256 |
| --- | --- |
| `p4-a-controlled-run1-1ffa669-890-e2e.json` | `bd84cbe5a3adbebf592ef6b19dae50cbb3dd0b82c1b2ee9d151bf93ad6f02282` |
| `p4-a-controlled-run2-1ffa669-890-e2e.json` | `dddbd6958d65592b7cbec8ce09cf17b1b3363f84598d35b4572a2b8099244f81` |
| `p4-b-run1-1ffa669-890-e2e.json` | `a98a1066f5215d33a43673c6bf732696a97406948937eb9050610bfa1cf59033` |
| `p4-b-run2-1ffa669-890-e2e.json` | `bd342c78f8784db28e5e09a31a0866be088b0023b11610267e88757b6ef330a9` |
| `p4-c-run1-1ffa669-890-e2e.json` | `724df4cae758a7e607cf4316db9c03acef9f0f6f58cc06cfdd47b2c11ad2c86e` |
| `p4-c-run2-1ffa669-890-e2e.json` | `cf7f2b9cf326a350252c3e30295c74ab5f2ea56f1c4c1b977ff66d58a1c920ac` |
| `p4-d-run1-1ffa669-890-e2e.json` | `c7308472b7f3d33f184420712f5fd781ce8b53a384c55ade0239b1fa715e0f21` |
| `p4-d-run2-1ffa669-890-e2e.json` | `0c25674093f9be8b86dd08cedaa1d911af5a707deb20e8a20c9fd72ada46dcbf` |
| `p4-a-events-diagnostic-1ffa669-890-e2e.json` | `496593a17b43e7b3fd6e83527751bf85277c95c3e687174ea5982c10de19b00d` |

P4 测量和完整性校验结束后 `8502` 已停止；正式 `8501` 在同步、启动、切换候选、测量和停止期间始终保持 `ready`。

## P5 关键路径调度（2026-08-15）

P5 调度实现提交为 `f0b59954a82d613a23dbebb0c8cdf2693bd156b7`。`submit_order` 和 `collect_order` 分别接受 Scene/Base/Specialist 的无重复全排列；P4 胜出的 `threaded_execute` 路径和保留用于诊断的 `unified_enqueue` 路径共用有序提交、全量 drain 和首异常传播逻辑。Base、Specialist、Scene 三个模型的 runtime role 也被显式传到 Ascend 后端，为条件式 stream priority 建立内部接口。配置枚举、异常恢复、顺序执行和 priority 支持/回退均有本地 fake runtime 覆盖。

为满足业务 JSON 零差异门禁，同一端到端基准从提交 `863a77b3637a10e285b68973c7544e39482bd0fa` 开始为每个请求保存剔除 `inference_ms`、`timings`、`queue_wait_ms` 和 `system_total_ms` 后的 canonical SHA256。首次板端启动工具时发现该实现间接导入 `httpx`，而板端环境没有该包；请求尚未开始，未生成性能报告。提交 `55f498bba537a6073578dbcf21b60d1c2866f68c` 将哈希函数下沉到零第三方依赖的 `fair_agent.core.hashes`，没有安装依赖或创建环境。修复后本地系统 Python 和 WSL `.venv` 均可直接启动基准工具，全量回归为 `241 passed, 1 skipped`。

### Stream priority 能力结论

目标 CANN 7.0.RC1 的 PyACL 存在 `acl.rt.create_stream_with_config`，随板 `acl_rt.h` 声明通用优先级参数范围 `0..7`，但没有 `get_stream_priority_range`。独立实探逐个创建 `0..7`：只有 priority `0` 成功创建和销毁，`1..7` 全部返回 `107000`。探针前后正式 `8501` 均为 `ready`。该板卡/运行时无法形成可区分的 high/normal/low 三档，因此 P5 按计划记录 `priority_range_api_unavailable` 并跳过 Specialist 高优先级和 Base 高优先级两个候选，没有引入兼容替代。实现只有在运行时能报告并实探通过至少三档时才映射 `high/normal/low`；否则三个模型统一使用普通 stream，避免部分应用 priority 的混合状态。

### 提交与收集顺序结果

P5 继续使用 P4 胜出的 `pageable + threaded_execute + detailed_event_timing=false`。每个候选在受控 screen saver 条件下执行 30 次预热加 10×89 单并发回环 HTTP keep-alive 请求；当前 Scene→Base→Specialist 提交及同序收集作为新鲜基线。三个 OM、AIPP、输入集和构建清单均未改变，板端未运行 Web pytest。

| 候选 | 提交顺序 | 收集顺序 | Mean / P95 / P99 | 相对基线均值 | 业务签名差异 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 当前 | Scene→Base→Specialist | Scene→Base→Specialist | `41.245 / 47.300 / 48.400 ms` | — | `0` | 新鲜基线 |
| 提交 1 | Specialist→Base→Scene | Scene→Base→Specialist | `41.202 / 47.300 / 48.211 ms` | 改善 `0.043 ms / 0.104%` | `0` | 未达到 `0.5 ms` 和 `1%` |
| 提交 2 | Base→Specialist→Scene | Scene→Base→Specialist | `41.240 / 47.300 / 48.200 ms` | 改善 `0.005 ms / 0.012%` | `0` | 未达到 `0.5 ms` 和 `1%` |
| 收集 1 | Scene→Base→Specialist | Base→Specialist→Scene | `41.227 / 47.400 / 48.200 ms` | 改善 `0.018 ms / 0.044%` | `0` | 未达到门槛，P95 略高 |
| 收集 2 | Scene→Base→Specialist | Specialist→Base→Scene | `41.288 / 47.200 / 48.211 ms` | 恶化 `0.044 ms / 0.106%` | `0` | 均值回退 |

5 份报告均包含 890 个成功请求，每份报告的 89 张图在 10 轮内业务签名完全稳定；四个候选相对当前基线按 `(round, image)` 对齐均为 0 差异。所有候选的 P95/P99 变化都在 `±2%` 内，但没有任何候选同时满足均值改善 `≥1%` 和 `≥0.5 ms`。P5 因此保留 **Scene→Base→Specialist 提交、Scene→Base→Specialist 收集、普通 stream**，P6 从该组合继续。

| 配置 | SHA256 |
| --- | --- |
| `p5-submit-scene-base-specialist.yaml` | `5c5c9fef4718af5d8bfcacedad041e2b0d29baf50ac49014daedf1d376597c1a` |
| `p5-submit-specialist-base-scene.yaml` | `e21e7af0018921fbb66eb1f52f743f6e589c3fb4e1f49e35f1265e736a3a3d9a` |
| `p5-submit-base-specialist-scene.yaml` | `efea5af0a2194a2dc11ae09269abbcb3d5150619a2ad925623dd5ff70ea120a1` |
| `p5-collect-base-specialist-scene.yaml` | `957c74076c6e9131174984d337722a35d152b41ae89a0e73942b9acc0eae3287` |
| `p5-collect-specialist-base-scene.yaml` | `79290b291858dd266e7f6385989183c7797edd996cab5ab0c5ccc77619cfdf08` |

| 报告 | SHA256 |
| --- | --- |
| `p5-submit-scene-base-specialist-55f498b-890-e2e.json` | `49daece8087bf778b3a4421e49ddf82d52d98ae2107b523a996689a5dabbea1f` |
| `p5-submit-specialist-base-scene-55f498b-890-e2e.json` | `78fb041a1d809ae586edf2be53bf00b606fec1ad1908f6af22e6bc473d441032` |
| `p5-submit-base-specialist-scene-55f498b-890-e2e.json` | `3fcc53d0348a68d0fe57f2c6aeb87b20fc3b2b2da7bd15d3b72640d8513ee95b` |
| `p5-collect-base-specialist-scene-55f498b-890-e2e.json` | `39ee9ab35ebcebde4e96e4a27af083b8f5b3b0b2ce627b3a486d550b1e226306` |
| `p5-collect-specialist-base-scene-55f498b-890-e2e.json` | `5ffd139a648dfd7307652df5737549dfa637c7d11e0ffcaf239e2f8fc59b2781` |

P5 仍未达到完整 API 均值 `≤33.33 ms` 和 P95 `≤35 ms`，并继续继承 P0 的阈值边界与 provenance 阻断，不能晋级正式版。测量结束后 `8502` 已停止，正式 `8501` 始终保持 `ready`。

## P6 设备侧 YOLO 解码与 NMS（2026-08-15）

P6 已实现 `raw_yolo_v1 | detections_v1` 显式输出契约。Ascend 后端只按配置选择路径，不依据输出 shape 猜测；`detections_v1` 固定校验 `boxes[max_det,4] float32`、`scores[max_det] float32`、`class_ids[max_det] int32` 和 `valid_count[1] int32`，并绑定设备候选阈值、IoU 和 `max_det`。发布校验同时核对配置、构建清单和设备后处理契约。原始 Host YOLO 路径保持不变，当前生产配置显式使用 `raw_yolo_v1`。

本机只使用 WSL 仓库现有 `.venv` 中的 `torch 2.5.1+cu124`、`torchvision 0.20.1+cu124` 和 `ultralytics 8.4.92` 做 ONNX 导出，没有安装依赖，也没有下载 CPU PyTorch。本机环境没有 `onnx`/`onnxruntime`；导出器只跳过 Torch 用于附加 ONNXScript 自定义函数的可选 hook，仍由 Torch 写出标准 ONNX protobuf。完整 Base/Specialist 实验 ONNX 的 SHA256 分别为：

| 产物 | SHA256 |
| --- | --- |
| `base_detector.onnx` | `ddba88c8bf25469119d0531faa2c0aec273923e372ea2d8b868b6e6dcb4535a0` |
| `incremental_detector.onnx` | `5765b3b762ee3ff0920abdb613a4aa2dd61ff501ca8589cfb251ff3fbed2f1bb` |

### CANN 7.0.RC1 严格语义探针

构造输入同时覆盖三个 `0.8` 同分候选、跨类重叠、同类完全重复的 `0.7` 框、严格阈值边界 `0.01` 和应保留的 `0.01001`。Host reference 应输出三个 `0.8` 框和 `0.01001` 框：IoU 恰好等于 `0.7` 时不抑制，完全重复的低分同类框被抑制。

| 探针 | 路径/实现 | 结果 |
| --- | --- | --- |
| v1 | 标准 ONNX `NonMaxSuppression` | ATC 映射为 `NonMaxSuppressionV7`，CANN 7.0.RC1 不支持该动态输出 shape，编译失败 |
| v2–v5 | 版本算子 `NPUNmsWithMask`，含 8/5 列 proposals 和关闭 fusion | 可编译，但输出仍包含完全重复的 `0.7` 框，`valid_count=5`；关闭 `NMSWithMaskFusionPass` 不改变结果 |
| v6 | ONNX 直接使用内部名 `BatchMultiClassNonMaxSuppression` | parser 未注册，ATC 明确拒绝 |
| v7–v8 | ONNX parser `BatchMultiClassNMS` | 可编译，严格 score 边界和重复框抑制正确，但把 IoU 恰好 `0.7` 的同分框也抑制，`valid_count=3` 而不是 `4` |
| v9 | 命令行选择 `norm_class` | CANN 7.0.RC1 的 `--op_select_implmode` 不接受该值 |
| v10 | `--op_precision_mode` 逐算子选择隐藏 `norm_class` | 可编译，但语义输出与默认实现相同，仍为 `valid_count=3` |

完整证据保存在板端 `/home/HwHiAiUser/agileagent/candidates/20260815-p6/probe-v1` 至 `probe-v10`，包括 ONNX、ATC 日志、OM、构造输入、输出 `.npy` 和报告。ATC 日志中的 `customize_impl/mrgba.py` 权限异常来自既有 root-only vendor 算子；成功探针仍明确结束为 `ATC run success` 并生成 OM，不影响上述结论。

P6 当时不能通过“严格 `score > threshold`、稳定 tie-break、class-aware NMS 和 `IoU > threshold`”零差异门禁。工程没有用 epsilon 改阈值、没有升级 CANN、没有启动 8502，也没有运行板端 Web pytest。该停止条件现在已降为诊断：可编译的 `BatchMultiClassNMS` 候选将在 P9-B 直接执行 89 图重新评分和 batch score gate，不能再仅凭构造边界差异淘汰。

P6 修订结论为：**双契约和回滚基础设施保留，旧拒绝结论撤销为“待计分口径复测”。** 当前正式仍使用 raw；只有设备 NMS 实测三项精度满分且 batch FPS 更高才切换。

## P7 统一检测器与共享上下文头（2026-08-15）

P7 首先修复并固化统一检测器实验基础：

- `expanded_single_student` 的三个旧类分类行在每步后同时恢复到 model 和 EMA，冻结 BatchNorm 保持 eval；
- `yolo_iod_lite` 同样在 model/EMA 中恢复受保护旧类行并冻结 BN 统计；
- 预检允许统一四类架构使用显存可承受的 batch 4，但继续保留原双检测器严格基准 `batch=32`；
- P7 统一使用 Base 的 `896` 输入，复用 production Base checkpoint；YOLO-IOD 额外复用现有 current teacher；
- 单检测器物理输出在 Agent 层继续把旧类标记为 `frozen_base_model`、warship 标记为 `incremental_model`，并生成原协议结构的逻辑审计结果；
- 两个专用配置的只读训练预检均为 `ready: true`，本机仍只使用 WSL 仓库现有 `.venv`，没有安装依赖或下载 CPU PyTorch。

### 候选 1：expanded_single_student

Run ID 为 `p7-expanded-20260815`。候选复用冻结 Base checkpoint，只在 warship 增量数据上训练新增类别通道；共享参数和旧类通道漂移均为 `0`。锁集结果为：

| 门禁 | 实测 | 要求 | 结果 |
| --- | ---: | ---: | --- |
| Base mAP50 | `0.8140883421` | `≥0.80` | 通过 |
| New-mAP50 | `0.0166666667` | `≥0.60` | 失败 |
| KRR | `1.0000000000` | `≥0.95` | 通过 |
| 新类 precision | `0.6666666667` | `≥0.90` | 失败 |
| 误激活率 | `0.0142857143` | `≤0.05` | 通过 |

部署阈值 `0.68` 下只有 `TP=2`、`FP=1`、`targets=76`。该候选完整保留旧知识，但 New-mAP50 明确未到 `0.60`，按计分门禁拒绝；precision 只作诊断。

### 候选 2：yolo_iod_lite

Run ID 为 `p7-yolo-iod-20260815`，完整训练 `80/80` epoch，最终 dev mAP50 为 `0.8281`。current teacher 复用 `models/production/incremental_detection/incremental_detector.pt`，SHA256 为 `d27bda7cb89375788deb1f29366b037757f23f7b32ddf6c11e1aa778384dc957`；教师只在增量 student 图像上推理，不读取旧类原始训练数据。

| 门禁 | 实测 | 要求 | 结果 |
| --- | ---: | ---: | --- |
| Base mAP50 | `0.8140883421` | `≥0.80` | 通过 |
| New-mAP50 | `0.5981284425` | `≥0.60` | 失败，差 `0.0018715575` |
| KRR | `0.3445057231` | `≥0.95` | 失败 |
| 新类 precision | `0.4166666667` | `≥0.90` | 失败 |
| 误激活率 | `0.4285714286` | `≤0.05` | 失败 |

该候选的旧类通道最大漂移为 `0`，共享参数相对漂移为 `0.0771120489`。部署阈值同为 `0.68`，锁集得到 `TP=70`、`FP=98`、`targets=76`，70 张负样本中 30 张产生误激活。`fusion decision_count=0`。按新口径，拒绝它的充分理由是 New-mAP50 `0.598128 <0.60` 且 KRR `0.344506 <0.95`；precision/FAR 不参与结论。

### 数据隔离、证据与阶段结论

两个 dataset manifest 的 `learning_data_scope` 均为 `incremental_dataset_only`，`old_raw_image_count`、`old_raw_label_count`、`old_feature_cache_count` 均为 `0`，`original_data_modified=false`，且锁集只在无标签预测冻结后物化。

| 候选 | student weight SHA256 | metrics SHA256 | dataset manifest SHA256 |
| --- | --- | --- | --- |
| expanded | `cd82013926bde21504315f99bb437821276cee3a24775901a544f602b82e1392` | `c52c3f59505e8bdab6dfdff61e28bfbef1791887a4e2e68ce1dcbad264b3adc6` | `94f3f6edd893192c06370ba80078976bf9d26093a158229bf43dc062401e23b3` |
| YOLO-IOD-lite | `889e43bf22059e5901ab7407d395d7824a4b6c2c3d54f018c79685c8694e12db` | `545d2c6221dc9bd07eadeb39298a58b147f84c545c5aeb77f027ace79a630586` | `40bed0ec033c11134f5b4be95af2b36556f30739f64c1dce848f6f1698f90405` |

P7 计分口径复核结论仍为：**两个统一检测器候选均拒绝晋级。** 两者都失败于 New-mAP50，第二个还失败于 KRR，并非由 precision/FAR 误杀。正式 release、双检测器 OM、Scene OM 和 `8501` 均保持原状。

## P8 板端环境固化（2026-08-15）

P8 实现提交为 `6aaff39f56188660fcd50a0dc405d359d98a12a7`；板端实测发现结束温度被错误当作开始门禁后，以 `5702511040986bbd7b3a37316db9d393746310bf` 修正为“开始温度闭锁、结束温度留证”。两个提交均已推送，板端 clean 工作副本通过 SHA256 校验的增量 Git bundle `--ff-only` 同步。现有 WSL `.venv` 全量回归为 `266 passed, 1 skipped`，未安装依赖或下载 CPU PyTorch；板端未运行 Web pytest。

环境治理结果：

- 停止 `xscreensaver`、`xscreensaver-systemd` 和 `xfce4-screensaver`，为三者写入用户级 XDG `Hidden=true` autostart override；
- 板端不存在 `/sys/devices/system/cpu/cpufreq/policy*`，因此 governor 记录为 `unsupported`，未伪造 `performance` 状态；
- pre-start/prerun guard 均确认 `8501 ready`、`8502` 端口状态符合阶段、NPU Health `OK`、三次 CPU 采样无非白名单 `>10%` 进程；
- Git HEAD、配置、build manifest、OM/ONNX/AIPP SHA256、Python、SoC 和 governor 均已绑定，两轮身份比较完全一致。

使用 P5 保留配置执行两轮独立 30 次预热 + 10×89 请求：

| P8 基线 | API mean | P95 | P99 | Engine mean | 开始/结束温度 | 门禁 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Run 1 | `41.1265 ms` | `47.200 ms` | `48.200 ms` | `38.0903 ms` | `59/67°C` | 通过 |
| Run 2 | `41.1347 ms` | `47.300 ms` | `48.211 ms` | `38.1017 ms` | `61/70°C` | 通过 |

Run 2 相对 Run 1 的 mean、P95、P99 差异约为 `0.020%/0.212%/0.023%`，全部 `≤2%`。P8 后续阶段基线固定为 Run 1 `41.1265/47.200/48.200 ms`；相对旧 P5 参考的 mean 变化仅约 `-0.287%`，说明环境治理主要提升了测量可重复性，没有伪装成模型收益。正式报告与运行前 guard 的 SHA256 为：

| 证据 | SHA256 |
| --- | --- |
| `p8-run1-5702511.json` | `4c11fa96e89d478cc8cd2fef6932cf5be1757e0261b1c162e2ef38be8dc7c79f` |
| `p8-run2-5702511.json` | `733742c99c0b3f206829e4c0851e51650b26fcb03f6656bca09388c69ffee68e` |
| `prerun1-5702511.json` | `a7fe5880c09221fd13f37934ab550c3978257bc63ba0cb785cbcbb3bc5d34197` |
| `prerun2-5702511.json` | `ee6ca9d11957928e2d4272d893fcf73f2a78826a7829cf2a2e02ca7fc44cc703` |

测量完成后只停止 `8502`；正式 `8501` 在候选启动前、两轮测量间和候选停止后均为 `ready`。P8 结论为：**环境固化和两轮重复性门禁通过，允许进入 P9；性能仍未达到最终 30 FPS 门禁。**

随后用正确评分阈值 `confidence=0.01` 重新冻结预测，P8 raw 得到 Base mAP50 `0.819415`、New-mAP50 `0.728761`、KRR `1.0`，精度三项满分。正式 20 图 batch 三轮的中位时长为 `921.3 ms`、`21.708 FPS`；上传解析约 `51.8 ms`、Host decode 约 `148–172 ms`、batch Engine 约 `690–705 ms`。因此 P8 是当前 score-aligned raw 基线，效率仅为 `7/10`。

## P9 设备后处理计分复核（2026-08-15）

P9 实现提交为 `23993fdf44a4996902039d0c16ef345b90f1ca5b`，业务 JSON 对比工具提交为 `4ee54dc`，均已推送。实现增加 `decoded_candidates_v1` 固定输出契约、anchor-major/class-minor 唯一 TopK 键、Host 稳定全局排序与 class-aware 严格 NMS、低阈值/overflow raw 回滚、选择性 D2H、manifest 绑定和可复现导出/探针/构建工具。完整 WSL `.venv` 回归为 `279 passed, 1 skipped`，没有安装依赖或下载 CPU PyTorch。

### CANN 7.0.RC1 严格探针

标准 ONNX TopK/gather 图可由 CANN `7.0.RC1` 编译。v1 的候选 boxes、scores、class/anchor ID 和 raw 输出正确，但设备把 int32 `ReduceSum` 错误计算为 `1`，导致 `valid_count/overflow` 失真。v2 改为 float32 求和再转 int32后，三个构造输入的 7 个输出全部逐项匹配 Host reference：

| 探针 | 预期候选数/overflow | 结果 |
| --- | ---: | --- |
| 严格语义 | `6/0` | 阈值相等排除、`0.01001` 保留、同分稳定顺序、跨类重叠、`IoU=0.5` 不抑制和 NMS 后补位全部通过 |
| capacity 边界 | `8/0` | 通过 |
| overflow | `8/1` | 通过，要求 raw 回滚 |

v2 语义报告 SHA256 为 `9aae98eb4744ecb7edf6aeffd8915826c197203c836b904307ea7dabbe31d68c`。板端探针 ONNX/OM SHA256 分别为 `f5fa2c63358c5eedd6e9e9e78b2cb4b0a8d9fe1eb5cdfdf647edd9c828fc55f6` 和 `ab40c31b1920688aa8caeee9629cc74b95711086de01cc673847616e7c3943ad`。

### 完整 OM、89 图评分与正式 batch score gate

本机使用现有 CUDA `.venv` 导出 Base `13524 anchors/capacity 4096` 与 Specialist `6720/capacity 2048` ONNX，SHA256 分别为 `02cd5f20237a83fa5b4f793166734f1d2303ca77d313a5eda559ffcd4ae03fb4`、`e663f63a13db79f65ebed50a59e0f88a24b705cda4d20a16af83c4bd85567958`。板端两份完整 OM 均得到 `ATC run success`，release manifest 验证为 `passed`：

| 产物 | SHA256 |
| --- | --- |
| Base OM | `5f02b38224a1a22bbe2e034b4f695c232428e7c9ad786a899edaf53c17993413` |
| Specialist OM | `9b303c04f2eb081671c078e69c7593b8213254304fbb1467d5fe448ae372b5f2` |
| build manifest | `5cda5197e2d408a29f897c46a44e791bc9bfadf32a7226867eb04b26fba00694` |
| candidate config | `901e83413f686bd90e28fb445cafdf5657b32217e13c11a47ee8c7dfdfc6b9bd` |

真实 89 图对照使用同一 P8 Scene/DVPP/调度配置。P8 raw 与 P9 decoded 均输出 `343` 个检测，检测数、类别数、置信度及 Scene/Sensor 上下文全部一致；mixed-fp16 图内坐标解码导致 `84/89` 图的框坐标不完全相同，最大绝对差为 `0.25 px`，业务 JSON SHA256 也有 `84/89` 不一致。该差异现仅作诊断。按 `confidence=0.01` 重新评分，P9 decoded 得到 Base mAP50 `0.819901`、New-mAP50 `0.728761`、KRR `1.0`，三项满分；冻结预测 SHA256 为 `7f57f7f012e6e94693b0a571efe97143cf01a23332e677b1db9b1d98927c0006`。

P9 decoded 已补跑两轮 30+890 单请求诊断：Run 1 `42.798/48.555/49.500 ms`，Run 2 `42.737/48.600/49.300 ms`。正式 20 图 batch 三轮为 `920.8/919.5/903.6 ms`，中位 `919.5 ms`、`21.751 FPS`；中位分解为上传解析 `51.936 ms`、Host decode `163.149 ms`、Engine `697.130 ms`、cache `6.025 ms`。相对 P8 `21.708 FPS` 只高约 `0.20%`，没有稳定效率收益，且仍为 `7/10`。报告 SHA256 为 `95104f39d1045bdafed80c11084125882ee65bca41015490e8fc601b42081805`。

P9-A 修订结论为：**精度三项满分，但效率计分不提升，回滚 `raw_yolo_v1`。** 拒绝理由不再包含逐框或 JSON 差异。P9-B 将重新开放 P6 可编译的 `detections_v1/BatchMultiClassNMS`，直接跑 89 图评分与 batch score gate。补测结束后 `8502` 已停止，正式 `8501` 为 `ready`。

## P10–P11 待执行方案

P10–P11 已按计分口径重排，尚未产生双逻辑检测头、共享上下文头或相应板端晋级结果。执行顺序固定为：P9-B 设备 NMS 计分复测、P10-A batch DVPP encoded 路径、P10-B 共享骨干双逻辑检测头、P11 共享 Scene/Sensor 上下文头。完整规则记录在 `docs/ascend-310b-p0-p3-improvement-plan.md`。

当前参考状态：

| 项目 | 当前值 | 目标或结论 |
| --- | ---: | --- |
| 正式 Base mAP50 | `0.819407` | 已达到满分档 |
| 正式 New-mAP50 | `0.728761` | 已达到满分档 |
| 正式 KRR | `1.000000` | 已达到满分档 |
| 正式新类 precision | `0.933333` | 非评分诊断 |
| 正式误激活率 | `0.014286` | 非评分诊断 |
| P8 raw 20 图 batch | `21.708 FPS` | 效率 `7/10`，未满分 |
| P9 decoded 20 图 batch | `21.751 FPS` | 与 P8 基本持平，回滚 |
| 最终性能预算 | batch `≤666.7 ms` | `≥30 FPS`，效率 `10/10` |

计划要点如下：

- P8 已持久禁用屏保进程并以温度、后台 CPU、NPU Health、端口、Git/配置/OM 哈希守卫两轮 30+890 新鲜基线；governor 在该板明确为 `unsupported`；
- P9-A decoded 已证明精度满分但 batch 无收益；P9-B 重新开放设备 NMS，不再因 NMS 边界或 JSON 差异提前停止；
- P10-A 让 `/api/batch` 复用 DVPP encoded 路径，优先消除每批 `148–172 ms` Host decode；
- P10-B 共享 Base backbone 与 neck/FPN，保留独立 old/new Detect head；只用 warship 数据训练 new head 或 residual adapter，Base/EMA 漂移必须为 `0`；
- P11 即使 P10 失败也继续挂到正式 Base 骨干，依次测试深层单尺度和 P3/P4/P5 多尺度上下文头，检测网络保持冻结；
- P8–P11 候选只使用 `8502`，正式 `8501` 不停止、不替换；板端不运行 Web pytest；CANN 固定 `7.0.RC1`，不使用 INT8、降分辨率、剪枝或跨请求流水线。

所有可运行候选均冻结 89 图预测并重算 Base `≥0.80`、New `≥0.60`、KRR `≥0.95`，再用三轮 20 图 batch 的中位 FPS 判定效率；四项满分即可晋级。precision/FAR、逐框/JSON、P95/P99、上下文 accuracy 和 P0 阈值边界只作诊断。provenance、数据隔离和资产哈希继续验证结果真实性。

## 环境迁移记录

板端 Python 环境已迁移到命名环境 `agileagent`。迁移前后使用固定 PNG 执行响应语义对照，检测数量、类别、框和置信度保持一致；切换后 health 返回 `ready`。

命名环境确认项：

- Python `3.9.2`；
- 181 个 Conda 记录；
- 176 个 `pip freeze --all` 条目；
- PyACL 与核心模块导入成功；
- 三个 OM 加载成功；
- 真实 PNG 推理成功。

## 自动验证

```bash
python -m pytest -q
python scripts/verify_release.py
```

当前本地 WSL 仓库既有 `.venv` 全量回归为 `280 passed, 1 skipped`；修订门禁后的定向回归为 `106 passed`。正式 release 的既有发布校验状态仍为 `passed`。P7 因计分精度失败保持拒绝；P9-A 精度满分但 batch FPS 未提升，回滚 raw；P6/P9-B 设备 NMS 重新开放待测。板端只执行探针、89 图评分和真实 API 端到端时长，没有运行 Web pytest。

## 运行态检查

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
npu-smi info
```
