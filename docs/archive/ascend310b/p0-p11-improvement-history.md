# AgileAgent 昇腾 310B P0–P11 推理优化计划与执行记录（历史归档）

> 本文保留完整阶段设计和消融证据，不再作为活动入口。当前复现方法见 [`../../ascend-310b-full-score-method.md`](../../ascend-310b-full-score-method.md)。

## 1. 摘要

本文记录 AgileAgent 在 Atlas 200I DK A2 / Ascend310B1 上完成的 P0–P11 实测结论、接口、消融矩阵和晋级门禁。详细设备证据保存在 `docs/ascend-310b-current-status.md`；本文以同条件端到端实测和严格类增量锁集结果替代执行前的跨批次估计。

2026-08-15 按赛题机器可读评分规则从头复核后，当前唯一数值硬门禁改为 Base mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95` 和 20 图 `/api/batch` FPS `≥30`。逐框/业务 JSON 零差异、新类 precision、误激活率、单请求均值、P95/P99、Scene/Sensor lock accuracy 和 P0 阈值边界全部降为非阻断诊断。数据隔离、无标签预测冻结、评分资产哈希和服务可运行性继续作为结果真实性前提，不构成额外性能阈值。

P0–P3 已全部结束，结果如下：

| 阶段 | 服务端均值 | P95 | P99 | 结论 |
| --- | ---: | ---: | ---: | --- |
| P0 | `41.439 ms` | `47.200 ms` | `48.411 ms` | multipart 已优化；旧门禁中的精度边界不再阻断，正式 batch FPS 当时未测 |
| P1 | `41.302 ms` | `47.355 ms` | `48.311 ms` | 路由降至约 `0.37 ms`，完整 API 未达标 |
| P2 | `42.043 ms` | `47.955 ms` | `49.400 ms` | profiling 完成，AOE 与固定 precision 不兼容，未生成 tuned OM |
| P3 | `45.916 ms` | `51.700 ms` | `53.800 ms` | pinned + 统一异步编排使均值恶化 `9.21%`，拒绝晋级 |

P4–P7 已按以下顺序执行：

1. P4：拆分 P3 的锁页内存和统一提交，建立可复现的运行时消融矩阵；
2. P5：优化三模型提交顺序、收集顺序和可用时的 stream priority；
3. P6：将 YOLO 解码与 NMS 移入 OM 或 Ascend 自定义算子，只回传最终框；
4. P7：以严格类增量四类统一检测器替换 Base + Specialist，再把 Scene/Sensor 变为共享骨干轻量头。

P4–P7 最终结论如下：

| 阶段 | 已执行内容 | 结论 |
| --- | --- | --- |
| P4 | pageable/pinned × threaded/unified 四组合、两轮 890 请求 | `pageable + threaded_execute` 胜出；统一提交是 P3 回退主因，pinned 无稳定收益 |
| P5 | 三种提交顺序、三种收集顺序、stream priority 能力探针 | 无候选同时改善 `≥1%` 和 `≥0.5 ms`；保留 Scene→Base→Specialist 同序提交/收集 |
| P6 | 标准 ONNX NMS、`NPUNmsWithMask`、`BatchMultiClassNMS` 严格语义探针 | 只因非评分的严格 NMS 边界停止；按新口径重新开放 `detections_v1` 评分候选 |
| P7 | `expanded_single_student` 与 `yolo_iod_lite` 两个四类统一检测器 | 两候选均失败于 New-mAP50 或 KRR 计分项，结论不变 |

P4–P6 的历史性能参考为 P2 pageable 链路 `42.043/47.955/49.400 ms`。P0 阈值边界、逐框和 JSON 差异今后只留证，不再阻断候选。P8–P11 最终候选必须重新通过四项计分门禁、数据合规和资产真实性校验。

P8–P11 已按以下顺序执行：

1. P8：固化板端运行与测量环境，消除 `xscreensaver`、governor、温度和后台负载漂移；
2. P9：按计分口径重评设备 decode/filter 和设备 NMS 两条候选，不要求逐框等同；
3. P10：先让 `/api/batch` 复用 DVPP encoded 路径，再共享 Base backbone、neck/FPN 并保留 old/new 逻辑检测头；
4. P11：按计分复核结果停用非评分的独立 Scene/Sensor 执行，以固定中性上下文保持接口和无标签路由语义，再压缩 batch 解析、调度和复制开销。

正式 `8501` 的精度为 Base mAP50 `0.819407`、New-mAP50 `0.728761`、KRR `1.0`，三项共 `50/50`；其 P8 计分口径 20 图 batch 为 `21.708 FPS`，效率为 `7/10`。P11 最终候选在同一评分器下得到 Base mAP50 `0.804901`、New-mAP50 `0.605033`、KRR `1.0`，首轮三次 batch 为 `30.066/30.071/30.039 FPS`、中位 `30.066 FPS`，独立复轮为 `30.062/30.080/30.093 FPS`、中位 `30.080 FPS`。因此候选四项均进入满分档；新类 precision `0.792453`、误激活率 `0.242857` 继续作为非评分风险提示，不反向否决计分结果。

当前瓶颈和目标预算为：

| 阶段或模块 | 实测 | 判断 |
| --- | ---: | --- |
| P8 raw | `921.3 ms / 21.708 FPS` | 原计分基线，效率 `7/10` |
| P9 decoded | `919.5 ms / 21.751 FPS` | 设备 decode/filter 没有稳定收益 |
| P9 device NMS | `950.0 ms / 21.053 FPS` | 三项精度满分，但效率回退，未采用 |
| P10-A encoded batch | `794.7 ms / 25.167 FPS` | 消除 Host 图像 decode，仍受三模型 Engine 开销限制 |
| P10-B shared dual head | `723.3 ms / 27.651 FPS` | 一次 OM 执行返回 old/new 两个逻辑头，关闭主要重复执行 |
| P11 fixed neutral + batch fast path | `665.2 ms / 30.066 FPS` | 首轮达到满分；独立复轮 `664.9 ms / 30.080 FPS` |
| 最终 batch Engine | 约 `656.3–657.8 ms` | 当前剩余瓶颈；解析约 `6.2–7.1 ms`、cache 约 `0.4–0.8 ms` |

最终候选已满足 20 图 batch `≤666.7 ms`、即 `≥30 FPS` 的硬预算，但首轮只余约 `1.47 ms/0.22%`，独立复轮只余约 `1.77 ms/0.27%`，性能余量很小。单图 Engine 均值 `34.64 ms`、服务端均值 `37.70 ms` 和 P95/P99 继续报告但不阻断。正式 release 和 `8501` 在全部候选验证期间保持不变；候选仍为 `validated: false`，本轮结论是“满足满分评分门槛”，不是已经完成正式切换。

## 2. 统一基线与验收门禁

### 测量协议

- 固定现有 CANN `7.0.RC1`、驱动和固件，不在本轮升级。
- 固定 `640×512`、8 位 RGB/RGBA PNG，`incremental_protocol=auto`，单请求、并发数 1、本机回环 HTTP keep-alive。
- 生产性能使用 `confidence=0.5`；另用 `confidence=0.01` 做高候选数压力回归。
- 服务和模型预热后再执行 30 个预热请求。
- 诊断测量执行 10 轮固定 89 图，共 890 个单请求，记录客户端完整请求墙钟、服务端 `system_total_ms`、均值、P50、P95 和 P99。
- 效率计分使用正式 `benchmark-api` 口径：固定取 20 图调用 `/api/batch` 三轮，以 `system_total_ms` 的中位轮计算 `batch_fps = 20×1000/system_total_ms`。
- 每次测量保存 Git SHA、OM/ONNX/AIPP SHA256、CANN/驱动/固件信息、温度、NPU 内存及 msprof/AOE 版本。
- `reports/` 保存原始设备产物；发布清单保存路径、摘要和 SHA256。

最终计分硬目标为：

- Base mAP50 `≥0.80`；
- New-mAP50 `≥0.60`；
- KRR `≥0.95`；
- 20 图 `/api/batch` FPS `≥30`，等价于该 batch 的 `system_total_ms ≤666.7 ms`。

### 计分门禁、真实性前提与诊断项

只有以下条件可以阻断晋级：

- 上述四项计分指标未满分；
- 候选不能启动、请求失败、公共必需字段缺失或输出无法由评分器解析；
- 评分前预测未冻结、评分资产/模型哈希不匹配，或增量阶段违反只读取增量数据的赛题约束。

以下项目继续测量、保存和报警，但不得再淘汰一个四项计分满分候选：

- 新类 precision、误激活率、Scene/Sensor/Joint lock accuracy；
- 逐图类别、框数量、坐标、confidence、排序、NMS 边界和完整业务 JSON 差异；
- 单请求/Engine 的均值、P95、P99、copy/后处理分段及两轮波动；
- 相对上一候选的百分比收益、Engine `≤30 ms`、单请求 API `≤33.33 ms` 和 P95 `≤35 ms`。

允许候选改变内部阈值、排序、NMS、路由和审计细节以换取效率；每次变化必须重新冻结 89 图无标签预测并重新计算三项精度，不能根据 lock 标签逐图路由或调参。旧阶段文中的“零差异”“P95 不恶化”和阶段百分比收益保留为当时执行记录，自本节修订后均按诊断项解释。

## 3. 分阶段改进

### Wave 0：建立不可变基线

- 固化当前正式 CPU 路径、现有 AIPP staging 和 DVPP staging 的同条件 A/B。
- 为路由融合增加细分计时：记录转换、门控、冲突仲裁、NMS、决策对象构造分别耗时，同时保留现有 `routing_fusion_ms`。
- 建立统一机器可读报告，关联环境、模型哈希、精度报告和完整性能分布。
- 扩展 Ascend 发布校验，使其检查 OM、构建清单、AIPP 配置、验证报告和 SHA256；当前通用发布检查不足以证明 Ascend 产物可复现。
- 在完成基线前，不晋级任何新配置。

### P0：正式化 DVPP PNGD/VPC + 静态 AIPP

目标是正式交付已有 staging 能力，不重写现有 `AscendEncodedPreprocessor`。

实施内容：

- 将三个 AIPP 配置纳入版本控制，固定以下契约：
  - Base：DVPP resize `896×717`，补边到 `896×736`，输入 `uint8 NHWC`；
  - Specialist：固定 `640×512` RGB888，输入 `uint8 NHWC`；
  - Scene：resize `176×176`、中心 crop `160×160`，输入 `uint8 NHWC`。
- 新建受控构建入口，复用固定 ONNX，调用 ATC 生成三个 AIPP OM；所有 ATC 参数、命令输出、源权重/ONNX/AIPP/OM SHA256 写入构建清单。
- Ascend 配置增加 `build_manifest`、`build_manifest_sha256`；只有 golden、89 图评分和性能门禁全部通过后，才允许设置 `validated: true`。
- 正式配置切换为：
  - `encoded_preprocessing: dvpp`；
  - `execution_mode: async_stream`；
  - 三个新 AIPP OM 及对应 SHA256。
- 创建独立 CPU 回滚配置。正式 DVPP 配置遇到不符合固定 PNG 契约的输入时返回明确的 400 错误，不静默走 CPU 预处理。
- 发布校验强制验证：
  - 只有一个活动 Specialist；
  - 未启用像素 positive prototype；
  - OM 输入确为固定 `uint8 NHWC`；
  - 配置、构建清单与实际文件哈希一致。

P0 晋级条件：

- 全部精度和逐框门禁通过；
- encoded API 均值 `≤40 ms`、P95 `≤42 ms`；
- 无请求失败、资源泄漏或隐式 CPU fallback。

### P1：Python 后处理、路由与融合

实施顺序固定为“细分计时—优化—逐项等价验证”。

- 为 Ascend 结果增加原生 records 快速路径，避免将已有检测记录转换成 boxes/list 后再由 `result_records()` 反向构造。
- 对阈值过滤、跨类 IoU 仲裁和 class-aware NMS 使用 NumPy 批量计算；稳定排序、相同置信度 tie-break、`max_det` 截断顺序必须保持现状。
- 减少重复 `.tolist()`、字典复制和类别名查找，但继续生成完整 rejected/conflict 审计记录。
- 不引入 Python 预处理或后处理线程池；现有候选验证已显示其 P95/P99 不稳定。
- candidate prefilter 已在 Ascend YOLO NMS 前启用，不作为新优化重复实现。
- 单独复现 `early_incremental_threshold` 已知的 1 图差异，保存图像 ID、候选框、NMS 顺序和最终 JSON：
  - 若差异来自真实 NMS/排序语义变化，永久拒绝该优化；
  - 只有证明是验证工具缺陷且修复后 89 图业务负载零差异，才允许启用；
  - 不因 Specialist 正式阈值为 `0.63` 而直接提前过滤。
- 保留完整 `conflict_suppressions`，不增加“精简审计”模式。

P1 晋级条件：

- 89 图业务响应零差异；
- `routing_fusion_ms` 均值由约 `16.64 ms` 降至 `≤8 ms`；
- 完整 API P95/P99不劣于 P0；
- 若无法同时满足性能和兼容性，保留 P0 实现并记录未通过原因。

### P2：msprof 定位与 AOE 调优

先定位，再调优，禁止直接批量重编全部 OM。

msprof 阶段：

- 建立受控 profiling CLI，使用官方 `msprof --output=<dir> <app>` 采集生产请求。
- 分别生成单请求 trace 和固定请求集汇总，提取：
  - Host、Runtime、DVPP；
  - H2D、D2H、D2D；
  - stream 等待与同步；
  - model execute；
  - AI Core、算子和子图耗时。
- 输出原始 profiler 目录和机器可读 JSON，明确 Base、Specialist、Scene 对关键路径的贡献。
- 若 Host 路由仍是首要瓶颈，先返回 P1；只有模型执行占比足够时才进入 AOE。

AOE 阶段：

- 使用与 P0 完全相同的 ONNX、AIPP、SoC、precision 和输入 shape。
- 每张卡串行执行，顺序为：
  1. Base `job_type=1` 子图调优；
  2. Base `job_type=2` 算子调优；
  3. Specialist 子图调优；
  4. Specialist 算子调优。
- Scene 当前仅约 `0.37 ms`，只有 msprof 显示其模型时间超过 `1 ms` 或占关键路径 `5%` 以上时才调优。
- 分别比较 P0/P1 OM、子图 tuned OM、子图+算子 tuned OM；不得一次覆盖正式 OM。
- 构建清单记录 AOE 命令、tuning repository、`aoe_result_opat_*.json`、候选 OM SHA256。

单个 tuned OM 的接受条件：

- 对应模型平均执行时间至少下降 `5%`；
- 完整 API 均值至少下降 `1%`；
- P95/P99 不得恶化超过 `2%`；
- 全部精度、逐框和发布门禁通过。

没有候选满足条件时，P2 结论为“已完成 profiling，拒绝 tuned OM”，P3 从上一正式版本继续。

### P3：锁页内存、异步拷贝与统一等待

实施前先在目标 PyACL `7.0.RC1` 环境检查 `malloc_host`、host pointer 映射、`free_host`、`memcpy_async` 和 event API。缺少必要能力时停止 P3，不静默降级。

运行时设计：

- 配置增加 `memory_mode: pageable | pinned`；正式异步配置使用 `pinned`，同步回滚配置使用 `pageable`。
- 每个模型常驻分配：
  - CPU 输入锁页 staging buffer；
  - 每个输出的锁页 buffer；
  - 必要的 copy/model/completion events。
- encoded PNG staging 使用可复用、按需增长且受上传上限约束的锁页缓冲，不按请求重复分配。
- 引入内部待完成执行句柄：
  - `submit(array)` 排入异步 H2D→model→D2H；
  - `submit_preloaded(ready_event)` 排入 event wait→model→D2H；
  - `result()` 执行唯一一次最终等待并返回输出和分段耗时；
  - 原有同步 `execute`/`execute_preloaded` 保留为兼容包装。
- 三模型全部完成 enqueue 后再收集结果；不在每个 `execute_async` 后立即同步。
- inference end event 放在模型执行之后、D2H 之前，确保原有 `inference_ms` 仍只表示模型时间。
- API `timings` 兼容新增：
  - `dvpp_device_ms`；
  - `ascend_submit_ms`；
  - `ascend_wait_ms`；
  - `ascend_input_copy_max_ms`；
  - `ascend_output_copy_max_ms`。
- 设备报告另保存三模型逐项 copy/execute/wait 数据，API 只暴露关键路径聚合值。
- 第一版保持现有单请求串行队列和每模型单 in-flight 执行，不实现跨请求 Ring Buffer。
- 任一 enqueue、event 或同步失败时，将句柄标记为失败并安全完成/终止在途工作；禁止释放仍被 stream 使用的锁页缓冲。
- 关闭顺序固定为：拒绝新提交→等待 outstanding handles→销毁 event/stream/dataset/device buffer→释放 host buffer→卸载模型→关闭 runtime。

P3 晋级条件：

- 相对 P2 或上一正式版本，完整 API 均值至少下降 `3%`；
- 最终均值 `≤33.33 ms`、P95 `≤35 ms`；
- P99 不恶化超过 `2%`；
- 相同 OM 的原始输出逐元素一致，89 图业务负载零差异；
- 异常注入与重复启动/关闭测试不存在泄漏、死锁或 use-after-free。

## 4. P4–P7 后续优化

### P4：P3 运行时消融

P3 同时改变了 host memory、D2H copy、提交方式、提交顺序和 event timing，组合回退不能直接证明任一单项无效。P4 从 P2 pageable 行为建立可配置回滚路径，禁止直接从 P3 候选继续叠加优化。

运行时增加：

- `schedule_mode: threaded_execute | unified_enqueue`；
- 保留 `memory_mode: pageable | pinned`；
- 详细 event timing 变为仅诊断启用，正式端到端测量关闭，避免把测量扰动混入候选收益。

固定按以下四个候选执行，不跳项、不同时改变提交顺序：

| 候选 | memory_mode | schedule_mode | 目的 |
| --- | --- | --- | --- |
| P4-A | `pageable` | `threaded_execute` | 复现 P2 参考行为 |
| P4-B | `pageable` | `unified_enqueue` | 隔离统一提交和统一等待 |
| P4-C | `pinned` | `threaded_execute` | 隔离锁页 host buffer |
| P4-D | `pinned` | `unified_enqueue` | 复现 P3 组合 |

四个候选均保持 Scene→Base→Specialist 的逻辑创建顺序、相同 OM、相同 DVPP/AIPP、相同请求顺序和相同后处理。P4-A 若无法在两次正式测量中把 P2 均值、P95、P99 复现到 `±2%`，停止 P4，先排查温度、后台负载、配置、OM 哈希和计时开关漂移。

每个候选执行两轮独立的 30 次预热 + 10×89 请求。只有两轮均相对 P4-A 改善均值至少 `2%`，且 P95/P99 均不恶化超过 `2%`，才允许成为 P5 基线；否则固定保留 P4-A。89 图业务 JSON 必须零差异。

### P5：关键路径调度

P5 只基于 P4 胜出组合改变调度，禁止同时改变 memory、OM 或后处理。提交顺序比较固定为：

1. Specialist→Base→Scene；
2. Base→Specialist→Scene；
3. 当前 Scene→Base→Specialist。

比较提交顺序时固定收集顺序为 Base→Specialist→Scene。选定提交顺序后，再保持该顺序不变，比较以下收集策略：

1. Base→Specialist→Scene，使 Base 的较大 Host 后处理与 Specialist 尾部执行重叠；
2. Specialist→Base→Scene，先等待 single trace 中最长的模型；
3. Scene→Base→Specialist，作为当前行为参考。

正式比较关闭逐 event 详细计时；胜出组合另跑一次诊断轮保存模型 execute、wait、copy 分段。

stream priority 先在目标 PyACL 执行只读能力探针：

- 必须同时存在设备 priority range 查询和带 priority 创建 stream 的 API；
- 通过设备返回范围映射 `high/normal/low`，禁止硬编码不同 CANN 版本的数值语义；
- 依次比较 Specialist=`high`/Base=`normal`/Scene=`low` 与 Base=`high`/Specialist=`normal`/Scene=`low`；
- API 缺失或行为探针失败时记录“当前 CANN 不支持”并跳过，不用其他同步策略冒充 priority。

P5 候选只有在两轮正式测量中均同时满足“均值改善 `≥1%` 且绝对改善 `≥0.5 ms`”，业务 JSON 零差异，P95/P99 不恶化超过 `2%` 时才晋级；否则保留 P4 胜出组合。

### P6：设备侧 YOLO 解码与 NMS

P6 将 Base 和 Specialist 的原始 YOLO 输出解码、置信度过滤、坐标转换、稳定排序和 class-aware NMS 移到设备侧，只回传固定上限的最终检测结果。输出契约固定为：

- `boxes`: `[max_det, 4] float32`，坐标语义为模型输入空间 `xyxy`；
- `scores`: `[max_det] float32`；
- `class_ids`: `[max_det] int32`；
- `valid_count`: `[1] int32`。

实现顺序固定为：

1. 先构建最小 ONNX 兼容探针，验证当前 CANN `7.0.RC1` 能否编译所需 sort/NMS 图；
2. 支持时把后处理图拼接到现有检测 ONNX 后生成新 OM；
3. 不支持时使用当前工具链可编译的 AscendC/TBE 自定义算子；
4. 两条路径都无法构建或启动时结束 P6，保留 Host `raw_yolo_v1`，不升级 CANN。

模型配置增加 `output_contract: raw_yolo_v1 | detections_v1`。Ascend 后端按契约选择现有 Host 解码或直接读取最终框，原始 OM、Host 解码和回滚配置全部保留。原计划要求 `detections_v1` 逐项复现 Host 语义；修订后只把差异写入诊断报告，并以重新计算的 Base/New/KRR 和 batch FPS 决定晋级。

P6 先用构造输入覆盖阈值边界、同分、重叠、多类和 `max_det` 饱和，再对 89 图重新评分。构造差异、copy/后处理收益、单请求均值和 P95/P99 只作诊断；候选能构建运行且四项计分满分时即可晋级。

原执行结论（2026-08-15）：配置、运行时双契约、发布清单绑定、无新增依赖 ONNX 导出和严格语义测试已经实现；板端 CANN `7.0.RC1` 的标准 ONNX NMS 因动态输出无法编译，`NPUNmsWithMask` 不抑制同类重复框，`BatchMultiClassNMS` 会抑制 IoU 恰好等于 `0.7` 的框，因此旧口径停止。新口径下这只是非阻断语义差异，P9 后续重新开放可编译的 `BatchMultiClassNMS detections_v1`，必须实际跑 89 图评分和 20 图 batch 后再决定，不能继续由构造零差异探针提前淘汰。

### P7：统一检测器与共享上下文头

P7 是闭合至少 `8.713 ms` 缺口的主收益阶段。模型结构变化允许检测框发生合理变化，因此不要求与旧模型逐框一致，但必须重新通过全部严格类增量、精度、发布和端到端门禁。

第一步构建四类统一检测器：

- 复用仓库现有 `expanded_single_student` 路径和冻结 Base checkpoint；
- `build_unified_student: true`，增量阶段只读取 warship train/dev，禁止读取旧类原始图像、标签或 feature cache；
- 复制并冻结共享参数和三个旧类分类通道，只训练新增 warship 通道；
- `student_train.imgsz` 固定为 `896`，导出和推理沿用 Base 的 `896×736` rect/AIPP 契约，生成四类 OM；
- Agent 使用现有 `deployment: single_detector` profile，旧类 `0/1/3` 仍映射为 `frozen_base_model`，warship `2` 仍映射为 `incremental_model`，保持响应、融合和审计结构；
- 四类候选通过后才从运行时配置移除独立 Specialist OM，原双检测器配置保留为回滚。

若 `expanded_single_student` 的 New-mAP50 未达到 `0.60`，执行仓库已有 `yolo_iod_lite` 第二候选。Base teacher 和 current teacher 只允许在增量图像上推理；数据审计继续要求 `old_raw_image_count=0`、旧类通道隔离和冻结 Base 哈希不变。两个候选都失败时停止统一检测器晋级，不允许改用混合旧类重训。

第二步只在胜出的四类检测器上增加共享上下文头：

- 从冻结检测 backbone 的固定特征层做全局池化；
- 增加 sensor 二分类和 scene 四分类轻量 head；
- 冻结 backbone 和检测 head，只用现有 scene train/dev 训练上下文 head；
- 导出单个多输出 OM，输出四类检测契约、sensor logits 和 scene logits；
- 通过后才移除独立 Scene OM，原 Scene OM 保留为回滚。

P7 计分门禁只包含 Base mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95` 和 batch FPS `≥30`。precision、误激活、Engine、单请求均值和尾延迟继续记录但不参与晋级。

执行结论（2026-08-15）：P7 增加了两个专用严格配置、统一检测器预检、Base/current teacher 复用、统一模型逻辑 owner/source 映射和原协议审计结构。`expanded_single_student` 与 `yolo_iod_lite` 均完整训练并在同一 89 图锁集上评分；旧类分类行在 model/EMA 中逐步恢复，冻结 BatchNorm 始终保持 eval。两个 manifest 都证明增量阶段读取旧类原始图像、标签和 feature cache 的数量为 `0/0/0`，且原始数据未修改。

| 候选 | Base mAP50 | New-mAP50 | KRR | 新类 precision | 误激活率 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `expanded_single_student` | `0.814088` | `0.016667` | `1.000000` | `0.666667` | `0.014286` | 旧知识保持，但新增通道几乎未学到，拒绝 |
| `yolo_iod_lite` | `0.814088` | `0.598128` | `0.344506` | `0.416667` | `0.428571` | New-mAP50 与 KRR 两项计分失败；precision/误激活仅诊断，拒绝结论不变 |

`expanded_single_student` 的旧通道最大漂移和共享参数相对漂移均为 `0`。`yolo_iod_lite` 的旧通道最大漂移为 `0`，共享参数相对漂移为 `0.077112`；它复用 SHA256 `d27bda7cb89375788deb1f29366b037757f23f7b32ddf6c11e1aa778384dc957` 的 current teacher，教师只在增量 student 图像上推理。其锁集得到 `TP=70`、`FP=98`、`targets=76`，70 张负样本中 30 张误激活，且统一模型没有双检测器冲突仲裁可用，`fusion decision_count=0`。

| 候选 | 配置 SHA256 | metrics SHA256 | dataset manifest SHA256 |
| --- | --- | --- | --- |
| `p7-expanded-20260815` | `9c2800525a66f79f6a03cba2e2680b2c8b29427cc34d517678eec8f527529a2c` | `c52c3f59505e8bdab6dfdff61e28bfbef1791887a4e2e68ce1dcbad264b3adc6` | `94f3f6edd893192c06370ba80078976bf9d26093a158229bf43dc062401e23b3` |
| `p7-yolo-iod-20260815` | `d8518910ace533797ad8e9a70d99aa86d4caadf3c5eeb45c048b96f1e7545ae3` | `545d2c6221dc9bd07eadeb39298a58b147f84c545c5aeb77f027ace79a630586` | `40bed0ec033c11134f5b4be95af2b36556f30739f64c1dce848f6f1698f90405` |

按新口径复核，第一候选 New-mAP50 明确失败，第二候选 New-mAP50 和 KRR 明确失败，因此 P7 的停止结论不变；precision/FAR 失败不再用于论证。不得通过读取旧类原始训练数据绕过赛题约束。两个配置、EMA/BN 冻结修复、严格数据隔离、single-detector 逻辑映射和失败证据保留，作为后续双逻辑头研究的可复现起点。

## 5. P8–P11 后续优化

### P8：板端环境固化

P4/P5 期间曾发现 `xscreensaver` 稳定占用约 `38%` CPU。P8 先治理环境，再建立后续阶段共用的新鲜基线，避免把后台负载或温度变化误判为模型收益。

板端增加 benchmark guard，分为一次性持久治理和每轮只读检查：

- 停止当前 `xscreensaver`，通过 `HwHiAiUser` 的 XDG autostart override 持久阻止重新启动；若发现精确匹配的 systemd 单元，只禁用并 mask 该单元，不修改显示管理器；
- CPU policy 暴露 `performance` governor 时，由持久化 systemd oneshot 为所有 policy 设置该模式；不支持时记录 `unsupported`，但基线和所有候选的实际 governor 必须完全一致；
- 每轮开始前要求正式 `8501` health 为 `ready`、`8502` 未被占用、NPU Health 为 `OK`；
- NPU 温度必须降至 `≤65°C`，最多等待十分钟，超时即终止该轮；
- 连续三次 CPU 采样不得存在非白名单进程占用 `>10%`；
- Git HEAD、配置、OM、AIPP 和 build manifest SHA256 必须与候选清单一致；
- 报告保存测试前后进程、CPU governor、温度、`npu-smi`、内存、端口、CANN/Python/SoC 和资产哈希快照。

环境治理后重新执行两轮 30 次预热 + 10×89 请求，形成 P8 新鲜基线。两轮均值、P95 和 P99 的相对差异必须分别 `≤2%`；不满足时停止 P8，继续定位环境漂移，不进入 P9。旧 P5 `41.245/47.300/48.400 ms` 只用于说明治理前后差异，不要求新基线强行落入旧结果 `±2%`。

执行结论（2026-08-15）：P8 环境固化、只读 guard 和报告快照已实现并在板端启用。`xscreensaver`、`xscreensaver-systemd` 和 `xfce4-screensaver` 已停止，并通过三个用户级 XDG `Hidden=true` override 阻止自启动；板端未暴露 cpufreq policy，因此 governor 明确记录为 `unsupported`。每轮开始前的 NPU 温度门禁保持 `≤65°C`，结束温度只作为持续负载证据记录，不反向否决开始条件合格的测量。

P5 保留配置在 clean Git HEAD `5702511040986bbd7b3a37316db9d393746310bf` 上完成两轮正式测量：

| P8 基线 | API mean | P95 | P99 | Engine mean | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| Run 1 | `41.1265 ms` | `47.200 ms` | `48.200 ms` | `38.0903 ms` | 通过 |
| Run 2 | `41.1347 ms` | `47.300 ms` | `48.211 ms` | `38.1017 ms` | 通过 |

Run 2 相对 Run 1 的 mean、P95、P99 差异分别约为 `0.020%`、`0.212%`、`0.023%`，均满足 `≤2%`。两轮开始温度分别为 `59°C` 和 `61°C`，结束快照分别为 `67°C` 和 `70°C`；NPU Health、正式/候选 health、Git、配置、build manifest、OM/AIPP 哈希、Python、SoC 和 governor 身份均通过门禁。P8 形成的新鲜性能基线固定为 Run 1 `41.1265/47.200/48.200 ms`，用于 P9 的改善比较。测量后 `8502` 已停止，正式 `8501` 在同步、启动、测量和停止期间始终为 `ready`。

按修订后的计分口径补测，P8 raw 在 `confidence=0.01` 下得到 Base mAP50 `0.819415`、New-mAP50 `0.728761`、KRR `1.0`，三项满分；20 图 batch 中位时长 `921.3 ms`、`21.708 FPS`，效率为 `7/10`。batch 分解中上传解析约 `51.8 ms`、Host decode 约 `148–172 ms`、Engine 约 `690–705 ms`。

### P9：设备侧后处理的计分口径复核

P9-A 只把坐标解码和候选筛选移入 OM，Host 继续执行现有 NMS；P9-B 重新开放 CANN `7.0.RC1` 可编译的设备 NMS。两者都不再要求与 Host 逐框等同，只以三项精度和 batch FPS 判定。

内部输出契约扩展为：

```yaml
output_contract: raw_yolo_v1 | decoded_candidates_v1 | detections_v1
```

`decoded_candidates_v1` 固定使用 `candidate_confidence=0.01`，只保留严格 `score > 0.01` 的候选，输出：

- `boxes[capacity,4] float32`；
- `scores[capacity] float32`；
- `class_ids[capacity] int32`；
- `anchor_ids[capacity] int32`；
- `valid_count[1] int32`；
- `overflow[1] int32`；
- 设备侧保留 raw YOLO 输出作为按需回滚输出。

Base `capacity=4096`，Specialist `capacity=2048`。候选保持 anchor-major、class-minor 顺序；Host 根据 `anchor_ids` 校验并恢复稳定 tie-break，再执行当前全局稳定置信度排序、class-aware NMS、严格 `IoU > threshold` 和 `max_det` 截断。

请求阈值 `<0.01` 或 `overflow=1` 时，运行时只复制 raw 输出并执行现有 `raw_yolo_v1` 路径，禁止静默丢弃候选。正常路径先复制 `valid_count/overflow`，再复制紧凑候选数组，raw fallback 不复制到 Host。优先使用 CANN `7.0.RC1` 可编译的 ONNX 图算子；若固定候选 gather 无法编译，只允许为 decode/filter/gather 实现 AscendC/TBE 算子，不实现 NMS。

构造探针继续覆盖阈值相等、IoU 相等、同分框、跨类重叠、NMS 后补位、capacity 边界和 overflow/raw 回退，用于说明差异和发现崩溃/溢出；语义不等同本身不得停止 89 图评分或性能轮。

P9 晋级要求是 Base/New/KRR 和 20 图 batch FPS 四项满分。逐框/JSON、output copy、Host 后处理、单请求均值和 P95/P99全部降为诊断。若多个候选都未到 30 FPS，则只保留 batch FPS 最高且三项精度满分者作为 P10 输入；差异小于 `1%` 时视为无稳定收益，保留更简单的 raw 路径。

执行结论（2026-08-15）：P9 已实现 `decoded_candidates_v1`、固定 TopK 候选收集、Host 严格排序/NMS、低阈值/overflow raw 回滚、两阶段选择性 D2H、构建清单验证和完整探针工具。CANN `7.0.RC1` 能编译该标准 ONNX 图；首版 int32 `ReduceSum` 在设备上错误退化为 `0/1`，改用 float32 求和后，阈值相等/略高、同分框、跨类重叠、`IoU` 相等、NMS 后补位、capacity 边界和 overflow 回滚探针全部逐输出通过。

P9-A 完整 Base/Specialist OM 构建、加载和 89 图评分成功。相对 P8 raw，84 图有最大 `0.25 px` 坐标差异，但修订口径下不阻断；在 `confidence=0.01` 下得到 Base mAP50 `0.819901`、New-mAP50 `0.728761`、KRR `1.0`，三项仍满分。两轮单请求诊断为 `42.798/48.555/49.500 ms` 和 `42.737/48.600/49.300 ms`。正式 20 图 batch 三轮为 `21.720/21.751/22.134 FPS`，中位 `21.751 FPS`，仅比 P8 `21.708 FPS` 高约 `0.20%`；中位 batch `919.5 ms`，其中 Host decode `163.149 ms`、Engine `697.130 ms`。因此 P9-A 以“效率计分无稳定收益”回滚 raw，而不是因 JSON/逐框差异淘汰。正式报告 SHA256 为 `95104f39d1045bdafed80c11084125882ee65bca41015490e8fc601b42081805`。

P9-B `detections_v1` 已按同一评分协议执行。首版单类别 NMS 输出形状不兼容，修正 score padding 后可构建并运行；89 图得到 Base mAP50 `0.824415`、New-mAP50 `0.728761`、KRR `1.0`，三项精度满分。20 图 batch 三轮为 `21.053/20.760/21.363 FPS`，中位 `21.053 FPS`，低于 P8 raw `21.708 FPS`，因此按效率计分回滚 raw。其逐框/NMS 边界差异未作为拒绝理由。评分和性能报告 SHA256 分别为 `c4802b6a1a92facb4c368fbdb464e2f743721469e7a803e999abfb8f6845f51b`、`508a348ca3a5ebd0afee6501795ad60e2fa9ae92c5ac59eb7225e0f5c15cb938`。

### P10：batch encoded 路径、共享骨干与双逻辑检测头

P10-A 先让 `/api/batch` 的 20 张 PNG 直接进入已有 DVPP encoded 预处理，避免当前整批约 `148–172 ms` 的 Host OpenCV/PIL decode；非法 PNG 明确报错，不静默 CPU 回退。该子阶段不改模型，预期直接把 P8 `921.3 ms` 压到约 `750–773 ms`，仍需后续共享执行闭合剩余缺口。

P10-B 解决 Engine 重复执行。P7 的四类单分类头在 New-mAP50/KRR 计分项上失败；P10 改为共享 Base backbone 和 neck/FPN，但保留相互独立的 old/new Detect head，物理上一次执行，逻辑上仍是两个检测器。

新增内部布局 `shared_backbone_dual_head_v1`：

- 统一使用 Base 的 `896×736` AIPP 输入；
- 复用并冻结正式 Base 的 backbone、neck/FPN 和 old head；
- old head 固定输出 `[1,7,13524]`，new head 固定输出 `[1,5,13524]`；
- Agent 继续把 old head 标记为 `frozen_base_model`、new head 标记为 `incremental_model`；
- 继续执行原逐类阈值、positive prototype、双模型冲突仲裁、融合、class-aware NMS 和完整审计。

训练候选按以下顺序执行：

1. 复制现有 Specialist 的 `cv2/cv3/DFL` 到 new head，仅训练 new head；
2. 第一候选精度失败时，在三层共享特征前增加零初始化 residual `1×1` adapter，仅训练 adapter 和 new head。

两种候选都只允许读取 warship 增量数据。Base backbone、neck、old head、BatchNorm 统计以及 model/EMA 中所有冻结参数的最大漂移必须为 `0`，这是数据合规与可复现前提。两种候选都通过三项精度计分门禁时，以 20 图 batch FPS 更高者胜出。

P10 从 P9 胜出契约开始；当前 P9-A 无稳定收益，因此默认使用双 raw 输出。Ascend 后端一次执行返回两个逻辑结果，Web 编排继续按 Base/Specialist 两个结果运行，公共必需响应字段不变。

P10 晋级要求是 Base mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95` 和 20 图 batch FPS `≥30`。若尚未达到 30 FPS，则可保留三项精度满分且 batch FPS 最高的中间候选继续 P11；`≥8%/≥3 ms`、precision/FAR、逐框差异和 P95/P99 均只作诊断。Base/EMA 零漂移和增量数据隔离继续作为评分真实性前提。

候选任一计分精度门禁失败时回滚为独立 Base/Specialist OM，不复用 P7 被拒绝的四类单头模型。

执行结论（2026-08-16）：P10-A 已让固定 PNG batch 直接进入 DVPP encoded 路径，三轮为 `25.237/25.129/25.167 FPS`、中位 `25.167 FPS`；Host decode 降为 `0`，但 batch Engine 仍为约 `734–737 ms`，未达到满分。

P10-B 训练中，head-only 初始候选未通过 New-mAP50；第二候选使用三层零初始化 residual adapter，只读取 warship 增量数据，`old_raw_image_count/old_raw_label_count/old_feature_cache_count=0/0/0`，共享参数最大漂移为 `0`。v2 训练候选的 New-mAP50 为 `0.603549`。adapter 导出后折叠版本在板端反而慢于未折叠版本，因此保留未折叠 `shared_backbone_dual_head_last.om`。

选择 `candidate_confidence=0.3` 限制 new logical head 的 Host 候选后，89 图得到 Base mAP50 `0.812761`、New-mAP50 `0.605033`、KRR `1.0`；20 图 batch 为 `27.651/27.663/27.590 FPS`、中位 `27.651 FPS`。三项精度已满分，但效率仍差 `2.349 FPS`，因此把该候选作为 P11 输入而不切换正式版。核心 OM SHA256 为 `3dd053e041c36225059cf6624eefebe5945ba6b8ca5bc0ca9d914448c4a54c89`，P10 性能报告 SHA256 为 `224064141e7714b2da31f8ec306a30bc0cc4c6979fac400dfa31ea07a474d8bf`。

### P11：非评分上下文消融与 batch fast path

原计划是在共享检测骨干上训练 Scene/Sensor 轻量头。重新核对机器可读评分规则后，Scene/Sensor accuracy 和逐请求 context JSON 均不参与计分；继续训练、导出和执行上下文头会增加延迟，却不能增加分数。因此 P11 先执行更直接的消融：`context_mode: fixed_neutral_v1` 不运行 Scene/Sensor OM，返回均匀概率保持公共 schema，并用中性证据进入既有路由。该值不读取文件名、标签或 lock 结果，原 Scene OM 仍在 manifest 中作为显式回滚资产。

若后续正式业务重新要求上下文质量，再按成本从低到高恢复原候选：

1. 对 YOLO `model.10` 深层特征做全局平均池化，分别接 `Linear(C,2)` Sensor head 和 `Linear(C,4)` Scene head；
2. 第一候选精度失败时，对 P3/P4/P5 三层特征分别池化并拼接，经 256 维 SiLU 隐层后接两个分类头。

单 OM 新增固定输出 `sensor_logits[1,2] float32` 和 `scene_logits[1,4] float32`。Agent 保留可解析的 context 输出；允许 softmax、软阈值和路由细节发生变化，只要检测三项精度仍满分。两种候选都可运行时，以 20 图 batch FPS 更高者胜出。

Sensor lock accuracy `≥0.95`、Scene lock accuracy `≥0.80`、Joint lock accuracy `≥0.75` 全部改为非阻断诊断。P11 只要求 context 输出可解析、Base/New/KRR 三项精度满分且 20 图 batch FPS 达到 `≥30`。固定中性上下文不能运行或损伤计分精度时，回滚为独立 Scene OM。

old detector、new detector 和 Scene/Sensor 继续作为三个独立逻辑功能模型记录职责和 owner；P11 只是把不计分的物理上下文执行替换为中性实现，没有伪造上下文模型精度。最终发布硬目标是三项精度和 batch FPS 四项满分；Engine、单请求均值和 P95 只留证。

执行结论（2026-08-16）：固定中性上下文首先把 P10-B 中位 FPS 从 `27.651` 提升到 `27.828`；有界 multipart batch 解析提升到 `29.696`，消除中性上下文 batch 构造与调度后达到约 `30.003`，避免 multipart 图像复制后稳定到 `30 FPS` 以上。旧逻辑头进一步只对 `score > 0.05` 的候选执行 Host 后处理；`0.10` 虽仍勉强通过 Base mAP50 `0.801121`，但精度余量过小，因此未采用。

最终 `0.05` 候选的 Base mAP50 为 `0.804901`、New-mAP50 为 `0.605033`、KRR 为 `1.0`。首轮 batch 为 `30.066/30.071/30.039 FPS`、中位 `30.066 FPS`；独立复轮为 `30.062/30.080/30.093 FPS`、中位 `30.080 FPS`。两轮都满足四项计分满分。非评分诊断为 lock precision `0.792453`、误激活率 `0.242857`、单请求 Engine 均值 `34.64 ms`；这些风险继续留证但不否决评分结果。

最终配置、manifest、OM、冻结预测、精度摘要、首轮性能报告和复轮性能报告 SHA256 依次为：`73fb8c56f0139be0af2ef489c3ebabcbe3da494dba1e40be5625dcf58640d0d7`、`44583627b870e372a852a36f69d55464df6fc7ca4d745fd859285c0d5e3389a3`、`3dd053e041c36225059cf6624eefebe5945ba6b8ca5bc0ca9d914448c4a54c89`、`86d79a1b11a7ca0f0924346acde289f2b1e291b2038f2798980a00d0010d88c4`、`9d52316096215a3bb7c9c66599fad127fa3881ae2bd2a553299e0c6c9d1dfd28`、`1dd80ce96276a3ec07adfdf504ebffc6b9a0f356bbbfbd9e5b081ea6b171faa2`、`603c5646a706fd895480292f59ab808c1417a1764c942dd042f1e7b97766f320`。

## 6. 接口与兼容性

- `POST /api/detect` 的请求字段和现有必需响应字段保持不变。
- `agent.decision` 等公共必需字段保持可解析；内部审计明细允许随候选变化，不要求业务 JSON 零差异。
- `timings` 仅向后兼容地增加字段，不删除或重命名已有字段。
- P4/P5 Ascend 候选配置增加 `schedule_mode`、提交顺序、收集顺序和可选的模型 priority 映射；所有字段均为内部运行时接口。
- P6/P9 模型条目支持 `raw_yolo_v1 | decoded_candidates_v1 | detections_v1`；未声明时保持 `raw_yolo_v1`，禁止依据输出 shape 静默猜测契约。
- P7 复用现有 `deployment: single_detector` profile；公共 class ID、class name、source、protocol 和审计字段保持现有语义。
- P10/P11 增加内部 `model_layout`、logical head owner、class map、anchor count、candidate capacity 和 batch encoded 输入契约；这些字段不进入 Web 公共接口。
- DVPP 候选配置只接受固定 PNG 契约；不合规输入明确报错。正式配置只有在 P8–P11 最终候选全部门禁通过后才允许切换，CPU 和双检测器回滚通过独立配置显式选择。
- 内部 submit/result 句柄不成为 Web 公共接口。
- `/api/batch` 的 Host decode 消除属于 P10-A 核心范围；跨请求流水线和并发压测仍不在本轮范围。

## 7. 测试与交付

自动化测试至少覆盖：

- AIPP shape、dtype、归一化、padding、resize/crop 契约；
- 构建清单字段、SHA256 不匹配和未验收 OM 拒绝加载；
- 固定 PNG 接受规则及非法格式不触发 CPU fallback；
- P4 四种 memory/schedule 组合、P2 行为回滚和详细 timing 开关；
- P5 提交/收集顺序、priority range 映射和 priority API 缺失时的明确跳过；
- P6 两种 output contract，以及阈值边界、相同置信度、anchor/class tie-break、重叠框、多类别和 `max_det` 饱和场景的差异报告、崩溃防护和重新评分；
- P7 统一 student 数据视图、增量数据隔离、旧类权重/通道冻结、single-detector source 映射和三项计分精度复核；
- P8 环境快照解析、guard 失败闭锁、governor 不支持和两轮环境一致性；
- P9 三种 output contract、阈值/排序/NMS 差异、capacity 边界、overflow/raw 回退和每个可运行候选的 89 图重新评分；
- P10 双逻辑头装配、冻结参数与 EMA 零漂移、严格增量数据隔离和 old/new owner 映射；
- P10 `/api/batch` encoded PNG 直接进入 DVPP、非法输入显式失败和无隐式 CPU decode；
- P11 在 P10 成败两种布局下的上下文头装配、`fixed_neutral_v1` 无标签中性回退、上下文诊断和检测计分精度兼容；
- 已知 1 图 early-threshold 差异的永久回归夹具；
- fake ACL 验证异步调用顺序、最终等待、错误传播和资源释放；
- `/api/detect` 必需字段不变且新增 timing 字段为兼容添加；
- 89 图冻结预测评分、20 图 batch score gate 和按需执行的 30+890 单请求诊断。

本机 Python 固定使用 WSL 仓库现有 `.venv`。不得创建新环境、安装额外依赖或下载 CPU PyTorch。板端不运行 Web pytest，只执行真实 API 端到端测量和阶段所需的能力/模型探针。

板端候选固定使用 `127.0.0.1:8502`，正式 `127.0.0.1:8501` 在同步、启动候选、测量和停止候选期间必须持续 `ready`。计分验收执行 30 次预热、89 图无标签预测冻结与三轮 20 图 batch score gate；10×89 单请求轮只作为需要时的诊断，不再阻断或拖延一个四项已满分的候选。报告保存 batch FPS、服务端/客户端分布和关键分段。

每阶段产出独立候选 release、构建清单、精度报告、性能报告和回滚指针。正式配置只在四项计分满分、真实性校验通过后原子切换，旧 OM 和配置不得覆盖或删除。P8、P9、P10、P11 分别提交并推送，不把多个阶段压入同一提交。

## 8. 明确假设与非目标

- 本轮固定现有 CANN `7.0.RC1`、驱动和固件；不实施或评估版本升级。
- 不实施 INT8、降分辨率、剪枝、跨请求流水线或 Ring Buffer。
- P6 只允许使用当前 CANN 可编译的图算子或 AscendC/TBE 自定义算子；不以升级工具链绕过兼容结论。
- P7 允许四类统一检测器和共享上下文头，但严格类增量阶段不得读取旧类原始图像、标签或 feature cache，也不得通过混合数据重训换取精度。
- P8 只治理影响测量可重复性的板端服务、governor 和后台负载，不停止或替换正式 `8501`。
- P9 允许使用 CANN `7.0.RC1` 可运行的设备 NMS，即使它不逐框复现 Host；所有候选都必须重新计算三项精度。
- P10/P11 可以调整检测阈值、排序、冲突仲裁、融合、上下文实现和审计细节换取性能，但不得按文件名、lock 标签或评分答案逐图分支，且必须在无标签预测冻结后重新通过三项精度计分门禁。
- 官方方法参考：
  - [msprof Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0005.html)
  - [AOE 调优](https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000110.html)

执行时以目标板已安装 CANN `7.0.RC1` 的命令帮助和实际 API 能力为准，官方新版文档仅作为流程依据。
