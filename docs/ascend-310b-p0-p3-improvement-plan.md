# AgileAgent 昇腾 310B P0–P11 推理优化计划与执行记录

## 1. 摘要

本文记录 AgileAgent 在 Atlas 200I DK A2 / Ascend310B1 上完成的 P0–P9 实测结论，以及 P10–P11 后续优化、接口、消融矩阵和晋级门禁。详细设备证据保存在 `docs/ascend-310b-current-status.md`；本文以同条件端到端实测和严格类增量锁集结果替代执行前的跨批次估计。

P0–P3 已全部结束，结果如下：

| 阶段 | 服务端均值 | P95 | P99 | 结论 |
| --- | ---: | ---: | ---: | --- |
| P0 | `41.439 ms` | `47.200 ms` | `48.411 ms` | multipart 已优化，但精度边界、provenance 和性能门禁失败 |
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
| P6 | 标准 ONNX NMS、`NPUNmsWithMask`、`BatchMultiClassNMS` 严格语义探针 | CANN `7.0.RC1` 无兼容实现；保留 `raw_yolo_v1`，未启动 8502 性能轮 |
| P7 | `expanded_single_student` 与 `yolo_iod_lite` 两个四类统一检测器 | 两候选均未通过完整精度门禁；停止共享上下文头、单 OM 和性能门禁 |

P4–P6 的性能参考固定为 P2 pageable 链路 `42.043/47.955/49.400 ms`。该链路仍继承 P0 的 Base 阈值边界差异，只能用于性能筛选，不能直接晋级正式 release。P8–P11 最终候选必须重新通过完整精度、P0 严格阈值边界、provenance、语义和发布门禁。

P8–P11 按以下顺序执行：

1. P8：固化板端运行与测量环境，消除 `xscreensaver`、governor、温度和后台负载漂移；
2. P9：设备侧只执行 YOLO 解码和候选筛选，Host 保留严格 NMS；
3. P10：共享 Base backbone、neck/FPN，保留相互独立的 old/new 逻辑检测头；
4. P11：将 Scene/Sensor 轻量头并入检测骨干，移除独立上下文 OM 的执行开销。

现有正式精度为 Base mAP50 `0.819407`、New-mAP50 `0.728761`、KRR `1.0`、新类 precision `0.933333`、误激活率 `0.014286`，均处于满分档。正式 API 约 `13.99 FPS`，尚未达到 30 FPS 满分档；P5 保留组合为 `41.245/47.300/48.400 ms`，约 `24.25 FPS`，作为 P8 固化环境前的最新候选参考。

当前瓶颈和目标预算为：

| 模块 | 实测或推导 | 判断 |
| --- | ---: | --- |
| P2 完整 API | `42.043 ms` | 距均值目标仍差 `8.713 ms`，即 `20.7%` |
| P2 Engine | `38.723 ms` | 占完整 API 约 `92.1%`，是主瓶颈 |
| Host/API 非 Engine | 约 `3.320 ms` | 即使完全消除也不能达到 `33.33 ms` |
| P2 single msprof Base | `21.138 ms` | 主要模型关键路径之一 |
| P2 single msprof Specialist | `26.263 ms` | 单请求 trace 中最慢，应优先提交 |
| P2 single msprof Scene | `3.007 ms` | 可通过共享骨干轻量头消除独立执行 |
| P3 最大输出复制 | `2.981 ms` | P6 设备侧后处理的直接优化对象 |
| 路由与融合 | 约 `0.374 ms` | 已不是主要瓶颈，不再投入大规模 Python 优化 |

为达到完整 API 均值 `≤33.33 ms`，最终 Engine 均值预算固定为 `≤30 ms`。P7 没有产生通过精度门禁的统一检测器，因此 P8–P11 从正式双检测器语义和 P5 保留调度继续；正式 release 和 `8501` 在全部候选验证期间保持不变。

## 2. 统一基线与验收门禁

### 测量协议

- 固定现有 CANN `7.0.RC1`、驱动和固件，不在本轮升级。
- 固定 `640×512`、8 位 RGB/RGBA PNG，`incremental_protocol=auto`，单请求、并发数 1、本机回环 HTTP keep-alive。
- 生产性能使用 `confidence=0.5`；另用 `confidence=0.01` 做高候选数压力回归。
- 服务和模型预热后再执行 30 个预热请求。
- 正式测量执行 10 轮固定 89 图，共 890 个请求，记录客户端完整请求墙钟、服务端 `system_total_ms`、均值、P50、P95、P99 和 FPS。
- 每次测量保存 Git SHA、OM/ONNX/AIPP SHA256、CANN/驱动/固件信息、温度、NPU 内存及 msprof/AOE 版本。
- `reports/` 保存原始设备产物；发布清单保存路径、摘要和 SHA256。

最终硬目标为：

- 完整 encoded PNG API 均值 `≤33.33 ms`；
- P95 `≤35.00 ms`；
- 任一候选不得以均值提升换取 P95/P99 超过上一正式版本 `2%` 的恶化。

### 精度与语义

所有阶段均须满足：

- Base mAP50 `≥0.80`；
- New-mAP50 `≥0.60`；
- KRR `≥0.95`；
- 新类 precision `≥0.90`；
- 误激活率 `≤0.05`。

P0、P2 等可能产生数值差异的阶段还须满足：

- 89 图逐图类别和检测框数量相同；
- 坐标最大绝对差 `≤1.0 px`；
- confidence 最大绝对差 `≤0.02`；
- Base/New-mAP50、KRR 相对当前正式值的绝对下降均不超过 `0.005`。

P1、P3 属于语义保持型优化，要求业务响应完全一致，包括：

- detections、class counts、context、协议状态、路由结果；
- 完整 `conflict_suppressions`、fusion summary 和审计详情；
- 排除耗时、queue wait、trace 等非确定字段后，89 图 JSON 业务负载零差异。

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
4. 两条路径都失败时结束 P6，保留 Host `raw_yolo_v1`，不升级 CANN、不改变阈值或 NMS 语义。

模型配置增加 `output_contract: raw_yolo_v1 | detections_v1`。Ascend 后端按契约选择现有 Host 解码或直接读取最终框，原始 OM、Host 解码和回滚配置全部保留。`detections_v1` 必须逐项复现当前严格 `score > confidence`、相同 confidence 的 anchor/class tie-break、class-aware NMS、坐标还原和 `max_det` 截断顺序。

P6 先用构造输入覆盖阈值边界、同分、重叠、多类和 `max_det` 饱和，再对 89 图比较最终检测 records 和完整业务 JSON。只有零差异、输出复制加后处理均值至少减少 `1.5 ms`、完整 API 均值至少改善 `3%`，且 P95/P99 不恶化超过 `2%` 时才晋级。

执行结论（2026-08-15）：配置、运行时双契约、发布清单绑定、无新增依赖 ONNX 导出和严格语义测试已经实现；板端 CANN `7.0.RC1` 的标准 ONNX NMS 因动态输出无法编译，`NPUNmsWithMask` 不抑制同类重复框，`BatchMultiClassNMS` 则错误抑制 IoU 恰好等于 `0.7` 的框。默认和 `norm_class` 两种实现均未通过现有严格 `IoU > threshold` 边界，因此按第 4 条结束 P6，不启动 8502 性能候选，继续保留 `raw_yolo_v1`。`detections_v1` 构建入口在没有显式通过语义门禁时会拒绝执行，不能用于发布。

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

P7 固定精度门禁为 Base mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95`、新类 precision `≥0.90`、误激活率 `≤0.05`。最终 Engine 均值必须 `≤30 ms`，完整 API 均值 `≤33.33 ms`、P95 `≤35 ms`，P99 相对 P6 或此前最快合格候选不得恶化超过 `2%`。

执行结论（2026-08-15）：P7 增加了两个专用严格配置、统一检测器预检、Base/current teacher 复用、统一模型逻辑 owner/source 映射和原协议审计结构。`expanded_single_student` 与 `yolo_iod_lite` 均完整训练并在同一 89 图锁集上评分；旧类分类行在 model/EMA 中逐步恢复，冻结 BatchNorm 始终保持 eval。两个 manifest 都证明增量阶段读取旧类原始图像、标签和 feature cache 的数量为 `0/0/0`，且原始数据未修改。

| 候选 | Base mAP50 | New-mAP50 | KRR | 新类 precision | 误激活率 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `expanded_single_student` | `0.814088` | `0.016667` | `1.000000` | `0.666667` | `0.014286` | 旧知识保持，但新增通道几乎未学到，拒绝 |
| `yolo_iod_lite` | `0.814088` | `0.598128` | `0.344506` | `0.416667` | `0.428571` | New-mAP50 仍低于门槛，且旧知识、precision 和误激活同时失败，拒绝 |

`expanded_single_student` 的旧通道最大漂移和共享参数相对漂移均为 `0`。`yolo_iod_lite` 的旧通道最大漂移为 `0`，共享参数相对漂移为 `0.077112`；它复用 SHA256 `d27bda7cb89375788deb1f29366b037757f23f7b32ddf6c11e1aa778384dc957` 的 current teacher，教师只在增量 student 图像上推理。其锁集得到 `TP=70`、`FP=98`、`targets=76`，70 张负样本中 30 张误激活，且统一模型没有双检测器冲突仲裁可用，`fusion decision_count=0`。

| 候选 | 配置 SHA256 | metrics SHA256 | dataset manifest SHA256 |
| --- | --- | --- | --- |
| `p7-expanded-20260815` | `9c2800525a66f79f6a03cba2e2680b2c8b29427cc34d517678eec8f527529a2c` | `c52c3f59505e8bdab6dfdff61e28bfbef1791887a4e2e68ce1dcbad264b3adc6` | `94f3f6edd893192c06370ba80078976bf9d26093a158229bf43dc062401e23b3` |
| `p7-yolo-iod-20260815` | `d8518910ace533797ad8e9a70d99aa86d4caadf3c5eeb45c048b96f1e7545ae3` | `545d2c6221dc9bd07eadeb39298a58b147f84c545c5aeb77f027ace79a630586` | `40bed0ec033c11134f5b4be95af2b36556f30739f64c1dce848f6f1698f90405` |

按既定顺序，第一步没有胜出的四类检测器后立即停止 P7：不训练 Scene/Sensor 共享上下文头，不导出单 OM，不启动 8502，也不执行 30+890 性能门禁。不得通过读取旧类原始训练数据、降低门禁或改变阈值/融合语义绕过失败。两个配置、EMA/BN 冻结修复、严格数据隔离、single-detector 逻辑映射和失败证据保留，作为下一轮算法研究的可复现起点；被拒绝的 profile 不能被正式加载。

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

### P9：设备侧解码 + Host 精确 NMS

P6 已证明 CANN `7.0.RC1` 的设备 NMS 无法同时复现严格 `IoU > threshold`、稳定 tie-break 和重复框抑制。P9 不再尝试设备 NMS，只把坐标解码和候选筛选移入 OM，Host 继续执行现有精确 NMS。

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

构造探针覆盖阈值相等、IoU 相等、同分框、跨类重叠、NMS 后补位、capacity 边界和 overflow/raw 回退。任一严格语义探针失败即停止 P9，继续使用 `raw_yolo_v1`，不进入 89 图或性能轮。

P9 晋级要求：

- 89 图检测记录和排除耗时字段后的业务 JSON 零差异；
- output copy 加 Host 后处理均值至少减少 `1.5 ms`；
- 两轮完整 API 均值相对 P8 新鲜基线均改善至少 `3%`；
- 两轮 P95/P99 均不恶化超过 `2%`。

执行结论（2026-08-15）：P9 已实现 `decoded_candidates_v1`、固定 TopK 候选收集、Host 严格排序/NMS、低阈值/overflow raw 回滚、两阶段选择性 D2H、构建清单验证和完整探针工具。CANN `7.0.RC1` 能编译该标准 ONNX 图；首版 int32 `ReduceSum` 在设备上错误退化为 `0/1`，改用 float32 求和后，阈值相等/略高、同分框、跨类重叠、`IoU` 相等、NMS 后补位、capacity 边界和 overflow 回滚探针全部逐输出通过。

完整 Base/Specialist OM 也构建和加载成功，但 89 图零差异门禁失败。相对同一 P8 raw 配置，89 图检测数均为 `343`，类别数、置信度和 Scene/Sensor 结果完全相同；设备侧 mixed-fp16 坐标解码仍使 84 图出现框坐标差异，最大绝对差 `0.25 px`。业务 JSON SHA256 同样有 `84/89` 不一致。该差异虽不改变检测数量或置信度，但违反 P9 的严格逐框和业务 JSON 零差异要求，因此 P9 拒绝晋级并回滚 `raw_yolo_v1`。按停止条件未执行两轮 30+890 性能门禁，也不以单轮诊断耗时宣称收益；P10 从 P8 raw 基线继续。

### P10：共享骨干与双逻辑检测头

P7 的四类单分类头会失去原双模型冲突仲裁，并在新增类学习、KRR、precision 或误激活上失败。P10 改为共享 Base backbone 和 neck/FPN，但保留相互独立的 old/new Detect head，物理上一次执行，逻辑上仍是两个检测器。

新增内部布局 `shared_backbone_dual_head_v1`：

- 统一使用 Base 的 `896×736` AIPP 输入；
- 复用并冻结正式 Base 的 backbone、neck/FPN 和 old head；
- old head 固定输出 `[1,7,13524]`，new head 固定输出 `[1,5,13524]`；
- Agent 继续把 old head 标记为 `frozen_base_model`、new head 标记为 `incremental_model`；
- 继续执行原逐类阈值、positive prototype、双模型冲突仲裁、融合、class-aware NMS 和完整审计。

训练候选按以下顺序执行：

1. 复制现有 Specialist 的 `cv2/cv3/DFL` 到 new head，仅训练 new head；
2. 第一候选精度失败时，在三层共享特征前增加零初始化 residual `1×1` adapter，仅训练 adapter 和 new head。

两种候选都只允许读取 warship 增量数据。Base backbone、neck、old head、BatchNorm 统计以及 model/EMA 中所有冻结参数的最大漂移必须为 `0`。两种候选都通过精度门禁时，以板端完整 API 均值更低者胜出。

若 P9 晋级，两个逻辑头均使用 `decoded_candidates_v1`；否则使用双 raw 输出。Ascend 后端一次执行返回两个逻辑结果，Web 编排继续按 Base/Specialist 两个结果运行，公共响应和审计字段不变。

P10 晋级要求：

- Base mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95`；
- 新类 precision `≥0.90`、误激活率 `≤0.05`；
- Base 冻结参数和 EMA 最大漂移为 `0`；
- 两轮完整 API 均值相对上一胜出组合均同时改善 `≥8%` 且 `≥3 ms`；
- 两轮 P95/P99 均不恶化超过 `2%`。

两个候选任一精度门禁失败，或没有候选达到性能门禁时，回滚为独立 Base/Specialist OM，不复用 P7 被拒绝的四类单头模型。

### P11：共享 Scene/Sensor 上下文头

P11 不严格依赖 P10。P10 晋级时，上下文头挂到双逻辑检测器的共享骨干；P10 失败时，上下文头直接挂到当前正式 Base 骨干，继续保留独立 Specialist OM。检测 backbone、neck 和所有检测头全部冻结，只训练上下文头；训练使用现有 scene/sensor 数据划分，不读取检测标签或生成检测 feature cache。

候选按成本从低到高执行：

1. 对 YOLO `model.10` 深层特征做全局平均池化，分别接 `Linear(C,2)` Sensor head 和 `Linear(C,4)` Scene head；
2. 第一候选精度失败时，对 P3/P4/P5 三层特征分别池化并拼接，经 256 维 SiLU 隐层后接两个分类头。

单 OM 新增固定输出 `sensor_logits[1,2] float32` 和 `scene_logits[1,4] float32`。Agent 复用当前 softmax、上下文概率、软阈值和路由语义；两种候选都通过时，以新增 Engine 时长更低者胜出。

P11 上下文门禁为 Sensor lock accuracy `≥0.95`、Scene lock accuracy `≥0.80`、Joint lock accuracy `≥0.75`，并重新通过 P10 的全部检测精度门禁。性能晋级还要求两轮 API 均值同时改善 `≥1%` 和 `≥0.5 ms`，P95/P99 不恶化超过 `2%`。两种候选均失败时保留独立 Scene OM。

即使 old detector、new detector 和 Scene/Sensor 最终物理合并到一个 OM，也必须作为三个独立逻辑功能模型记录职责、输出、owner 和审计证据。最终发布仍须达到 Engine 均值 `≤30 ms`、完整 API 均值 `≤33.33 ms`、P95 `≤35 ms`。

## 6. 接口与兼容性

- `POST /api/detect` 的请求字段和现有必需响应字段保持不变。
- `agent.decision.conflict_suppressions` 等完整审计响应保持不变。
- `timings` 仅向后兼容地增加字段，不删除或重命名已有字段。
- P4/P5 Ascend 候选配置增加 `schedule_mode`、提交顺序、收集顺序和可选的模型 priority 映射；所有字段均为内部运行时接口。
- P6/P9 模型条目支持 `raw_yolo_v1 | decoded_candidates_v1 | detections_v1`；未声明时保持 `raw_yolo_v1`，禁止依据输出 shape 静默猜测契约。
- P7 复用现有 `deployment: single_detector` profile；公共 class ID、class name、source、protocol 和审计字段保持现有语义。
- P10/P11 增加内部 `model_layout`、logical head owner、class map、anchor count 和 candidate capacity；这些字段不进入 Web 公共接口。
- DVPP 候选配置只接受固定 PNG 契约；不合规输入明确报错。正式配置只有在 P8–P11 最终候选全部门禁通过后才允许切换，CPU 和双检测器回滚通过独立配置显式选择。
- 内部 submit/result 句柄不成为 Web 公共接口。
- 批量 API、多请求并发和响应审计精简不在本轮范围。

## 7. 测试与交付

自动化测试至少覆盖：

- AIPP shape、dtype、归一化、padding、resize/crop 契约；
- 构建清单字段、SHA256 不匹配和未验收 OM 拒绝加载；
- 固定 PNG 接受规则及非法格式不触发 CPU fallback；
- P4 四种 memory/schedule 组合、P2 行为回滚和详细 timing 开关；
- P5 提交/收集顺序、priority range 映射和 priority API 缺失时的明确跳过；
- P6 两种 output contract，以及阈值边界、相同置信度、anchor/class tie-break、重叠框、多类别和 `max_det` 饱和场景下与 Host reference 完全一致；
- P7 统一 student 数据视图、增量数据隔离、旧类权重/通道冻结、single-detector source 映射和共享上下文头不改变检测输出；
- P8 环境快照解析、guard 失败闭锁、governor 不支持和两轮环境一致性；
- P9 三种 output contract、严格阈值/排序/NMS、capacity 边界和 overflow/raw 回退；
- P10 双逻辑头装配、冻结参数与 EMA 零漂移、严格增量数据隔离和 old/new owner 映射；
- P11 在 P10 成败两种布局下的上下文头装配、上下文精度门禁和检测输出兼容；
- 已知 1 图 early-threshold 差异的永久回归夹具；
- fake ACL 验证异步调用顺序、最终等待、错误传播和资源释放；
- `/api/detect` 必需字段不变且新增 timing 字段为兼容添加；
- raw/device-NMS golden 对齐、89 图冻结预测评分和每候选两轮 890 请求性能测试。

本机 Python 固定使用 WSL 仓库现有 `.venv`。不得创建新环境、安装额外依赖或下载 CPU PyTorch。板端不运行 Web pytest，只执行真实 API 端到端测量和阶段所需的能力/模型探针。

板端候选固定使用 `127.0.0.1:8502`，正式 `127.0.0.1:8501` 在同步、启动候选、测量和停止候选期间必须持续 `ready`。每个正式性能报告均使用 30 次预热、10×89 请求、单并发和 HTTP keep-alive，并保存服务端、客户端和关键分段的均值、P50、P95、P99。

每阶段产出独立候选 release、构建清单、精度报告、性能报告和回滚指针。正式配置只在全部门禁通过后原子切换，旧 OM 和配置不得覆盖或删除。P8、P9、P10、P11 分别提交并推送，不把多个阶段压入同一提交。

## 8. 明确假设与非目标

- 本轮固定现有 CANN `7.0.RC1`、驱动和固件；不实施或评估版本升级。
- 不实施 INT8、降分辨率、剪枝、跨请求流水线或 Ring Buffer。
- P6 只允许使用当前 CANN 可编译的图算子或 AscendC/TBE 自定义算子；不以升级工具链绕过兼容结论。
- P7 允许四类统一检测器和共享上下文头，但严格类增量阶段不得读取旧类原始图像、标签或 feature cache，也不得通过混合数据重训换取精度。
- P8 只治理影响测量可重复性的板端服务、governor 和后台负载，不停止或替换正式 `8501`。
- P9 不再实现或近似设备 NMS；候选溢出或低于候选阈值的请求必须显式回到 raw Host 路径。
- P10/P11 不通过改变检测阈值、类别映射、冲突仲裁或上下文软阈值语义换取性能。
- 不改变检测阈值、融合策略、类别映射或审计语义来换取性能。
- 官方方法参考：
  - [msprof Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0005.html)
  - [AOE 调优](https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000110.html)

执行时以目标板已安装 CANN `7.0.RC1` 的命令帮助和实际 API 能力为准，官方新版文档仅作为流程依据。
