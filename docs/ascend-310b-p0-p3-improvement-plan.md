# AgileAgent 昇腾 310B P0–P3 推理优化计划

## 1. 摘要

本文规划 AgileAgent 在 Atlas 200I DK A2 / Ascend310B1 上的四阶段优化。本文仅描述改进计划，不直接实施工程改动。

优化顺序固定为：

1. 基线与验收夹具；
2. P0：正式化 DVPP + AIPP；
3. P1：优化 Python 后处理、路由与融合；
4. P2：msprof 定位后执行 AOE；
5. P3：锁页内存、异步拷贝与延迟同步。

每阶段只基于上一个“已通过验收”的版本继续。候选未通过时保留负面结论并回退，不降低精度或语义门槛。

当前瓶颈判断：

| 模块 | 已有测量 | 判断 |
| --- | ---: | --- |
| 完整 CPU 路径 | `71.491 ms` | CPU 解码、预处理和 Web 开销明显 |
| DVPP encoded API | `37.124 ms` | P0 是目前收益最确定的方向 |
| Agent 核心 | `32.148 ms` | 距离完整 30 FPS 已较近 |
| 路由与融合 | `16.64 ms` | 当前最大 Host 侧热点 |
| Base NPU | `20.25 ms` | P2 首要 AOE 对象 |
| Specialist NPU | `9.28 ms` | P2 第二优先级 |
| Scene NPU | `0.37 ms` | 默认不做 AOE |
| 同步拷贝/等待 | 尚未独立测量 | 由 P2 定位，P3 处理 |

上述数据来自不同批次，不能直接视为严格 A/B；所有晋级结论以新建的同条件基准为准。

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

## 4. 接口与兼容性

- `POST /api/detect` 的请求字段和现有必需响应字段保持不变。
- `agent.decision.conflict_suppressions` 等完整审计响应保持不变。
- `timings` 仅向后兼容地增加字段，不删除或重命名已有字段。
- DVPP 正式配置只接受固定 PNG 契约；不合规输入明确报错。
- Ascend 配置新增：
  - `build_manifest`；
  - `build_manifest_sha256`；
  - `memory_mode`。
- `encoded_preprocessing` 正式值切换为 `dvpp`；CPU 回滚通过独立配置显式选择。
- 内部 submit/result 句柄不成为 Web 公共接口。
- 批量 API、多请求并发和响应审计精简不在本轮范围。

## 5. 测试与交付

自动化测试至少覆盖：

- AIPP shape、dtype、归一化、padding、resize/crop 契约；
- 构建清单字段、SHA256 不匹配和未验收 OM 拒绝加载；
- 固定 PNG 接受规则及非法格式不触发 CPU fallback；
- P1 在阈值边界、相同置信度、重叠框、多类别、`max_det` 饱和场景下与 reference 完全一致；
- 已知 1 图 early-threshold 差异的永久回归夹具；
- fake ACL 验证异步调用顺序、最终单次等待、错误传播和资源释放；
- `/api/detect` 必需字段不变且新增 timing 字段为兼容添加；
- 三个 OM golden 对齐、89 图冻结预测评分和 890 请求性能测试。

每阶段产出独立候选 release、构建清单、精度报告、性能报告和回滚指针。正式配置只在全部门禁通过后原子切换，旧 OM 和配置不得覆盖或删除。

## 6. 明确假设与非目标

- 本轮固定现有 CANN `7.0.RC1`、驱动和固件；不实施或评估版本升级。
- 固件升级不能被预设为必然提速。未来如重启版本调研，必须作为独立环境矩阵进行同硬件 A/B，不能与 AOE、代码优化同时切换。
- 不实施 INT8、降分辨率、剪枝、模型结构调整、C++ 重写、跨请求流水线或 Ring Buffer。
- 不改变检测阈值、融合策略、类别映射或审计语义来换取性能。
- 官方方法参考：
  - [msprof Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0005.html)
  - [AOE 调优](https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000110.html)

执行时以目标板已安装 CANN `7.0.RC1` 的命令帮助和实际 API 能力为准，官方新版文档仅作为流程依据。
