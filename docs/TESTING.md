# 测试指南

AgileAgent 使用 Pytest 覆盖配置、数据、增量生命周期、模型代际、Web API、推理融合和 Ascend 契约。

## 完整回归

```bash
.venv/bin/python -m pytest -q
```

测试数量会随功能增长，提交记录应写明本次实际通过数量，不在本文维护容易过期的固定计数。

## 发布校验

```bash
.venv/bin/python scripts/verify_release.py
```

发布校验核对：

- schema 3 主配置；
- 发布模型路径与 SHA256；
- `models/manifest.json`；
- `models/generations.json`；
- 三个功能模型、Base/Increment 场景先验及其证据；
- production 类别所有权和指标；
- 增量检测策略、轮次登记/汇总/晋级源码入口。

当前发布校验状态为 `passed`。

## 模型冒烟

```bash
.venv/bin/python scripts/smoke_models.py
```

该脚本加载 Scene-SensorNet、四类 Base 检测器和二类增量专家，并执行当前 4+2 production 模型组合冒烟。当前编排中，场景头输出 air/forest/sea/urban 已知类概率，Base 与 Increment 都必须执行；配置与推理测试另行覆盖逐类 `soft_threshold_penalty`、固定 owner，以及不使用文件名、标签或场景硬路由的约束。

## 4+2 场景门控证据

当前 scene-aware production 的证据链位于 `models/production/incremental_detection/evidence/`：

- `scene_aware_dev_search.json/.md` 与候选必须声明 `phase: system_calibration`、`counted_as_incremental_learning: false` 和 `detector_weights_updated: false`；
- `scene_aware_dev_search.json/.md` 记录 mixed dev 上的候选与约束，并明确 `lock_labels_read: false`；
- `scene_aware_candidate.json` 是由 dev 冻结的 `guarded_precision` 参数；
- `scene_aware_lock_recheck.json` 记录冻结候选的一次性 mixed lock 复核，并声明 `phase: joint_evaluation` 与 `model_selection_allowed: false`；
- `base_context_prior.json` 只来自 Base train 正样本，`incremental_context_prior.json` 只来自 Increment train 正样本；

## 4+2 两轮类别增量证据

`configs/incremental_round_registry_4plus2.yaml` 是轮次、类别映射和父子代际的单一来源。发布校验会静态检查：

- 至少两轮且每轮注入不同新类别；
- Round 1/2 的 train/dev/lock 各自非空、互斥，合并后等于固定 Increment 总清单；
- Scene-SensorNet 明确属于 `system_calibration`，不计入增量学习；
- 训练和评估入口从注册表读取类别，源码不使用固定 `4/5` 作为协议参数。

实际训练后，每轮 `tools/08_evaluate_4plus2.py` 必须生成 `lineage`、`round_metrics.new_map50/krr/full_map50` 和 `predictions_frozen_before_lock_labels: true`。随后 `tools/13_register_incremental_round_candidate.py` 核对 selection、evaluation、历史专家和父代 owner，只登记 candidate。`tools/12_summarize_incremental_rounds.py` 只在两轮指标及 `models/generations.json` 都满足父子链、模型身份、零旧样本及不同新类别约束时生成完整证据；`tools/10` 缺少该证据时不得晋级。
- `operating_point_diagnostics.md` 给出逐类 mAP50、TP、FP、precision、recall 和误激活率。

静态发布校验同时检查四阶段范围契约。严格实验档加载器要求当前 schema 的 `calibration.json` 属于 `system_calibration`、不更新检测器权重，并准确声明 Base train、Increment train 与 mixed dev 的用途。参数搜索只允许读取 dev；lock 模式必须接收已冻结 candidate，并在读取 lock 标签前固定候选与模型预测。

## 测试范围

| 范围 | 主要测试 |
| --- | --- |
| 配置与 CLI | `test_configuration_runtime.py` |
| 数据与固定划分 | `test_strict_3plus1_splits.py`、`test_strict_incremental.py` |
| 增量工作台 | `test_incremental_workbench.py`、`test_incremental_lifecycle_v2.py` |
| 模型代际 | `test_runtime_maturity.py`、`test_incremental_lifecycle_v2.py` |
| 推理与融合 | `test_web_inference.py`、`test_incremental_rejection.py`；覆盖 Base/Increment 逐类场景软门控、每图执行与固定 owner |
| Web API 与 UI | `test_agent_workbench.py`、`test_web_ui_flow.py` |
| Ascend | `test_ascend_acl.py`、`test_ascend_preflight.py` |
| Ascend 满分方法 | `test_ascend_full_score_workflow.py`、`test_shared_dual_head.py`、`test_ascend_release.py` |
| Ascend 预构建发布包 | `test_ascend_packaged_release.py` |
| 多批次增量 | `test_incremental_guardian.py`、`test_incremental_experiment.py` |

## 聚焦运行

单文件：

```bash
.venv/bin/python -m pytest -q tests/test_web_inference.py
```

单节点：

```bash
.venv/bin/python -m pytest -q \
  tests/test_configuration_runtime.py::test_default_config_loads
```

关键字过滤：

```bash
.venv/bin/python -m pytest -q -k "generation or incremental"
```

Ascend 契约：

```bash
.venv/bin/python -m pytest -q \
  tests/test_ascend_acl.py \
  tests/test_runtime_maturity.py
```

满分方法和候选选择：

```bash
.venv/bin/python -m pytest -q \
  tests/test_ascend_full_score_workflow.py \
  tests/test_shared_dual_head.py \
  tests/test_configuration_runtime.py \
  tests/test_ascend_release.py \
  tests/test_strict_incremental.py
bash -n scripts/build_ascend_dual_head_om.sh
bash -n scripts/run_ascend310b_score_gate.sh
bash -n scripts/manage_ascend310b_primary_route.sh
bash -n scripts/install_ascend310b_primary_services.sh
```

克隆后零训练部署包：

```bash
.venv/bin/python -m pytest -q tests/test_ascend_packaged_release.py
bash -n scripts/materialize_ascend310b_full_score_release.sh
(
  cd models/ascend310b/full-score/20260816-full-score-1493b04
  sha256sum -c SHA256SUMS
)
```

该测试核对完整文件清单、全部 SHA256、正式配置到 OM/manifest/report 的路径映射、满分指标、`ready_without_training` 标记和物化脚本的端口无副作用。它不加载 PyTorch、不连接板卡，也不重新运行 ATC。

### 满分工作流覆盖范围

| 契约 | 覆盖内容 |
| --- | --- |
| 方法配置 | schema、参考指标、无绝对板端路径、训练/导出/运行时/评分协议 |
| 训练 | residual adapter、Base/BN/EMA 零漂移、best/last 授权、增量数据隔离 |
| 导出和构建 | 双输出 shape、owner/class map、CANN/AIPP/ATC 与资产 SHA256 |
| 候选生成 | `8501` 保护、强制 `8502`、哈希不匹配拒绝、`validated: false` |
| 评分 | score schema v2、benchmark schema v5、三项精度和三轮 batch FPS |
| 选择器 | 四项门禁、诊断项不阻断、确定性排名、无满分候选分支 |
| 正式物化 | `tools/111`、release-local 资产、验证摘要、诊断项不阻断 |
| 仓库发布包 | OM/ONNX/checkpoint/AIPP/ATC/报告齐全、26 项 SHA256、固定 release 路径、零训练物化和 `8501/8502` 无副作用 |
| 原子提升/回滚 | 主/回滚 systemd service、精确 iptables rule、失败自动撤销和 `8502` 保留 |

评分报告的阻断边界：

- `competition_gates` 只包含 Base mAP50、New-mAP50 和 KRR；
- benchmark 只用三轮 20 图 batch 的中位 FPS 作为性能门禁；
- `diagnostic_warnings` 可以包含逐框/业务 JSON、precision、误激活率、Scene/Sensor 或单请求时延问题，但不得改变 `full_score`；
- `predictions_frozen_before_labels`、增量数据隔离、Base 零漂移和资产哈希属于有效性前置条件，失败时结果无效。

### 本机与板端测试边界

| 环境 | 执行内容 | 禁止事项 |
| --- | --- | --- |
| WSL 现有 `.venv` | Pytest、Ruff、配置/文档/哈希校验、训练与 ONNX 导出 | 不安装依赖，不下载 CPU PyTorch |
| 新 310B（环境已配置） | 校验仓库模型包、物化 release、运行 `tools/95 --require-validation`、启动 health/冒烟 | 不训练，不运行 ATC，不安装依赖 |
| Ascend 候选 `8502` | 探针、无标签预测冻结、三项精度评分、30 次预热、三轮 20 图 batch | 不运行 Web pytest，不停止或替换 `8501` |
| 正式公共 `8501` | 检查 `/api/health` 的 `validated/model_layout/context_mode`，执行发布后 `30 + 3×20` batch | 不直接停止旧监听器，不把 `8502` 纳入正式路由 |

正式板端还应确认 `8501` 三 OM 回滚监听器、`18501` 共享双头主实例和三个 systemd unit 同时存在；公共 `8501` 健康请求应经精确路由返回共享双头身份。执行 `manage_ascend310b_primary_route.sh remove 18501` 时应恢复旧服务，重新 `apply` 后回到主线。正式验收不需要在板端运行 Web pytest。

板端输入必须是评分目录根层的 `640×512`、8-bit RGB/RGBA PNG，stem 唯一。任一输入契约、CANN 版本、候选端口或 manifest 哈希不一致时，score gate 在正式测量前停止并保留已有报告。

## 测试夹具

测试使用 `tmp_path` 创建临时配置、注册表、批次目录和模型占位文件；使用 Pillow 生成小型 PNG；使用 ZIP 构造增量上传包；使用假引擎验证 Web 与代际切换。所有测试状态保留在测试临时目录中。

## 本地发布门禁

仓库不启用 GitHub Actions。发布前必须在 WSL 仓库现有 `.venv` 中执行完整回归与发布校验，不安装依赖，也不下载 CPU 版 PyTorch：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_release.py
```

模型冒烟、Ascend 包哈希和 Shell 语法检查仍按本文前述流程执行；本地验证记录是发布判断依据。
