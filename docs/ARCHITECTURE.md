# 系统架构

AgileAgent 将配置、模型代际、在线推理、增量学习、审计和设备部署组织为一套统一运行时。

## 组件分层

| 层级 | 主要模块 | 职责 |
| --- | --- | --- |
| 接口层 | `fair_agent/web/`、`fair_agent/cli.py` | Web API、工作台、CLI 和健康检查 |
| 编排层 | `fair_agent/modules/web_inference.py` | production 代际解析、三模型执行、类别映射与融合 |
| 增量层 | `incremental_workbench.py`、`incremental_lifecycle.py` | 数据审计、训练、校准、复核、注册与切换 |
| 模型层 | `fair_agent/models/`、`models/` | Scene-SensorNet、基础检测器、增量检测器和模型元数据 |
| 后端层 | `fair_agent/backends/` | Ultralytics CUDA 与 Ascend ACL/OM 运行时 |
| 状态层 | `data/`、`reports/`、`models/generations.json` | 批次状态、审计事件、报告和 production 代际 |

## 在线推理流程

```text
图像输入
  -> production 代际解析
  -> Scene-SensorNet 场景与传感器认知
  -> 三类基础检测器
  -> 活动增量检测器
  -> 全局类别映射
  -> 逐类阈值与软上下文
  -> 框级冲突仲裁
  -> class-aware NMS
  -> API、黑板与审计事件
```

每张图像使用同一 production 代际。基础检测器负责 soldier、small_aircraft 和 tank，当前增量检测器负责 warship。Scene-SensorNet 输出场景与传感器概率，融合阶段把这些概率转换为逐类软阈值证据。

Ascend 满分候选保留相同的三个逻辑职责，但物理执行不同：Base backbone/neck 只运行一次，一个 `shared_backbone_dual_head_v1` OM 同时返回 old/new raw head；`fixed_neutral_v1` 用 Sensor `0.5/0.5`、Scene 四类各 `0.25` 的均匀上下文保持响应和审计结构。context OM 仍加载用于回滚，但正常路径不执行其前向推理；old/new 仍分别记录为 `frozen_base_model` 和 `incremental_model`。

### 正式链路与满分候选拓扑

```text
                         POST /api/detect 或 /api/batch
                                      |
                              FastAPI / Engine
                                      |
                 +--------------------+--------------------+
                 |                                         |
          正式 8501（三 OM）                         候选 8502（比赛评分）
                 |                                         |
      +----------+-----------+                    DVPP encoded batch
      |          |           |                             |
    Base    Incremental    Scene                  shared backbone/neck
      |          |           |                      /             \
      +----------+-----------+                  old head       new head
                 |                                |                |
            融合与 NMS                    frozen_base_model  incremental_model
                 |                                \                /
                 |                          fixed_neutral_v1
                 +--------------------+------------+------------+
                                      |
                         同一响应 schema、owner 和审计证据
```

候选结构的“物理合并”不改变逻辑责任：

| 逻辑功能 | 物理实现 | owner/输出语义 |
| --- | --- | --- |
| 旧类检测 | 冻结 Base backbone、neck/FPN 和 old Detect head | `frozen_base_model`，局部 `0/1/2` 映射全局 `0/1/3` |
| 新类检测 | 共享特征上的 residual `1×1` adapter 与 new Detect head | `incremental_model`，当前局部 `0` 映射全局 `2` |
| Scene/Sensor | `fixed_neutral_v1`；context OM 作为回滚资产加载但不前向 | Sensor `0.5/0.5`、Scene 四类各 `0.25` |

old/new 的 `candidate_confidence` 是 Host 运行时参数，不参与 ONNX/OM 身份。更换数据集时可以复用同一 build manifest 搜索阈值，但不能改变 logical owner、class map、anchor 数或 output contract。

## 增量学习流程

```text
ZIP 上传
  -> 安全解压与五列 YOLO 校验
  -> 数据血缘审计
  -> train/dev/lock 固定拆分
  -> 训练快照
  -> GPU 训练
  -> dev 阈值校准与混淆图
  -> 候选代际登记
  -> lock 指标复核
  -> shadow 预热
  -> production 原子切换
```

训练快照绑定数据清单、类别注册表、父代际和训练配置。模型权重与阈值冻结后进入 lock 复核，复核结果与哈希写入候选代际。

## 模型代际

`models/generations.json` 维护：

- production 与 candidate 通道；
- 父子代际关系；
- 基础类、新增类和更新类集合；
- 每个类别的模型所有权；
- 权重、配置和证据文件哈希；
- Base mAP50、New-mAP50、KRR、precision 和误激活率；
- 冲突融合策略与场景软阈值配置。

Web 服务启动时加载 production 代际，代际切换时构建并预热新引擎，再更新进程内运行实例。

## 配置系统

`fair_agent/core/config.py` 负责 schema 3 加载、环境变量展开、路径解析、命令行覆盖、字段校验和敏感值脱敏。默认配置用于 x86/CUDA 开发与训练，Ascend 配置用于板端 OM 推理。

## Ascend 310B 运行时

`fair_agent/backends/ascend_acl.py` 使用 PyACL/AscendCL 完成：

- CANN 初始化与设备上下文；
- 正式三 OM 或共享双逻辑头候选的加载和静态输入契约校验；
- 图像预处理与输入缓冲区管理；
- Base、Incremental、Scene 三模型执行，或单次共享骨干双 head 执行；
- YOLO 输出解码、类别映射和耗时统计；
- Web 健康状态与请求级指标。

正式板端配置记录三个回滚 OM 的路径和 SHA256。满分候选使用 `raw_dual_head_v1`、DVPP encoded batch、pageable memory、threaded execution 和固定中性上下文；build manifest 同时登记 dual OM 与 context 回滚资产。结构、阈值搜索和评分门禁见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。

### 满分候选的资产与控制流

```text
full_score_method.yaml
  -> tools/107：训练 best/last + 数据隔离/零漂移报告
  -> tools/108：dual-head ONNX + export manifest
  -> build_ascend_dual_head_om.sh：CANN 7.0.RC1 OM + build manifest
  -> tools/109：注入 owner/class map/Host 阈值并生成 8502 配置
  -> run_ascend310b_score_gate.sh：冻结预测 -> 三项精度 -> 三轮 batch
  -> tools/110：按精度余量、FPS 波动、中位 FPS 选择候选
```

构建阶段把方法配置、training report、export manifest、source checkpoint、ONNX、AIPP、OM、ATC 命令和 context 回滚资产通过 SHA256 串成一条证据链。新 training report schema v2 授权同轮 best/last 且要求二者共享参数漂移均为零；2026-08-16 参考候选因为历史报告是 schema v1，只允许方法配置中固定的 report/export/`last.pt` 哈希组合兼容。

评分阶段只把 Base mAP50、New-mAP50、KRR 与三轮 20 图 batch 中位 FPS 作为比赛门禁。数据隔离、预测先冻结和资产哈希是结果有效性的前置条件；precision、误激活率、逐框/JSON 差异和单请求时延继续记录，但不改变四项满分判定。

## 审计与证据

运行事件写入 `reports/agent_logs/`，并使用 `trace_id`、`batch_id`、`job_id` 和 `generation_id` 串联。增量批次在 `data/incremental_batches/` 保存源包、训练视图、封存样本、快照、任务记录和类别注册表。

## 目录结构

```text
fair_agent/core/       配置、哈希、运行日志与审计基础设施
fair_agent/backends/   CUDA 与 Ascend 推理适配器
fair_agent/modules/    数据、训练、评测、代际和部署流程
fair_agent/web/        FastAPI 服务与前端资源
configs/               主配置、模型配置与实验配置
models/                发布模型与代际元数据
scripts/               环境准备、启动和发布校验
tools/                 数据、实验和设备验收工具
tests/                 单元测试与集成回归
splits/                固定数据清单
```
