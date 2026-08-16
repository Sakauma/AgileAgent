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
