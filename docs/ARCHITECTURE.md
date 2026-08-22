# 系统架构

AgileAgent 将配置、模型代际、在线推理、增量学习、审计和设备部署组织为一套统一运行时。

## 组件分层

| 层级 | 主要模块 | 职责 |
| --- | --- | --- |
| 接口层 | `fair_agent/web/`、`fair_agent/cli.py` | Web API、工作台、CLI 和健康检查 |
| 编排层 | `fair_agent/modules/web_inference.py` | production 代际解析、三模型执行、类别映射与融合 |
| 生命周期层 | `incremental_workbench.py`、`incremental_lifecycle.py` | 编排数据审计、增量训练、系统校准、联合复核、注册与切换 |
| 模型层 | `fair_agent/models/`、`models/` | Scene-SensorNet、基础检测器、增量检测器和模型元数据 |
| 后端层 | `fair_agent/backends/` | Ultralytics CUDA 与 Ascend ACL/OM 运行时 |
| 状态层 | `data/`、`reports/`、`models/generations.json` | 批次状态、审计事件、报告和 production 代际 |

## 在线推理流程

```text
图像输入
  -> production 代际解析
  -> Scene-SensorNet 场景与传感器认知
  -> 四类 Base 检测器
  -> 二类增量专家
  -> 全局类别映射
  -> 六类逐类场景先验亲和度
  -> 六类逐类 dev 基础阈值 + 场景软惩罚
  -> 固定 owner 直接合并
  -> class-aware NMS
  -> API、黑板与审计事件
```

每张图像使用同一 production 代际。四类 Base 固定负责 soldier、small_aircraft、warship 和 tank（全局类 0–3），二类增量专家固定负责 patrol_boat 与 armored_vehicle（全局类 4–5）。两个检测 owner 对每张图并行推理，专家局部类 `0/1` 映射为全局类 `4/5`。当前基础阈值为 `0=.21, 1=.14, 2=.36, 3=.05, 4=.57, 5=.82`，逐类最大场景惩罚为 `0=.15, 1=.88, 2=.26, 3=.19, 4=.65, 5=0`。

Scene-SensorNet 对 air/forest/sea/urban 四个已知场景做闭集概率预测。Base 类先验只从 Base train 正样本学习，新增类先验只从 Increment train 正样本学习；在线亲和度由场景概率与对应类别先验计算，有效阈值为 `min(1, 基础阈值 + 最大惩罚 × (1 - 亲和度))`。该信号会同时影响新旧类，但只软抑制低亲和度候选：不读取文件名或真值标签、不改变 owner、不做场景硬路由，也不跳过 Base 或 Increment。

当前 x86/CUDA 4+2 production 与下述 Ascend 结构属于不同发布代际。Ascend 不可变包仍是已验证的 3+1 板端 release；在新的 4+2 head 完成 ONNX/ATC 构建与板端评分前，不把 `.pt` 指针变化解释为 OM 已更新。

历史 3+1 Ascend 正式满分主线保留相同的三个逻辑职责，但物理执行不同：Base backbone/neck 只运行一次，一个 `shared_backbone_dual_head_v1` OM 同时返回 old/new raw head；`fixed_neutral_v1` 用 Sensor `0.5/0.5`、Scene 四类各 `0.25` 的均匀上下文保持响应和审计结构。context OM 仍加载用于资产回滚，但正常路径不执行其前向推理；old/new 仍分别记录为 `frozen_base_model` 和 `incremental_model`。

### 历史 3+1 Ascend 正式主线、回滚与候选拓扑

```text
                    POST /api/detect 或 /api/batch
                                  |
                     公共入口 127.0.0.1:8501
                                  |
                 loopback NAT 精确原子路由（新连接）
                    /                              \
        无规则：即时回滚                       有规则：正式主线
              |                                      |
    三 OM 监听器 :8501                     满分实例 :18501
    Base + Incremental + Scene                DVPP encoded batch
                                                    |
                                           shared backbone/neck
                                             /             \
                                         old head       new head
                                            |                |
                                  frozen_base_model  incremental_model
                                             \              /
                                           fixed_neutral_v1

                    :8502 始终留给下一轮隔离候选
```

该 Ascend 主线结构的“物理合并”不改变其 3+1 逻辑责任：

| 逻辑功能 | 物理实现 | owner/输出语义 |
| --- | --- | --- |
| 旧类检测 | 冻结 Base backbone、neck/FPN 和 old Detect head | `frozen_base_model`，局部 `0/1/2` 映射全局 `0/1/3` |
| 新类检测 | 共享特征上的 residual `1×1` adapter 与 new Detect head | `incremental_model`，当前局部 `0` 映射全局 `2` |
| Scene/Sensor | `fixed_neutral_v1`；context OM 作为回滚资产加载但不前向 | Sensor `0.5/0.5`、Scene 四类各 `0.25` |

old/new 的 `candidate_confidence` 是 Host 运行时参数，不参与 ONNX/OM 身份。更换数据集时可以复用同一 build manifest 搜索阈值，但不能改变 logical owner、class map、anchor 数或 output contract。

## 学习、校准与评估阶段

```text
base_learning
  Base train/dev -> Base 检测器权重
        |
        v
incremental_learning
  当轮 Increment train/dev -> 新类专家权重与全局 ID 映射
  Base 检测器权重冻结
        |
        v
system_calibration
  Base/Increment train/dev + mixed dev
  -> Scene-SensorNet、逐类场景先验、阈值与场景惩罚
  -> 不更新 Base 或 Increment 检测器权重
        |
        v
joint_evaluation
  参数冻结 -> mixed lock/test 六类联合评分
  -> 禁止训练和选参 -> 候选登记、shadow、production 切换
```

`incremental_learning` 的定义只覆盖新类检测器训练、新类映射及新类专属学习。Scene-SensorNet 训练、Base/Increment 场景先验学习和六类门控搜索属于独立的 `system_calibration`；它们即使读取 Base train/dev，也不构成旧类样本回放，因为两个检测器权重都保持冻结。工作台状态机会串联这些步骤，但编排范围不等于增量学习统计范围。

训练快照绑定数据清单、类别注册表、父代际和训练配置。Base 类先验只来自 Base train，新增类先验只来自 Increment train，门控参数只由 mixed dev 选择。模型权重与校准参数冻结后进入 mixed lock/test 联合复核，复核结果写入候选代际。

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
- 三 OM 回滚或共享双逻辑头主线/候选的加载和静态输入契约校验；
- 图像预处理与输入缓冲区管理；
- Base、Incremental、Scene 三模型执行，或单次共享骨干双 head 执行；
- YOLO 输出解码、类别映射和耗时统计；
- Web 健康状态与请求级指标。

正式板端配置已切换为 `raw_dual_head_v1`、DVPP encoded batch、pageable memory、threaded execution 和固定中性上下文；release manifest 同时登记 dual OM 与 context 回滚资产。原三 OM release 由独立 systemd 服务保留，公共 `8501` 通过精确 loopback NAT 路由到满分实例 `18501`。结构、阈值搜索和评分门禁见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。

### 可移植正式模型包

`models/ascend310b/full-score/20260816-full-score-1493b04/` 是正式 release 的版本化副本，不是需要继续加工的训练目录。它把两个 OM 与其 source checkpoint、ONNX、AIPP、ATC 日志、训练/导出/build manifest、字节级配置和原始验收报告绑定在同一 SHA256 清单中。

```text
Git clone
  -> SHA256SUMS 校验
  -> materialize_ascend310b_full_score_release.sh
  -> 固定 release 根 /home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04
  -> tools/95 --require-validation
  -> 新板直接 :8501，或既有板主实例 :18501 + :8501 回滚 listener
```

这条消费路径只加载已构建资产，不调用训练、ONNX 导出或 ATC。固定绝对 release 根是正式配置和 manifest 身份的一部分；物化器拒绝覆盖已有目录，避免把不同字节伪装成同一 release。

### 新数据集的候选到正式控制流

```text
full_score_method.yaml
  -> tools/107：训练 best/last + 数据隔离/零漂移报告
  -> tools/108：dual-head ONNX + export manifest
  -> build_ascend_dual_head_om.sh：CANN 7.0.RC1 OM + build manifest
  -> tools/109：注入 owner/class map/Host 阈值并生成 8502 配置
  -> run_ascend310b_score_gate.sh：冻结预测 -> 三项精度 -> 三轮 batch
  -> tools/110：按精度余量、FPS 波动、中位 FPS 选择候选
  -> tools/111：物化 validated release 与不可变证据
  -> systemd 双实例 + loopback NAT：8501 原子提升/即时回滚
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
