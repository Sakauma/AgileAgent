# 系统架构

AgileAgent 将配置、模型代际、在线推理、增量学习、系统校准、审计和 Ascend 部署组织为统一运行时。

## 组件分层

| 层级 | 主要模块 | 职责 |
| --- | --- | --- |
| 接口层 | `fair_agent/web/`、`fair_agent/cli.py` | Web API、工作台、CLI 和健康检查 |
| 编排层 | `fair_agent/modules/web_inference.py` | production 代际解析、模型执行、门控、类别映射与融合 |
| 生命周期层 | `incremental_workbench.py`、`incremental_lifecycle.py` | 数据审计、增量训练、系统校准、联合复核、注册与切换 |
| 模型层 | `fair_agent/models/`、`models/` | Scene-SensorNet、Base、增量专家和发布元数据 |
| 后端层 | `fair_agent/backends/` | Ultralytics CUDA 与 Ascend ACL/OM |
| 状态层 | `data/`、`reports/`、`models/generations.json` | 批次状态、审计事件、指标和 production 代际 |

## 固定类别所有权

| 全局类 | 名称 | owner |
| ---: | --- | --- |
| 0 | soldier | `frozen_base_model` |
| 1 | small_aircraft | `frozen_base_model` |
| 2 | warship | `frozen_base_model` |
| 3 | tank | `frozen_base_model` |
| 4 | patrol_boat | `incremental_model` |
| 5 | armored_vehicle | `incremental_model` |

Base 与增量专家的输出先映射为全局 ID，再做冲突仲裁和 class-aware NMS。场景模型不会改变 owner。

## x86/CUDA 在线流程

```text
图像
  -> production 代际解析
  -> Scene-SensorNet、四类 Base、二类专家
  -> 全局类别映射
  -> 六类场景先验亲和度
  -> dev 冻结的逐类基础阈值 + 场景软惩罚
  -> 固定 owner 合并
  -> class-aware NMS
  -> API 与审计事件
```

x86 使用六类场景软阈值。Scene-SensorNet 对 air/forest/sea/urban 做闭集概率预测；Base 类先验只来自 Base train 正样本，新增类先验只来自 Increment train 正样本。在线不读取文件名或真值标签。

## Ascend310B1 正式流程

当前 release：

```text
/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
```

```text
640×512 PNG
  -> bounded multipart
  -> DVPP encoded preprocessing
  -> 并发提交 Scene-SensorNet 与四类 Base YOLO26s
  -> 收集真实场景概率与 Base 检测
  -> 双证据执行门控
       air >= 0.5 且 Base 检出 small_aircraft：跳过专家
       其他情况：执行/收集二类专家
  -> 固定 owner、冲突仲裁、class-aware NMS
  -> 六类响应与审计
```

门控输入只有 `scene_probabilities` 和 `base_detections`，不读取标签或文件名。它属于冻结后的 system calibration，不更新任何检测器权重。

### Ascend 模型契约

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| Base YOLO26s | uint8 NHWC `[1,608,736,3]` | E2E `[1,300,6]`，4 类 |
| Incremental YOLO26s | uint8 NHWC `[1,608,736,3]` | E2E `[1,300,6]`，2 类 |
| Scene-SensorNet | uint8 NHWC `[1,160,160,3]` | sensor 与 scene 概率 |

`fair_agent/backends/ascend_acl.py` 负责 CANN 初始化、ACL context、OM 加载、DVPP、统一 enqueue、E2E 输出接收和耗时统计。`fair_agent/modules/web_inference.py` 负责全局类别、内容门控、融合和 API 语义。

### 正式拓扑

```text
                     公共 127.0.0.1:8501
                               |
             精确 loopback NAT，comment 固定
                     /                     \
             无规则：回滚              有规则：正式
                  |                         |
        旧 listener :8501           4+2 主实例 :18501
                                           |
                          Base + Specialist + Scene 三 OM

                   :8502 仅用于隔离候选
```

三个 systemd unit 分别管理主实例、回滚 listener 与路由。正式提升保留旧 listener 的物理监听，删除唯一规则即可即时回滚。

## 学习、校准与评估阶段

```text
base_learning
  Base train/dev -> 四类 Base 权重
        |
incremental_learning
  当轮 Increment train/dev -> 当轮新类专家
  Base 与历史专家冻结
        |
system_calibration
  Scene-SensorNet、场景先验、门控与阈值
  不更新任何检测器
        |
joint_evaluation
  参数冻结 -> 截至当轮全部类别的 lock/test
  -> New-mAP50、KRR、Full-mAP50 与父子代际
```

`incremental_learning` 只包括新类检测器训练、新类映射及新类专属学习。Scene-SensorNet 和场景门控属于独立功能模型校准。完整契约见 `docs/compliant-incremental-learning.md`。

正式 4+2 注册表按 patrol_boat → armored_vehicle 记录两个新类别轮次。每轮只读当轮 Increment train/dev；`tools/13_register_incremental_round_candidate.py` 逐轮登记候选，`tools/12_summarize_incremental_rounds.py` 验证父子链，最终由 `tools/10_promote_scene_aware_4plus2.py` 切换 production。

## 模型代际

`models/generations.json` 维护：

- production 与 candidate 通道；
- 父子代际和轮次；
- 类别 owner；
- 权重、配置和证据身份；
- New-mAP50、KRR、Full-mAP50；
- 阈值、场景配置和执行门控；
- 数据隔离、Base 冻结与验收状态。

Web 服务启动时加载 production 代际。新代际必须先构建并预热引擎，再原子替换进程内运行实例。

## 配置系统

`fair_agent/core/config.py` 负责 schema 3 加载、环境变量展开、路径解析、命令行覆盖、字段校验和敏感值脱敏：

- `configs/agent_pipeline.yaml`：x86/CUDA production；
- `configs/agent_pipeline_ascend310b.yaml`：当前 Ascend 正式配置；
- `configs/ascend310b/full_score_method.yaml`：训练、导出、ATC、运行时与评分契约；
- release-local 配置：不可变发布身份。

## Ascend 候选到正式控制流

```text
4+2 冻结权重
  -> 两个 YOLO26 E2E ONNX
  -> build_ascend_yolo26_e2e_oms.sh
  -> Base / Incremental OM + 真实 Scene 资产 + build manifest
  -> tools/112：生成 :8502 候选与候选代际
  -> run_ascend310b_score_gate.sh
  -> tools/110：候选排序
  -> tools/111：validated release
  -> systemd + loopback NAT：公共 :8501 提升
  -> 公共入口部署后 FPS 复验
```

结果有效性要求数据隔离、预测先冻结、Base 权重冻结与资产身份一致。比赛淘汰门禁只使用 Base mAP50、New-mAP50、KRR 和 batch FPS；precision、误激活率和单请求延迟保留为诊断。

## 可移植正式模型包

`models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/` 保存三个 OM、source checkpoint、ONNX、AIPP、ATC 日志、构建清单、正式配置、冻结预测和原始验收报告。

```text
Git clone
  -> 包完整性检查
  -> materialize_ascend310b_full_score_release.sh
  -> 固定 release 根
  -> tools/95 --require-validation
  -> 新板直接 :8501，或 :18501 主实例 + :8501 回滚 listener
```

该消费路径不训练、不导出 ONNX、不运行 ATC、不升级 CANN。

## 审计与证据

运行事件写入 `reports/agent_logs/`，用 `trace_id`、`batch_id`、`job_id` 和 `generation_id` 串联。增量批次在 `data/incremental_batches/` 保存源包、训练视图、封存样本、快照、任务记录和类别注册表。板端 release 的正式证据位于包内 `provenance/` 与 `validation/`。

## 目录结构

```text
fair_agent/core/       配置、哈希、运行日志与审计基础设施
fair_agent/backends/   CUDA 与 Ascend 推理适配器
fair_agent/modules/    数据、训练、评测、代际和部署流程
fair_agent/web/        FastAPI 服务与前端资源
configs/               主配置、模型配置与实验配置
models/                发布模型与代际元数据
scripts/               环境准备、启动和发布校验
tools/                 数据、实验、导出和设备验收工具
tests/                 单元测试与集成回归
splits/                固定数据清单
```
