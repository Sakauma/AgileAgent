# 配置参考

AgileAgent 使用 UTF-8 YAML 配置。主运行配置采用 schema 3，并由 `fair_agent/core/config.py` 统一加载、校验和解析。

## 配置文件

| 文件 | 用途 |
| --- | --- |
| `configs/agent_pipeline.yaml` | x86/CUDA 开发、训练和 Web 服务 |
| `configs/agent_pipeline_ascend310b.yaml` | Ascend 310B OM 推理服务 |
| `models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/configs/agent_pipeline_ascend310b.yaml` | 当前 4+2 满分 release 的字节级正式配置；由物化脚本复制，不手工修改 |
| `configs/ascend310b/full_score_method.yaml` | Ascend 310B 满分结构、阈值搜索、评分门禁和参考证据 |
| `configs/functional_models.yaml` | 三个功能模型及发布资产 |
| `configs/incremental_detection_policy.yaml` | `base_learning`、`incremental_learning`、`system_calibration`、`joint_evaluation` 的统一范围契约 |
| `configs/incremental_round_registry_4plus2.yaml` | 两轮类别注册、逐轮 split、局部/全局类别映射及父子代际契约 |
| `configs/strict_class_incremental_3plus1.yaml` | 旧 3+1 兼容实验参数，不是当前 production |
| `configs/scene_sensor_model.yaml` | 旧 3+1 Scene-SensorNet 兼容参数 |
| `configs/scene_sensor_model_4plus2.yaml` | 当前 4+2 Scene-SensorNet 训练参数 |
| `configs/local_infer_gpu.yaml` | 本机 GPU 推理参数 |
| `configs/submission_infer_base_4class.yaml` | 当前四类 Base 检测提交参数 |
| `configs/submission_infer_base_3class.yaml` | 旧三类 Base 兼容参数；对应权重已归档 |

## 增量协议字段

`configs/incremental_detection_policy.yaml` 的 `scope_definition` 是术语单一来源：

| `phase` | `counted_as_incremental_learning` | 权重更新与数据口径 |
| --- | --- | --- |
| `base_learning` | `false` | Base train/dev，只更新 Base 检测器 |
| `incremental_learning` | `true` | 只用当轮 Increment train/dev，只更新 Increment 检测器，Base 冻结 |
| `system_calibration` | `false` | 可用 Base/Increment train/dev 和 mixed dev，两个检测器都冻结 |
| `joint_evaluation` | `false` | 在全已学类别 lock/test 上只评分，禁止训练和选参 |

`configs/scene_sensor_model_4plus2.yaml` 的 `protocol.phase` 固定为 `system_calibration`。新生成的场景模型、门控候选、`calibration.json` 和联合评估证据均显式记录 `phase`、`counted_as_incremental_learning` 与 `detector_weights_updated`。其中 `calibration.json.data_scope` 分别声明 Scene-SensorNet、Base 先验、Increment 先验和 mixed dev 门控搜索的数据用途，不再把 mixed dev 校准误写成 `incremental_dataset_only`。

`incremental.round_registry` 是正式多轮源码入口。训练和评估工具不接受源码内固定的新类集合，而是从注册表读取 `round_id`、`new_class_ids`、`local_to_global`、`parent_generation_id` 与 `generation_id`。当前顺序为 `round_01_patrol_boat` → `round_02_armored_vehicle`；每轮 Base 和历史专家冻结，只使用该轮 Increment train/dev。`incremental.round_candidate_registration_tool`、`round_summary_tool` 与 `strict_promotion_tool` 固定登记、汇总和晋级入口；运行时模型唯一来源为 `models/generations.json`。Scene-SensorNet、场景先验与门控搜索仍属于 `system_calibration`，可以使用其声明的数据范围，不受增量检测器“不得回放 Base”规则约束。

候选登记只允许更新 `channels.candidate`。只有最终轮次已登记、两轮 `round_evidence.json` 完整且最终 scene-aware lock 通过时，晋级工具才会更新 `channels.production`；旧联合二类代际随后标记为 `retired_baseline`。实验档 schema 2 同时登记 `specialist_models[]`，命令行 `--profile incremental-detection` 与默认 Web 运行时都按相同的多专家 owner 加载。

## 加载顺序

配置加载器依次执行：

1. 读取 YAML；
2. 展开完整字符串形式的 `${ENV_NAME}`；
3. 应用 `--set key=value` 临时覆盖；
4. 校验 schema、章节和字段；
5. 将相对路径解析到仓库根目录；
6. 生成脱敏的有效配置视图。

## 环境变量

| 变量 | 用途 |
| --- | --- |
| `AGILE_AGENT_PYTHON` | 为环境引导与启动脚本指定 Python 解释器 |
| `PYTHON_BIN` | 为环境引导指定 Python 3.10–3.12 可执行文件 |
| `AGILE_RUNTIME_PYTHON` | 作为 YAML 中 `${AGILE_RUNTIME_PYTHON}` 的运行时解释器值 |

环境准备完成后，解释器绝对路径写入 `.agent-python`，日常启动脚本直接读取该文件。

## 主配置章节

| 章节 | 内容 |
| --- | --- |
| `runtime` | 运行模式、解释器、设备、监听地址和端口 |
| `web` | 代际注册表、production 通道和功能模型注册表 |
| `inference` | 后端、输入尺寸、批量、置信度、预热和并行参数 |
| `routing` | 类别所有权、冲突阈值、融合和软上下文参数 |
| `decoding` | 图像解码与线程设置 |
| `storage` | 数据、报告和模型目录 |
| `incremental_workbench` | 上传限制、数据拆分、训练和生命周期参数 |
| `gates` | Base mAP50、New-mAP50、KRR 和质量阈值 |
| `incremental_guardian` | 数据审计、混淆图和候选评估 |
| `generation` | 代际注册表、通道、recheck 和 shadow 参数 |
| `ascend_backend` | OM 路径、SHA256、输入契约与执行模式 |
| `logging` | 日志目录、轮转和请求记录设置 |
| `decision` | 自动动作及输入输出 |

## 配置命令

校验与查看：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent config show --config configs/agent_pipeline.yaml --effective
agile-agent config get routing.conflict_iou
```

持久修改：

```bash
agile-agent config set routing.conflict_iou 0.50
agile-agent config unset runtime.local_python
```

单次运行覆盖：

```bash
agile-agent --set inference.confidence_default=0.60 serve
```

代际命令管理 production 通道、类别所有权和模型身份：

```bash
agile-agent generation recheck --candidate CANDIDATE_ID
agile-agent generation promote \
  --candidate CANDIDATE_ID \
  --manifest reports/generation_audit/CANDIDATE_ID/recheck_manifest.json
agile-agent generation rollback --to GENERATION_ID
```

## 当前默认值

| 配置 | 值 |
| --- | --- |
| `schema_version` | `3` |
| `inference.backend` | `ultralytics_cuda` |
| `inference.imgsz` / `specialist_imgsz` | `1280 / 1280` |
| `inference.confidence_default` | `0.01` |
| `inference.batch_size` | `32` |
| `incremental_workbench.validation_fraction` | `0.20` |
| `incremental_workbench.lock_fraction` | `0.20` |
| `incremental_workbench.split_seed` | `20260821` |
| `web.generation_channel` | `production` |

当前 CUDA production 的固定 owner 为 Base `0–3`、Increment `4–5`。`models/generations.json` 提供六类逐类基础阈值：`0=.21, 1=.14, 2=.36, 3=.05, 4=.57, 5=.82`；逐类最大场景惩罚为 `0=.15, 1=.88, 2=.26, 3=.19, 4=.65, 5=0`。Base 的 `context_prior` 绑定 `base_context_prior.json` 且 `learning_data_scope=base_train_only`，Increment 绑定 `incremental_context_prior.json` 且 `learning_data_scope=incremental_train_only`。

两个 `context_gate` 均使用 `soft_threshold_penalty`，线上输入为 Scene-SensorNet 的 air/forest/sea/urban 已知场景概率。有效阈值按 `min(1, 基础阈值 + 最大场景惩罚 × (1 - 场景亲和度))` 计算；`hard_routing: false` 保证场景只影响候选阈值，不跳过 Base 或 Increment。`routing.conflict_iou: 1.0` 关闭跨 owner 冲突抑制，`context_evidence_weight: 0.0` 不参与可选专家排名；class-incremental 专家仍是每图必执行。

Ascend 配置将 `inference.backend` 设为 `ascend_acl`，登记 Base、Incremental、Scene 三个正式 OM、release manifest 和验证摘要。板端主实例监听内部 `18501`，公共 `8501` 由精确 loopback 路由提供；x86 的 `configs/agent_pipeline.yaml` 仍独立监听本机 `8501`。

仓库根部的 `configs/agent_pipeline_ascend310b.yaml` 与当前正式包配置保持一致，也是下一轮候选生成的基础 schema 3 配置。零训练部署仍应使用物化后的 release-local 配置；固定 release 路径、资产身份、`validated: true` 和 validation summary 构成同一验证链，不能只复制 OM 后手工改路径。

## Ascend 满分方法配置

`configs/ascend310b/full_score_method.yaml` 不是可直接启动的服务配置，而是训练、导出、ATC、运行时和评分的单一机器可读契约。

| 章节 | 当前固化内容 | 主要消费者 |
| --- | --- | --- |
| `target` | Ascend310B1、CANN `7.0.RC1`、`mixed_float16`、正式/候选端口 | build、materialize、score gate |
| `competition` | Base/New/KRR 与三轮 20 图 batch 门槛；诊断和有效性前置条件 | score、benchmark、selector |
| `training` | 独立 YOLO26s、1280 训练、500 epoch、patience 50、阶段数据范围 | x86 训练与文档核验 |
| `export` | `independent_yolo26_e2e_v1`、`608×736`、两个 `[1,300,6]` E2E 输出、类别映射 | build、materialize |
| `runtime` | DVPP、async/unified enqueue、真实 context 和双证据执行门控 | materialize、Ascend backend |
| `threshold_search` | 评分阈值、old/new 搜索序列与选优顺序 | materialize、selector |
| `benchmark` | 30 次预热、`3×20` batch、PNG 契约和端口 | score gate |
| `reference_result` | 当前正式精度、诊断与公共 `8501` FPS | 文档和发布核验 |

不可变约束：

- `target.candidate_port=8502`，`target.formal_port=8501`；
- `export.model_layout=independent_yolo26_e2e_v1`；
- `export.output_contract=yolo26_e2e_v1`；
- 检测输入 NCHW `[1,3,608,736]`，AIPP 输入 NHWC `[1,608,736,3]`；
- Base 局部类 `0–3` 映射全局 `0–3`，Specialist 局部类 `0/1` 映射全局 `4/5`；
- `runtime.context_mode=model` 与 `schedule_mode=unified_enqueue`；
- 门控必须同时使用 Scene air 概率和 Base 全局类 1 检测，且不得读取标签或文件名；
- 候选必须保持 `validated:false`、`validation_candidate:true`；
- 只有正式提升工具可以生成 `validated:true` release。

当前六类活动阈值和计分请求阈值均为 `0.10`。更换权重或数据集后按 `threshold_search` 重新搜索，不把该值视为通用阈值。

### 方法配置到候选配置

`tools/112_materialize_ascend_yolo26_candidate.py` 合并：

1. schema 3 基础配置；
2. `full_score_method.yaml`；
3. Base、Specialist、Scene 三个 OM；
4. build manifest；
5. production 代际注册表；
6. old/new 阈值。

生成器写入固定 owner、E2E 输出契约、真实 context、双证据门控、`8502` 和候选代际。它拒绝资产身份、类别映射、门控或端口不一致的输入。

### 候选到正式配置

`tools/111_promote_ascend_full_score_release.py` 读取候选、score schema v2、benchmark schema v5 和复轮报告，复制三组 OM/provenance/validation，重写 release-local 路径并生成验证摘要。正式配置边界：

- `runtime.server_port: 18501`；
- 公共客户端仍访问 `127.0.0.1:8501`；
- `validated:true`、`validation_candidate:false`；
- `model_layout:independent_yolo26_e2e_v1`；
- `context_mode:model`；
- 旧 listener 由独立回滚 service 保留；
- `8502` 不进入正式路由。

systemd 安装器管理 main、rollback 和 route 三个 unit；路由脚本只管理一条带固定 comment 的 loopback NAT 规则。

相关环境变量：

| 环境变量 | 默认值/用途 |
| --- | --- |
| `AGILE_AGENT_FULL_SCORE_METHOD` | 覆盖方法 YAML；默认仓库内 `full_score_method.yaml` |
| `AGILE_AGENT_ASCEND_PYTHON` | 板端构建/评分解释器；默认命名环境 Python |
| `AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION` | 仅 score gate 临时设为 `1`，授权候选加载 |
| `AGILE_AGENT_ASCEND_RELEASE` | 启停脚本的 release 根；默认当前 4+2 release |
| `AGILE_AGENT_ASCEND_PORT` | 直启监听端口；systemd 主实例固定使用 `18501` |

## 配置验证

```bash
.venv/bin/python -m fair_agent.cli --config configs/agent_pipeline.yaml config validate
.venv/bin/python -m fair_agent.cli --config configs/agent_pipeline_ascend310b.yaml config validate
.venv/bin/python scripts/verify_release.py
```

发布校验同时核对主配置、模型资产、模型 manifest、功能模型注册表和 production 代际。

预构建 Ascend 发布包校验：

```bash
cd models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2
sha256sum -c SHA256SUMS
cd ../../../..
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

这两步均不训练、不运行 ATC，也不启动或停止 `8501/8502`。

满分候选生成和选优入口：

```bash
.venv/bin/python tools/112_materialize_ascend_yolo26_candidate.py --help
.venv/bin/python tools/110_select_ascend_full_score_candidate.py --help
.venv/bin/python tools/111_promote_ascend_full_score_release.py --help
bash -n scripts/build_ascend_yolo26_e2e_oms.sh
bash -n scripts/run_ascend310b_score_gate.sh
bash -n scripts/manage_ascend310b_primary_route.sh
bash -n scripts/install_ascend310b_primary_services.sh
```

完整参数顺序和候选索引格式见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。
