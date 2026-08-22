# 配置参考

AgileAgent 使用 UTF-8 YAML 配置。主运行配置采用 schema 3，并由 `fair_agent/core/config.py` 统一加载、校验和解析。

## 配置文件

| 文件 | 用途 |
| --- | --- |
| `configs/agent_pipeline.yaml` | x86/CUDA 开发、训练和 Web 服务 |
| `configs/agent_pipeline_ascend310b.yaml` | Ascend 310B OM 推理服务 |
| `models/ascend310b/full-score/20260816-full-score-1493b04/configs/agent_pipeline_ascend310b.yaml` | 已验证满分 release 的字节级正式配置；由物化脚本复制，不手工修改 |
| `configs/ascend310b/full_score_method.yaml` | Ascend 310B 满分结构、阈值搜索、评分门禁和参考证据 |
| `configs/functional_models.yaml` | 三个功能模型及发布资产 |
| `configs/incremental_detection_policy.yaml` | 增量数据范围与评测策略 |
| `configs/strict_class_incremental_3plus1.yaml` | 旧 3+1 兼容实验参数，不是当前 production |
| `configs/scene_sensor_model.yaml` | 旧 3+1 Scene-SensorNet 兼容参数 |
| `configs/scene_sensor_model_4plus2.yaml` | 当前 4+2 Scene-SensorNet 训练参数 |
| `configs/local_infer_gpu.yaml` | 本机 GPU 推理参数 |
| `configs/submission_infer_base_4class.yaml` | 当前四类 Base 检测提交参数 |
| `configs/submission_infer_base_3class.yaml` | 旧三类 Base 兼容参数；对应权重已归档 |

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

Ascend 配置将 `inference.backend` 设为 `ascend_acl`，使用 `batch_size: 1`，并登记正式共享双逻辑头 OM、context 回滚 OM、release manifest 和验证摘要。板端进程监听内部 `18501`，公共 `8501` 由精确 loopback 路由提供；x86 的 `configs/agent_pipeline.yaml` 仍独立监听本机 `8501`。

仓库根部的 `configs/agent_pipeline_ascend310b.yaml` 是开发和候选生成的源配置；零训练部署必须使用预构建模型包内的正式配置。该配置中的固定 release 绝对路径、资产 SHA256、`validated: true` 和 validation summary 构成同一验证链，不能复制 OM 后手工改路径。执行 `scripts/materialize_ascend310b_full_score_release.sh` 会把整包安装到固定路径，并在启动前完成正式 release 校验。

## Ascend 满分方法配置

`configs/ascend310b/full_score_method.yaml` 不是可直接启动的 schema 3 服务配置，也不保存本机或板端绝对路径。它是比赛方法的单一配置源，结构如下：

| 章节 | 固化内容 | 主要消费者 |
| --- | --- | --- |
| `target` | `Ascend310B1`、CANN `7.0.RC1`、`mixed_float16`、正式/候选端口 | build、materialize、score gate |
| `competition` | Base/New/KRR 与 20 图三轮 batch 满分门槛；诊断项和有效性前置条件 | score、benchmark、selector |
| `training` | residual adapter、冻结范围、输入尺寸、优化器、增强、best/last 策略 | `tools/107_train_shared_dual_head.py` |
| `export` | `shared_backbone_dual_head_v1`、`raw_dual_head_v1`、输入/输出 shape、AIPP、owner/class map | `tools/108`、build、materialize |
| `runtime` | DVPP encoded、pageable、threaded execution、fixed-neutral 与 fast path | materialize、Ascend backend |
| `threshold_search` | 评分请求阈值、old/new 搜索序列与确定性选优顺序 | materialize、selector、操作手册 |
| `benchmark` | `30` 次预热、`3×20` batch、PNG 输入契约、端口 URL | score gate、benchmark |
| `reference_result` | 当前满分指标、提交和核心资产 SHA256 | 文档事实核验、历史 schema v1 兼容 |

关键不可变约束：

- `target.candidate_port` 必须为 `8502`，`target.formal_port` 必须为 `8501`；
- `export.model_layout/output_contract` 必须为 `shared_backbone_dual_head_v1/raw_dual_head_v1`；
- 输入固定为 NCHW `[1,3,736,896]`，AIPP 输入为 NHWC `[1,736,896,3]`；
- old owner 为 `frozen_base_model`，new owner 为 `incremental_model`；
- training report 必须证明增量数据隔离和共享参数零漂移；
- 候选配置必须保持 `validation_candidate: true`、`validated: false`；只有 `tools/111` 验证四项满分和有效性前置条件后才能生成 `validated: true` 的正式 release。

当前 `old=0.05`、`new=0.30` 是 Host 运行时搜索种子，不是永久阈值，也不属于 ONNX/OM 身份。更换数据集时先更新类别映射和训练输入，再按方法中的搜索序列生成多份候选 YAML；同一 dual OM 可用于不同 Host 阈值。

### 方法配置到候选配置

`tools/109_materialize_ascend_full_score_candidate.py` 合并四类输入：

1. `configs/agent_pipeline_ascend310b.yaml` 基础 schema 3 配置；
2. `full_score_method.yaml` 方法契约；
3. dual/context OM 与 SHA256；
4. build manifest 中的 training/export/method 证据。

生成结果会写入 `ascend_backend.model_layout`、单一 dual model、`logical_heads`、Host 阈值、context 回滚资产、运行时快路径和 `8502` 端口。生成器拒绝指向 `8501`、哈希不一致、owner/class map 不一致或已标记 validated 的候选。

### 候选到正式配置

`tools/111_promote_ascend_full_score_release.py` 读取胜出候选、score schema v2、benchmark schema v5 和可选复轮报告，将 OM、训练/导出证据、原始 build manifest、方法配置及评分报告复制到不可变 release，重写 release-local 资产路径并生成验证摘要。正式配置具有以下边界：

- `runtime.server_port: 18501` 是板端主实例的实际监听端口；
- 客户端仍访问公共 `127.0.0.1:8501`；
- `ascend_backend.validated: true`、`validation_candidate: false`；
- `model_layout: shared_backbone_dual_head_v1`、`context_mode: fixed_neutral_v1`；
- 原三 OM release 不写入主配置，由独立回滚 service 继续监听 `8501`；
- `8502` 不进入正式路由，继续作为下一轮候选端口。

正式提升由 `scripts/install_ascend310b_primary_services.sh` 安装主/回滚两个 systemd service，并通过 `scripts/manage_ascend310b_primary_route.sh` 管理一条带固定 comment 的精确 loopback NAT 规则。删除该规则即可把新连接立即恢复到三 OM 监听器。

相关环境变量只用于选择已有解释器或方法文件，不会写回方法 YAML：

| 环境变量 | 默认值/用途 |
| --- | --- |
| `AGILE_AGENT_FULL_SCORE_METHOD` | 覆盖方法 YAML 路径；默认使用仓库内 `full_score_method.yaml` |
| `AGILE_AGENT_ASCEND_PYTHON` | 板端构建/评分解释器；默认 `/usr/local/miniconda3/envs/agileagent/bin/python` |
| `AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION` | 仅由受控评分链路设置为 `1`，授权加载未发布候选 |

## 配置验证

```bash
.venv/bin/python -m fair_agent.cli --config configs/agent_pipeline.yaml config validate
.venv/bin/python -m fair_agent.cli --config configs/agent_pipeline_ascend310b.yaml config validate
.venv/bin/python scripts/verify_release.py
```

发布校验同时核对主配置、模型资产、模型 manifest、功能模型注册表和 production 代际。

预构建 Ascend 发布包校验：

```bash
cd models/ascend310b/full-score/20260816-full-score-1493b04
sha256sum -c SHA256SUMS
cd ../../../..
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

这两步均不训练、不运行 ATC，也不启动或停止 `8501/8502`。

满分候选生成和选优入口：

```bash
.venv/bin/python tools/107_train_shared_dual_head.py --help
.venv/bin/python tools/108_export_ascend_dual_head.py --help
.venv/bin/python tools/109_materialize_ascend_full_score_candidate.py --help
.venv/bin/python tools/110_select_ascend_full_score_candidate.py --help
.venv/bin/python tools/111_promote_ascend_full_score_release.py --help
bash -n scripts/build_ascend_dual_head_om.sh
bash -n scripts/run_ascend310b_score_gate.sh
bash -n scripts/manage_ascend310b_primary_route.sh
bash -n scripts/install_ascend310b_primary_services.sh
```

完整参数顺序和候选索引格式见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。
