<!-- generated-by: gsd-doc-writer -->
# 配置参考

AgileAgent 以 UTF-8 YAML 管理运行、训练、增量协议和 Ascend310B 发布契约，以受版本控制的 JSON 注册表管理当前 production 代际、类别 owner 与逐类阈值。主运行配置的 schema 版本为 `3`，由 `fair_agent/core/config.py` 统一加载和校验。

## 配置源

| 路径 | 格式 | 用途 |
| --- | --- | --- |
| `configs/agent_pipeline.yaml` | YAML, schema 3 | x86/NVIDIA CUDA 服务、增量工作台、路由、性能和发布门禁的主配置 |
| `configs/agent_pipeline_ascend310b.yaml` | YAML, schema 3 | Ascend310B1 正式服务配置，登记 Base、Incremental 和 Scene-SensorNet 三个 OM |
| `models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/configs/agent_pipeline_ascend310b.yaml` | YAML, schema 3 | Ascend310B v2 发布包内的已验证部署配置 |
| `configs/functional_models.yaml` | YAML, schema 2 | Scene-SensorNet、四类 Base 检测器和二类 Incremental 专家的功能模型注册表 |
| `configs/incremental_detection_policy.yaml` | YAML, schema 4 | Base learning、Incremental learning、System calibration 和 Joint evaluation 的数据与权重更新边界 |
| `configs/incremental_round_registry_4plus2.yaml` | YAML, schema 1 | 4+2 类别注册表、两轮注入顺序、split、局部/全局 ID 映射和父子代际 |
| `configs/scene_sensor_model_4plus2.yaml` | YAML | 封闭集场景/传感器模型的数据、训练、输出与验收参数 |
| `configs/local_infer_gpu.yaml` | YAML | x86 GPU 本地四类 Base 批量推理和结果打包 |
| `configs/submission_infer_base_4class.yaml` | YAML | 四类 Base 数据提交推理的输入、阈值和输出格式 |
| `configs/ascend310b/full_score_method.yaml` | YAML, schema 1 | Ascend310B v2 的训练、E2E 导出、ATC、内容门控、阈值搜索、评分和性能契约 |
| `configs/ascend310b/aipp/*.cfg` | ATC AIPP | Base、Incremental 和 Scene-SensorNet 的板端图像预处理参数 |
| `models/generations.json` | JSON, schema 2 | x86/CUDA production 代际、六类 owner、模型身份、阈值、场景先验和验收结果 |
| `models/profiles/incremental-detection/active.json` | JSON, schema 1 | `incremental-detection` 运行档的展开快照 |
| `models/production/incremental_detection/calibration.json` | JSON, schema 4 | 新类逐类阈值和场景软惩罚的 system-calibration 结果 |

`models/manifest.json`、`models/SHA256SUMS.txt` 和 Ascend v2 包内的 `release.json`、`SHA256SUMS` 负责发布资产身份与完整性，不作为运行时调参入口。

## 加载与覆盖顺序

`load_config()` 按以下顺序得到有效配置：

1. 选择主配置。调用方显式传入 `--config PATH` 时固定使用该文件；默认 `--config auto` 先接受 `AGILE_AGENT_CONFIG`，否则将 `x86_64/AMD64` 映射到 `configs/agent_pipeline.yaml`，将 `aarch64/ARM64` 映射到 `configs/agent_pipeline_ascend310b.yaml`。ARM 设置 `AGILE_AGENT_ASCEND_RELEASE` 时优先使用该 release 内的配置。
2. 对值完全等于 `${ENV_NAME}` 的字符串做环境变量展开；受控主配置当前采用显式值，设备专用副本可使用该语法注入路径。
3. 应用重复的 `--set KEY=VALUE` 或 `AGILE_AGENT_OVERRIDES` 中的临时覆盖。值使用 YAML 语义解析，因此数字、布尔值和 `null` 保留其类型。
4. 校验 schema、已知字段、数值范围、后端契约和发布资产身份。
5. 添加 `_config_path`、`_config_overrides`、`_config_sha256` 和 `_runtime_platform` 运行时元数据。后者记录主机架构、后端、设备族、模型格式及选择来源，不写回 YAML，也不参与配置 SHA256。

相对路径以仓库根目录为基准。`runtime.local_python: null` 表示使用当前 Python 解释器。

常用查看和校验命令：

```bash
agile-agent config validate
agile-agent config get inference.backend
agile-agent --config configs/agent_pipeline.yaml config validate
agile-agent --config configs/agent_pipeline_ascend310b.yaml config validate
agile-agent --config configs/agent_pipeline.yaml config show --effective
agile-agent --config configs/agent_pipeline.yaml config get inference.backend
agile-agent --config configs/agent_pipeline.yaml \
  --set runtime.server_port=8503 \
  --set inference.confidence_default=0.05 \
  config diff
```

持久修改会先完整校验新配置，再将原文件备份到 `reports/config_audit/backups/` 并记录 `reports/config_audit/events.jsonl`：

```bash
agile-agent --config configs/agent_pipeline.yaml config set runtime.server_port 8503
agile-agent --config configs/agent_pipeline.yaml config unset runtime.local_python
```

模型哈希、代际注册表、production 通道及已验收后端属性是受保护配置，持久变更由代际或发布工具完成。

## 主配置格式

主 YAML 是完整配置，不与隐式默认值深度合并。新建设备配置时，复制对应的受控主文件并修改所需字段。下面是需要修改的局部示例：

```yaml
schema_version: 3

runtime:
  mode: local
  local_python: null
  default_device: "1"
  server_host: 127.0.0.1
  server_port: 8503

inference:
  backend: ultralytics_cuda
  imgsz: 1280
  specialist_imgsz: 1280
  batch_size: 32
  confidence_default: 0.01
```

| 章节 | 作用 |
| --- | --- |
| `runtime`, `web` | 解释器、设备、回环监听地址、端口和 production 注册表 |
| `inference`, `routing`, `decoding` | 推理后端、输入尺寸、阈值边界、类别 owner、证据权重、并行和图像解码 |
| `storage`, `logging`, `ui` | 结果缓存上限/TTL、日志轮转和 Web 界面行为 |
| `incremental_workbench` | 增量数据包限额、train/dev/lock 拆分、血缘、训练和生命周期参数 |
| `performance` | API 性能目标、预热、并发、batch probe 和报告路径 |
| `generation`, `gates`, `incremental_guardian` | 代际复核/提升、官方硬门禁、诊断阈值和恢复动作 |
| `native_backend`, `tensorrt_backend`, `ascend_backend` | CUDA native/TensorRT/Ascend ACL 的设备、引擎或 OM、精度、输出契约和验收状态 |
| `model`, `assets`, `functional_models` | 模型入口、必需资产和三个功能模型注册表 |
| `automation`, `submission`, `blackboard`, `decision` | 受控动作、提交输入、黑板报告与决策输出 |
| `detector`, `inputs`, `incremental` | Base 检测器证据、数据审计输入和增量协议入口 |

## 必填与可选设置

主配置加载器要求 `schema_version: 3` 以及以下映射：`runtime`、`web`、`inference`、`routing`、`decoding`、`storage`、`logging`、`incremental_workbench`、`ui`、`performance`、`generation`、`gates`、`incremental_guardian`、`native_backend`、`ascend_backend`、`tensorrt_backend`、`model`、`assets`、`functional_models`、`incremental` 和 `automation`。`decision.actions` 也必须非空。

关键启动约束：

- `runtime.mode` 必须为 `local`，`runtime.server_host` 必须是回环地址，`default_device` 必须是非负设备编号。
- `inference.backend` 可以是 `ultralytics_cuda`、`tensorrt_engine`、`tensorrt_native` 或 `ascend_acl`；置信度默认值必须位于 `confidence_min` 和 `confidence_max` 之间。
- `routing.detection_evidence_weight + routing.context_evidence_weight` 必须等于 `1.0`。
- `routing.cross_class_suppression` 仅接受 `highest_confidence` 与 `all_classes`；`iou`、`smaller_box_coverage` 和 `incremental_over_base_margin` 都必须位于对应的 `[0,1]` 范围。
- 启用 `routing.score_calibration` 时，必须同时登记 `frozen_base_model` 与 `incremental_model` 的 logit-affine temperature/bias，并标明 `source_split: mixed_dev_only`。
- 隔离演示配置通过 `routing.edge_incremental_adapter` 登记已验收 manifest、协议 ID 和加载要求；`WebInferenceEngine` 在冻结 score calibration 之前执行 Adapter 置信度更新。
- `logging.request_bodies` 必须为 `false`；上传的图像和数据包不写入请求日志。
- `incremental.learning_data_scope` 必须为 `incremental_dataset_only`，支持模式必须同时包含 `class_incremental` 和 `target_incremental`。
- `model.expected_sha256`、已验收后端的模型资产和发布报告必须登记有效身份。
- `inference.backend: ascend_acl` 需要 `ascend_backend.validated: true` 或在受控评分进程中使用 `validation_candidate: true`。`independent_yolo26_e2e_v1` 布局要求恰好两个 YOLO26 E2E 检测 OM 和一个 context OM。
- `inference.backend: tensorrt_engine` 需要设备匹配的 TensorRT 版本、GPU 计算能力、engine 身份和验收报告。

可选值在受控文件中也显式写出。当前使用的 `null` 语义如下：

| 字段 | `null` 含义 |
| --- | --- |
| `runtime.local_python` | 使用当前运行 Agent 的 Python |
| `inference.quantize` | 不在 Ultralytics 运行时动态量化 |
| `tensorrt_backend.expected_version` / `expected_compute_capability` | 当 TensorRT engine 后端未启用且未验收时不绑定设备档 |
| `submission.official_format` | 提交格式由提交编排层在准备完成后填入 |

## 默认值（x86/CUDA）

| 配置 | 当前值 |
| --- | --- |
| 服务 | `127.0.0.1:8501` |
| 推理后端 | `ultralytics_cuda` |
| Base / Incremental 输入尺寸 | `1280 / 1280` |
| batch | `32` |
| IoU / 请求默认置信度 | `0.70 / 0.01` |
| 图像解码 | OpenCV，`4` workers，OpenCV 内部线程数 `0` |
| 模型并行 | Base、Incremental 和 context 并行，最多 `6` 个 model workers |
| 类别 owner | Base 固定负责全局类 `0–3`，Incremental 专家固定负责 `4–5` |
| 全类别重叠抑制 | 对所有输入启用；跨类别 `IoU >= 0.50` 或小框覆盖率 `>= 0.95` 时保留最高置信度框 |
| 性能目标 | API `30 FPS`，p95 `50 ms`，`8` 并发请求 |
| 评分门禁 | Base mAP50 `>=0.80`，New-mAP50 `>=0.60`，KRR `>=0.95`，old-data overlap `=0` |

`models/generations.json` 的 `channels.production` 和 `channels.candidate` 均指向 `incremental_detection_generation_4plus2`。当前六类基础阈值为：

| 全局类 | 名称 | owner | 基础阈值 | 最大场景惩罚 |
| ---: | --- | --- | ---: | ---: |
| 0 | `soldier` | Base | 0.21 | 0.15 |
| 1 | `small_aircraft` | Base | 0.14 | 0.88 |
| 2 | `warship` | Base | 0.36 | 0.26 |
| 3 | `tank` | Base | 0.05 | 0.19 |
| 4 | `patrol_boat` | Incremental | 0.57 | 0.65 |
| 5 | `armored_vehicle` | Incremental | 0.82 | 0.00 |

x86 的 `soft_threshold_penalty` 根据 Scene-SensorNet 的 `air/forest/sea/urban` 已知类概率调整逐类有效阈值，`hard_routing: false` 使 Base 和 Incremental 类别 owner 保持固定。

`routing.cross_class_suppression` 是数据来源无关的正式后处理：Web、CLI、训练集回放、dev、lock 和未来无标签图像走同一规则，不读取文件名或标签，也不维护类别对白名单。同类重复框仍由模型后端和 class-aware NMS 处理；这一层只在不同类别框高度重叠时按校准后置信度仲裁。Ascend 736 OM 的冻结参数为 `iou=0.90`、`smaller_box_coverage=0.95`。

`routing.edge_incremental_adapter` 在标准 production 配置中保持关闭，由 `promote_demo.py` 为通过门禁的 310B 离线演示生成独立配置。manifest 固化 `run_id`、协议、类别顺序、有效权重、精度结果和运行时验收状态，使 CLI/Web 可以明确显示 Adapter 是否已经接入。

增量工作台的默认训练值为 `imgsz=1280`、`batch=18`、`epochs=500`、`patience=50`、`optimizer=AdamW`、`lr0=0.001`、`seed=20260821`、`deterministic=true` 和 `amp=true`。数据拆分使用 `validation_fraction=0.20`、`lock_fraction=0.20` 与 `split_seed=20260821`。

Scene-SensorNet 的单独训练配置使用 `224` 输入、`batch=256`、`epochs=200`、`patience=30` 和 `seed=20260821`；它的 `protocol.phase` 为 `system_calibration`。

## 4+2 增量协议配置

`configs/incremental_detection_policy.yaml` 直接定义四个阶段：

| 阶段 | 是否计入增量学习 | 数据与更新边界 |
| --- | --- | --- |
| `base_learning` | 否 | 仅 Base 数据，更新 Base 检测器 |
| `incremental_learning` | 是 | 仅当轮 Increment train/dev，只更新当轮新类专家，Base 和已学轮次专家冻结 |
| `system_calibration` | 否 | 检测器冻结，训练 Scene-SensorNet 并选择场景先验、逐类阈值与门控参数 |
| `joint_evaluation` | 否 | 对截至当轮的全部已学类别评分，无梯度、无选参、无权重更新 |

`configs/incremental_round_registry_4plus2.yaml` 将 Base 类登记为 `0–3`，将 `patrol_boat` 登记为 round 1 的全局类 `4`，将 `armored_vehicle` 登记为 round 2 的全局类 `5`。每轮都指定独立 train/dev/lock split、父子 generation ID 和专家局部到全局映射。

当前 x86 production 代际使用一个二类 Incremental 专家同时负责全局类 `4` 和 `5`；两轮注册表是新一轮训练、候选登记和逐轮证据的权威输入。

## Ascend310B v2 当前值

正式发布 ID 为 `20260824-4plus2-yolo26-runtime-calibration-v1`。仓库根 Ascend 配置与发布包内配置登记同一组 release-local OM、构建清单、验证摘要和冻结运行时策略。根配置另保留当前 strict 4+2 工作台与 x86/TensorRT 默认值，因此不要以整文件字节相等代替 release 资产验证。

| 配置 | 当前值 |
| --- | --- |
| SoC / CANN | `Ascend310B1` / `7.0.RC1` |
| 服务监听 | 正式主实例 `127.0.0.1:18501`，公共入口由路由脚本绑定到 `8501` |
| 推理后端 | `ascend_acl` |
| 模型布局 | `independent_yolo26_e2e_v1` |
| 检测输入 | NCHW `[1,3,608,736]`，AIPP NHWC `[1,608,736,3]` |
| 检测输出 | Base 和 Incremental 均为 `[1,300,6]` (`yolo26_e2e_v1`) |
| context | 真实 `scene_sensor_net.om`，`context_mode=model` |
| 预处理/调度 | `dvpp`、`async_stream`、`unified_enqueue`、`pageable` |
| 请求默认置信度 | `0.01` |
| batch | `1` |
| 性能目标 | `30 FPS`，p95 `35 ms`，30 次预热 |

| OM | owner/功能 | 全局类 |
| --- | --- | --- |
| `om/base_detector.om` | 冻结 Base 检测器 | `0–3` |
| `om/incremental_detector.om` | Incremental 检测器 | 局部 `0/1` 映射到全局 `4/5` |
| `om/scene_sensor_net.om` | 已知类场景/传感器模型 | `air/forest/sea/urban` 与 `ir/sar` |

Ascend 内容执行门控使用 `skip_specialist_on_scene_and_base_evidence_v1`：当 `air` 概率至少 `0.5` 且 Base 出现全局类 `1` 证据时跳过 Incremental 专家。线上输入只包含场景概率和 Base 检测结果。

当前 x86 根配置使用全类别 `IoU=0.50` 与小框覆盖率 `0.95` 抑制。Ascend 在 mixed dev 上冻结为 `IoU=0.90`、小框覆盖率 `0.95`，并在阈值与仲裁前分别对 Base 和 Specialist 执行 logit-affine 校准。Base 为 temperature `1.5` / bias `0`，Specialist 为 temperature `1.0` / bias `-0.5`。

`configs/ascend310b/full_score_method.yaml` 的训练契约使用独立 YOLO26s、`input_size=[608,736]`、`epochs=500`、`patience=50`、`seed=3407`、`optimizer=AdamW`、`lr0=0.001` 和确定性 AMP 训练。候选端口为 `8502`，正式公共端口为 `8501`；系统在 mixed dev 上搜索 `5,476` 组参数，不读取 lock 选参。

## 环境变量

工程以受版本控制的 YAML 和发布清单作为配置主源；默认 x86 启动直接使用这些显式配置。环境变量用于选择配置、解释器、发布目录、板端训练环境或构建/评分入口。

| 变量 | 必需性 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `AGILE_AGENT_CONFIG` | 可选 | `auto` | 覆盖 Web、CLI 与 app 的自动主配置选择；Ascend systemd 服务将它设为 release-local YAML |
| `AGILE_AGENT_ASCEND_RELEASE` | ARM 可选 | 未设置；专用 Ascend 启停脚本使用当前正式 release | `auto` 模式下从指定 release 读取 Ascend 配置与 OM 映射 |
| `AGILE_AGENT_OVERRIDES` | 可选 | 空列表 | JSON 字符串数组，例如 `["runtime.server_port=8503"]` |
| `AGILE_AGENT_PYTHON` | 可选 | x86 读取 `.agent-python`/现有环境，ARM 读取 `AGILE_AGENT_ASCEND_ENV` | 显式指定启动、`doctor`、环境引导与 TensorRT 导出所用 Python |
| `PYTHON_BIN` | 可选 | 自动寻找 Python 3.12/3.11/3.10 | `bootstrap_x86.sh` 创建 `.venv` 时的 Python |
| `VIRTUAL_ENV` | 可选 | 未设置 | `bootstrap_x86.sh` 可复用的已激活 venv |
| `CONDA_PREFIX` | 可选 | 未设置 | `bootstrap_x86.sh` 可复用的已激活 Conda 环境 |
| `PYTORCH_INDEX_URL` | 可选 | `https://download.pytorch.org/whl/cu124` | x86 bootstrap 的 PyTorch wheel 索引 |
| `TORCH_VERSION` | 可选 | `2.5.1+cu124` | x86 bootstrap 在 CUDA 栈不可用时安装的 PyTorch 版本 |
| `TORCHVISION_VERSION` | 可选 | `0.20.1+cu124` | 与 `TORCH_VERSION` 配套的 torchvision 版本 |
| `AGILE_AGENT_ASCEND_ENV` | 可选 | `/usr/local/miniconda3/envs/agileagent` | Ascend 服务的 Conda 环境 |
| `AGILE_AGENT_CANN_ENV` | 可选 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | 统一启动脚本在 ARM 上加载的 CANN 环境脚本 |
| `AGILE_AGENT_ASCEND_PID_FILE` | 可选 | `$AGILE_AGENT_ASCEND_RELEASE/agent-web.pid` | Ascend 启停脚本的 PID 文件 |
| `AGILE_AGENT_ASCEND_PORT` | 可选 | `8501` | Ascend 直接启动端口；systemd 正式主实例设为 `18501` |
| `AGILE_AGENT_ASCEND_USER` | 可选 | `HwHiAiUser` | 安装 Ascend 主/回滚 systemd 服务的运行用户 |
| `AGILE_AGENT_ASCEND_PYTHON` | 可选 | `/usr/local/miniconda3/envs/agileagent/bin/python` | Ascend OM 构建和 score gate 使用的 Python |
| `AGILE_AGENT_FULL_SCORE_METHOD` | 可选 | `configs/ascend310b/full_score_method.yaml` | 替换 score gate 的方法契约 |
| `AGILE_AGENT_AIPP_DIR` | 可选 | `configs/ascend310b/aipp` | OM 构建使用的 AIPP 配置目录 |
| `AGILE_AGENT_RESUME` | 可选 | `0` | 设为 `1` 时复用命令和成功日志均匹配的现有 OM |
| `AGILE_AGENT_IPTABLES` | 可选 | `/usr/sbin/iptables-legacy` | Ascend 公共 `8501` 回环路由工具 |
| `AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION` | score gate 内部必需 | 未设置 | 候选验证进程由 score gate 设为 `1`，使 `validation_candidate: true` 配置可加载 |
| `AGILE_AGENT_BUILD_ROOT` | 构建脚本内部 | 仓库根目录 | 传给 OM 构建清单生成器的源码根路径 |
| `AGILE_AGENT_ONNX_DIR` | 构建脚本内部 | 第一个位置参数 | 已解析的 Base/Incremental ONNX 目录 |
| `AGILE_AGENT_OUTPUT_DIR` | 构建脚本内部 | 第二个位置参数 | OM、ATC 日志和构建清单输出目录 |
| `AGILE_AGENT_CONTEXT_BUILD_MANIFEST` | 构建脚本内部 | 第三个位置参数 | Scene-SensorNet 父构建清单路径 |
| `AGILE_EDGE_PRODUCTION_PYTHON` | 310B 增量演示可选 | `/usr/local/miniconda3/envs/agileagent/bin/python` | 运行评估、ATC 编排和正式推理复测的 production Python |
| `AGILE_EDGE_TRAINING_PYTHON` | 310B 增量演示可选 | 自动选择 `~/agileagent/envs/agileagent_train` 或已验证实验环境 | 执行 `torch_npu` 真实反向传播的独立训练 Python |
| `AGILE_EDGE_CANN_ENV` | 310B 增量演示可选 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | 一键演示和底层流水线加载的 CANN 环境脚本 |
| `AGILE_EDGE_WHEEL_DIR` | 训练环境准备可选 | 由操作者指定 | 离线准备板端训练环境时使用的 aarch64 wheel 目录 |

`build_ascend_yolo26_e2e_oms.sh` 在同一进程内设置上述四个内部变量，并将已解析的路径传给构建清单生成器。

## 按环境选择配置

仓库使用独立 YAML 表示 x86、Ascend、候选和 release-local 平台配置，运行环境通过配置选择变量组合这些文件。

- x86/CUDA 日常运行使用 `configs/agent_pipeline.yaml`。
- x86/CUDA 设备专用 TensorRT 档从主配置复制，填入当前 TensorRT 版本、GPU 计算能力和 engine 路径后，交给 `scripts/export_tensorrt_engines.sh <profile.yaml>`。
- Ascend310B v2 源码校验使用 `configs/agent_pipeline_ascend310b.yaml`；物化后的服务使用 release-local `configs/agent_pipeline_ascend310b.yaml`。
- Ascend 候选验证由 `run_ascend310b_score_gate.sh` 在 `8502` 启动独立进程；通过的发布配置由提升工具生成。
- Ascend 断网增量演示由 `run_ascend310b_incremental_demo.sh` 生成 run-local 配置，并通过 `AGILE_AGENT_CONFIG` 交给 CLI/Web 复现本次 Adapter 结果。

临时差异使用 `--set`，长期平台差异使用独立 YAML。配置更改后重启对应服务，使新的有效配置与运行时注册表同步。
