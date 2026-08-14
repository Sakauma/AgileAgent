<!-- generated-by: gsd-doc-writer -->
# 配置参考

AgileAgent 使用 YAML 文件保存运行、推理、增量学习和后端参数。默认运行配置是 [`configs/agent_pipeline.yaml`](../configs/agent_pipeline.yaml)；它是完整配置，不是可与默认值合并的局部覆盖文件。配置加载顺序为：读取 YAML、展开值为完整 `${NAME}` 形式的环境变量占位符、应用进程级 `key=value` 覆盖，最后执行模式与范围校验。

## 环境变量

项目当前没有 `.env` 或 `.env.example`，也没有加载 dotenv 文件。需要使用环境变量时，应由当前 shell、服务管理器或部署平台显式注入；仓库的 `.gitignore` 会忽略 `.env` 和 `.env.*`，但明确允许将来提交不含密钥的 `.env.example`。

| 变量 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `AGILE_AGENT_CONFIG` | 否 | `configs/agent_pipeline.yaml` | 为调用 `load_config()` 且未显式传入路径的进程选择完整运行配置。Ascend 启动脚本会将其设置为发布目录中的 `configs/agent_pipeline_ascend310b.yaml`。CLI 命令应优先使用 `--config` 明确选取配置。相对路径按仓库根目录解析。 |
| `AGILE_AGENT_OVERRIDES` | 否 | 空列表 | JSON 字符串数组，每一项必须是 `key=value`，例如 `["inference.confidence_default=0.60"]`。值按 YAML 标量解析；格式错误、不是字符串数组或键不存在时加载失败。CLI 命令应优先使用可重复的 `--set`。 |
| `AGILE_AGENT_PYTHON` | 否 | 见说明 | 为 `bootstrap_x86.sh`、`start_agent.sh` 和 TensorRT 导出脚本选择 Python。`bootstrap_x86.sh` 会强制校验可执行文件、Python 3.10–3.12 和 `pip`；启动与导出脚本只预检其可执行性，因此正常流程应先通过 bootstrap。未设置时，启动脚本先读取 `.agent-python`，再使用 `.venv/bin/python`；引导脚本还会依次检查现有 `.venv`、`VIRTUAL_ENV`、`CONDA_PREFIX` 和其他可用解释器。 |
| `PYTHON_BIN` | 否 | 自动查找 `python3.12`、`python3.11`、`python3.10` | 仅供 `scripts/bootstrap_x86.sh` 创建 `.venv`。只有流程没有选中更高优先级的现有 `.venv`、`VIRTUAL_ENV` 或 `CONDA_PREFIX` 并实际进入 `PYTHON_BIN` 分支时，无效命令或不受支持的版本才会使脚本退出。 |
| `VIRTUAL_ENV` | 否 | 未设置 | `scripts/bootstrap_x86.sh` 在仓库 `.venv` 不可用时，将其下的 `bin/python` 作为候选解释器。 |
| `CONDA_PREFIX` | 否 | 未设置 | `scripts/bootstrap_x86.sh` 在仓库 `.venv` 和 `VIRTUAL_ENV` 均不可用时，将其下的 `bin/python` 作为候选解释器。 |
| `PYTORCH_INDEX_URL` | 否 | `https://download.pytorch.org/whl/cu124` | x86 引导脚本安装 PyTorch/CUDA 组合时使用的包索引。 |
| `TORCH_VERSION` | 否 | `2.5.1+cu124` | x86 引导脚本安装的 PyTorch 版本。 |
| `TORCHVISION_VERSION` | 否 | `0.20.1+cu124` | x86 引导脚本安装的 torchvision 版本。 |
| `AGILE_AGENT_ASCEND_RELEASE` | 否 | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` | `start_agent_ascend310b.sh` 与 `stop_agent_ascend310b.sh` 使用的板端发布根目录。启动脚本要求其中存在 `src/`、`conda-env/bin/python` 和发布配置；停止脚本只用该根目录定位 `agent-web.pid`。 |

任何 YAML 字符串都可以使用一个完整值占位符，例如：

```yaml
runtime:
  local_python: "${AGILE_RUNTIME_PYTHON}"
```

此处的 `AGILE_RUNTIME_PYTHON` 是配置作者自行定义的变量。占位符只能占据整个字符串；`prefix-${NAME}` 不会展开。变量缺失时，加载器报错 `配置引用的环境变量不存在：<键路径> -> <变量名>`。环境展开早于 `--set`，因此不能依靠后续覆盖绕过缺失占位符。

严格 3+1 训练配置还会把以下固定值写入当前训练进程；它们来自 [`configs/strict_class_incremental_3plus1.yaml`](../configs/strict_class_incremental_3plus1.yaml)，无需在 shell 中预设：

| 变量 | 配置值 | 用途 |
| --- | --- | --- |
| `NCCL_P2P_DISABLE` | `1` | 禁用 NCCL 点对点传输。 |
| `NCCL_IB_DISABLE` | `1` | 禁用 NCCL InfiniBand 传输。 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少长训练中的 CUDA 保留段碎片。 |

Ascend 启动脚本还会读取目标机的 `/usr/local/Ascend/ascend-toolkit/set_env.sh`。该外部脚本不在仓库中，因此它额外导出的 CANN 环境变量无法从本项目枚举。

<!-- VERIFY: 目标 Ascend 板上的 /usr/local/Ascend/ascend-toolkit/set_env.sh 存在，并提供当前 CANN 运行时所需的环境变量 -->

## 配置文件格式

所有项目配置均为 UTF-8 YAML。运行配置顶层必须是映射，`schema_version` 必须为 `2`，未知顶层键以及受约束章节中的未知键会被拒绝。相对文件路径统一相对仓库根目录解析；绝对路径保持不变。

### 配置文件清单

| 文件 | 作用 | 主要顶层键 |
| --- | --- | --- |
| [`configs/agent_pipeline.yaml`](../configs/agent_pipeline.yaml) | 默认 x86/NVIDIA 运行配置，也是配置迁移的当前模板。 | `runtime`、`web`、`inference`、`routing`、`storage`、`logging`、`incremental_workbench`、各推理后端、模型与代际管理等。 |
| [`configs/agent_pipeline_ascend310b.yaml`](../configs/agent_pipeline_ascend310b.yaml) | Ascend 310B1 板端完整运行配置；选择 `ascend_acl` 并登记 OM 路径、哈希与验收报告。 | 与默认运行配置相同。 |
| [`configs/incremental/warship_3plus1.yaml`](../configs/incremental/warship_3plus1.yaml) | 舰船 3+1 实验入口，定义数据来源、轮次、训练适配器和验收条件。 | `experiment`、`dataset`、`audit`、`partition`、`training`、`acceptance`、`diagnostics`、`integrity`。 |
| [`configs/strict_class_incremental_3plus1.yaml`](../configs/strict_class_incremental_3plus1.yaml) | 严格类别增量训练模板。 | `paths`、`runtime`、`training_policy`、训练阶段、融合、校准、上下文门控、验收和 `protocols`。 |
| [`configs/incremental_detection_policy.yaml`](../configs/incremental_detection_policy.yaml) | 增量学习数据边界、模式契约和验收策略。 | `learning_phase`、`derived_artifacts`、`mode_contracts`、`evaluation_phase`、`acceptance`。 |
| [`configs/functional_models.yaml`](../configs/functional_models.yaml) | 功能模型注册表及模型间协作关系。 | `generation_registry`、`models`、`collaboration`。 |
| [`configs/scene_sensor_model.yaml`](../configs/scene_sensor_model.yaml) | Scene-SensorNet 数据、训练和输出配置。 | `data`、`model`、`train`、`output`、`acceptance`。 |
| [`configs/local_infer_gpu.yaml`](../configs/local_infer_gpu.yaml) | 本地 GPU 基础检测与模型冒烟检查参数。 | `model`、`source`、`predict`、`output`、`names`。 |
| [`configs/submission_infer_base_3class.yaml`](../configs/submission_infer_base_3class.yaml) | 三类基础检测器的提交推理与导出参数。 | `model`、`source`、`predict`、`output`、`names`。 |

运行时生成的 `dataset.yaml`、`batch.yaml`、`training_adapter.yaml` 和实验快照属于可审计产物，不是全局配置入口，通常不应手工编辑。

### 主运行配置结构

主配置的顶层职责如下：

| 分组 | 说明 |
| --- | --- |
| `schema_version`、`seed` | 配置模式版本和全局随机种子。 |
| `runtime`、`web`、`ui` | Python、设备、监听地址、Web 注册表和界面保留策略。 |
| `inference`、`routing`、`decoding`、`storage`、`performance` | 推理后端、阈值、模型协作、解码并发、临时缓存和性能门槛。 |
| `logging`、`incremental_workbench` | 结构化日志、上传限制、训练、数据血缘和生命周期参数。 |
| `generation`、`gates`、`incremental_guardian` | 代际注册、复核门槛、自动晋升和失败恢复动作。 |
| `native_backend`、`tensorrt_backend`、`ascend_backend` | NVIDIA 原生/TensorRT 兼容路径以及 Ascend ACL/OM 运行参数。即使某后端未启用，其章节仍须满足基础结构校验。 |
| `model`、`assets`、`functional_models` | 主权重身份、必需资产清单和功能模型注册表。 |
| `automation`、`blackboard`、`decision` | 允许的自动动作、黑板输出和决策动作定义。 |
| `submission`、`detector`、`inputs`、`modules`、`policies`、`thresholds`、`incremental` | 提交状态、检测器证据、模块清单以及增量学习合同。 |

局部 YAML 不会自动继承默认配置。创建可运行的设备配置时，应复制完整模板，再修改所需键：

```bash
PROFILE=configs/agent_pipeline.local.yaml
cp configs/agent_pipeline.yaml "$PROFILE"
agile-agent config set --config "$PROFILE" inference.confidence_default 0.60
agile-agent config validate --config "$PROFILE"
agile-agent config show --config "$PROFILE" --effective
```

`configs/agent_pipeline.local.yaml` 符合仓库现有的 `/configs/*` 忽略规则，不会被默认纳入版本控制。`config set` 会先解析 YAML 标量、校验完整配置、备份旧文件，再以临时文件原子替换目标配置。

## 必需与可选设置

### 启动时必需

- `schema_version` 必须严格等于 `2`。
- 下列章节必须为映射并满足其字段校验：`runtime`、`inference`、`routing`、`decoding`、`storage`、`logging`、`incremental_workbench`、`ui`、`performance`、`generation`、`gates`、`incremental_guardian`、`native_backend`、`ascend_backend`、`tensorrt_backend`、`model`、`assets`、`functional_models`、`incremental` 和 `automation`。`decision.actions` 也必须是非空映射。
- `runtime.mode` 只能是 `local`，`runtime.default_device` 必须是非负设备编号，`runtime.server_host` 只能是 `127.0.0.1`、`localhost` 或 `::1`，端口范围为 1–65535。
- `inference.backend` 必须是 `ultralytics_cuda`、`tensorrt_engine`、`tensorrt_native` 或 `ascend_acl`；置信度默认值必须位于最小值与最大值之间。
- `routing.detection_evidence_weight` 与 `routing.context_evidence_weight` 之和必须为 `1`。
- `logging.root` 和 `incremental_workbench.root` 不能为空；`logging.request_bodies` 必须为 `false`。
- `model.weights`、64 位十六进制 `model.expected_sha256`、`assets.manifest`、`assets.checksums` 和非空 `assets.required` 均为必需项。
- `performance` 必须给出 `benchmark_split` 和 `report_root`；`generation` 必须给出 `registry`、`runtime_registry`、`recheck_lock_split` 和 `report_root`。
- 增量工作台的 `training`、`lineage` 和 `lifecycle` 子映射必须存在。训练配置必须包含 Python、初始权重、设备、尺寸、批量、轮数、优化器、学习率、随机种子以及确定性/AMP 开关。

这些是配置加载器的模式检查。文件路径是否存在、权重是否可加载、GPU/CUDA 是否可用等运行条件由 `agile-agent doctor`、模型冒烟检查和具体后端初始化继续验证。

### 条件必需

| 条件 | 额外要求 |
| --- | --- |
| 任意值写成 `${NAME}` | 启动前必须定义 `NAME`；缺失即停止加载。 |
| `inference.backend: ascend_acl` | `ascend_backend.validated` 必须为 `true`；每个检测 OM 与上下文 OM 必须登记路径和 64 位 SHA256。 |
| `inference.backend: tensorrt_engine` | 必须登记 TensorRT 版本、GPU 计算能力、各 engine 路径和 SHA256；已验收配置还必须有 `validation_report`。 |
| `inference.backend: tensorrt_native` | `native_backend.library`、`base_engine` 和 `context_engine` 必须非空；当 `native_backend.validated: true` 时，各 engine 必须有 SHA256。 |
| `tensorrt_backend.precision: int8` | `int8_calibration.enabled` 必须为 `true`，并提供代表集、阈值集、缓存目录和有效批量参数。 |

默认配置中 `runtime.local_python: null`、未启用后端的 `validated: false`、对应的校验报告或 engine 哈希为 `null` 是有意的可选状态。`configured_python()` 在 `runtime.local_python` 为空时使用当前 Python 解释器。`ascend_backend.execution_mode` 和 `encoded_preprocessing` 如果省略，源代码回退到 `synchronous` 和 `cpu`；其他大多数字段没有代码级补全，删除后会校验失败。

### 配置修改保护

`config set` 与 `config unset` 不能修改以下受保护状态：模型预期哈希、资产校验和、代际注册表与通道、代际运行状态、TensorRT engine/上下文 engine 身份和验收字段。`generation recheck/promote/rollback` 管理代际注册表，TensorRT 导出与验收流程管理 engine 身份和验收字段；仓库目前没有自动写回 `model.expected_sha256` 或 `assets.checksums` 的入口，这两个值必须作为单独评审的发布资产变更维护。普通配置命令不能绕过这些边界。

持久修改会在 `reports/config_audit/backups/<时间戳>/` 保存修改前的 YAML，并向 `reports/config_audit/events.jsonl` 和结构化运行日志写入哈希与操作记录。命令输出的 `restart_required: true` 表示已运行的服务不会热加载新配置。

## 默认值

下表记录默认 x86 运行配置 [`configs/agent_pipeline.yaml`](../configs/agent_pipeline.yaml) 的主要值。这些值由完整 YAML 提供；除上节明确列出的回退项外，不能通过删除键来“恢复默认”。

| 区域 | 默认值 |
| --- | --- |
| 配置选择 | `configs/agent_pipeline.yaml`，无进程级覆盖。 |
| 运行时 | `mode: local`；`local_python: null`；设备 `0`；监听 `127.0.0.1:8501`。 |
| 推理 | `ultralytics_cuda`；基础尺寸 `896`；专家尺寸 `640`；IoU `0.70`；默认置信度 `0.50`；最大检测数 `300`；批量 `20`；不量化。 |
| 路由 | 启用增量与验收门禁；最多 `4` 个专家；模型、上下文和上下文批处理并行均启用；最多 `6` 个模型工作线程。 |
| 解码 | OpenCV；`4` 个解码工作线程；OpenCV 内部线程数 `0`。 |
| 临时存储 | 最多 `4` 项、TTL `1800` 秒、总量 `536870912` 字节（512 MiB）。 |
| 日志 | `reports/agent_logs`；单文件 `10485760` 字节（10 MiB）；保留 `14` 份；不记录请求体。 |
| 增量工作台 | 根目录 `data/incremental_batches`；上传上限 2 GiB；解压上限 5 GiB/20000 文件；最少 2 张图；验证集与锁定集比例均为 `0.20`；轮询间隔 `2000` 毫秒。 |
| 增量训练 | 当前解释器；设备 `0`；`imgsz: 640`；批量 `32`；`80` epochs；AdamW；`lr0: 0.001`；确定性和 AMP 均开启。 |
| UI | 历史 `20` 条；完整结果缓存 `10` 份；健康轮询 `15000` 毫秒；提示持续 `4200` 毫秒；默认视图 `detect`。 |
| 性能 | API 目标 `30` FPS、P95 `50` 毫秒；允许自动启动服务；启动/请求超时均为 `180` 秒；基准轮数 `3`。 |
| 代际 | production 通道；候选 `incremental_detection_generation`；校准阈值 `0.63`；`auto_promote: true`；冒烟图像数 `1`。 |
| 后端状态 | 原生 TensorRT、Ascend 和 TensorRT engine 均 `validated: false`；默认实际后端仍为 `ultralytics_cuda`。 |

## 按环境覆盖

仓库没有 `.env.development`、`.env.test`、`.env.production`，也不根据类似 `NODE_ENV` 的变量自动合并配置。不同设备或部署环境应使用独立的完整 YAML 配置，并通过以下三种机制之一选择或覆盖。

### 选择完整配置

对 CLI 使用 `--config`：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent --config configs/agent_pipeline.local.yaml serve
```

对直接导入 Web 应用的进程，可设置运行配置环境变量：

```bash
export AGILE_AGENT_CONFIG="$PWD/configs/agent_pipeline.local.yaml"
python -m uvicorn fair_agent.web.app:app --host 127.0.0.1 --port 8501
```

### 仅覆盖当前进程

`--set` 可重复使用，且不会写回 YAML。键必须已经存在，值按 YAML 解析：

```bash
agile-agent \
  --config configs/agent_pipeline.local.yaml \
  --set inference.confidence_default=0.60 \
  --set performance.auto_start_server=false \
  config diff
```

需要给不便传 CLI 参数的默认加载进程注入相同覆盖时，使用 JSON 数组：

```bash
export AGILE_AGENT_OVERRIDES='["inference.confidence_default=0.60","performance.auto_start_server=false"]'
```

有效值按“YAML → 完整环境变量占位符 → `--set`/`AGILE_AGENT_OVERRIDES`”顺序产生。可用以下命令查看脱敏后的结果：

```bash
agile-agent \
  --config configs/agent_pipeline.local.yaml \
  --set inference.confidence_default=0.60 \
  config show --effective
```

### 持久化设备配置

对于需要复用的本地设备参数，先复制模板，再用 `config set` 修改。TensorRT 导出脚本会拒绝直接修改或使用默认发布配置，因此必须传入设备专用副本。旧版本配置可迁移到当前完整模式：

```bash
agile-agent config migrate \
  --input configs/agent_pipeline.old.yaml \
  --output configs/agent_pipeline.migrated.yaml
```

当前仓库提供两套已提交的运行基线：

| 环境 | 配置 | 选择方式 |
| --- | --- | --- |
| x86/NVIDIA 开发与默认运行 | `configs/agent_pipeline.yaml` | CLI 默认值或显式 `--config`。 |
| Ascend 310B1 | `configs/agent_pipeline_ascend310b.yaml` | `scripts/start_agent_ascend310b.sh` 设置 `AGILE_AGENT_CONFIG`。 |
| x86/TensorRT 兼容实验 | 从默认配置复制的设备专用 YAML | 使用 `--config`，并把该文件传给 `scripts/export_tensorrt_engines.sh`。 |
| 测试 | 默认配置或测试创建的临时副本 | 没有专用 `.env.test` 或自动测试覆盖层。 |

Ascend 配置包含板端发布目录、CANN 版本、OM 路径、SHA256 和验收报告路径。复制到另一块板或新的发布目录时，必须重新核对并重新生成与验收相绑定的值。

截至本轮审查，配置中还有两处旧板前元数据尚未修正：两份主配置的 `policies.end_device` 都是 `paused_until_ascend_board_ready`，而 Ascend 配置中 `decision.actions.refresh_blackboard.required_artifacts` 仍指向 `configs/agent_pipeline.yaml`，不是当前选中的 Ascend 配置。它们不会改变 `inference.backend: ascend_acl` 或启动脚本选择的配置，但会原样进入决策展示和必需产物列表；在 release manifest 与功能模型注册表完成证据绑定后，应通过单独实现变更修正并补回归测试。

<!-- VERIFY: configs/agent_pipeline_ascend310b.yaml 中的绝对发布路径、CANN 版本、OM 文件和验收报告在目标板及当前发布版本上仍然有效 -->
