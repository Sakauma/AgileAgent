<!-- generated-by: gsd-doc-writer -->
# 开发指南

## 本地开发环境

AgileAgent 的 x86 开发环境面向 x86-64 Linux/WSL、Python `>=3.10,<3.13` 和可用的 NVIDIA CUDA 驱动。`scripts/bootstrap_x86.sh` 会选择现有兼容解释器或创建 `.venv`，检查 CUDA PyTorch/torchvision，安装 workbench、inference、dev 依赖，以 editable 模式注册 `agile-agent`，并把实际解释器路径写入 `.agent-python`。

```bash
git clone git@github.com:Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

后续命令统一读取已经登记的解释器：

```bash
AGENT_PYTHON="$(< .agent-python)"
"$AGENT_PYTHON" -m fair_agent.cli doctor
```

在已经配置好 CUDA PyTorch 的环境中，也可以手动安装项目依赖：

```bash
python -m pip install -c constraints-agent.txt -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
```

板端开发使用 `/usr/local/miniconda3/envs/agileagent`、现有 CANN/PyACL 运行时以及 `configs/requirements-ascend310b.txt` 中的 Web 与验证依赖。板端服务入口为 `scripts/start_agent_ascend310b.sh`，正式配置为 `configs/agent_pipeline_ascend310b.yaml`。

## 常用开发命令

项目使用 setuptools 和 `pyproject.toml`，没有单独的任务运行器。以下命令覆盖日常安装、启动、静态校验和正式资产检查：

| 命令 | 用途 |
| --- | --- |
| `./scripts/bootstrap_x86.sh` | 准备或复用 x86 CUDA 开发环境并注册 CLI |
| `./scripts/start_agent.sh` | 自动识别 x86/ARM、选择 CUDA/PT 或 Ascend/OM，校验环境并启动 Starlette 工作台 |
| `./scripts/start_agent.sh --cli` | 使用同一自动设备与模型选择启动终端工作台 |
| `"$AGENT_PYTHON" -m fair_agent.cli config validate --config configs/agent_pipeline.yaml` | 校验 x86 schema 3 配置 |
| `"$AGENT_PYTHON" scripts/verify_release.py --config configs/agent_pipeline.yaml` | 检查 x86 发布配置和正式资产 |
| `"$AGENT_PYTHON" scripts/smoke_models.py --load-only` | 在 CUDA 上加载三个正式功能模型 |
| `"$AGENT_PYTHON" tools/03_split_r2_4plus2.py --verify-only` | 验证 strict 4+2 固定清单 |
| `"$AGENT_PYTHON" -m pytest -q` | 运行完整 Python 回归；测试范围见 `docs/TESTING.md` |
| `bash -n scripts/*.sh` | 静态检查维护中的 Shell 脚本 |

只检查 Python 语法时可运行：

```bash
"$AGENT_PYTHON" -m compileall -q fair_agent tools scripts/smoke_models.py scripts/verify_release.py
```

## 代码组织

| 路径 | 开发责任 |
| --- | --- |
| `fair_agent/core/` | schema 3 配置、路径解析、审计、运行日志和状态基础设施 |
| `fair_agent/backends/` | Ultralytics CUDA、TensorRT 与 Ascend ACL 推理适配器 |
| `fair_agent/models/` | Scene-SensorNet 定义、加载和上下文预测 |
| `fair_agent/modules/` | 检测融合、增量批次、生命周期、代际、评分和发布业务逻辑 |
| `fair_agent/web/` | Starlette API、静态工作台和运行时原子切换 |
| `fair_agent/cli.py` | `agile-agent` 命令入口 |
| `configs/` | x86、Ascend、功能模型、场景模型和两轮增量配置 |
| `splits/strict_4plus2/` | Base、Increment、mixed 及逐轮 train/dev/lock 固定清单 |
| `models/production/incremental_detection/` | x86 正式 Base、增量专家、门控参数和冻结证据 |
| `models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/` | Ascend310B1 v2 三-OM 正式包 |
| `tools/` | 编号化数据、训练、评估、候选和验收入口 |
| `scripts/` | 环境、服务、OM 构建、评分门禁、物化和原子路由 |
| `tests/` | 配置、运行时、增量与 Ascend 发布回归 |

## 4+2 x86/CUDA 开发流程

### 1. 数据与固定清单

`splits/strict_4plus2/manifest.json` 描述 600/75/75 张 Base train/dev/lock、112/14/14 张 Increment train/dev/lock，以及两轮单类视图。修改数据划分工具后先执行只读校验：

```bash
"$AGENT_PYTHON" tools/03_split_r2_4plus2.py --verify-only
```

逐轮清单由类别注册表和合法数据根共同生成：

```bash
"$AGENT_PYTHON" tools/11_prepare_incremental_round_splits.py \
  --data-root /path/to/tiaozhanbei_4plus2_dataset_20260821
```

`configs/incremental_round_registry_4plus2.yaml` 是六类 ID、两轮顺序、父子代际、局部到全局映射和阶段数据范围的权威配置。

### 2. 四类 Base

Base 训练和选模入口如下：

| 入口 | 作用 | 关键输入 |
| --- | --- | --- |
| `tools/04_train_base_4plus2.py` | 使用一个或多个随机种子训练 Base 候选 | `--data-root`、`--model`、`--model-tag`、`--project`、`--batch` |
| `tools/05_select_base_4plus2.py` | 等待、复评并选择 Base 候选 | `--project`、`--dataset-yaml`、`--tags`、`--seeds` |

正式训练默认使用 1280 输入、500 epoch 和 50 轮早停；批量大小由可用显存确定。Base 只读取 `base_train.txt` 和 `base_dev.txt`，选定后冻结权重，`base_lock.txt` 用于参数冻结后的复核。

### 3. 两轮类别增量

轮次按注册表顺序运行：

```text
round_01_patrol_boat
  -> round_02_armored_vehicle
```

| 入口 | 作用 |
| --- | --- |
| `tools/06_train_incremental_4plus2.py` | 按 `--round-id` 和 `--seeds` 训练当轮单类专家 |
| `tools/07_select_incremental_4plus2.py` | 只用当轮 Increment dev 复评并选择专家 |
| `tools/08_evaluate_4plus2.py` | 在冻结参数下累计评估截至当轮的全部已学类别 |
| `tools/13_register_incremental_round_candidate.py` | 登记权重、类别 owner、指标和父子代际候选 |
| `tools/12_summarize_incremental_rounds.py` | 联合校验两轮代际链和逐轮证据 |

每轮训练视图只包含当前新增类别的图像和投影标签。四类 Base 与此前轮次专家权重保持冻结；当前轮次的 train/dev 用于训练和选模，lock 在候选参数冻结后用于累计评估。

### 4. Scene-SensorNet 与系统校准

Scene-SensorNet 是独立功能模型，配置位于 `configs/scene_sensor_model_4plus2.yaml`：

```bash
"$AGENT_PYTHON" tools/60_train_scene_sensor.py \
  --config configs/scene_sensor_model_4plus2.yaml \
  --run-dir /path/to/scene-candidate \
  --data-root /path/to/tiaozhanbei_4plus2_dataset_20260821

"$AGENT_PYTHON" tools/61_select_scene_sensor_4plus2.py \
  --project /path/to/scene-runs \
  --seeds 20260821
```

`tools/09_optimize_scene_aware_4plus2.py` 先以 `dev` 模式搜索逐类阈值、场景先验和门控参数，再以 `lock` 模式复核冻结候选。`tools/10_promote_scene_aware_4plus2.py` 在两轮证据、dev 搜索和 lock 结果完整时更新 production 代际及正式资产。

这部分属于 `system_calibration`；检测器训练发生在 Base 与两轮 `incremental_learning` 阶段，冻结后的六类评分属于 `joint_evaluation`。

## Ascend310B1 v2 开发流程

正式运行结构为 `independent_yolo26_e2e_v1`：四类 Base YOLO26s、二类 Incremental YOLO26s 和 Scene-SensorNet 分别使用独立 OM。检测 OM 的 uint8 NHWC 输入为 `[1,608,736,3]`，E2E 输出为 `[1,300,6]`；Scene OM 输入为 `[1,160,160,3]`。

当前开发链路如下：

| 阶段 | 入口 | 产物 |
| --- | --- | --- |
| 检测 OM 构建 | `scripts/build_ascend_yolo26_e2e_oms.sh` | Base/Incremental OM 与 build manifest |
| 三-OM 候选物化 | `tools/112_materialize_ascend_yolo26_candidate.py` | 候选配置和候选代际注册表 |
| 候选冻结评分 | `scripts/run_ascend310b_score_gate.sh` | 冻结预测、score、benchmark 和验证证据 |
| 正式包生成 | `tools/111_promote_ascend_full_score_release.py` | validated release 目录 |
| release 复核 | `tools/95_verify_ascend_release.py --require-validation` | 配置、OM、清单和报告一致性结果 |
| 板端物化 | `scripts/materialize_ascend310b_full_score_release.sh` | `/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2` |
| 服务提升 | `scripts/install_ascend310b_primary_services.sh` | `18501` 主实例、`8501` 公共路由和 `8502` 候选隔离 |

构建两个检测 OM：

```bash
./scripts/build_ascend_yolo26_e2e_oms.sh \
  /path/to/onnx \
  /path/to/output \
  /path/to/context-build-manifest.json
```

候选评分使用固定五参数接口：

```bash
./scripts/run_ascend310b_score_gate.sh \
  /path/to/candidate.yaml \
  /path/to/images \
  /path/to/mixed_lock.txt \
  /path/to/base_lock.txt \
  /path/to/score-output
```

内容执行门控读取 Scene 概率和 Base 检测：当 `air >= 0.5` 且 Base 检出 `small_aircraft` 时跳过增量专家。类别 owner 仍由代际注册表固定，阈值保存在 Host 配置中。

## 配置与正式资产同步

配置变更按责任同步：

- 通用 schema 或运行字段：更新 `fair_agent/core/config.py`、`configs/agent_pipeline.yaml`、`configs/agent_pipeline_ascend310b.yaml` 和配置测试。
- 功能模型职责或协作关系：更新 `configs/functional_models.yaml`、`models/manifest.json` 和对应加载/验证逻辑。
- 类别、轮次或 owner：更新 `configs/incremental_round_registry_4plus2.yaml`、`models/generations.json`、逐轮清单与增量回归。
- x86 production 模型：更新 `models/production/incremental_detection/` 中的权重、profile、calibration、metrics 和冻结证据。
- Ascend v2 正式包：更新 `models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/` 中的配置、OM、provenance、validation 和 `release.json`。

正式模型包更新时同步维护包内 `SHA256SUMS`、`models/SHA256SUMS.txt`、`models/manifest.json`、`configs/functional_models.yaml` 与 `models/generations.json`。普通源码和文档修改只执行与改动范围对应的语法、配置和路径检查。

`.gitignore` 默认排除数据集、运行目录、报告和一般模型二进制，并对白名单内的正式 x86 权重及 Ascend v2 包开放版本控制。提交正式二进制前使用以下命令确认跟踪状态：

```bash
git check-ignore -v path/to/artifact || true
git ls-files --error-unmatch path/to/artifact
```

## 代码风格

仓库未配置 Black、Ruff、isort、mypy 或 `.editorconfig`；现有代码约定如下：

- Python 使用四空格缩进、类型注解和 `pathlib.Path`，导入按标准库、第三方库和项目模块分组。
- 模块、函数、变量和 YAML 键使用 `snake_case`；类使用 `PascalCase`；常量使用 `UPPER_SNAKE_CASE`。
- 公共配置边界优先使用 `Mapping[str, Any]` / `Dict[str, Any]`，进入业务逻辑后立即规范化 ID、路径和数值类型。
- CLI、Web 响应和运维错误使用简体中文；协议 ID、类别名、配置键和文件名保持稳定英文标识。
- 可调运行参数进入 YAML 或环境变量，模型职责、类别 owner 和轮次进入对应注册表。
- Shell 脚本使用 Bash、`set -euo pipefail`、显式参数校验和非交互式执行。
- 新增测试沿用 `tests/test_<能力>.py` 与 `test_<预期行为>()` 命名。

## 分支与提交约定

默认分支为 `main`。仓库没有强制分支命名规则；需要通过 PR 协作时使用能表达范围的短名称。提交历史采用简短英文主题，祈使句和 `fix:` 等常规前缀均可，仓库未配置自动提交格式校验。

每个提交围绕一个可独立审阅的目标组织。涉及模型或发布包的提交在正文记录平台、输入资产、配置、验证命令和结果；源码提交不包含数据集、凭据、板端环境文件或本地运行目录。

## PR 流程

仓库未提供 PR 模板。提交 PR 时：

- 从最新 `main` 开始，并将改动限制在一个明确目标内。
- 在说明中列出影响的 x86/CUDA、两轮增量、Scene 系统校准或 Ascend v2 边界。
- 运行与改动范围对应的配置、语法、测试或 release 验证，并粘贴实际命令与结果。
- 模型变更同时提交注册表、manifest、冻结证据和正式包中受影响的文件。
- 确认 diff 不包含数据集、密码、SSH 配置、临时报告或未登记模型资产。
