<!-- generated-by: gsd-doc-writer -->
# 测试与验收

AgileAgent 将验证分为三层：默认静态/CPU 回归、x86/NVIDIA GPU 模型与训练验收、Ascend310B1 板端候选与正式 release 验收。Pytest 测试使用仿真后端和临时文件验证契约；真实 CUDA、ACL、DVPP、CANN 与性能结果由对应硬件上的专用工具验证。

## 本次验证状态

2026-08-24 已在 Ascend310B1 正式环境对当前 4+2 production 与不可变 release `20260824-4plus2-yolo26-runtime-calibration-v1` 完成最终验收：

- Ascend 定向回归：`100 passed`；
- 全量回归：`283 passed in 34.19s`；
- `scripts/verify_release.py`：PASS；
- `bash scripts/materialize_ascend310b_full_score_release.sh --verify-existing`：33 项发布资产全部 PASS；
- 三个 systemd unit 均为 active，公共 `8501` 健康检查返回 `status=ready`、`backend=ascend_acl`、`validated=true`；
- 冻结 release 的 Base-mAP50、New-mAP50、KRR 与公共 `8501` FPS 四项满分门禁全部通过。

本轮没有重新训练模型。精度、误激活和性能结论采用当前冻结 release 的候选、lock、正式提升后复验及部署后 benchmark 证据；下文命令仍是后续代码、模型或发布变更时的标准验收入口。

## 测试框架与安装

项目使用 `pytest>=8.0`，`pyproject.toml` 将收集根设为 `tests`。测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。仓库没有全局 `conftest.py`，各测试文件直接构造本地 fixture。

在 WSL/Linux 仓库根目录使用已配置的 Python 3.10–3.12 环境：

```bash
./scripts/bootstrap_x86.sh
python -m pip install -c constraints-agent.txt -e ".[dev,workbench,inference]"
```

`dev` 提供 Pytest 和 TestClient 所需的 `httpx2`，`workbench` 提供 Starlette/Uvicorn/multipart，`inference` 提供 NumPy、OpenCV 和 Ultralytics。`bootstrap_x86.sh` 还会校验 CUDA 版 PyTorch/torchvision 组合。

## 默认静态与 CPU 回归

完整 Pytest 命令是：

```bash
python -m pytest -q
```

这一层不调用真实 GPU 或 Ascend ACL 设备。Ascend async 测试使用 fake runtime/stream/memory，Web 测试使用 fake engine，TensorRT 分支通过 monkeypatch 验证合约。发布权重存在时，部分节点会核对实际文件和登记身份，但不加载模型到加速器。

静态发布与固定划分验证：

```bash
python scripts/verify_release.py
python tools/03_split_r2_4plus2.py --verify-only
bash -n scripts/*.sh
git diff --check
```

`scripts/verify_release.py` 加载 schema 3 主配置，核对必需资产、`models/manifest.json`、`models/generations.json`、三个功能模型、当前 production owner/阈值、增量阶段契约与两轮注册表。`--verify-only` 对受控 4+2 train/dev/lock 列表、计数、互斥性和清单做只读核对。

## 当前 Pytest 覆盖范围

| 范围 | 当前测试文件 | 主要契约 |
| --- | --- | --- |
| 配置、CLI 与运行成熟度 | `test_configuration_runtime.py`、`test_runtime_maturity.py`、`test_agent_workbench.py` | schema/override/脱敏、回环服务、黑板、production 代际、TensorRT 档和发布静态验收 |
| 4+2 固定划分与评分 | `test_strict_4plus2_splits.py`、`test_strict_incremental.py` | Base/Increment 完整互斥划分、两个不同新类轮次、AP50/KRR 重算、固定 owner 融合和 production profile |
| 顺序增量生命周期 | `test_incremental_workbench.py`、`test_incremental_lifecycle_v2.py`、`test_incremental_guardian.py` | 数据包审计、自动 lock、父子代际、累积 lock 链、候选重检、提升/回滚与 Base/New/KRR 门禁 |
| 场景校准与误激活诊断 | `test_incremental_rejection.py`、`test_unlabeled_inference.py`、`test_functional_models.py` | 场景先验来源、软阈值、新类原型/冲突处理、无标签全图推理、Scene-SensorNet 验收和三功能模型注册 |
| Web 推理和 UI | `test_web_inference.py`、`test_web_ui_flow.py` | 六类映射、内容双证据门控、并行/有序模型执行、批量 ZIP、health、单图/批量 API 和前端交互 |
| 提交输出安全 | `test_submission_safety.py` | 输出根范围、重复 stem、GPU 设备强制和推理结果计数 |
| Ascend ACL 与输入/输出 | `test_ascend_acl.py`、`test_ascend_acl_async.py` | `640×512` PNG、AIPP NHWC、`608×736` 检测张量、YOLO26 E2E `[1,300,6]`、async stream 序列与失败恢复 |
| Ascend 对齐与性能契约 | `test_ascend_alignment.py`、`test_ascend_benchmark.py` | 逐类 IoU 对齐、业务签名硬门禁、DVPP PNG 格式和 20 图 batch FPS 计算 |
| Ascend310B v2 满分工作流 | `test_ascend_full_score_workflow.py`、`test_ascend_release.py` | `independent_yolo26_e2e_v1`、三 OM 身份、候选 `8502`、三轮 FPS 门禁、候选授权和正式提升 |
| Ascend310B v2 正式包 | `test_ascend_packaged_release.py` | 31 项完整性清单、release-local 配置/清单、三个 OM、零训练物化、启停默认发布 ID 和提升后证据 |

## 聚焦运行

单文件：

```bash
python -m pytest -q tests/test_strict_4plus2_splits.py
```

单节点：

```bash
python -m pytest -q \
  tests/test_strict_incremental.py::test_ap50_and_krr_are_recomputed_from_frozen_prediction_rows
```

配置、4+2 划分和增量评分：

```bash
python -m pytest -q \
  tests/test_configuration_runtime.py \
  tests/test_strict_4plus2_splits.py \
  tests/test_strict_incremental.py
```

顺序增量、场景校准和无标签运行：

```bash
python -m pytest -q \
  tests/test_incremental_workbench.py \
  tests/test_incremental_lifecycle_v2.py \
  tests/test_incremental_guardian.py \
  tests/test_incremental_rejection.py \
  tests/test_unlabeled_inference.py \
  tests/test_functional_models.py
```

Web/CLI：

```bash
python -m pytest -q \
  tests/test_agent_workbench.py \
  tests/test_runtime_maturity.py \
  tests/test_web_inference.py \
  tests/test_web_ui_flow.py \
  tests/test_submission_safety.py
```

Ascend310B v2 的 CPU/契约层：

```bash
python -m pytest -q \
  tests/test_ascend_acl.py \
  tests/test_ascend_acl_async.py \
  tests/test_ascend_alignment.py \
  tests/test_ascend_benchmark.py \
  tests/test_ascend_full_score_workflow.py \
  tests/test_ascend_release.py \
  tests/test_ascend_packaged_release.py
```

## x86/NVIDIA GPU 模型验收

Pytest 通过后，使用 CUDA 加载当前 Base、Incremental 和 Scene-SensorNet 资产：

```bash
python scripts/smoke_models.py --load-only
```

拥有完整 4+2 数据根时，增加真实 mixed-lock 场景复核和 Agent 编排冒烟：

```bash
DATA_ROOT=/path/to/tiaozhanbei_4plus2_dataset_20260821
python scripts/smoke_models.py --data-root "$DATA_ROOT"
```

GPU 冒烟必须确认：

- Base/Incremental YOLO26s 权重和 Scene-SensorNet 均加载到 `cuda:<device>`；
- Base 批量输出数与 `configs/local_infer_gpu.yaml` 中的 `batch` 一致；
- 当 mixed lock 图像数与参考证据可比时，Scene-SensorNet 的三项指标与当前证据一致；
- Agent 输出包含当前 generation ID、已执行/跳过协议和融合摘要。

## GPU 训练与顺序增量验收

训练入口使用 `tools/04_train_base_4plus2.py`、`tools/06_train_incremental_4plus2.py` 和 `tools/60_train_scene_sensor.py`。Base 和 Incremental 队列要求每个进程通过 `CUDA_VISIBLE_DEVICES` 只暴露一张物理 GPU，进程内使用 `--device 0`。

训练前只生成/核对数据视图：

```bash
python tools/04_train_base_4plus2.py --help
python tools/06_train_incremental_4plus2.py --help
python tools/60_train_scene_sensor.py \
  --config configs/scene_sensor_model_4plus2.yaml \
  --data-root "$DATA_ROOT" \
  --check-only
```

完整训练和选模的验收链是：

1. `tools/04_train_base_4plus2.py` 生成多随机种子 Base 队列，`tools/05_select_base_4plus2.py` 仅使用 Base dev 复评并选择权重。
2. `tools/11_prepare_incremental_round_splits.py` 从类别注册表生成逐轮 split；`tools/06_train_incremental_4plus2.py` 每轮仅读取当轮 Increment train/dev，`tools/07_select_incremental_4plus2.py` 复评当轮专家。
3. `tools/08_evaluate_4plus2.py` 在读取 lock 标签前冻结预测，输出每轮 New-mAP50、KRR、Full-mAP50 和 lineage。
4. `tools/13_register_incremental_round_candidate.py` 登记当轮候选，`tools/12_summarize_incremental_rounds.py` 核对两个不同新类的父子代际和累积指标。
5. `tools/60_train_scene_sensor.py` 与 `tools/61_select_scene_sensor_4plus2.py` 训练/选择封闭集 Scene-SensorNet；`tools/09_optimize_scene_aware_4plus2.py` 先在 dev 冻结候选，再在 lock 模式做一次复核。
6. `tools/10_promote_scene_aware_4plus2.py` 同时验证候选、dev search、lock result 和顺序 round evidence，通过后更新 production 资产。
7. `tools/14_evaluate_all_images_4plus2.py` 在不训练、不选参的前提下，用冻结 lock 预测生成一号正式结果，并在 Base/Increment train、dev、lock 全部图像上生成二号诊断结果。

全量诊断首次运行会按 `--batch` 显式分块执行两个检测器；预测缓存只能写入本地工作区。后续可使用 `--reuse-cache` 重算后处理与指标：

```bash
"$AGENT_PYTHON" tools/14_evaluate_all_images_4plus2.py \
  --data-root /path/to/tiaozhanbei_4plus2_dataset_20260821 \
  --output-dir /path/to/local_workspace/all_images_x86 \
  --device 0 \
  --batch 8
```

二号结果包含训练图像，不得用于独立测试集或泛化性能声明，也不参与模型、阈值和门控参数选择。

训练验收以各 run 的 `args.yaml`、`results.csv`、`weights/best.pt`、队列 summary、selection JSON/Markdown、冻结 predictions 和逐轮 evaluation JSON 为证据。不以“训练进程正常退出”代替 dev 复评、lock 冻结和代际链核对。

## Ascend310B v2 本地静态验证

不连接板卡时，Pytest 契约层覆盖 ACL 假运行时、DVPP/AIPP 输入、YOLO26 E2E 输出、候选与提升 schema、评分门禁和发布包自包含性。包完整性的独立检查命令是：

```bash
(
  cd models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1
  sha256sum -c SHA256SUMS
)
```

这一层只复核已提交的 31 项文件、release-local 路径、配置、清单和验证报告，不调用 ACL、ATC 或板端性能测量。

## Ascend310B v2 板端验收

板端验收在已配置 CANN `7.0.RC1` 和 `/usr/local/miniconda3/envs/agileagent` 的 Ascend310B1 上执行。首次物化与已有 release 只读复核分别使用：

```bash
./scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

物化脚本会检查预构建包，复制受控源码/资产，并执行正式 release 验证器。也可在 release 目录上直接复核：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
  /usr/local/miniconda3/envs/agileagent/bin/python \
  "$RELEASE/src/tools/95_verify_ascend_release.py" \
  --config "$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
  --require-validation
```

`tools/95_verify_ascend_release.py` 核对 release build manifest、Base/Incremental/Scene 三个 OM、`independent_yolo26_e2e_v1`、`context_mode=model`、冻结预测、精度报告和性能报告的身份链。

服务启动后检查正式主实例和公共入口：

```bash
curl -fsS http://127.0.0.1:18501/api/health
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

health 必须报告 `status=ready`、`backend=ascend_acl`、`device=ascend:0`、`validated=true`、`validation_candidate=false`、`model_layout=independent_yolo26_e2e_v1`、`context_mode=model` 和当前 4+2 production generation ID。

新候选的完整 score gate 入口是：

```bash
./scripts/run_ascend310b_score_gate.sh \
  "$CANDIDATE_CONFIG" \
  "$IMAGE_ROOT" \
  "$MIXED_SPLIT" \
  "$BASE_SPLIT" \
  "$OUTPUT_DIR"
```

该 gate 在 `8502` 启动受控候选，冻结无标签预测，计算 Base mAP50、New-mAP50、KRR，并执行 30 次预热和三轮 20 图 batch FPS。执行前后正式 `8501` 都必须 ready，输入必须是根目录内 stem 唯一的 `640×512`、8-bit 灰度/RGB/RGBA PNG。

## 编写新测试

- 在 `tests/test_<area>.py` 中添加职责单一的 `test_<behavior>` 函数。
- 使用 `tmp_path` 复制配置、注册表和证据后再做变异测试，不修改仓库内 production 文件。
- 使用 `monkeypatch` 替换子进程、TensorRT/Ultralytics 模块和引擎；Ascend 单元测试扩展现有 fake ACL runtime，不要求开发机安装 CANN。
- Web 路由使用 `starlette.testclient.TestClient` 与 fake engine；图像使用 Pillow 在内存中生成。
- 多个合法/非法契约变体使用 `pytest.mark.parametrize`，错误边界使用 `pytest.raises(..., match=...)` 核对。
- 涉及 lock 的测试先冻结候选与预测，再读取 lock 标签；顺序增量节点从 `configs/incremental_round_registry_4plus2.yaml` 读取轮次和类别。

## 覆盖率要求

仓库没有 `.coveragerc`、`pytest-cov` 依赖或 `coverageThreshold` 配置，因此当前没有自动化覆盖率百分比门禁。验收以契约测试、静态发布校验、GPU 模型冒烟和板端 score gate 为准。

| 类型 | 最低覆盖率 |
| --- | --- |
| Lines | 未配置 |
| Branches | 未配置 |
| Functions | 未配置 |
| Statements | 未配置 |

## CI 集成

仓库当前没有 `.github/workflows/` 或其他受版本控制的 CI 工作流，因此不会在 push 或 Pull Request 上自动运行测试。默认回归在 WSL/Linux 环境手动执行，GPU 和 Ascend310B 验收分别在对应硬件上记录产物和报告。
