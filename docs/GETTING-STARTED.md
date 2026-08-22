# 快速上手

AgileAgent 的本机开发环境运行于 WSL/Linux，训练与模型冒烟使用 NVIDIA GPU，Web 工作台与 CLI 共享同一套配置、模型代际和审计状态。

## 1. 准备仓库

```bash
cd "/mnt/d/Ajax Mao/研二/近期工作/研二下/tiaozhanbei/AgileAgent"
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

引导脚本完成以下工作：

- 选择 Python 3.10–3.12 解释器；
- 准备 CUDA PyTorch、TorchVision 和项目依赖；
- 以 editable 模式注册当前仓库；
- 将解释器路径写入 `.agent-python`；
- 执行发布校验和模型加载冒烟。

已验证参考组合：

| 组件 | 版本 |
| --- | --- |
| Python | `3.10.19` |
| PyTorch | `2.5.1+cu124` |
| TorchVision | `0.20.1+cu124` |
| Ultralytics | `8.4.92` |

## 2. 启动 Web 工作台

```bash
./scripts/start_agent.sh
```

访问：

```text
http://127.0.0.1:8501
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8501/api/health
```

## 3. 使用 CLI

```bash
./scripts/start_agent.sh --cli
```

常用命令：

```bash
agile-agent doctor
agile-agent status --format json --refresh
agile-agent detect --source path/to/image.png --confidence 0.50
agile-agent logs --limit 100
```

## 4. 准备基础数据

基础数据放置于：

```text
datasets_r1_base_train/
```

数据体检与固定划分生成：

```bash
.venv/bin/python tools/00_check_dataset.py
.venv/bin/python tools/02_split_dataset.py --increment-class warship
```

数据体检输出写入 `reports/`，固定清单写入 `splits/`。

## 5. 运行增量学习

增量 ZIP 使用图像与同 stem 五列 YOLO 标签。完整生命周期命令：

```bash
agile-agent incremental audit --batch /path/to/new_batch.zip
agile-agent incremental run --batch BATCH_ID
agile-agent incremental status --run-id TRAIN_JOB_ID
```

Web 工作台的“注入并训练”调用同一生命周期，并实时展示任务状态、阶段指标和审计事件。

## 6. 运行验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_release.py
.venv/bin/python scripts/smoke_models.py
```

测试数量随功能增长，不在本页固定计数；提交说明记录当次实际结果。发布资产校验状态为 `passed`。

## 7. Ascend 310B 零训练部署

以下步骤假定板端已经配置好 CANN `7.0.RC1` 和命名环境 `/usr/local/miniconda3/envs/agileagent`。仓库已经包含通过满分门禁的 OM、源 checkpoint、ONNX、构建证据和原始验收报告；部署当前方案不需要训练、ATC 或联网安装依赖。

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"

curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

物化脚本先用包内 `SHA256SUMS` 校验全部资产，再生成固定 release 并执行 `tools/95_verify_ascend_release.py --require-validation`。目标目录已存在时默认拒绝覆盖；可用 `--verify-existing` 做只读复核。脚本本身不启动或停止服务。

新板可以让满分 release 直接监听 `8501`。已有旧 listener 的板使用双实例拓扑：公共 `8501` 精确路由到 4+2 主实例 `18501`，旧 listener 仍物理监听 `8501`，删除路由即可即时回滚；`8502` 只用于下一轮候选。健康响应应包含 `validated: true`、`model_layout: independent_yolo26_e2e_v1` 和 `context_mode: model`。

无竞赛数据集时仍可完成模型包哈希、release 校验、服务启动和历史报告核对。重新测量 batch FPS 需要 20 张符合契约的 PNG；重新计算 Base/New/KRR 需要合法取得的 89 图和标签。包内已带同一候选的冻结预测，因此有标签后无需重新推理或训练即可重新评分。

## 8. 为新数据集重新训练与选优

本节只适用于比赛更换数据集、需要重新训练和搜索阈值的情况，不是部署当前满分模型的前置步骤。满分方法开发使用当前 WSL 仓库已有 `.venv`，不要为了运行这些入口重新安装依赖或下载 CPU 版 PyTorch：

```bash
.venv/bin/python tools/04_train_base_4plus2.py --help
.venv/bin/python tools/06_train_incremental_4plus2.py --help
.venv/bin/python tools/112_materialize_ascend_yolo26_candidate.py --help
.venv/bin/python tools/110_select_ascend_full_score_candidate.py --help
.venv/bin/python tools/111_promote_ascend_full_score_release.py --help
```

开始新数据集前先阅读 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)，并确认以下边界：

1. 公共 `8501` 已路由到正式 4+2 主线 `18501`；旧 listener 继续作为即时回滚，候选只能使用 `8502`。
2. Base 阶段只用 Base train/dev；增量阶段只用当轮 Increment train/dev，并保持 Base 与历史专家冻结。
3. Base 和 Specialist 分别导出 `608×736`、`[1,300,6]` 的 YOLO26 E2E ONNX。
4. OM 只在板端 CANN `7.0.RC1` 下以 `mixed_float16` 构建，不升级 CANN，也不替换关键运行依赖。
5. 候选必须使用真实 Scene-SensorNet 和双证据执行门控，线上不得读取标签或文件名。
6. 每个阈值候选依次执行无标签预测冻结、Base/New/KRR 评分、30 次预热和三轮 20 图 batch。
7. 只有四项比赛指标同时满分才可宣称满分候选；precision、误激活率和单请求时延只作诊断。
8. 胜出候选由 `tools/111` 物化并通过正式 release 校验后，才允许使用 systemd 和精确路由提升；不要手工把 `validated` 改为 `true`。

候选配置由方法 YAML、基础 Ascend 配置、Base/Specialist/Scene OM、候选代际和 build manifest 生成，不手工复制板端绝对路径到 `full_score_method.yaml`。

## 下一步

- 日常开发和提交规范见 [`DEVELOPMENT.md`](DEVELOPMENT.md)。
- 测试矩阵与板端验收边界见 [`TESTING.md`](TESTING.md)。
- 正式/候选部署职责见 [`ascend-310b-deployment.md`](ascend-310b-deployment.md)。
