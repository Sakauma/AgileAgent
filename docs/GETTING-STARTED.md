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

## 7. Ascend 310B 服务

板端使用 `configs/agent_pipeline_ascend310b.yaml` 和命名环境 `agileagent`：

```bash
./scripts/start_agent_ascend310b.sh
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

服务启动时完成 OM 哈希校验、三模型加载、预热和健康状态登记。

## 8. 复现 Ascend 满分候选

满分方法开发使用当前 WSL 仓库已有 `.venv`，不要为了运行这些入口重新安装依赖或下载 CPU 版 PyTorch：

```bash
.venv/bin/python tools/107_train_shared_dual_head.py --help
.venv/bin/python tools/108_export_ascend_dual_head.py --help
.venv/bin/python tools/109_materialize_ascend_full_score_candidate.py --help
.venv/bin/python tools/110_select_ascend_full_score_candidate.py --help
```

开始新数据集前先阅读 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)，并确认以下边界：

1. 正式 `8501` 继续运行三 OM 回滚链路；候选只能使用 `8502`。
2. Base backbone、neck/FPN、old head、BN 统计和 EMA 必须冻结；训练输入只来自新增类数据。
3. 先训练并记录 best/last，再选择一个 checkpoint 导出双输出 ONNX；当前历史满分参考实际使用 `last.pt`。
4. OM 只在板端 CANN `7.0.RC1` 下以 `mixed_float16` 构建，不升级 CANN、不用 INT8、不降分辨率。
5. 每个阈值候选依次执行无标签预测冻结、Base/New/KRR 评分、30 次预热和三轮 20 图 batch。
6. 只有四项比赛指标同时满分才可宣称满分候选；其他检测和时延指标只作诊断。

候选配置由方法 YAML、基础 Ascend 配置、dual/context OM 和 build manifest 生成，不手工复制板端绝对路径到 `full_score_method.yaml`。

## 下一步

- 日常开发和提交规范见 [`DEVELOPMENT.md`](DEVELOPMENT.md)。
- 测试矩阵与板端验收边界见 [`TESTING.md`](TESTING.md)。
- 正式/候选部署职责见 [`ascend-310b-deployment.md`](ascend-310b-deployment.md)。
