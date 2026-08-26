# 现场 4+2+n 一键增量学习

本功能用于现场临时收到一个或多个真正新类别时，将当前六类 production 扩展为 `4+2+n`。统一入口为：

```bash
agile-agent incremental onsite --bundle /path/to/onsite_increment.zip
```

命令连续执行数据审计、全局类别注册、train/dev/lock 封存、新专家训练、dev 阈值校准、累计 lock 验收、候选 FPS 验收、候选部署和 production 晋级。只有通过全部硬门禁的候选才原子切换为 production，父代际贯穿全流程并可自动恢复。

阶段状态会实时写到标准错误，最终机器可读 JSON 单独写到标准输出。训练超过 30 秒后每 30 秒输出一次耗时与任务状态，因此现场既能展示过程，也可以安全重定向最终验收结果。

## 现场路径选择

| 现场输入 | 学习形式 | 主入口 | 已实现验收 |
| --- | --- | --- | --- |
| 赛题已有两类，演示 `4→4+1→4+2` | 310B `npu:0` 训练轻量置信度 Adapter | [`run_ascend310b_incremental_demo.sh`](../scripts/run_ascend310b_incremental_demo.sh) | 真实 NPU、OM、隔离部署、精度与完整链路 30 FPS |
| production 之后的真正新类别 `4+2+n` | CUDA 训练具备新定位能力的检测专家 | `agile-agent incremental onsite` | 数据审计、动态类别、累计 lock、FPS、原子晋级与回滚契约 |
| 已学习类别追加样本 | 目标增量更新 | 增量工作台批次生命周期 | 批次审计、冻结父代和候选代际 |

`4+2+n` 入口实现以下固定契约：

- 每包登记一个或多个 production 尚未拥有的类别，并从当前类别注册表继续分配全局 ID；
- 新专家在装有 PyTorch、Ultralytics 和 NVIDIA GPU 的 CUDA 节点训练，设备预检固定为 CUDA；
- Base、当前二类专家和历史专家保持冻结，训练阶段只读取本轮 Increment train/dev；
- Scene-SensorNet、场景门控和累计评分分别记为系统校准与联合评估；
- x86 运行时并存当前专家和现场新专家，默认每图最多运行 16 个专家；一包多类由同一个本轮专家负责；
- Ascend310B 部署编排完成新专家 ONNX/OM、隔离候选、累计精度、30 FPS 和原子晋级，并在预检中核对每个阶段及板端产物身份。

## 数据包格式

推荐将一次现场轮次打成一个 ZIP：

```text
onsite_increment.zip
├── classes.yaml
├── images/
│   ├── train/
│   ├── val/       # 可省略，由工作台自动拆分
│   └── lock/      # 可省略，由工作台自动封存
└── labels/
    ├── train/
    ├── val/
    └── lock/
```

`classes.yaml` 使用本轮局部类别 ID：

```yaml
names:
  0: new_vehicle_a
  1: new_vehicle_b
```

标签为标准五列 YOLO 格式：

```text
class_id x_center y_center width height
```

也可使用 `data.yaml` / `dataset.yaml` 的 `names`。若数据包没有类别名称，命令行必须显式提供：

```bash
agile-agent incremental onsite \
  --bundle /path/to/onsite_increment.zip \
  --class-names new_vehicle_a,new_vehicle_b
```

同时兼容赛题当前数据集的平铺格式：图像与同 stem 的 `.txt` 标签直接位于同一目录，根目录使用每行一个类别名称的 `classes.txt`。`data.yaml` / `classes.txt` 可以列出截至本轮的完整类别表；预检只读取标签中实际出现的类别 ID，不会把未出现在本轮标签里的旧类误算为新增类。

系统根据当前 production 和已保留的现场批次自动分配全局类别 ID，不要求赛题方把标签改成 `6/7/...`。例如当前类别为 `0–5`，本轮局部类 `0/1` 会登记为全局类 `6/7`。

## 演示前预检

先生成只读执行计划，核对完成后再进入正式运行：

```bash
agile-agent incremental onsite \
  --bundle /path/to/onsite_increment.zip \
  --plan-only
```

输出会明确列出：

- 当前 production、已有类别数和专家数；
- 本轮声明的新类别和预计全局 ID；
- 实际训练 Python、PyTorch/Ultralytics、CUDA 卡数和目标 GPU；
- 最终类别数与专家预算；
- 部署目标及 Ascend 编排是否完整；
- 全部阶段和任何阻断原因。

只有 `ready: true` 才开始正式一键运行。

## x86/CUDA 一键运行

在 4090/CUDA 节点激活工程环境后执行：

```bash
agile-agent incremental onsite \
  --bundle /path/to/onsite_increment.zip \
  --name onsite-round-01 \
  --target x86
```

默认会自动部署。若只想训练、校准和完成累计 lock 验收，不切换 production：

```bash
agile-agent incremental onsite \
  --bundle /path/to/onsite_increment.zip \
  --target x86 \
  --no-deploy
```

如果 `127.0.0.1:<server_port>` 上已经运行当前工程的正式 Web/CLI 服务，命令会通过本机专用控制通道在服务进程内 shadow 加载候选，并由 `AtomicEngineProvider` 原子替换内存中的正式引擎；运行中请求继续使用父代，新请求切到子代。预检同时核对服务是否支持运行时代际控制；空闲状态下则原子更新 production 注册表，下一次 Web/CLI 进程启动时加载新代际。

候选晋级前必须同时满足：

| 门禁 | 默认要求 |
| --- | --- |
| Base mAP50 | `>= 0.80` |
| 本轮 New-mAP50 | `>= 0.60` |
| 截至本轮全部旧类 KRR | `>= 0.95` |
| 候选完整图像推理 FPS | `>= 30` |
| 数据范围 | 旧图、旧标签和旧缓存交集均为 0 |

Full-mAP50、逐类 AP50、precision、误激活率和混淆诊断会一并记录，但仍沿用当前赛题口径，不替代三项精度硬门禁。

## Ascend310B 一键部署

真实新类的推荐链路是“CUDA 训练、310B 编译与验收”。同一个现场命令在 CUDA 节点运行，并增加：

```bash
agile-agent incremental onsite \
  --bundle /path/to/onsite_increment.zip \
  --name onsite-round-01 \
  --target ascend310b \
  --deployment-spec /secure/site/ascend310b-onsite.yaml
```

部署编排不得包含密码；SSH 密钥、目标主机和现场目录通过环境变量提供。编排 schema：

```yaml
schema_version: 1
target: ascend310b

stages:
  - id: export
    command: [python, /secure/site/export_candidate.py, "{candidate_weight}", "{run_root}"]

  - id: candidate_deploy
    command: [python, /secure/site/deploy_candidate.py, "{run_root}"]

  - id: accuracy_gate
    command: [python, /secure/site/score_candidate.py, "{run_root}"]
    report: "{run_root}/ascend/score.json"
    require:
      score_passed: true

  - id: fps_gate
    command: [python, /secure/site/benchmark_candidate.py, "{run_root}"]
    report: "{run_root}/ascend/benchmark.json"
    require:
      competition.batch_fps_passed: true

  - id: promote
    command: [python, /secure/site/promote_candidate.py, "{run_root}"]

rollback:
  command: [python, /secure/site/rollback_candidate.py, "{parent_generation_id}"]
```

命令参数按列表直接执行，不经过 shell。支持的内置变量包括：

| 变量 | 含义 |
| --- | --- |
| `{repo_root}` | CUDA 节点工程根目录 |
| `{run_root}` / `{run_id}` | 本次不可覆盖运行目录和编号 |
| `{candidate_weight}` | 本轮通过 dev 选择的 `best.pt` |
| `{generation_registry}` | 含候选代际的注册表 |
| `{candidate_generation_id}` | 本轮候选代际 |
| `{parent_generation_id}` | 失败时回滚目标 |
| `{lock_root}` | 本轮封存 lock 根目录 |
| `{new_class_ids}` | 逗号分隔的全局新类 ID |
| `{target_fps}` | 当前性能硬门禁 |

`accuracy_gate` 的 JSON 必须包含 `score_passed: true`；`fps_gate` 必须包含 `competition.batch_fps_passed: true` 且 `competition.batch_fps >= target_fps`。流程固定按以下顺序执行：

```text
export
  -> candidate_deploy
  -> accuracy_gate
  -> fps_gate
  -> promote
```

只有隔离候选同时通过累计精度和 30 FPS 才执行板端原子晋级。`candidate_deploy` 之后任一步失败都会调用 `rollback.command`；CUDA 侧 production 在板端门禁全部通过前也不会切换。

板端实现可以选择一个合并后的 `2+n` 增量专家 OM，或扩展为多专家 OM 运行时，但必须满足两个条件：旧 `4/5` 输出保持冻结，且新类加入后的完整链路仍通过真实 30 FPS 门禁。不能为了演示把第 7 类权重覆盖到现有二类专家并丢失旧类。

## 状态、产物和恢复

每次运行使用独立目录：

```text
runs/onsite_incremental/<run_id>/
├── state.json
├── round_contract.yaml
├── candidate-fps.json
└── deployment/
    └── <stage>.log
```

`round_contract.yaml` 固化父代际、本轮新类、局部到全局映射、零旧样本和冻结规则。`state.json` 记录每一步、候选指标、FPS、晋级或回滚结果。

查询状态：

```bash
agile-agent incremental onsite-status --run-id onsite-YYYYMMDD-HHMMSS-xxxxxx
```

终态含义：

- `PROMOTED`：累计精度、FPS、部署和 lineage 收尾全部完成；
- `CANDIDATE_ACCEPTED`：使用 `--no-deploy`，候选通过但未切换；
- `REJECTED`：门禁未通过，production 未改变；
- `CANCELLED`：用户中断后已请求停止训练，production 未改变或已恢复；
- `FAILED`：训练、导出或部署异常，production 保持或已成功恢复；
- `ROLLBACK_FAILED`：自动恢复也失败，必须立即按状态文件中的父代际人工处置。

## 现场演示顺序

1. 演示前用一个格式相同的小包完整演练 `--plan-only` 和正式命令。
2. 现场包尽量显式提供 train/val/lock；样本很少时至少保证每个新类在三部分都有样本。
3. 保持 Base、Scene-SensorNet、旧类阈值和旧类场景先验冻结。
4. 先展示 `round_contract.yaml` 的零旧样本与冻结证据，再展示 loss、New-mAP50、KRR、Full-mAP50 和 FPS。
5. 由 30 FPS 门禁决定候选晋级；未达线的运行保留完整训练与候选验收证据，并继续展示当前 4+2 production。
