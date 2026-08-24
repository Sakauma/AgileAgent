# CLI 主界面使用指南

AgileAgent CLI 是无显示器设备、SSH 会话和普通本地终端的主要操作界面。它面向识别任务本身，而不是后台参数设置：模型、设备、置信度、Scene-SensorNet、增量专家、场景门控和后处理策略均由当前 production 配置自动确定。

## 进入主界面

在仓库根目录运行：

```bash
./scripts/start_agent.sh --cli
```

完成 `bootstrap_x86.sh` 或板端环境登记后，也可以直接运行：

```bash
agile-agent
```

不带子命令的 `agile-agent` 会自动打开交互主界面。310B 上启动脚本会复用 `/usr/local/miniconda3/envs/agileagent` 和 CANN 环境，不安装或替换 CANN、PyTorch、CUDA 等关键依赖。

主菜单如下：

```text
[1] 单图识别    [2] 批量识别    [3] 最近结果
[4] 运行状态    [5] 模型信息    [0] 返回首页
[r] 刷新状态    [h] 使用帮助    [q] 退出
```

- `1`：粘贴一张图像路径并执行正式识别。
- `2`：输入图像目录，可选择是否递归扫描子目录。
- `3`：查看最近八次识别的时间、图像数、目标数和保存目录。
- `4`：查看架构、设备、推理后端、配置来源与运行门禁。
- `5`：查看当前 production 代际和三个功能模型，只读且不提供人工切换。
- `r`：重新采集运行状态；不会重新训练或修改模型。

路径包含空格时可以直接粘贴，也可以使用单引号或双引号包裹。识别结束后按 Enter 返回首页。

## 直接执行识别命令

单张图像：

```bash
agile-agent detect --source /path/to/image.png
```

当前目录中的所有受支持图像：

```bash
agile-agent detect --source /path/to/images
```

包含子目录：

```bash
agile-agent detect --source /path/to/images --recursive
```

指定本次结果目录：

```bash
agile-agent detect \
  --source /path/to/image.png \
  --output /home/HwHiAiUser/results/case-001
```

显式结果目录必须不存在或为空，CLI 不会覆盖既有识别结果。不指定 `--output` 时，结果自动写入：

```text
runs/cli_detections/detect_<输入名>_<时间>/
```

默认终端输出是适合 SSH 阅读的中文表格。脚本需要机器可读输出时使用：

```bash
agile-agent detect --source /path/to/image.png --format json
```

`--format json` 只改变标准输出格式，识别产物仍会正常保存。CLI 不提供 `--confidence` 或模型 profile 选项，避免终端结果和正式 Web/评测链路产生口径分裂。

## 保存的结果

每次识别创建独立运行目录：

```text
detect_<输入名>_<时间>/
├── annotated/
│   └── 001_<图像名>.png
├── predictions/
│   └── 001_<图像名>.txt
├── detections.csv
├── results.json
└── summary.txt
```

| 文件 | 内容 |
| --- | --- |
| `annotated/*.png` | 带类别、置信度和彩色检测框的标注图 |
| `predictions/*.txt` | 每行 `class_id center_x center_y width height confidence`，坐标按图像宽高归一化 |
| `detections.csv` | 图像名、类别、置信度、像素坐标、Base/Incremental 来源和 owner 模型 |
| `results.json` | 场景概率、六类检测、执行轨迹、耗时、门控与融合信息的完整机器可读结果 |
| `summary.txt` | 与终端一致的易读识别摘要 |

保存的是正式后处理后的最终结果，不是模型尚未经过阈值、场景门控和跨类别抑制的原始候选框。

## 310B 上的执行方式

CLI 会自动检查当前配置的本机服务端口：

1. 服务为 `ready`、后端与当前配置一致且已经验收时，CLI 直接复用正式 `18501/8501` 服务，避免再次占用 Ascend 设备内存。
2. 没有匹配的本机服务时，CLI 才在当前进程加载同一 production 引擎。

终端摘要中的“执行”字段会明确显示“复用本机正式服务”或“本进程直接推理”。该选择完全自动，不改变模型、阈值或检测结果口径。

## 将结果复制到 SSH 客户端

识别结果首先保存在运行 CLI 的设备上。退出板端 SSH 会话后，在自己的电脑终端执行：

```bash
scp -r \
  HwHiAiUser@<310B-IP>:/home/HwHiAiUser/agileagent/repo/runs/cli_detections/detect_xxx \
  ./
```

也可以只复制标注图或 JSON：

```bash
scp -r HwHiAiUser@<310B-IP>:/path/to/result/annotated ./annotated
scp HwHiAiUser@<310B-IP>:/path/to/result/results.json ./results.json
```

CLI 菜单 `3` 会列出可直接用于 `scp` 的完整板端路径。

## 状态与自动化命令

人类可读状态：

```bash
agile-agent status
```

机器可读状态：

```bash
agile-agent status --format json --refresh
```

显示命令帮助：

```bash
agile-agent --help
agile-agent detect --help
```

## 常见问题

### 提示结果目录非空

CLI 默认拒绝覆盖。删除 `--output` 让系统自动创建新目录，或者指定另一个不存在/为空的目录。

### 首次识别显示“正在加载本地推理引擎”

说明当前配置的本机 Web/Ascend 服务不可用或身份不匹配。CLI 会安全回退到同一 production 的直接推理，因此首次加载会更慢。在 310B 正式部署中可先检查：

```bash
curl -fsS http://127.0.0.1:18501/api/health
curl -fsS http://127.0.0.1:8501/api/health
```

### SSH 终端表格错位

确保客户端使用 UTF-8，并使用支持中英文等宽显示的终端字体。建议终端宽度至少为 80 列。

### 长批次担心 SSH 中断

在 `tmux` 会话中运行 CLI。识别完成后所有产物都保留在运行目录中，可以从菜单 `3` 重新找到。
