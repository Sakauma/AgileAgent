# 配置参考

AgileAgent 使用 UTF-8 YAML 配置。主运行配置采用 schema 3，并由 `fair_agent/core/config.py` 统一加载、校验和解析。

## 配置文件

| 文件 | 用途 |
| --- | --- |
| `configs/agent_pipeline.yaml` | x86/CUDA 开发、训练和 Web 服务 |
| `configs/agent_pipeline_ascend310b.yaml` | Ascend 310B OM 推理服务 |
| `configs/functional_models.yaml` | 三个功能模型及发布资产 |
| `configs/incremental_detection_policy.yaml` | 增量数据范围与评测策略 |
| `configs/strict_class_incremental_3plus1.yaml` | 固定 3+1 训练和评分参数 |
| `configs/scene_sensor_model.yaml` | Scene-SensorNet 训练参数 |
| `configs/local_infer_gpu.yaml` | 本机 GPU 推理参数 |
| `configs/submission_infer_base_3class.yaml` | 三类基础检测提交参数 |

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
| `inference.confidence_default` | `0.50` |
| `inference.batch_size` | `32` |
| `incremental_workbench.validation_fraction` | `0.20` |
| `incremental_workbench.lock_fraction` | `0.20` |
| `incremental_workbench.split_seed` | `20260705` |
| `web.generation_channel` | `production` |

Ascend 配置将 `inference.backend` 设为 `ascend_acl`，使用 `batch_size: 1`，并登记三个正式 OM 的路径与 SHA256。

## 配置验证

```bash
python -m fair_agent.cli --config configs/agent_pipeline.yaml config validate
python -m fair_agent.cli --config configs/agent_pipeline_ascend310b.yaml config validate
python scripts/verify_release.py
```

发布校验同时核对主配置、模型资产、模型 manifest、功能模型注册表和 production 代际。
