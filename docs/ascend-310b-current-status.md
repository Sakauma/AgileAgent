# Ascend 310B 工程记录

本文记录截至 2026-08-14 已完成的 Ascend 310B 实现、部署、精度复核和性能测量。

## 板端环境

| 项目 | 已确认值 |
| --- | --- |
| 开发套件 | Atlas 200I DK A2，`aarch64` |
| 芯片 | Ascend310B1，NPU Health `OK` |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1` |
| Conda | Miniconda `23.5.0` |
| Python 环境 | `/usr/local/miniconda3/envs/agileagent`，Python `3.9.2` |
| 正式 release | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` |
| 服务 | `127.0.0.1:8501`，health `ready` |
| 驻留快照 | NPU 内存 `9479 / 11577 MB`，温度 `61°C` |

## 已实现链路

- `fair_agent/backends/ascend_acl.py` 使用 PyACL/AscendCL 加载和执行三个 OM；
- `fair_agent/modules/web_inference.py` 编排 Base、Incremental 和 Scene 三模型；
- `fair_agent/web/app.py` 提供 health、单图检测和批量检测 API；
- `configs/agent_pipeline_ascend310b.yaml` 登记设备、CANN、执行模式、OM 路径和 SHA256；
- `scripts/start_agent_ascend310b.sh` 使用命名环境 `agileagent` 启动 Uvicorn；
- 模型输出经过全局类别映射、逐类阈值、场景软证据、冲突仲裁和 class-aware NMS。

## 正式模型

| 模型 | 输入 | 产物 |
| --- | --- | --- |
| 三类基础检测器 | `1×3×736×896` | `base_detector.om` |
| 增量检测器 | `1×3×512×640` | `incremental_detector.om` |
| Scene-SensorNet | `1×3×160×160` | `scene_sensor_net.om` |

三个 OM 使用 `mixed_float16` 编译，配置记录对应 SHA256。每张图像执行三个模型并生成统一检测结果。

## 正式 release 精度

正式 release 已在 89 张固定混合测试集上完成无标签推理、预测冻结和评分：

| 指标 | 结果 |
| --- | ---: |
| Base mAP50 | `0.819407` |
| New-mAP50 | `0.728761` |
| KRR | `1.000000` |
| 新类 precision | `0.933333` |
| 误激活率 | `0.014286` |

基础目标检测、New-mAP 和 KRR 三项合计取得 `50/50` 精度分档结果。

## 性能测量

| 测量对象 | 样本量 | 已记录结果 |
| --- | ---: | --- |
| 正式 release 完整 89 图 | 89 | 引擎均值 `57.849 ms`、墙钟均值 `71.491 ms`、引擎 `17.29 FPS`、墙钟 `13.99 FPS` |
| 已解码 Agent 核心 | 200 | 均值 `32.148 ms`、P95 `33.193 ms`、`31.11 FPS` |
| AIPP staging multipart PNG API | 1,068 | 均值 `51.203 ms`、P95 `63.9 ms`、`19.53 FPS` |
| DVPP 编码输入测量 | 240 | 均值 `37.124 ms`、P95 `38.154 ms`、`26.94 FPS` |

这些记录分别覆盖完整 89 图运行、已解码核心、真实 multipart PNG API 和编码输入路径。

## 环境迁移记录

板端 Python 环境已迁移到命名环境 `agileagent`。迁移前后使用固定 PNG 执行响应语义对照，检测数量、类别、框和置信度保持一致；切换后 health 返回 `ready`。

命名环境确认项：

- Python `3.9.2`；
- 181 个 Conda 记录；
- 176 个 `pip freeze --all` 条目；
- PyACL 与核心模块导入成功；
- 三个 OM 加载成功；
- 真实 PNG 推理成功。

## 自动验证

```bash
python -m pytest -q
python scripts/verify_release.py
```

当前完整回归确认 `214` 项通过，发布校验状态为 `passed`。

## 运行态检查

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
npu-smi info
```
