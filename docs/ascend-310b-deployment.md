# Ascend 310B 部署实现

AgileAgent 已在 Atlas 200I DK A2 上完成正式三模型 OM 推理，以及隔离的共享双逻辑头满分候选验证。本页区分当前正式回滚服务和比赛候选，避免把候选误写成已经发布。

## 当前正式回滚结构

```text
真实 PNG 请求
  -> FastAPI / Uvicorn
  -> 图像解码与矩形预处理
  -> Base Detector OM
  -> Incremental Detector OM
  -> Scene-SensorNet OM
  -> YOLO 解码与全局类别映射
  -> 场景软阈值与冲突仲裁
  -> class-aware NMS
  -> JSON 响应与审计事件
```

Python 编排层位于 `fair_agent/modules/web_inference.py`，Ascend 运行时位于 `fair_agent/backends/ascend_acl.py`，Web 服务位于 `fair_agent/web/app.py`。

## 满分候选结构

```text
20 图 multipart batch
  -> 有界解析与 DVPP encoded 预处理
  -> shared_backbone_dual_head_v1 OM
     -> old head / frozen_base_model
     -> new head / incremental_model
  -> fixed_neutral_v1（不执行 Scene OM）
  -> 原融合、审计与 API schema
```

候选固定使用 `8502`、`896×736` AIPP、`raw_dual_head_v1`、pageable memory 和 threaded execution。old/new 当前参考阈值为 `0.05/0.30`，但更换数据集后必须重新搜索。完整方法见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。

## 设备与运行环境

| 项目 | 当前值 |
| --- | --- |
| 设备 | Atlas 200I DK A2 |
| SoC | Ascend310B1 |
| CANN | `7.0.RC1` |
| Python | `/usr/local/miniconda3/envs/agileagent/bin/python` |
| 配置 | `configs/agent_pipeline_ascend310b.yaml` |
| 服务地址 | `127.0.0.1:8501` |

## 模型契约

| 模型 | 输入张量 | 输出 |
| --- | --- | --- |
| Base Detector | `1,3,736,896` FP32 | `1,7,13524` YOLO 原始输出 |
| Incremental Detector | `1,3,512,640` FP32 | `1,5,6720` YOLO 原始输出 |
| Scene-SensorNet | `1,3,160,160` FP32 | sensor logits 与 scene logits |

正式三个模型使用固定 `batch=1`，ATC 以 `mixed_float16` 生成 OM。候选单个 dual OM 输出 old `[1,7,13524]` 和 new `[1,5,13524]`；context OM 会加载并登记为回滚资产，但 `fixed_neutral_v1` 正常路径不执行它的前向推理。配置和 build manifest 都记录路径、SHA256、logical owner 和类别映射。

## 图像预处理

基础图像尺寸为 `640×512` PNG。

Base Detector 预处理：

1. RGB 解码；
2. 等比例缩放到 `896×717`；
3. 按 stride 32 补边为 `896×736`；
4. 转换为 `1×3×736×896` FP32；
5. 将数值归一化到 `[0,1]`。

Incremental Detector 使用 `640×512` 固定输入并转换为 `1×3×512×640` FP32。Scene-SensorNet 将图像缩放与中心裁剪为 `160×160`，随后执行标准化。

## 后处理与融合

Base Detector 的局部类别 `0/1/2` 映射到全局 `0/1/3`，Incremental Detector 的局部类别 `0` 映射到全局 `2`。后处理依次执行：

1. YOLO 输出解码；
2. 置信度筛选；
3. 坐标还原；
4. 类别所有权映射；
5. Scene-SensorNet 软阈值调整；
6. 框级冲突仲裁；
7. class-aware NMS；
8. 生成检测记录与模型轨迹。

## Release 目录

```text
/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2/
├── src/
├── om/
│   ├── base_detector.om
│   ├── incremental_detector.om
│   └── scene_sensor_net.om
├── validation/
└── agent-web.pid
```

## 启动服务

```bash
cd /home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2/src
./scripts/start_agent_ascend310b.sh
```

启动脚本加载 CANN 环境、登记配置路径、写入 PID 文件并启动 Uvicorn。

## API

健康检查：

```bash
curl -fsS http://127.0.0.1:8501/api/health
```

单图检测：

```bash
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

批量检测由 `POST /api/batch` 接收图像集合并返回逐图结果、汇总与耗时。

## 精度复核

单 OM 静态输入复核：

```bash
python tools/92_run_ascend_om.py \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --output-dir reports/ascend310b/om_output
```

89 图 Agent 预测冻结后执行评分：

```bash
python tools/94_score_ascend_agent.py \
  --predictions reports/ascend310b/predictions_frozen \
  --mixed-split splits/strict_3plus1/mixed_test.txt \
  --base-split splits/strict_3plus1/base_test.txt \
  --output reports/ascend310b/score.json
```

正式 release 结果：

| 指标 | 数值 |
| --- | ---: |
| Base mAP50 | `0.819407` |
| New-mAP50 | `0.728761` |
| KRR | `1.000000` |
| 新类 precision | `0.933333` |
| 误激活率 | `0.014286` |

## 性能记录

| 测量 | 样本量 | 已记录结果 |
| --- | ---: | --- |
| 正式 release 完整 89 图 | 89 | 墙钟均值 `71.491 ms`、`13.99 FPS` |
| 已解码 Agent 核心 | 200 | 均值 `32.148 ms`、P95 `33.193 ms`、`31.11 FPS` |
| AIPP staging 真实 PNG API | 1,068 | 均值 `51.203 ms`、P95 `63.9 ms`、`19.53 FPS` |
| DVPP 编码输入 | 240 | 均值 `37.124 ms`、P95 `38.154 ms`、`26.94 FPS` |
| 共享双头满分候选 | 两组 3×20 batch | 中位 `30.066/30.080 FPS`；Base/New/KRR 同时满分 |

满分候选尚未替换 `8501`。其单请求均值/P95/P99 和逐框差异继续留作诊断，但正式计分只使用 Base mAP50、New-mAP50、KRR 和三轮 20 图 batch 中位 FPS。

## 运行监测

```bash
npu-smi info
curl -fsS http://127.0.0.1:8501/api/health
ps -ef | grep 'uvicorn fair_agent.web.app:app'
```

请求级日志记录 `trace_id`、generation、执行模型、检测数量和分段耗时，发布验证记录模型路径、SHA256、配置与指标。
