<!-- generated-by: gsd-doc-writer -->
# Web API

AgileAgent 的 Web 服务由 `fair_agent/web/app.py` 提供。x86/CUDA 与 Ascend310B v2 使用同一组 HTTP 路由、六类全局 ID 和类别 owner；运行后端与模型布局通过健康响应区分。

默认地址为 `http://127.0.0.1:8501`。默认监听和运行时代际控制均采用 loopback 边界；需要跨主机访问时，由受控反向代理提供 TLS、身份认证和访问策略。所有响应包含 `X-Request-ID`，请求也可以主动传入该头以串联审计日志。

## 核心路由

| 方法与路径 | 输入 | 成功响应 |
| --- | --- | --- |
| `GET /api/health` | 无 | 后端、设备、队列、代际和活动类别 |
| `GET /api/config/public` | 无 | 自动识别的运行平台、置信度范围、UI、上传限制和标签字典 |
| `GET /api/capabilities` | 无 | 当前运行平台、代际、模型格式、增量协议和冻结指标 |
| `POST /api/detect` | multipart：`file`，可选 `confidence` | 单图场景、检测、执行轨迹和耗时 |
| `POST /api/batch` | multipart：一个或多个 `files`，可选 `confidence` | 批次摘要、逐图结果、预览和下载地址 |
| `GET /api/batch/{batch_id}/preview/{index}` | 路径参数 | 标注 PNG |
| `GET /api/batch/{batch_id}/download` | 路径参数 | 结果 ZIP |

## 健康检查

```bash
curl -fsS http://127.0.0.1:8501/api/health
```

公共字段：

```json
{
  "status": "ready",
  "device": "cuda:0",
  "backend": "ultralytics_cuda",
  "architecture": "x86",
  "machine": "x86_64",
  "device_family": "x86_cuda",
  "model_format": "pt",
  "config_selection": "architecture",
  "validated": false,
  "validation_candidate": false,
  "model_layout": "independent_models_v1",
  "context_mode": "model",
  "queue": {},
  "inference_replicas": 1,
  "generation_id": "incremental_detection_generation_4plus2",
  "generation_name": "4+2 增量检测生产代际",
  "runtime_generation_control": "onsite_generation_v1",
  "classes": [
    "soldier",
    "small_aircraft",
    "warship",
    "tank",
    "patrol_boat",
    "armored_vehicle"
  ]
}
```

Ascend310B v2 正式实例返回 `architecture: "arm"`、`backend: "ascend_acl"`、`device: "ascend:0"`、`model_format: "om"`、`validated: true`、`model_layout: "independent_yolo26_e2e_v1"` 和 `context_mode: "model"`。Web 页头使用这些字段显示 `x86 · CUDA` 或 `ARM · Ascend`；模型路径不会通过公共 API 暴露。

模型初始化失败时返回 HTTP `503`：

```json
{"status": "error", "error": "模型服务初始化失败：..."}
```

## 单图检测

```bash
curl -fsS \
  -F "file=@/path/to/image.png;type=image/png" \
  -F "confidence=0.10" \
  http://127.0.0.1:8501/api/detect
```

`confidence` 必须位于 `/api/config/public` 返回的 `confidence.min` 与 `confidence.max` 之间。当前 Ascend 发布包的输入契约为 `640×512`、8-bit 灰度/RGB/RGBA PNG；x86 解码器接受其配置允许的图像格式。单文件内存快速路径上限为 2 MiB。

成功响应的稳定顶层字段如下：

| 字段 | 含义 |
| --- | --- |
| `filename`、`image_width`、`image_height` | 输入身份与尺寸 |
| `context` | Scene-SensorNet 的 sensor、scene 与概率 |
| `detections` | 六类检测框、置信度、全局类别和来源 owner |
| `class_counts`、`detection_count` | 汇总计数 |
| `confidence_threshold` | 本次请求阈值 |
| `inference_ms`、`system_total_ms`、`timings` | 推理与分阶段耗时 |
| `agent.models_used` | 实际执行的模型 |
| `agent.protocols` | 增量专家输出摘要 |
| `agent.decision` | 内容门控、融合、冲突抑制、代际和 owner |

`agent.decision.content_execution_gates` 记录 Ascend v2 的场景与 Base 双证据门控。`executed_protocols` 和 `skipped_protocols` 直接反映增量 OM 是否执行。通过板端增量门禁的隔离演示配置还会在 `agent.decision.edge_incremental_adapter` 返回 Adapter 的活动状态、轮次协议和运行身份。x86/CUDA 与 Ascend 的 `timings` 子字段不同，调用方应按字段名读取。

## 批量检测

```bash
curl -fsS \
  -F "files=@/path/to/a.png;type=image/png" \
  -F "files=@/path/to/b.png;type=image/png" \
  -F "confidence=0.10" \
  http://127.0.0.1:8501/api/batch
```

响应包含 `batch_id`、`image_count`、`detection_count`、`results` 和 `download_url`。每个 `results` 元素沿用单图结果契约，并增加 `preview_url`。批次结果存放在有容量和 TTL 限制的内存缓存中；过期的预览或下载请求返回 `404`。

## 增量工作台路由

| 方法与路径 | 用途 |
| --- | --- |
| `GET /api/incremental/batches` | 列出已审计批次 |
| `POST /api/incremental/batches` | 上传 ZIP；字段为 `file`、可选 `name` 和 `class_names` |
| `GET /api/incremental/batches/{batch_id}` | 读取批次详情 |
| `PATCH /api/incremental/batches/{batch_id}/classes` | JSON `{"names": {"源类别ID": "名称"}}` |
| `GET /api/incremental/batches/{batch_id}/images/{index}` | 读取预览图像 |
| `POST /api/incremental/batches/{batch_id}/inject` | 固化拆分并注入工作台 |
| `POST /api/incremental/batches/{batch_id}/train` | 创建训练任务，返回 HTTP `202` |
| `GET /api/incremental/jobs?batch_id=...` | 列出任务 |
| `GET /api/incremental/jobs/{job_id}?batch_id=...` | 任务详情 |
| `GET /api/incremental/jobs/{job_id}/logs?batch_id=...&tail=...` | 文本日志 |
| `POST /api/incremental/jobs/{job_id}/cancel?batch_id=...` | 取消任务 |

正式 4+2 顺序增量的类别、轮次和父子代际以 `configs/incremental_round_registry_4plus2.yaml` 为准。工作台上传接口负责批次审计与任务编排；正式候选登记和晋级仍由 `tools/13_register_incremental_round_candidate.py`、`tools/12_summarize_incremental_rounds.py` 与 `tools/10_promote_scene_aware_4plus2.py` 完成。

`POST /api/runtime/generation` 是 `agile-agent incremental onsite` 使用的本机原子代际接口。它同时校验实际 TCP 客户端为 loopback、专用请求契约、运行中父代和候选复核清单，仅接受 `promote` / `rollback`；候选在服务进程内完成 shadow 加载后由 `AtomicEngineProvider` 原子换代，并保留上一代作为回滚目标。现场一键 CLI 负责生成和提交完整请求契约。

## 访问控制边界

- Web/API 默认绑定 `127.0.0.1`，适合板端 CLI、本机浏览器和 SSH 端口转发。
- 跨主机部署由反向代理终止 TLS，并按现场网络策略增加身份认证、来源限制与速率限制。
- `/api/runtime/generation` 额外验证真实 TCP 客户端、父代身份、候选 manifest 和验收清单，形成独立于代理层的晋级门禁。
- 公共配置和健康响应只公开运行身份与能力，不公开权重绝对路径。

## 审计日志

```text
GET /api/logs?batch_id=BATCH_ID&limit=200&level=info&component=inference&trace_id=TRACE_ID&job_id=JOB_ID
```

`batch_id` 为必填项。响应的 `events` 只公开时间、级别、组件、事件、trace/batch/job ID、耗时和消息；完整日志通过 CLI 查询。

## 错误与通用响应头

| HTTP 状态 | 场景 |
| ---: | --- |
| `400` | multipart、图像、置信度、查询参数或批次状态无效 |
| `404` | 路由、批次、任务、预览或下载不存在 |
| `422` | 增量 ZIP 已接收但审计未通过 |
| `503` | 模型服务或增量工作台暂时不可用 |

错误响应统一包含 `error`；健康初始化失败响应同时包含 `status: "error"`。服务还设置 `X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer` 与 `Cache-Control: no-store`。
