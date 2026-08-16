# Ascend 310B 部署实现

AgileAgent 已在 Atlas 200I DK A2 上把共享双逻辑头满分方案提升为正式主线，同时保留原三模型 OM 作为即时回滚。仓库现已包含该正式 release 的预构建模型与验证证据；在 CANN/Python 环境已配置好的 310B 上可零训练、零 ATC 部署。本页区分新板直启、公共入口、主实例、回滚 listener 和下一轮候选。

## 三 OM 即时回滚结构

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

## 正式满分主线结构

```text
20 图 multipart batch
  -> 有界解析与 DVPP encoded 预处理
  -> shared_backbone_dual_head_v1 OM
     -> old head / frozen_base_model
     -> new head / incremental_model
  -> fixed_neutral_v1（不执行 Scene OM）
  -> 原融合、审计与 API schema
```

正式主实例使用内部 `18501`、`896×736` AIPP、`raw_dual_head_v1`、pageable memory 和 threaded execution，公共请求通过 `8501` 的精确 loopback NAT 进入。`8502` 继续专用于隔离候选。old/new 当前参考阈值为 `0.05/0.30`，但更换数据集后必须重新搜索。完整方法见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。

## 设备与运行环境

| 项目 | 当前值 |
| --- | --- |
| 设备 | Atlas 200I DK A2 |
| SoC | Ascend310B1 |
| CANN | `7.0.RC1` |
| Python | `/usr/local/miniconda3/envs/agileagent/bin/python` |
| 正式配置 | `configs/agent_pipeline_ascend310b.yaml` |
| 公共地址 | `127.0.0.1:8501` |
| 主实例内部地址 | `127.0.0.1:18501` |
| 回滚 listener | 原三 OM，物理监听 `127.0.0.1:8501` |
| 后续候选 | `127.0.0.1:8502` |

## 从仓库零训练部署

仓库模型包位于：

```text
models/ascend310b/full-score/20260816-full-score-1493b04/
```

它包含两个正式 OM、source checkpoint、ONNX、AIPP、ATC 日志、training/export/build manifest、正式配置和原始 validation 报告。单文件均小于 GitHub 100 MB 限制，使用普通 Git 版本化；克隆后不需要 Git LFS。

在已安装 CANN `7.0.RC1` 和命名环境 `agileagent` 的新板上执行：

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh
```

脚本依次校验包内 `SHA256SUMS`、复制 Git 跟踪的运行源码和预构建资产、调用 `tools/95_verify_ascend_release.py --require-validation`。它不训练、不导出 ONNX、不运行 ATC、不安装依赖，也不操作任何服务端口。固定安装目录是：

```text
/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04
```

目标已存在时默认拒绝覆盖；只读验证已有副本：

```bash
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

新板没有旧回滚服务时，可直接让满分 release 监听 `8501`：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

配置文件内部的 `runtime.server_port: 18501` 是正式双实例拓扑的身份字段；启动脚本通过显式 `AGILE_AGENT_ASCEND_PORT=8501` 决定新板监听端口。已有旧三 OM release 的板不要停止旧 listener，继续使用后文的 systemd 双实例安装：满分主实例监听 `18501`，公共 `8501` 经精确路由进入主实例，删除规则即回滚。两种拓扑使用相同模型、配置和验证身份。

启动后验证：

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

健康响应必须包含 `status:"ready"`、`validated:true`、`validation_candidate:false`、`model_layout:"shared_backbone_dual_head_v1"` 和 `context_mode:"fixed_neutral_v1"`。

### 指标复现所需数据

仓库不分发受授权约束的竞赛原始图像和标签，因此“核对原始证据”和“重新计算指标”必须区分：

| 可用输入 | 可完成的复现 |
| --- | --- |
| 仅克隆仓库 | 校验所有模型/证据 SHA256，验证 release，启动服务，查看包内原始 score/benchmark 报告 |
| 20 张符合契约的 PNG | 重新执行 30 次预热和三轮 20 图 batch，复测 FPS |
| 合法取得的 89 图与标签 | 重新冻结预测并计算 Base mAP50、New-mAP50、KRR |
| 仅取得同版 89 图标签 | 直接对包内 `validation/frozen-predictions.jsonl` 重新评分，无需再次推理或训练 |

原始报告位于模型包 `validation/`，记录 Base `0.8049006528`、New `0.6050327631`、KRR `1.0`，候选两次 batch 中位 `30.066/30.080 FPS`，发布后公共 `8501` 三轮 `30.234/30.243/30.294 FPS`。

## 模型契约

| 正式主线模型 | 输入张量 | 输出 |
| --- | --- | --- |
| Shared dual detector | `1,3,736,896` FP32/AIPP | old `[1,7,13524]`、new `[1,5,13524]` |
| Context 回滚资产 | `1,3,160,160` FP32/AIPP | 正常 fixed-neutral 路径不执行前向 |

主线单个 dual OM 输出两个 logical head；context OM 会加载并登记为回滚资产，但 `fixed_neutral_v1` 正常路径不执行它的前向推理。配置和 release manifest 都记录路径、SHA256、logical owner 和类别映射。原 Base/Incremental/Scene 三 OM 的输入输出契约保持不变，由独立回滚 service 使用。

## 图像预处理

基础图像尺寸为 `640×512` PNG。

正式 shared detector 预处理：

1. RGB 解码；
2. 等比例缩放到 `896×717`；
3. 按 stride 32 补边为 `896×736`；
4. 转换为 `1×3×736×896` FP32；
5. 将数值归一化到 `[0,1]`。

old/new logical head 共享同一个 `896×736` 输入和特征金字塔。回滚链路的 Incremental Detector 仍使用 `640×512`，Scene-SensorNet 仍使用 `160×160`。

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

## Release 目录与服务

```text
/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04/
├── src/
├── om/
│   ├── shared_backbone_dual_head.om
│   └── scene_sensor_net.om
├── provenance/
│   ├── release-build-manifest.json
│   ├── training_report.json
│   └── export_manifest.json
├── configs/agent_pipeline_ascend310b.yaml
├── validation/
│   ├── score.json
│   ├── benchmark.json
│   └── validation-summary.json
├── release.json
└── agent-web.pid
```

原三 OM release 仍位于 `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2`。

## 服务状态与回滚

```bash
systemctl status agileagent-ascend310b-main.service
systemctl status agileagent-ascend310b-rollback.service
systemctl status agileagent-ascend310b-route.service
curl -fsS http://127.0.0.1:8501/api/health
```

健康响应应包含 `validated:true`、`validation_candidate:false`、`model_layout:"shared_backbone_dual_head_v1"` 和 `context_mode:"fixed_neutral_v1"`。

即时回滚与重新提升：

```bash
sudo /usr/local/sbin/agileagent-ascend310b-primary-route remove 18501
sudo /usr/local/sbin/agileagent-ascend310b-primary-route apply 18501
```

规则只匹配 `127.0.0.1:8501` 并带 comment `AGILE_AGENT_ASCEND310B_PRIMARY`。删除规则后新连接直接进入仍在监听的三 OM 服务；已有连接按内核连接状态自然结束。

## 新数据集：构建与验收满分候选

本节只用于更换数据集或训练新 release，不是部署仓库内当前正式模型的前置步骤。候选不复用正式配置，也不直接启动在 `8501`。完整顺序为：

1. 在 WSL 既有 `.venv` 中训练 residual adapter/new head，生成同时登记 best/last 的 training report；
2. 选择一个已授权 checkpoint 导出 dual-head ONNX 和 export manifest；
3. 将 ONNX、source checkpoint、training/export manifest 与 context build manifest 按 SHA256 同步到板端；
4. 在 CANN `7.0.RC1` 环境构建 OM 和新的 build manifest；
5. 由 `tools/109` 生成只监听 `8502` 的候选配置；
6. 运行 score gate，结束后只停止其启动的精确 `8502` 进程，并再次确认 `8501 ready`。

板端构建示例：

```bash
cd /home/HwHiAiUser/agileagent
AGILE_AGENT_ASCEND_PYTHON=/usr/local/miniconda3/envs/agileagent/bin/python \
./scripts/build_ascend_dual_head_om.sh \
  /path/to/shared_backbone_dual_head.onnx \
  /path/to/EXPORT_CHECKPOINT.pt \
  /path/to/training-report.json \
  /path/to/export-manifest.json \
  /path/to/formal-context-build-manifest.json \
  /path/to/build-output
```

生成候选配置时必须同时提供 build manifest 中登记的 dual/context OM；生成器会核对方法配置、training/export manifest 和 OM 哈希：

```bash
/usr/local/miniconda3/envs/agileagent/bin/python \
  tools/109_materialize_ascend_full_score_candidate.py \
  --dual-om /path/to/shared_backbone_dual_head.om \
  --context-om /path/to/scene_sensor_net.om \
  --build-manifest /path/to/build-manifest.json \
  --old-threshold 0.05 \
  --new-threshold 0.30 \
  --output /path/to/candidate-8502.yaml
```

评分命令：

```bash
./scripts/run_ascend310b_score_gate.sh \
  /path/to/candidate-8502.yaml \
  /path/to/mixed-images \
  /path/to/mixed-test.txt \
  /path/to/base-test.txt \
  /path/to/score-gate-output
```

score gate 在启动候选前检查正式 `8501 ready`、`8502` 未占用、CANN 版本、PNG 输入契约和 build manifest。它先在短生命周期引擎中冻结无标签预测，再打开标签评分，最后启动 HTTP 候选执行 30 次预热和三轮 20 图 batch；板端不运行 Web pytest。

## 满分候选提升为正式主线

候选四项满分后，先物化不可变正式 release；不得直接编辑候选 YAML 的 `validated`：

```bash
/usr/local/miniconda3/envs/agileagent/bin/python \
  tools/111_promote_ascend_full_score_release.py \
  --candidate-config /path/to/candidate-8502.yaml \
  --score /path/to/score-v2.json \
  --benchmark /path/to/benchmark.json \
  --repeat-benchmark /path/to/benchmark-repeat.json \
  --release-root /home/HwHiAiUser/agileagent/releases/RELEASE_ID \
  --internal-port 18501
```

工具重新验证 Base/New/KRR/batch FPS、预测冻结、增量数据隔离、Base 零漂移及资产哈希，并复制 OM、training/export/build/method/score/benchmark 证据，生成 `validated: true` 的 release-local 配置。逐框/业务 JSON、precision、误激活率、Scene/Sensor 和单请求时延仍记录为诊断，不参与正式淘汰。

随后由 root 安装主/回滚/路由三个 systemd unit：

```bash
sudo /home/HwHiAiUser/agileagent/releases/RELEASE_ID/src/scripts/install_ascend310b_primary_services.sh \
  /home/HwHiAiUser/agileagent/releases/RELEASE_ID \
  /home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2 \
  18501
```

脚本先在 `18501` 启动并验证主实例，确认 ready 后才插入原子路由；任一步失败都会删除规则并恢复旧服务。`8502` 不会被 systemd 或正式路由占用。

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
/usr/local/miniconda3/envs/agileagent/bin/python tools/92_run_ascend_om.py \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --output-dir reports/ascend310b/om_output
```

89 图 Agent 预测冻结后执行评分：

```bash
/usr/local/miniconda3/envs/agileagent/bin/python tools/94_score_ascend_agent.py \
  --predictions reports/ascend310b/predictions_frozen \
  --method-config configs/ascend310b/full_score_method.yaml \
  --mixed-split splits/strict_3plus1/mixed_test.txt \
  --base-split splits/strict_3plus1/base_test.txt \
  --output reports/ascend310b/score.json
```

当前正式满分 release 结果：

| 指标 | 数值 |
| --- | ---: |
| Base mAP50 | `0.804901` |
| New-mAP50 | `0.605033` |
| KRR | `1.000000` |
| 新类 precision（诊断） | `0.792453` |
| 误激活率（诊断） | `0.242857` |

## 性能记录

| 测量 | 样本量 | 已记录结果 |
| --- | ---: | --- |
| 三 OM 回滚 release 完整 89 图 | 89 | 墙钟均值 `71.491 ms`、`13.99 FPS` |
| 已解码 Agent 核心 | 200 | 均值 `32.148 ms`、P95 `33.193 ms`、`31.11 FPS` |
| AIPP staging 真实 PNG API | 1,068 | 均值 `51.203 ms`、P95 `63.9 ms`、`19.53 FPS` |
| DVPP 编码输入 | 240 | 均值 `37.124 ms`、P95 `38.154 ms`、`26.94 FPS` |
| 共享双头候选 | 两组 3×20 batch | 中位 `30.066/30.080 FPS`；Base/New/KRR 同时满分 |
| 公共 `8501` 发布后复核 | 30 次预热 + 3×20 batch | `30.234/30.243/30.294 FPS`，中位 `30.243 FPS` |

共享双头已成为正式主线。其单请求均值/P95/P99、逐框差异、precision 和误激活率继续留作诊断；正式计分与 release 提升只阻断 Base mAP50、New-mAP50、KRR、三轮 20 图 batch 中位 FPS，以及数据隔离/零漂移/资产哈希等结果有效性前置条件。

## 运行监测

```bash
npu-smi info
curl -fsS http://127.0.0.1:8501/api/health
ps -ef | grep 'uvicorn fair_agent.web.app:app'
```

请求级日志记录 `trace_id`、generation、执行模型、检测数量和分段耗时，发布验证记录模型路径、SHA256、配置与指标。
