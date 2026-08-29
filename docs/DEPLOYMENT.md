<!-- generated-by: gsd-doc-writer -->
# 部署

AgileAgent 当前提供两条正式部署路径：x86/CUDA 运行 production PT 权重，Ascend310B v2 物化预构建三-OM release。两条路径共享六类全局 ID、类别 owner、Scene-SensorNet 输出和 Web API。

## x86/CUDA

### 环境与配置

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

默认配置为 `configs/agent_pipeline.yaml`，production profile 为 `incremental-detection`。所需资产位于：

```text
models/production/incremental_detection/four_class_base_detector.pt
models/production/incremental_detection/incremental_detector.pt
models/context/scene_sensor_net.pt
models/profiles/incremental-detection/active.json
models/generations.json
```

配置检查与模型加载检查：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent doctor
python scripts/verify_release.py
python scripts/smoke_models.py --load-only
```

### 启动

```bash
./scripts/start_agent.sh
```

终端工作台使用：

```bash
./scripts/start_agent.sh --cli
```

通用启动脚本和 `agile-agent` CLI 默认使用 `--config auto`：x86 选择 `ultralytics_cuda` 与三个 `.pt` 资产，ARM 选择 `ascend_acl` 与三个 `.om` 资产。`AGILE_AGENT_CONFIG` 或顶层 `--config PATH` 可固定配置；自动选择从不在 CUDA 与 Ascend 之间做失败回退。

服务默认监听 `127.0.0.1:8501`。运行状态以 `/api/health` 返回的 `generation_id`、`backend`、活动类别和模型队列为准。

## Ascend310B v2

正式包：

```text
models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/
```

目标环境为 Ascend310B1、CANN `7.0.RC1` 和 `/usr/local/miniconda3/envs/agileagent`。release 已包含 Base、Incremental、Scene-SensorNet 三个 OM 及构建和验收证据，可直接物化为确定性的正式运行目录。

### 正式 release 物化

```bash
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh
```

固定目标目录：

```text
/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
```

目标已存在时执行只读复核：

```bash
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

### 直接监听 8501

未配置回滚 listener 的设备可以直接启动正式实例：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

### 主实例与回滚实例

已配置回滚能力的设备采用固定拓扑：

| 职责 | 地址 |
| --- | --- |
| 公共入口 | `127.0.0.1:8501` |
| 4+2 主实例 | `127.0.0.1:18501` |
| 回滚 listener | 物理监听 `127.0.0.1:8501` |
| 候选验证 | `127.0.0.1:8502` |

安装或更新服务：

```bash
PRIMARY=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
ROLLBACK=/home/HwHiAiUser/agileagent/releases/ROLLBACK_RELEASE

sudo "$PRIMARY/src/scripts/install_ascend310b_primary_services.sh" \
  "$PRIMARY" "$ROLLBACK" 18501
```

三个 systemd unit 为：

```text
agileagent-ascend310b-main.service
agileagent-ascend310b-rollback.service
agileagent-ascend310b-route.service
```

路由管理：

```bash
sudo /usr/local/sbin/agileagent-ascend310b-primary-route status 18501
sudo /usr/local/sbin/agileagent-ascend310b-primary-route remove 18501
sudo /usr/local/sbin/agileagent-ascend310b-primary-route apply 18501
```

### 断网一键增量演示

板端同时提供与 production 隔离的 `4→4+1→4+2` 增量学习演示。production 环境负责 ACL 推理，独立 `agileagent_train` 环境负责 `torch_npu` 反向传播；现场只需提供当前 Increment 数据目录：

```bash
cd ~/agileagent/repo

./scripts/run_ascend310b_incremental_demo.sh \
  /path/to/datasets_r2_inc_train
```

命令自动完成输入审计、轮次对齐、两轮 Adapter 训练、dev 选择、mixed lock 验收、ONNX/OM 导出、ACL 数值验证、隔离部署和启用 Adapter 后的完整图像链路 FPS 门禁。演示前可只生成执行计划：

```bash
./scripts/run_ascend310b_incremental_demo.sh \
  /path/to/datasets_r2_inc_train \
  --plan-only
```

通过门禁后，运行目录中的 `demo_report.json` 给出 `demo_config`。使用该配置启动 CLI 即可展示学习后的隔离代际：

```bash
AGILE_AGENT_CONFIG=/absolute/run/deployment/agent_pipeline_ascend310b_demo.yaml \
  ./scripts/start_agent.sh --cli
```

2026-08-26 实机整链验收结果为 Base mAP50 `0.816663`、New-mAP50 `0.624935`、KRR `1.000000`、Full-mAP50 `0.726497`，热态完整命令耗时 `1007.07 秒`；当时的性能报告未包含正式结果落盘，现已归档。当前正式 release 于 2026-08-29 再次按“总帧数 ÷ 全流程总耗时”复核为 `33.897326 FPS`。完整环境准备、产物结构和冷/热态时间见 [`ascend-310b-offline-incremental-demo.md`](ascend-310b-offline-incremental-demo.md)。

## 验收

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS http://127.0.0.1:18501/api/health
ss -H -ltn 'sport = :8501 or sport = :18501 or sport = :8502'
```

Ascend 正式健康响应必须包含：

```json
{
  "status": "ready",
  "backend": "ascend_acl",
  "device": "ascend:0",
  "validated": true,
  "validation_candidate": false,
  "model_layout": "independent_yolo26_e2e_v1",
  "context_mode": "model",
  "generation_id": "incremental_detection_generation_4plus2"
}
```

真实图像冒烟：

```bash
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

正式验收同时保存 release verifier、健康响应、冻结精度报告和公共入口 batch FPS 报告。候选使用 `8502` 完成隔离评分，通过后由 `tools/111_promote_ascend_full_score_release.py` 生成新 release，再经服务安装器提升。平台由架构和显式配置确定：x86 使用 PT/CUDA，ARM 使用 OM/ACL，启动过程不会跨平台静默回退。

详细的板端目录、路由、构建候选、评分和监测命令见 [`ascend-310b-deployment.md`](ascend-310b-deployment.md)，当前 v2 证据索引见 [`ascend-310b-current-status.md`](ascend-310b-current-status.md)。HTTP 契约见 [`API.md`](API.md)。
