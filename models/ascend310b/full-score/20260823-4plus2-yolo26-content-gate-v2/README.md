# 2026-08-23 Ascend 310B 4+2 满分发布包

这是当前正式 4+2 板端 release 的可版本化副本，对应板端目录：

```text
/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
```

该包已经在 Atlas 200I DK A2 / Ascend310B1、CANN `7.0.RC1` 上完成候选评分、独立复跑、正式提升和公共 `8501` 部署后复验。

## 满分结果

| 赛题计分项 | 实测 | 满分门槛 |
| --- | ---: | ---: |
| Base mAP50 | `0.8256706047` | `≥0.80` |
| New-mAP50 | `0.6188591828` | `≥0.60` |
| KRR | `1.0000000000` | `≥0.95` |
| 候选 20 图 batch 中位 FPS | `39.3468` / `39.4244` | `≥30` |
| 公共 `8501` 部署后中位 FPS | `39.5726` / `39.5883` | `≥30` |

Full-mAP50 为 `0.7249274787`。旧类 mAP50 在增量前后均为 `0.7779616266`，冻结旧类预测完全等价。

lock precision `0.677551`、lock recall `0.661111`、误激活率 `0.466667` 是保留的部署诊断，不属于四项赛题淘汰门槛。

## 运行结构

- `independent_yolo26_e2e_v1`：四类 Base YOLO26s OM 与二类增量 YOLO26s OM 分开执行并保持固定类别所有权。
- `context_mode: model`：真实 Scene-SensorNet 输出 `air/forest/sea/urban` 与 IR/SAR 概率。
- Base、Scene 先并发；仅当 `air ≥ 0.5` 且 Base 同时检出 `small_aircraft` 时跳过增量专家。
- 在线执行门控只读取模型输出，不读取标签或文件名。
- AIPP 输入为 `1×608×736×3`，两个检测 OM 均输出 `[1,300,6]` 端到端候选。
- 计分请求阈值和六类活动阈值均为 `0.10`。

## 包内容

```text
configs/       validated:true 的正式 Agent 配置
om/            Base、Incremental、Scene-SensorNet 三个 OM
provenance/    源 checkpoint、ONNX、AIPP、ATC 日志和构建清单
validation/    冻结预测、精度报告、候选/复跑/公共8501性能报告
release.json   正式 release 摘要
SHA256SUMS     物化前的一次性完整性清单
```

仓库不包含竞赛原始图像或标签。仅部署和查看既有报告不需要数据集；重新测 FPS 需要至少 20 张符合 `640×512`、8-bit 灰度/RGB/RGBA PNG 契约的图像；重新计算精度需要合法取得的 89 图标签。

## 零训练物化

在已经配置 CANN 和 `/usr/local/miniconda3/envs/agileagent` 的 310B 上执行：

```bash
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

物化脚本不训练、不导出 ONNX、不运行 ATC、不升级 CANN，也不自动修改服务端口。已有旧 listener 的板应按 [`docs/ascend-310b-deployment.md`](../../../../docs/ascend-310b-deployment.md) 使用 `18501` 主实例、`8501` 回滚 listener 和原子路由。
