<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 预构建模型

当前活动模型包是 [`full-score/20260824-4plus2-yolo26-runtime-calibration-v1/`](full-score/20260824-4plus2-yolo26-runtime-calibration-v1/README.md)。它包含已经在 Ascend310B1 公共 `8501` 上完成部署后复验的 4+2 三-OM release，可直接物化为正式运行目录。板端服务层进一步使用 `20260824-4plus2-yolo26-replica-pool-v1` 组织三个同构推理实例。

活动结构：

- 四类 Base YOLO26s OM：`soldier/small_aircraft/warship/tank`；
- 二类增量 YOLO26s OM：`patrol_boat/armored_vehicle`；
- Scene-SensorNet OM：真实场景与 IR/SAR 概率；
- 双证据执行门控：只有空域概率和 Base 小型飞行器检测同时成立才跳过增量专家；
- mixed dev 冻结的 Base/Specialist logit 校准、逐类阈值和全类重叠仲裁；
- 正式公共 `8501` 经原子路由进入内部 `18501`，回滚 listener 保留，`8502` 用于候选验证。

正式部署：

```bash
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

活动包的 Base mAP50、New-mAP50、KRR 分别为 `0.816663/0.611461/1.0`；2026-08-29 当前三实例公共 `8501` 的 schema v8 全流程复核为 `33.897326 FPS`，生成并验证 60 个正式六列 TXT。lock 新类误激活从上一代的 `35/75` 降至 `17/75`。

同一模型包还作为断网板端增量演示的冻结父代。2026-08-26 的隔离 Adapter 候选达到 Base/New/KRR `0.816663/0.624935/1.0`，并保持正式包身份不变；其旧性能数据已归档。当前四项评分证据见 [`reports/ascend310b/20260829-full-score-recheck-v1`](../../reports/ascend310b/20260829-full-score-recheck-v1/README.md)，完整部署状态见 [`docs/ascend-310b-current-status.md`](../../docs/ascend-310b-current-status.md)。
