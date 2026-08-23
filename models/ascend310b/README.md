<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 预构建模型

当前活动包是 [`full-score/20260824-4plus2-yolo26-runtime-calibration-v1/`](full-score/20260824-4plus2-yolo26-runtime-calibration-v1/README.md)。它包含已经在 Ascend310B1 公共 `8501` 上完成部署后复验的 4+2 三-OM release，可直接零训练物化。

活动结构：

- 四类 Base YOLO26s OM：`soldier/small_aircraft/warship/tank`；
- 二类增量 YOLO26s OM：`patrol_boat/armored_vehicle`；
- Scene-SensorNet OM：真实场景与 IR/SAR 概率；
- 双证据执行门控：只有空域概率和 Base 小型飞行器检测同时成立才跳过增量专家；
- mixed dev 冻结的 Base/Specialist logit 校准、逐类阈值和全类重叠仲裁；
- 正式公共 `8501` 经原子路由进入内部 `18501`，回滚 listener 保留，`8502` 用于候选验证。

零训练部署：

```bash
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

活动包的 Base mAP50、New-mAP50、KRR 分别为 `0.816663/0.611461/1.0`；公共 `8501` 部署后 `30 + 3×20` 的中位 FPS 为 `38.662`。lock 新类误激活从上一代的 `35/75` 降至 `17/75`。完整证据和部署边界见 [`docs/ascend-310b-current-status.md`](../../docs/ascend-310b-current-status.md)。
