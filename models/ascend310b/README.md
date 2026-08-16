# Ascend 310B 预构建模型

`full-score/20260816-full-score-1493b04/` 是已在 Ascend310B1、CANN `7.0.RC1` 上通过满分门禁的不可变发布资产。它不是训练输出目录，也不是候选模型；其中的 OM 可直接用于板端部署。

包内容包括：

- 正式 `shared_backbone_dual_head.om` 和 context 回滚 OM；
- 与 OM 严格关联的 ONNX、源 checkpoint、AIPP 配置、ATC 日志和构建清单；
- 字节级正式配置、验证摘要、三项精度报告、两轮候选性能报告和发布后性能报告；
- 89 图冻结预测，便于在合法获得标签后重新计分；
- `SHA256SUMS`，用于在启动 ACL 前验证所有资产。

零训练部署：

```bash
./scripts/materialize_ascend310b_full_score_release.sh
RELEASE=/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

物化脚本不启动服务；上面的直接 `8501` 方式适合没有旧回滚 listener 的新板。已有正式旧三 OM 服务的板使用 `18501` 主实例加 `8501` 回滚 listener 的双实例拓扑。完整步骤见 [`docs/ascend-310b-deployment.md`](../../docs/ascend-310b-deployment.md)。

仓库不发布竞赛原始图像和标签。不需要数据集即可完成资产哈希、release 验证和服务健康检查；重新计算 Base/New/KRR 需要按文档放置有权使用的 89 图及标签，重新测量 FPS 需要 20 张符合输入契约的 PNG。
