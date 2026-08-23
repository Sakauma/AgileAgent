# 2026-08-16 Ascend 310B 满分发布包

该目录是板端正式 release `/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04` 的可版本化资产副本。两个 OM 、provenance 和 validation 文件都保持板端原始字节；额外的 `validation/frozen-predictions.jsonl` 来自同一胜出候选，其 SHA256 已在历史证据中登记。

关键结果：

- Base mAP50: `0.8049006528`
- New-mAP50: `0.6050327631`
- KRR: `1.0`
- 候选两次 20 图 batch 中位 FPS: `30.066` / `30.080`
- 发布后公共 `8501` 三轮 FPS: `30.234` / `30.243` / `30.294`

构建身份：Ascend310B1、CANN `7.0.RC1`、`mixed_float16`、`896×736` AIPP、`raw_dual_head_v1`。

文件中的发布绝对路径是验证链的一部分。请使用物化脚本安装到固定目录，不要手工改写 manifest 或把 `validated` 翻为 `true`。

```bash
./scripts/materialize_ascend310b_full_score_release.sh
```

该脚本先执行本目录的 `sha256sum -c SHA256SUMS`，再复制运行源码并执行正式 release 校验；不会训练、导出 ONNX、调用 ATC、安装依赖或触碰服务端口。
