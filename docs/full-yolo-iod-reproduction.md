# 完整 YOLO-IOD 舰船 3+1 复现指南

## 实验边界

本实验使用官方 [YOLO-IOD](https://github.com/qiangzai-lv/YOLO-IOD) 提交 `3d9d05a6e88561c88916657367412f9adb7341a7`，基座为 YOLO-World(X)，不再是 YOLO11s 轻量适配。实验独立运行，不修改默认 Web 权重，也不把未通过门禁的模型注册到 Agent。

严格 p02 将类别顺序重排为：

```text
0 soldier
1 small aircraft
2 tank
3 warship（新增类）
```

评测时通过 manifest 映射回赛题全局 ID。基础训练完全排除舰船图像；增量训练和验证只读取舰船增量图像。lock-val 在最终权重冻结后才物化。
原始 YOLO 标签没有实例分割标注，转换器会为每个框生成等价的矩形 polygon，使检测框能通过官方 YOLO-World mask-refine 与 CopyPaste 管线而不被丢弃。

## 完整组件

- **CPR**：r03 在没有旧类 GT 共现的舰船增量图像上生成了 420 个旧类伪框，导致监督与数据不匹配。r04/r05 要求 `incremental_old_class_gt_count=0` 并将 CPR 记为 `skip_by_policy`，最终训练直接读取 `current_train.json`。
- **IKS**：直接使用官方 GPS Loop，根据梯度和参数变化选择卷积核单元。
- **CAKD**：直接使用官方旧教师/当前教师检测头完成 Cross-Stage Asymmetric Knowledge Distillation。

官方固定提交在单类 GPS 阶段可能遇到没有梯度重要性条目的参数，但正式更新时仍强制索引掩码；同时，GPS 重要性扫描会消耗梯度累积计数，导致真实训练最后一轮的 `loss_factor` 变成 0。仓库内的 `patches/yolo_iod_single_class_grad_mask.patch` 同时启用掩码保护并在扫描后恢复训练迭代计数，runner 会校验补丁状态并记录 SHA256。

## 环境与预检

服务器使用独立环境：

```text
/project/IDIP/CONDA_ENV/envs/yolo-iod-full
```

代码与数据仍从 AgileAgent 工程发起。运行前执行：

```bash
cd /project/IDIP/MAJ/code/tiaozhanbei
/project/IDIP/CONDA_ENV/envs/yolo-iod-full/bin/python \
  tools/72_run_full_yolo_iod.py \
  --config configs/full_yolo_iod_warship_gpu3.yaml \
  --check-only
```

必须看到 `ready: true`，且官方提交、MMCV 依赖、三份 split 和输出目录均通过检查。
预训练权重按 YAML 中的地址顺序下载，支持超时重试和断点续传；国内服务器默认先尝试 Hugging Face 镜像，训练时 CLIP 文本编码器也通过 `runtime.env.HF_ENDPOINT` 使用同一镜像。
官方 MMDetection 3.0 数据管线固定使用 `albumentations==1.3.1` 与 `numpy==1.26.4`；预检会拒绝 1.4 之后严格校验未知元数据键的 Albumentations 版本。

## 一键运行

配置确认后执行：

```bash
/project/IDIP/CONDA_ENV/envs/yolo-iod-full/bin/python \
  tools/72_run_full_yolo_iod.py \
  --config configs/full_yolo_iod_warship_gpu3.yaml
```

脚本依次完成：三类基础模型、当前舰船教师、CPR 数据门禁、完整 YOLO-IOD、基础/最终 lock-val 评测。r05 所有阶段固定使用 GPU 3；配置校验会拒绝多卡或阶段间设备不一致。所有阶段写入独立日志和 `action_log.jsonl`，任一步失败立即停止。

显存允许的 micro-batch 分别为 base/current 的 4 和 final 的 2。YAML 使用梯度累积 `4/4/8`，使三个阶段的有效 batch 都等于 16；command manifest 会记录 micro-batch、累积步数和有效 batch。base、current 和 final 都会识别 `last_checkpoint` 并使用显式检查点路径续训。

r04/r05 的 final 验证配置只引用 `current_dev.json`，不读取旧类验证图像。旧类保持率只能在权重冻结后通过 lock-val 评分，不能用于调参或早停。

## 验收

最终比较以下指标：

```text
New-mAP50 >= 0.60
KRR >= 0.95
四类 mAP50 >= 0.80（内部观察目标）
old_raw_image_count = 0
```

同时报告 CPR 旧类伪标签数量、基础/当前/最终权重 SHA256、旧类与新增类逐类 AP50。完整模型明显大于 YOLO11s，结果只用于方法有效性比较；未经后续压缩和部署复核，不作为 310B 或默认 Web 候选。
