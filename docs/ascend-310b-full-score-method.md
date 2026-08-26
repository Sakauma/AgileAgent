<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 4+2 满分方法与复现手册

本文描述当前正式 4+2 Ascend310B1 方法。唯一机器可读契约是 `configs/ascend310b/full_score_method.yaml`，正式发布包是：

```text
models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/
```

## 1. 当前结论

| 赛题硬指标 | Ascend310B1 实测 | 满分门槛 |
| --- | ---: | ---: |
| Base mAP50 | `0.8166630282` | ≥0.80 |
| New-mAP50 | `0.6114608956` | ≥0.60 |
| KRR | `1.0000000000` | ≥0.95 |
| 候选 / 独立复跑中位 FPS | `37.3571 / 37.9696` | ≥30 |
| 公共 `8501` mixed 20 图中位 FPS | `38.2175` | ≥30 |
| 公共 `8501` 纯增量 140 图中位 FPS | `37.3997` | ≥30 |

Full-mAP50 为 `0.7220053258`。旧类 mAP50 增量前后均为 `0.7772775409`，冻结旧类预测完全等价。

公共 `8501` mixed 20 图逐轮 FPS：

```text
35.3751 / 38.6201 / 38.2175
```

precision `0.729167`、recall `0.612698`、误激活率 `0.226667` 是非阻断诊断。75 张不含新增类的图像中有 17 张至少误激活一个新增类。

性能运行时使用三个同构 Ascend 引擎均衡分片；单实例内部将下一帧 NPU 推理与上一帧 CPU 融合/NMS 重叠。它不修改模型资产、阈值、场景门控或融合结果。FPS 按完整图像推理耗时计算，不包含 HTTP 上传解析与结果保存。

## 2. 协议口径

赛题检测训练与场景功能模型分开记账：

| 阶段 | 可用数据 | 可更新内容 |
| --- | --- | --- |
| Base learning | Base train/dev | 四类 Base 检测器 |
| Incremental learning | 当轮 Increment train/dev | 当轮新增类专家、类别映射与新类专属参数 |
| System calibration | Base/Increment train/dev、mixed dev | Scene-SensorNet、场景先验、执行门控；不更新检测器 |
| Joint evaluation | 截至当前轮的冻结测试/lock | 只评分，不训练、不选参 |

Base 检测器在增量阶段冻结。Scene-SensorNet 属于独立功能模型，不计作增量学习；它的输出会影响在线执行和阈值，但不会改变类别 owner。逐轮协议和证据格式见 `docs/compliant-incremental-learning.md`。

当前板端 release 使用同一两类专家承载 patrol_boat 与 armored_vehicle，但代码交付已经通过 `configs/incremental_round_registry_4plus2.yaml`、`tools/11`–`tools/13` 支持按轮次和类别注册表记录顺序注入证据。

## 3. 模型结构

| 模型 | 来源权重 | 全局类别 |
| --- | --- | --- |
| Base YOLO26s | `models/production/incremental_detection/four_class_base_detector.pt` | 0–3 |
| Incremental YOLO26s | `models/production/incremental_detection/incremental_detector.pt` | 4–5 |
| Scene-SensorNet | `models/context/scene_sensor_net.pt` | IR/SAR 与四类已知场景 |

检测器训练输入为 1280，500 epoch 上限，patience 50；正式导出采用 `608×736` 静态 E2E 输入。训练胜出信息：

- Base：YOLO26s，seed `8675309`，dev mAP50 `0.913454`，best epoch 24，共运行 74 轮；
- Incremental：YOLO26s，seed `20260821`，dev mAP50 `0.983917`，best epoch 209，共运行 259 轮；
- Scene-SensorNet：seed `20260821`，best epoch 81。

Ascend 输出契约：

```text
Base:        [1,608,736,3] uint8 NHWC -> [1,300,6] float32
Incremental: [1,608,736,3] uint8 NHWC -> [1,300,6] float32
Scene:       [1,160,160,3] uint8 NHWC -> sensor + scene probabilities
```

两个检测 OM 均为 `yolo26_e2e_v1`，框解码和 NMS 已包含在图中；Python 侧只做阈值、全局类别映射、冲突仲裁与最终 class-aware NMS。

## 4. 端侧结构选择

| 目标 | 当前实现 |
| --- | --- |
| 复用预处理 | bounded multipart、DVPP 解码与一次 letterbox，Specialist 直接复制 Base 设备输入 |
| 提高并行度 | Base 与 Scene 首先并发提交，按内容证据决定是否收集 Specialist |
| 保持类别所有权 | 四类 Base 与二类 Specialist 使用独立 E2E OM，类别 owner 固定 |
| 使用场景信息 | Scene-SensorNet 输出真实 IR/SAR 与四类已知场景概率 |
| 控制增量计算 | 双证据内容门控只依赖 Scene 概率与 Base 检测 |
| 降低 Host 后处理 | 两个检测 OM 均直接输出 E2E `[1,300,6]` 检测结果 |

正式运行结构为“独立 E2E 检测器 + Scene-SensorNet + 内容执行门控”。

## 5. 板端性能优化

### 5.1 静态输入与 AIPP

原图是 `640×512`。检测模型使用 stride 对齐的 `608×736` 静态输入，ATC 固定：

```text
--input_format=NCHW
--input_shape=images:1,3,608,736
--soc_version=Ascend310B1
--precision_mode_v2=mixed_float16
```

AIPP 接受 NHWC uint8，避免 Python 侧生成 FP32 NCHW 大数组。Scene 模型使用独立 `160×160` AIPP。

### 5.2 统一 enqueue

`execution_mode: async_stream` 与 `schedule_mode: unified_enqueue` 让 DVPP、Scene、Base 和可选 Specialist 使用同一调度路径。Base 与 Scene 首先并发提交；在需要 Specialist 时才收集其结果。配置同时关闭详细事件计时，避免正式计分路径的细粒度同步开销。

### 5.3 双证据执行门控

门控定义：

```yaml
policy: skip_specialist_on_scene_and_base_evidence_v1
action: skip_specialist
scene: air
scene_probability_min: 0.5
base_evidence_class_ids: [1]
base_evidence_mode: any
online_inputs: [scene_probabilities, base_detections]
label_aware_online_routing: false
filename_aware_online_routing: false
```

只有 Scene-SensorNet 的 air 概率达到 0.5，且 Base 同时检出全局类 1（small_aircraft）时，系统才跳过海面/陆地二类专家。单独依赖场景或单独依赖 Base 都不能跳过专家。这利用了“场景是已知类识别”的信息，同时避免使用真值或文件名。

### 5.4 阈值

当前计分请求置信度为 `0.01`，六类冻结阈值分别为 `0=.075, 1=.05, 2=.05, 3=.05, 4=.20, 5=.50`。候选先对 Base 使用 temperature `1.5` / bias `0`、对 Specialist 使用 temperature `1.0` / bias `-0.5` 执行 logit-affine 校准，再应用阈值与跨模型仲裁。全部参数在 mixed dev 上选择并冻结，lock 只评分。

## 6. 训练与导出

基础和增量训练分别使用：

```text
tools/04_train_base_4plus2.py
tools/05_select_base_4plus2.py
tools/06_train_incremental_4plus2.py
tools/07_select_incremental_4plus2.py
```

严格逐轮证据使用：

```text
tools/11_prepare_incremental_round_splits.py
tools/08_evaluate_4plus2.py
tools/13_register_incremental_round_candidate.py
tools/12_summarize_incremental_rounds.py
```

Scene 与冻结校准：

```text
tools/60_train_scene_sensor.py
tools/61_select_scene_sensor_4plus2.py
tools/09_optimize_scene_aware_4plus2.py
tools/10_promote_scene_aware_4plus2.py
```

正式权重训练为 1280 输入，导出到板端前必须生成两个静态 E2E ONNX：

```text
base.onnx        class_count=4, max_det=300, output=[1,300,6]
specialist.onnx  class_count=2, max_det=300, output=[1,300,6]
```

ONNX 与实际 source checkpoint 必须同时进入构建清单，不能只保存 OM。

## 7. ATC 构建

在 CANN `7.0.RC1` / Ascend310B1 上：

```bash
./scripts/build_ascend_yolo26_e2e_oms.sh \
  /path/to/onnx-dir \
  /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/build \
  /home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1/provenance/release-build-manifest.json
```

`onnx-dir` 必须包含 `base.onnx` 和 `specialist.onnx`。第三个参数提供一个已验证 Scene 资产组；脚本复核 source weight、ONNX、AIPP、OM 和 ATC log 后才生成新的 `build-manifest.json`。

构建产物至少包含：

```text
base_detector.om
incremental_detector.om
atc_base_detector.log
atc_incremental_detector.log
build-manifest.json
```

## 8. 候选物化

```bash
/usr/local/miniconda3/envs/agileagent/bin/python \
  tools/112_materialize_ascend_yolo26_candidate.py \
  --base-om /path/to/base_detector.om \
  --specialist-om /path/to/incremental_detector.om \
  --context-om /path/to/scene_sensor_net.om \
  --build-manifest /path/to/build-manifest.json \
  --report-root /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/reports \
  --output /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/candidate-8502.yaml \
  --output-registry /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/generations.json
```

生成器验证：

- layout、输入和输出 shape；
- Base `0–3` 与 Specialist `4–5` 映射；
- 三个 OM 与构建清单身份；
-真实 context model；
- 双证据门控的完整字段；
- 正式 `8501` 与候选 `8502` 的端口边界；
- Base/New/KRR/FPS 满分门槛。

候选始终是 `validated:false`、`validation_candidate:true`。

## 9. 评分与阈值搜索

单候选：

```bash
./scripts/run_ascend310b_score_gate.sh \
  /path/to/candidate-8502.yaml \
  /path/to/mixed-images \
  /path/to/mixed-test.txt \
  /path/to/base-test.txt \
  /path/to/output
```

score gate 顺序：

1. 确认正式 `8501 ready` 且 `8502` 空闲；
2. 核对 CANN、PNG 和 build manifest；
3. 在无标签阶段冻结全部预测；
4. 读取标签并计算 Base mAP50、New-mAP50、KRR 与诊断；
5. 启动候选 HTTP 服务；
6. 预热 30 次；
7. 执行三轮 20 图 batch；
8. 精确停止自己启动的 `8502` 进程；
9. 再次确认正式 `8501 ready`。

约束搜索来自方法配置：

```text
selection split: mixed_dev_only
objective: minimize new-class false activation
constraints: Base>=0.80, New>=0.60, KRR>=0.95
dimensions: per-class thresholds, scene soft penalties,
            Base/Specialist logit calibration,
            conflict IoU/margin and overlap geometry
evaluated: 5,476; passing: 4,467
```

选择顺序是先满足三项精度约束，再最小化新类误激活，然后依次最大化 New-mAP50 和 Full-mAP50。首次 lock 暴露 Base 记录来源标识问题后，只修复来源标识使原 dev 候选校准在真实 OM 路径生效，再使用同一冻结候选完成复验。

## 10. 正式 release 与提升

```bash
/usr/local/miniconda3/envs/agileagent/bin/python \
  tools/111_promote_ascend_full_score_release.py \
  --candidate-config /path/to/candidate-8502.yaml \
  --score /path/to/score.json \
  --benchmark /path/to/benchmark.json \
  --repeat-benchmark /path/to/benchmark-repeat-1.json \
  --release-root /home/HwHiAiUser/agileagent/releases/NEW_RELEASE_ID \
  --internal-port 18501
```

提升工具重新验证四项门槛、预测冻结、增量数据隔离、Base 冻结以及三组资产身份。不得手工把候选 YAML 的 `validated` 改为 true。

随后安装主/回滚/路由 unit：

```bash
sudo /home/HwHiAiUser/agileagent/releases/NEW_RELEASE_ID/src/scripts/install_ascend310b_primary_services.sh \
  /home/HwHiAiUser/agileagent/releases/NEW_RELEASE_ID \
  /home/HwHiAiUser/agileagent/releases/ROLLBACK_RELEASE \
  18501
```

正式验收必须从公共 `8501` 重新执行两轮 `30 + 3×20`，不能只引用候选端口结果。

## 11. 当前证据位置

```text
models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/
├── configs/agent_pipeline_ascend310b.yaml
├── om/
│   ├── base_detector.om
│   ├── incremental_detector.om
│   └── scene_sensor_net.om
├── provenance/
│   ├── source-build-manifest.json
│   ├── release-build-manifest.json
│   ├── full_score_method.yaml
│   └── generations.json
├── validation/
│   ├── score.json
│   ├── score-dev.json
│   ├── score-post-promotion.json
│   ├── benchmark.json
│   ├── benchmark-repeat-1.json
│   ├── benchmark-post-promotion.json
│   ├── frozen-predictions-post-promotion.jsonl
│   ├── runtime-calibration-search.json
│   └── validation-summary.json
└── release.json
```

仓库不包含竞赛原始图像或标签。仅部署、完整性检查和查看既有报告不需要数据集；重新测 FPS 需要至少 20 张契约 PNG，重新计算精度需要同版 89 图和标签。

## 12. 验收清单

- `Base mAP50 >= 0.80`
- `New-mAP50 >= 0.60`
- `KRR >= 0.95`
- 三轮 20 图 batch 中位 `FPS >= 30`
- `validated:true` 且 `validation_candidate:false`
- `model_layout: independent_yolo26_e2e_v1`
- `context_mode: model`
- 六类映射与固定 owner 正确
- 在线门控不读取标签或文件名
- 公共 `8501` 和内部 `18501` 均 ready
- `8502` 正式状态下空闲
- 三个 systemd unit active
- 公共入口完成部署后 FPS 复验

## 13. 板端离线增量扩展

正式三-OM release 同时作为端侧增量演示的冻结父代。现场已有两类使用一条命令重放 `4→4+1→4+2`：

```bash
./scripts/run_ascend310b_incremental_demo.sh \
  /path/to/datasets_r2_inc_train
```

流水线在独立 `torch_npu` 环境中更新每类 8 参数 Adapter，在 production PyACL/CANN 环境中完成冻结候选、mixed lock、ONNX/OM、ACL 数值和完整图像链路 FPS 验收。胜出候选写入独立演示配置，正式三-OM release 继续作为父代和默认启动身份。

2026-08-26 `board-full-check-v6` 的 Base/New/KRR/Full-mAP50 为 `0.816663 / 0.624935 / 1.000000 / 0.726497`，三轮完整链路为 `39.05 / 38.70 / 37.92 FPS`，Adapter OM 最大绝对误差 `5.96e-08`，候选状态 `accepted`。数据协议、冷/热态时间和演示 CLI 入口见 [`ascend-310b-offline-incremental-demo.md`](ascend-310b-offline-incremental-demo.md)。
