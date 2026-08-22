# Ascend 310B 4+2 满分方法与复现手册

本文描述当前正式 4+2 Ascend310B1 方法。唯一机器可读契约是 `configs/ascend310b/full_score_method.yaml`，正式发布包是：

```text
models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/
```

旧 `20260816-full-score-1493b04` 共享双头 3+1 release 只保留为历史参考，不是当前 production。

## 1. 当前结论

| 赛题硬指标 | Ascend310B1 实测 | 满分门槛 |
| --- | ---: | ---: |
| Base mAP50 | `0.8256706047` | ≥0.80 |
| New-mAP50 | `0.6188591828` | ≥0.60 |
| KRR | `1.0000000000` | ≥0.95 |
| 候选 batch 中位 FPS | `39.3468 / 39.4244` | ≥30 |
| 公共 `8501` 中位 FPS | `39.5726 / 39.5883` | ≥30 |

Full-mAP50 为 `0.7249274787`。旧类 mAP50 增量前后均为 `0.7779616266`，冻结旧类预测完全等价。

公共 `8501` 两轮逐轮 FPS：

```text
39.5726 / 39.5804 / 39.3933
39.5883 / 39.5023 / 39.6668
```

precision `0.677551`、recall `0.661111`、误激活率 `0.466667` 是非阻断诊断。75 张不含新增类的图像中有 35 张至少误激活一个新增类。

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

## 4. 为什么没有照搬 YOLO11 优化

历史 YOLO11 方案依赖共享 backbone 的双 raw head、固定中性 context 和 Python raw-head 解码。YOLO26 的正式方案保留了其中可迁移的系统级思路，但没有复用模型结构：

| 历史思路 | YOLO26 处理 |
| --- | --- |
| 减少重复预处理 | 继续使用 bounded multipart、DVPP 与图像复制消除 |
| 并发执行 | Base 与 Scene 先并发提交 |
| 合并检测计算 | 不强行拼接两个已独立训练的 YOLO26 权重；保持固定 owner 的两个 E2E OM |
| 固定中性场景 | 改为真实 Scene-SensorNet，支持已知场景对新旧类的影响 |
| 每图执行全部模型 | 引入只依赖模型输出的双证据执行门控 |
| raw head Python 解码 | 使用 E2E `[1,300,6]` 输出，降低后处理成本 |

因此当前优化是“独立 E2E 检测器 + 真实 context + 内容执行门控”，不是历史共享双头方案的换名版本。

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

当前计分请求阈值以及六类活动阈值都为 `0.10`。阈值在候选 dev 上选择并冻结，lock/test 只评分。更换模型或数据集后必须重新搜索，不得把当前阈值当作通用常数。

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
tools/61_select_scene_sensor.py
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
  /home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2/provenance/release-build-manifest.json
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

阈值矩阵来自方法配置：

```text
new: 0.05, 0.075, 0.10, 0.125, 0.15
old: 0.05, 0.075, 0.10, 0.125, 0.15
```

先固定 old 搜索 new，再固定胜出 new 搜索 old，最后复核头部组合。选择顺序是最小精度余量最大、FPS 波动最小、batch 中位 FPS 最高。四项未同时达标的候选只能标记为 intermediate。

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
models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/
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
│   ├── benchmark.json
│   ├── benchmark-repeat-1.json
│   ├── benchmark-post-promotion.json
│   ├── benchmark-post-promotion-repeat.json
│   └── frozen-predictions.jsonl
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
