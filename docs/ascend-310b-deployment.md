# Ascend 310B 部署实现

截至 2026-08-23，Ascend310B1 production 是 4+2 独立 YOLO26s 三-OM release：

```text
/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
```

它已经完成 ATC 转换、候选评分、独立复跑、正式提升以及公共 `8501` 两轮部署后复验。旧三-OM listener 仍物理监听 `8501`，只承担即时回滚；公共请求由一条精确 loopback NAT 规则进入 `18501` 主实例，`8502` 保留给后续候选。

## 正式运行结构

```text
640×512 PNG
  -> bounded multipart + DVPP encoded preprocessing
  -> 并发提交 Scene-SensorNet 与四类 Base YOLO26s
  -> 收集 Scene 概率和 Base 检测
  -> air >= 0.5 且 Base 检出 small_aircraft
       是：跳过二类增量专家
       否：执行/收集二类增量专家
  -> 固定类别 owner、冲突仲裁、class-aware NMS
  -> 六类 JSON 与审计事件
```

在线门控只读取 `scene_probabilities` 与 `base_detections`，不读取文件名、数据划分或真值标签。Base 固定负责全局类 0–3，增量专家固定负责全局类 4–5。

| 组件 | 类别或输出 | Ascend 契约 |
| --- | --- | --- |
| Base YOLO26s | soldier、small_aircraft、warship、tank | uint8 NHWC `[1,608,736,3]`；E2E `[1,300,6]` |
| Incremental YOLO26s | patrol_boat、armored_vehicle | uint8 NHWC `[1,608,736,3]`；E2E `[1,300,6]` |
| Scene-SensorNet | IR/SAR 与 air/forest/sea/urban 概率 | uint8 NHWC `[1,160,160,3]` |

两个检测 OM 的 ATC 输入声明为 NCHW `images:1,3,608,736`，AIPP 接收 NHWC uint8。六类活动阈值与计分请求阈值均为 `0.10`。

实现入口：

- 编排与门控：`fair_agent/modules/web_inference.py`
- ACL、DVPP 和统一 enqueue：`fair_agent/backends/ascend_acl.py`
- Web API：`fair_agent/web/app.py`
- 机器可读方法：`configs/ascend310b/full_score_method.yaml`
- 正式板端配置：`configs/agent_pipeline_ascend310b.yaml`

## 正式拓扑

| 职责 | 地址 | 当前状态 |
| --- | --- | --- |
| 公共入口 | `127.0.0.1:8501` | 精确路由到主实例 |
| 4+2 主实例 | `127.0.0.1:18501` | `independent_yolo26_e2e_v1` |
| 回滚 listener | 物理监听 `127.0.0.1:8501` | 旧三-OM 服务，正常被路由旁路 |
| 候选 | `127.0.0.1:8502` | 正式状态下无 listener |

三个 unit：

```text
agileagent-ascend310b-main.service
agileagent-ascend310b-rollback.service
agileagent-ascend310b-route.service
```

正式提升不终止回滚 listener。删除唯一带 `AGILE_AGENT_ASCEND310B_PRIMARY` comment 的规则后，新连接会直接进入旧 listener。

## 从仓库零训练物化

可版本化模型包：

```text
models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/
```

包内包含三个 OM、对应 source checkpoint/ONNX/AIPP/ATC 日志、构建清单、正式配置、冻结预测、精度报告、候选性能报告以及两轮公共 `8501` 性能报告。单文件均低于 GitHub 100 MB，不需要 Git LFS。

在已安装 CANN `7.0.RC1` 和命名环境 `agileagent` 的板端执行：

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh
```

物化脚本只做包完整性检查、源码与资产复制以及 `tools/95_verify_ascend_release.py --require-validation`。它不训练、不导出 ONNX、不运行 ATC、不安装依赖，也不修改服务或端口。目标已存在时只读复核：

```bash
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

新板没有回滚 listener 时可以直接监听公共端口：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

已有正式回滚 listener 的设备使用后续 systemd 拓扑，不直接占用 `8501`。

## 安装或更新双实例服务

以 root 权限执行安装器，第二个参数必须是已经验证且能够在 `8501` ready 的回滚 release：

```bash
PRIMARY=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
ROLLBACK=/home/HwHiAiUser/agileagent/releases/ROLLBACK_RELEASE

sudo "$PRIMARY/src/scripts/install_ascend310b_primary_services.sh" \
  "$PRIMARY" "$ROLLBACK" 18501
```

安装器的顺序是：

1. 移除旧主线路由；
2. 启动并等待回滚 listener ready；
3. 启动并等待 `18501` 新主实例 ready；
4. 安装并应用精确路由；
5. 通过公共 `8501` 再次验证主实例身份。

任一步失败都会移除主线路由并恢复回滚 listener。安装器不会占用 `8502`。

## 健康检查与服务复核

```bash
systemctl is-active agileagent-ascend310b-main.service
systemctl is-active agileagent-ascend310b-rollback.service
systemctl is-active agileagent-ascend310b-route.service

curl -fsS http://127.0.0.1:8501/api/health
curl -fsS http://127.0.0.1:18501/api/health
ss -H -ltn 'sport = :8501 or sport = :18501 or sport = :8502'
sudo /usr/local/sbin/agileagent-ascend310b-primary-route status 18501
```

公共 `8501` 与内部 `18501` 的健康响应都必须包含：

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

`8501` 和 `18501` 应各有物理 listener；`8502` 应为空。不能仅凭进程存在判定部署成功。

真实 PNG 冒烟：

```bash
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

## 即时回滚与重新提升

回滚：

```bash
sudo /usr/local/sbin/agileagent-ascend310b-primary-route remove 18501
curl -fsS http://127.0.0.1:8501/api/health
```

重新提升：

```bash
sudo /usr/local/sbin/agileagent-ascend310b-primary-route apply 18501
curl -fsS http://127.0.0.1:8501/api/health
```

路由脚本同时检查公共 listener、主实例 health 与支持的正式 layout。回滚之后公共 health 应显示旧 release 身份；重新提升后必须恢复 4+2 正式身份。

## 当前正式指标

| 指标 | 实测 | 满分门槛 |
| --- | ---: | ---: |
| Base mAP50 | `0.8256706047` | ≥0.80 |
| New-mAP50 | `0.6188591828` | ≥0.60 |
| KRR | `1.0000000000` | ≥0.95 |
| 候选 batch 中位 FPS | `39.3468 / 39.4244` | ≥30 |
| 公共 `8501` 中位 FPS | `39.5726 / 39.5883` | ≥30 |

Full-mAP50 为 `0.7249274787`。公共 `8501` 两轮逐轮 FPS：

- `39.5726 / 39.5804 / 39.3933`
- `39.5883 / 39.5023 / 39.6668`

诊断项为 precision `0.677551`、recall `0.661111`、误激活率 `0.466667`。误激活率表示 75 张不含新增类的图像中有 35 张至少激活一个新增类；它不属于四项赛题淘汰门槛。

原始证据：

```text
models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/validation/
```

## 重新测量所需数据

| 可用输入 | 能完成的复核 |
| --- | --- |
| 仅仓库 | 完整性、release verifier、配置和历史报告 |
| 至少 20 张契约 PNG | 30 次预热与三轮 20 图 batch FPS |
| 89 图及 YOLO 标签 | 重新冻结预测并计算 Base/New/KRR/Full-mAP50 |
| 同版 89 图标签 | 对包内 `frozen-predictions.jsonl` 重新计分 |

输入契约为 `640×512`、8-bit 灰度/RGB/RGBA PNG。

## 构建新的 4+2 候选

以下流程用于新数据或新权重，不是部署当前包的前置条件。

### 1. 构建两个检测 OM

准备 `ONNX_DIR/base.onnx` 与 `ONNX_DIR/specialist.onnx`，两者必须是 `608×736`、max_det 300 的 YOLO26 E2E ONNX。Scene 资产从一个已验证 release 的构建清单复用：

```bash
./scripts/build_ascend_yolo26_e2e_oms.sh \
  /path/to/onnx-dir \
  /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/build \
  /home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2/provenance/release-build-manifest.json
```

脚本固定 `Ascend310B1`、`mixed_float16`、检测输入 `images:1,3,608,736`，并验证 Base、Specialist、Scene 三组来源资产。

### 2. 物化隔离候选

```bash
/usr/local/miniconda3/envs/agileagent/bin/python \
  tools/112_materialize_ascend_yolo26_candidate.py \
  --base-om /path/to/build/base_detector.om \
  --specialist-om /path/to/build/incremental_detector.om \
  --context-om /path/to/scene_sensor_net.om \
  --build-manifest /path/to/build/build-manifest.json \
  --report-root /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/reports \
  --output /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/candidate-8502.yaml \
  --output-registry /home/HwHiAiUser/agileagent/candidates/CANDIDATE_ID/generations.json
```

生成器强制 `8502`、`validated:false`、真实 context model、固定类别映射和双证据执行门控。

### 3. 运行四项 score gate

```bash
./scripts/run_ascend310b_score_gate.sh \
  /path/to/candidate-8502.yaml \
  /path/to/mixed-images \
  /path/to/mixed-test.txt \
  /path/to/base-test.txt \
  /path/to/score-output
```

score gate 在执行前后都要求正式 `8501 ready`，并要求 `8502` 初始为空。它先冻结无标签预测，再读取标签评分，最后执行 30 次预热和三轮 20 图 batch。

### 4. 物化与提升胜出 release

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

只有 Base mAP50、New-mAP50、KRR、batch FPS 与有效性前置条件全部通过，工具才生成 `validated:true` 的不可变 release。随后使用 systemd 安装器提升，并必须再从公共 `8501` 复跑 FPS。

## 监测

```bash
npu-smi info
curl -fsS http://127.0.0.1:8501/api/health
journalctl -u agileagent-ascend310b-main.service -n 100 --no-pager
journalctl -u agileagent-ascend310b-route.service -n 100 --no-pager
```

请求日志记录 trace、generation、执行模型、门控结果、检测数和分段耗时。正式验收应同时保留 score、候选 benchmark、公共入口 benchmark、健康响应与 unit 状态。
