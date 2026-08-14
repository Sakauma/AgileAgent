# Ascend 310B 工程记录

本文记录截至 2026-08-15 已完成的 Ascend 310B 实现、部署、精度复核和性能测量。

## 板端环境

| 项目 | 已确认值 |
| --- | --- |
| 开发套件 | Atlas 200I DK A2，`aarch64` |
| 芯片 | Ascend310B1，NPU Health `OK` |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1` |
| Conda | Miniconda `23.5.0` |
| Python 环境 | `/usr/local/miniconda3/envs/agileagent`，Python `3.9.2` |
| 正式 release | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` |
| 服务 | `127.0.0.1:8501`，health `ready` |
| 驻留快照 | NPU 内存 `9479 / 11577 MB`，温度 `61°C` |

## 已实现链路

- `fair_agent/backends/ascend_acl.py` 使用 PyACL/AscendCL 加载和执行三个 OM；
- `fair_agent/modules/web_inference.py` 编排 Base、Incremental 和 Scene 三模型；
- `fair_agent/web/app.py` 提供 health、单图检测和批量检测 API；
- `configs/agent_pipeline_ascend310b.yaml` 登记设备、CANN、执行模式、OM 路径和 SHA256；
- `scripts/start_agent_ascend310b.sh` 使用命名环境 `agileagent` 启动 Uvicorn；
- 模型输出经过全局类别映射、逐类阈值、场景软证据、冲突仲裁和 class-aware NMS。

## 正式模型

| 模型 | 输入 | 产物 |
| --- | --- | --- |
| 三类基础检测器 | `1×3×736×896` | `base_detector.om` |
| 增量检测器 | `1×3×512×640` | `incremental_detector.om` |
| Scene-SensorNet | `1×3×160×160` | `scene_sensor_net.om` |

三个 OM 使用 `mixed_float16` 编译，配置记录对应 SHA256。每张图像执行三个模型并生成统一检测结果。

## 正式 release 精度

正式 release 已在 89 张固定混合测试集上完成无标签推理、预测冻结和评分：

| 指标 | 结果 |
| --- | ---: |
| Base mAP50 | `0.819407` |
| New-mAP50 | `0.728761` |
| KRR | `1.000000` |
| 新类 precision | `0.933333` |
| 误激活率 | `0.014286` |

基础目标检测、New-mAP 和 KRR 三项合计取得 `50/50` 精度分档结果。

## 性能测量

| 测量对象 | 样本量 | 已记录结果 |
| --- | ---: | --- |
| 正式 release 完整 89 图 | 89 | 引擎均值 `57.849 ms`、墙钟均值 `71.491 ms`、引擎 `17.29 FPS`、墙钟 `13.99 FPS` |
| 已解码 Agent 核心 | 200 | 均值 `32.148 ms`、P95 `33.193 ms`、`31.11 FPS` |
| AIPP staging multipart PNG API | 1,068 | 均值 `51.203 ms`、P95 `63.9 ms`、`19.53 FPS` |
| DVPP 编码输入测量 | 240 | 均值 `37.124 ms`、P95 `38.154 ms`、`26.94 FPS` |

这些记录分别覆盖完整 89 图运行、已解码核心、真实 multipart PNG API 和编码输入路径。

## Wave 0 / P0 执行记录（2026-08-15）

本轮按照 `ascend-310b-p0-p3-improvement-plan.md` 建立了独立候选，不覆盖正式 release，也未修改 `8501` 服务。候选目录为：

```text
/home/HwHiAiUser/agileagent/candidates/20260814-wave0-p0-7c61f2b
```

### 板端 Git 工作副本同步

板端正式 release 和旧候选最初都是不含 `.git` 的文件快照，不能证明与远程仓库一致。现已在不修改正式 release 的前提下建立独立工作副本：

```text
/home/HwHiAiUser/agileagent/repo
```

板端无法直连 GitHub `443`，因此先在已与远程核对一致的本地仓库生成完整 Git bundle，经 SHA256 校验后在板端 clone；后续提交通过带 prerequisite 的增量 bundle fetch，并以 `--ff-only` 更新。当前工作副本跟踪 `origin/agent/ascend-310b-wave0-p0`，P0 复测提交为 `eac65cc42ed77fa9f7de4468b64089bd8ceb4941`，`origin` 仍为 `https://github.com/Sakauma/AgileAgent.git`，工作树 clean。完整初始 bundle SHA256 为 `f6bffae8d76265de4d9c788febdba1c6b6f757d047d7704306635aad8526b388`。同步前后正式 `8501` 均保持 `ready`。

### 受控构建与门禁

- 三份 AIPP 配置已纳入 `configs/ascend310b/aipp/`；
- `scripts/build_ascend_aipp_oms.sh` 固定 ONNX、SoC、precision、输入 shape，拒绝覆盖，并记录 ATC 命令、日志和全部 SHA256；
- 完整 P0-r3 构建清单 SHA256 为 `a62131586d33ade4090dbf925fb1adca3ad9a852049d1780fe4b990097c3d1d4`；
- 早期 P0-r3 构建清单中的 `git_sha` 错记为不存在于当前仓库的 `7c61f2b4308bc146009df260984a1506a6274737`；实际起点和当时 `origin/main` 均为 `7c61f2b9ec0004aec5a0f3c2ab8858a2f229c5e3`。两者前 7 位碰巧相同，旧候选目录短名未暴露该错误。该清单不得作为可晋级 provenance；后续设备报告从板端 clean Git 工作副本直接记录完整 `HEAD`、分支、远程和工作树状态；
- Base、Specialist、Scene OM SHA256 分别为：
  - `2bc60b224ba3702232f6e35363199ae2b2f3b7382498340a719bf093f80a8851`；
  - `69957129b060295736e9812b459588147f7f0dee7d35b1e600196d077a431b7a`；
  - `c1902bc8f66e8036e70a9ba12a3b91dc5711837408eca819eb02b6afddc1f1a`；
- PyACL 输入契约已复核为 Base `[1,736,896,3] uint8`、Specialist `[1,512,640,3] uint8`、Scene `[1,160,160,3] uint8`；
- 未验收候选必须同时设置 `validation_candidate: true` 和进程级 `AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1`，正式 `validated: true` 仍要求构建清单及 golden、精度、性能报告哈希全部闭环。

### 89 图数值与语义结论

旧比较逻辑按响应顺序配对框，曾把最大坐标差误报为 `120.67 px`。新工具先按类别和 IoU 稳定匹配，再计算数值差。原始 P0-r3 在生产阈值 `0.5` 下的真实结果为：

| 项目 | 原始单级 Scene DVPP | 多级 Scene DVPP |
| --- | ---: | ---: |
| 逐图检测数量/类别不一致 | `1` | `1` |
| Scene 标签不一致 | `18` | `0` |
| Sensor 标签不一致 | `6` | `0` |
| 最大坐标绝对差 | `0.54 px` | `0.54 px` |
| 最大 confidence 绝对差 | `0.003418` | `0.003418` |

设备输入回读证明 Specialist PNGD 输出和 Base resize/letterbox 与 CPU 契约逐字节一致。Scene 单次 `640×512 → 176×176` 双线性缩小缺少 Pillow 下采样抗锯齿：代表性 SAR 图平均像素差 `11.19`、最大差 `88`，从而翻转场景标签。候选增加以下 DVPP 多级缩放后，89 图 Scene/Sensor 标签全部恢复一致：

```text
640×512 → 208×192 → 288×230 → 176×176 → center crop 160×160
```

旧 Scene AIPP V1 归一化在 89 图上造成 `55` 个 Scene 和 `16` 个 Sensor 标签变化，已明确拒绝。Base AIPP 将 `1/255` 下调一个 FP16 ULP 的实验又造成 `15` 张图检测数量变化、最大 confidence 差 `0.061584`，同样明确拒绝。

当前唯一剩余的生产阈值差异是 `ir_r1_base_urban_000149.png`：正式 Base 的 tank confidence 为 `0.500000`，AIPP 为 `0.501465`，导致 `2 → 3` 个框。不得通过修改阈值或后处理容差掩盖该差异。

### 同口径 890 请求性能

两次候选均按 30 次预热、10 轮固定 89 图、单并发、板端回环 HTTP keep-alive 和 `confidence=0.5` 执行：

| 候选 | 服务端均值 | P95 | Engine 均值 | Multipart 解析均值 | P0 `40/42 ms` |
| --- | ---: | ---: | ---: | ---: | --- |
| 原始单级 Scene DVPP | `48.76 ms` | `65.90 ms` | `38.00 ms` | `9.31 ms` | 未通过 |
| 多级 Scene DVPP | `49.27 ms` | `66.71 ms` | `38.13 ms` | `9.69 ms` | 未通过 |
| 多级 Scene DVPP + 有界 multipart 快速解析 (`eac65cc`) | `41.439 ms` | `47.200 ms` | `38.371 ms` | `1.640 ms` | 未通过 |

复测表明 Starlette/python-multipart 通用逐块解析是确定热点：原多级候选上传解析均值/P95 约 `9.69/22.71 ms`；对带明确 `Content-Length` 且不超过 `2 MiB + 64 KiB` 的单图请求启用有界快速解析后降至 `1.640/3.491 ms`，完整 API 均值和 P95 分别改善约 `15.9%` 和 `29.3%`。较大或非标准请求仍走原解析器，公共 multipart 字段保持不变。

细分计时显示 `routing_fusion_ms` 的常态均值约 `0.36 ms`，旧记录中的 `16.64 ms` 已不再是当前候选热点。P0 最终受 Engine P95 `42.709 ms`、完整 API `41.439/47.200 ms` 和 Base 阈值边界差异共同阻断，继续微调 HTTP 层不能同时闭合精度与 `40/42 ms` 门禁。

### 晋级结论

P0 当前状态为 **拒绝晋级**：多级 Scene DVPP 和 multipart 快速解析保留为候选实现，但 Base 阈值边界、构建 provenance 和完整 API 性能门禁仍未通过。候选配置保持 `validated: false`，`8502` 已停止，正式 `8501` 保持 `ready`。后续 P1/P2/P3 只能从上一已验收正式版本建立独立实验候选，不得把该 P0 候选视为已晋级基线。

关键板端证据均保存在候选的 `validation/` 目录：

| 报告 | SHA256 |
| --- | --- |
| `p0-0.5-iou-alignment.json` | `e30fcd2180c47b0a6c79922fb8fdc9f51034b49038ad24b09070810c8e537319` |
| `p0-dvpp-input-alignment-multistage.json` | `545eebe7c91550b426fc3ab8987de5a0f12a9fcdf7fb731ded7cfff01d28b89d` |
| `p0-multistage-0.5-alignment.json` | `7b12b0ded32a6d3940cef04723f2700563aab24521881c9a3dd9375552497ec7` |
| `p0-multistage-890-characterization.json` | `18ec053814007ecea59297f1a3c2b31f3087511365f670529d8aa62a6fd03e61` |
| `p0-fast-multipart-eac65cc-890-characterization.json` | `756a537c54576bf0e665cad46de8b00b164cf6e7871fef76709126034bd247c0` |

## 环境迁移记录

板端 Python 环境已迁移到命名环境 `agileagent`。迁移前后使用固定 PNG 执行响应语义对照，检测数量、类别、框和置信度保持一致；切换后 health 返回 `ready`。

命名环境确认项：

- Python `3.9.2`；
- 181 个 Conda 记录；
- 176 个 `pip freeze --all` 条目；
- PyACL 与核心模块导入成功；
- 三个 OM 加载成功；
- 真实 PNG 推理成功。

## 自动验证

```bash
python -m pytest -q
python scripts/verify_release.py
```

当前本地 WSL 仓库既有 `.venv` 全量回归为 `223 passed, 1 skipped`；板端 Ascend、对齐和发布门禁定向回归为 `20 passed`。正式 release 的既有发布校验状态仍为 `passed`，P0 候选因上述门禁失败保持未验收。

## 运行态检查

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
npu-smi info
```
