<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 当前工程状态

本文记录截至 2026-08-14 的 Ascend 310B 工程状态，并严格区分仓库实现、正式板端只读复核、已批准精度、不同测量边界的性能，以及尚未进入正式配置的候选方案。结论是：板端三模型推理和前三项官方精度指标已经跑通，可用于功能演示与继续优化；端到端 FPS 尚未达到30 FPS的10分满分线，长期稳定性和可复现发布证据也未闭环，不能签署为最终性能验收通过。

## 总览

| 层级 | 当前结论 |
|---|---|
| 仓库实现 | 已具备 AscendCL/OM 后端、三模型编排、Web API、启动脚本、资产哈希校验和可选 DVPP 编码预处理。 |
| 正式板端部署 | 正式 release 为 `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2`，服务绑定 `127.0.0.1:8501`；部署记录显示 release verification 已通过，`/api/health` 为 `ready`。 |
| 正式执行语义 | 每张无标签图像都执行 Base、Incremental 和 Scene；不按标签、文件名或测试清单分流，也没有 CPU、CUDA 或 PyTorch 模型推理回退。 |
| 正式 release 精度 | Base mAP50 `0.819407`、New mAP50 `0.728761`、KRR `1.0`，前三项官方精度为 `50/50`；新类 precision `0.933333`、误激活率 `0.014286` 两项内部门禁也通过。 |
| AIPP staging 候选精度 | Base mAP50 `0.819415`、New mAP50 `0.728761`、KRR `1.0`，前三项官方精度为 `50/50`；两项内部门禁也通过，但没有切换正式 release。 |
| 正式 release 89图运行 | 平均引擎耗时 `57.849 ms/图`，平均墙钟 `71.491 ms/图`；这是完整89图评测记录，不是 HTTP 压测。 |
| 已解码 Agent 核心 | 200 个样本：均值 `32.148 ms`、P95 `33.193 ms`、`31.11 FPS`；该结果不含 multipart 解析和 PNG 解码。 |
| 真实 PNG API 候选 | AIPP staging 服务的 1,068 次 multipart PNG 请求：服务端均值 `51.203 ms`、P95 `63.9 ms`、`19.53 FPS`。按官方效率分档落入 `FPS ≥ 10` 的4分档，尚未达到20 FPS的7分档或30 FPS的10分档。 |
| DVPP 候选 | 240 个样本：均值 `37.124 ms`、P95 `38.154 ms`、`26.94 FPS`；仍为实验候选，默认关闭。 |
| ATC Base 候选 | P95 仅改善约 `0.234 ms`，但 89 张中有 2 张的检测数量发生变化，已拒绝。 |

本轮审查开始前已执行 `git fetch --prune origin`：本机 `main` 与 `origin/main` 均为 `7d3d10911b59a22ef7a348edf7efd55a08007dfc`，ahead/behind 为 `0/0`，因此审查基线与远端完全同步。正式 release 目录名是仓库中的旧提交 `212705a26d4414eff4e00604ce37c54d2ae729b2`，但板端 release 源码与该审查基线并非逐文件一致；目录名本身也不能证明板端文件与任一 checkout 完全相同。

板端精度、性能和实时健康状态来自本轮 SSH 只读复核及既有部署报告。原始板端日志尚未纳入仓库，不能仅靠当前 checkout 复现。审查时另执行了一次真实 PNG smoke：返回4个检测，Base、Incremental、Scene 三模型均执行，引擎 `62.985 ms`、系统 `78.3 ms`，随后 health 仍为 `ready`；该单次值只证明链路可运行，不作为性能结论。

<!-- VERIFY: 交付或复验时应在目标板重新确认 /api/health 仍为 ready，并归档对应板端日志；仓库 checkout 无法证明实时运行态。 -->

## 工程成熟度评估

| 维度 | 评价 | 依据与边界 |
|---|---|---|
| 在线架构与功能链路 | 可用 | Web/CLI、代际解析、三模型编排、软上下文、融合与审计路径均有实现和自动化契约测试。 |
| x86 训练与模拟增量 | 已形成受控基线 | 750张固定3+1模拟划分、权重/阈值/指标/哈希已固化；不代表官方隐藏数据成绩。 |
| 310B 运行时 | 板端可用 | PyACL/AscendCL 加载三个 OM，真实 PNG 推理与 health 检查通过，无模型推理回退。 |
| 310B 精度 | 当前正式 OM 为前三项官方精度满分档 | Base、New、KRR 合计 `50/50`，两项内部质量门禁也通过；任何 OM、AIPP、DVPP 或 ATC 变更都需独立重跑。 |
| 310B 性能 | 未达到效率满分档 | 正式89图和 staging API 证据均未达到端到端30 FPS；只有不含 multipart/PNG 的核心候选超过30 FPS，不能获得官方端到端FPS分数。 |
| 自动化测试与 CI | 部分充分 | 共收集221项，本轮按板端范围执行33项并通过；无覆盖率门槛、静态检查，且最小 CI 依赖未显式声明 `torch`。 |
| 发布可复现性 | 主要风险 | 板端 source/OM/config/报告未由一个提交内的 manifest 统一绑定，正式 release 与当前 main 也不是同一源码快照。 |
| 运维与长期运行 | 待补齐 | 服务可启动/停止并报告健康，但一小时稳定性、热降频、持续内存和原子回滚演练尚未签字。 |

### 板端环境快照

| 项目 | 只读观测 |
|---|---|
| 开发套件 / 架构 | Atlas 200I DK A2，`aarch64` |
| 芯片 | Ascend310B1，NPU Health `OK` |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1`；本轮未升级、替换或混装 CANN、驱动和固件 |
| NPU 内存 / 温度 | 空闲观测为 `6905 / 11577 MB`、`63°C`；这是单次快照，不是峰值或稳定性报告 |
| 正式服务 | `127.0.0.1:8501`，health `ready` |

## 仓库实现与关键模块

- [`configs/agent_pipeline_ascend310b.yaml`](../configs/agent_pipeline_ascend310b.yaml) 固定 `ascend_acl`、`batch_size: 1`、正式 OM 路径与 SHA256，并将 `encoded_preprocessing` 默认设为 `cpu`。这里的 CPU 只用于编码图像解码和预处理，不是模型推理回退。
- [`fair_agent/backends/ascend_acl.py`](../fair_agent/backends/ascend_acl.py) 负责 PyACL/AscendCL 运行时、OM 加载、静态输入契约、异步 stream、后处理和实验性 DVPP/VPC 路径；OM 缺失、哈希不匹配或契约不符时直接失败。
- [`fair_agent/modules/web_inference.py`](../fair_agent/modules/web_inference.py) 并行编排 Scene、Base 和 class-incremental specialist。当前增量模型是每图必跑的类别所有者；Scene 只影响软阈值，不进行场景硬路由。
- [`fair_agent/web/app.py`](../fair_agent/web/app.py) 提供 `GET /api/health`、`POST /api/detect` 和 `POST /api/batch`。检测 API 始终传入 `auto`，不会接受客户端字段来切换正式增量链路。
- [`scripts/start_agent_ascend310b.sh`](../scripts/start_agent_ascend310b.sh) 从正式 release 的隔离环境启动 Uvicorn，并固定监听 `127.0.0.1:8501`。

最小运行态检查示例：

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" http://127.0.0.1:8501/api/detect
```

## 官方四项指标与当前自测档位

性能指标共60分，四项分别为基础目标检测30分、New-mAP 10分、KRR 10分、310B端到端FPS 10分。赛题原文使用 `mAP`；下表中的精度实测来自仓库当前 `mAP@0.5` 实现，最终仍应以官方评分程序的IoU口径为准。

| 官方指标 | 完整分档 | 正式 release 证据 | 正式 release 自测档位 | AIPP staging 证据 | AIPP staging 自测档位 |
|---|---|---:|---|---:|---:|
| 基础目标检测 mAP（30分） | `≥0.80:30`；`≥0.70:25`；`≥0.65:20`；`≥0.60:15`；`≥0.50:10`；`≥0.40:5`；否则`0` | `0.819407` | `30` | `0.819415` | `30` |
| New-mAP（10分） | `≥0.60:10`；`≥0.50:7`；`≥0.40:4`；否则`0` | `0.728761` | `10` | `0.728761` | `10` |
| KRR（10分） | `≥0.95:10`；`≥0.90:7`；`≥0.80:4`；否则`0` | `1.000000` | `10` | `1.000000` | `10` |
| 端到端FPS（10分） | `≥30:10`；`≥20:7`；`≥10:4`；否则`0` | 89图墙钟约 `13.99 FPS`，但不是HTTP/官方压测 | 暂不正式计分；若该边界被接受则为`4` | 真实 multipart PNG API `19.53 FPS` | `4` |

因此，正式 release 当前可确认的前三项精度为 `50/50`；由于尚无同一正式 release 的官方口径端到端压测，总分不能写成正式成绩。AIPP staging 若仅按现有自测证据估算为 `30+10+10+4=54/60`，但它不是 production，也不是官方数据集成绩。`19.53 FPS` 严格低于20，只能落入4分档；已解码核心的 `31.11 FPS` 省略了 multipart 与PNG处理，不能替代第四项。

短期性能优化若把真实API提升到 `≥20 FPS` 且前三项精度不变，自测档位可从 `54/60` 提升到 `57/60`；达到 `≥30 FPS` 才能进入 `60/60` 满分档。内部 `≥36 FPS`、P95 `≤33.33 ms` 仍只是为温度和抖动预留余量的工程目标。

## 三项官方精度与两项内部质量门禁

| 指标 | 正式 release | AIPP staging 候选 |
|---|---:|---:|
| Base mAP50 | `0.819407` | `0.819415` |
| New mAP50 | `0.728761` | `0.728761` |
| KRR | `1.0` | `1.0` |
| 新类 precision | `0.933333` | `0.933333` |
| 误激活率 | `0.014286` | `0.014286` |

两套结果都通过“Base、New、KRR三项官方精度 + precision、误激活率两项内部质量门禁”，但只有左列属于当前正式 release；右列用于说明 AIPP 候选未破坏精度和质量，不能据此声称 production 已切换。precision 与误激活率不增加官方分数，第四项官方FPS也不在本表中。两者不应与 [`models/generations.json`](../models/generations.json) 中较早的仓库基线混写为同一次评测。

## 性能边界

| 测量对象 | 样本量 | 均值 | P95 | FPS | 判定 |
|---|---:|---:|---:|---:|---|
| 正式 release 完整89图 | 89 | 引擎 `57.849 ms`；墙钟 `71.491 ms` | 未记录 | 引擎约 `17.29`；墙钟约 `13.99` | 不是HTTP/官方压测；若官方接受墙钟边界则落入4分档。 |
| 已解码 Agent 核心 | 200 | `32.148 ms` | `33.193 ms` | `31.11` | 非端到端，不能据此领取FPS 10分。 |
| DVPP 编码输入候选 | 240 | `37.124 ms` | `38.154 ms` | `26.94` | 非完整API且默认关闭，不能据此领取FPS 7分。 |
| AIPP staging 真实 multipart PNG API | 1,068 | `51.203 ms` | `63.9 ms` | `19.53` | staging自测落入4分档；低于20 FPS且未切换正式服务。 |

这些数字不可直接互换：真实 API 还包含 multipart 解析、PNG 处理、排队、编排和响应生成；已解码核心数据不能用于宣称端到端达标，staging 数字也不能写成正式 release 数字。

## 候选方案决策

### DVPP

DVPP preflight 的 12/12 样本保持了检测数量、类别序列和 context 标签，但尚未执行完整 89 图精度门禁。因此 `encoded_preprocessing: dvpp` 只能作为候选，正式配置继续使用默认值 `cpu`。

### ATC Base

被测 ATC Base 候选的 P95 仅改善约 `0.234 ms`，同时在 89 张图中的 2 张改变了检测数量。该候选已拒绝，不得以微小延迟收益替换正式 Base OM。

## 测试与验证范围

本轮在 WSL Python `3.10.19` 环境中成功收集 `221` 个 pytest 用例。按用户指定的板端范围执行以下集合，实测为 `33 passed, 1 warning`：

```bash
python -m pytest -q \
  tests/test_ascend_acl.py \
  tests/test_web_ui_flow.py \
  tests/test_runtime_maturity.py::test_static_release_verification_passes
```

其中包括 Ascend 张量/后处理与 DVPP 输入契约、Web/API 自动模式与健康接口，以及静态 release verification。唯一警告来自 Starlette `TestClient` 与 httpx 兼容层的上游弃用提示。它们验证代码契约，不等同于目标板 CANN 执行、89 图完整板端精度或真实 multipart 压测；x86/CUDA、训练和完整工作台用例本轮没有全部执行。

`python scripts/verify_release.py` 也已通过，但其语义需要特别说明：当前 [`configs/functional_models.yaml`](../configs/functional_models.yaml) 仍把三个功能模型的 `ascend_310b` 标为 `false`，所以校验输出中的 `all_ascend_310b_ready` 仍为 `false`，阻塞项仍包含 `ascend_310b_not_ready`。因此“脚本通过”只证明默认 x86 配置、受保护资产和公开证据一致，不能证明板端 release 已完成验收；这也是当前仓库元数据与板端事实尚未统一的直接证据。

## 已知风险

- 当前正式 release 的完整89图平均引擎耗时为 `57.849 ms/图`；性能候选的真实 multipart PNG API 只有 `19.53 FPS`，仅处于4分档，距离20 FPS的7分档约 `0.47 FPS`、距离30 FPS的10分档约 `10.47 FPS`。
- 板端基准、精度和健康值是部署记录；缺少随仓库保存的原始请求明细、环境清单和完整日志，复现性依赖板端证据归档。
- `configs/functional_models.yaml` 与静态发布校验仍保留板前 `ascend_310b: false` 状态，和已能运行的板端 release 不一致；在证据绑定完成前不应直接改为 `true`。
- 两份主配置的 `policies.end_device` 仍写着 `paused_until_ascend_board_ready`；Ascend 配置的 `refresh_blackboard.required_artifacts` 还指向默认 x86 配置。这不会阻止 PyACL 服务启动，但会让决策展示和证据依赖保持旧板前语义。
- CI 安装的 `.[dev,workbench]` 未显式声明 `torch`，而 `tests/test_strict_incremental.py` 在收集阶段直接导入它；干净 runner 的测试可复现性存在隐式依赖风险。
- DVPP 仅通过 12 图 preflight，尚缺 89 图完整精度门禁和正式 API 级复测。
- ATC 候选已出现逐图输出变化；后续任何 OM、AIPP、ATC 参数或 CANN 版本变更都必须重新执行完整精度验收。
- 正式 release 路径与当前仓库 HEAD 不同，若不归档 source/OM/config 哈希清单，容易产生代码与测量结果错配。

## 下一步优先级

1. 将正式板端的源码、配置、三个 OM、CANN/ATC 版本、SHA256、精度报告和性能原始日志绑定到同一 release 清单。
2. 用该清单更新功能模型注册表、`policies.end_device`、Ascend `refresh_blackboard.required_artifacts` 与发布校验语义，消除旧板前元数据和实际部署之间的分裂；证据不齐时保持 fail-closed。
3. 对 DVPP 候选执行完整 89 图精度门禁；只有逐图与批准口径通过后，才进行同规模真实 multipart PNG API 复测。
4. 以真实 API 的 P95 和 FPS 为优化目标，先稳定跨过 `20 FPS` 的7分线，再争取 `30 FPS` 的10分线和内部 `36 FPS` 余量线；不得用已解码核心 `31.11 FPS` 替代端到端结果。
5. 保持当前 ATC Base 候选为拒绝状态；仅在 89 图输出等价且完整精度门禁通过时重新评估。
6. 修复测试环境对未声明 `torch` 的依赖，并为关键配置、后端和 Web 链路建立最小覆盖率/静态检查门禁。
7. 每次正式 release 变更后重跑板端相关回归、release verification、`/api/health`、完整89图精度与真实 API 基准。
