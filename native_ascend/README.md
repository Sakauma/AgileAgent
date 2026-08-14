# Ascend 原生后端契约

该目录固定 AgileAgent 可选 C++ AscendCL 整体管线的 C ABI。当前仍只提供不依赖 CANN 的 contract stub，用于验证构建、符号、Not Ready 状态、错误传播以及“禁止 CPU 模型回退”的约束；stub 不执行推理，也不提供任何性能结论。

当前可运行的 310B 后端不在本目录，而是 [`fair_agent/backends/ascend_acl.py`](../fair_agent/backends/ascend_acl.py) 中的 PyACL 实现。据本轮 SSH 只读复核，开发板正式 release 已加载三个 OM 并提供 `127.0.0.1:8501` 推理服务；对应实时日志尚未随仓库归档。仓库 `main` 还包含默认关闭的 DVPP 编码图像路径。本目录因此是 ABI 回归夹具和未来 C++ 迁移边界，不能再描述为当前 production 的实现入口。

若后续实现 C++ 动态库，一个 handle 必须同时持有：

- 固定 `736×896` 的基础检测 OM；
- 固定 `512×640` 的全部活动增量检测 OM；
- 固定 `160×160` 的 Scene-SensorNet OM；
- DVPP/VPC/AIPP 状态、独立 streams 和 2 至 3 槽复用缓冲区；
- 冻结的类别映射、阈值、NMS 与框级融合规则。

每个请求都必须运行基础 owner 和所有活动增量 owner。文件名、标签、测试清单和场景硬路由不属于 ABI 输入。

本目录及当前 PyACL 后端都不得链接 TensorRT/CUDA，也不得加载 `.engine`。板端模型只接受 ATC 生成的 OM，并通过 CANN/AscendCL 执行；CPU 图像预处理与 DVPP/VPC/AIPP 的选择必须由独立精度和性能门禁决定。

## 官方四项计分边界

竞赛性能指标共60分，前三项是检测与增量学习精度，第四项是 Ascend 310B 完整单帧链路效率。官方检测精度口径为 `mAP@0.5`（`mAP50`）；仓库评分实现和本目录的精度复核均采用这一官方口径。

| 官方指标 | 分值 | 完整评分档位 |
| --- | ---: | --- |
| 基础目标检测 mAP | 30 | `≥0.80:30`；`≥0.70:25`；`≥0.65:20`；`≥0.60:15`；`≥0.50:10`；`≥0.40:5`；`<0.40:0` |
| New-mAP | 10 | `≥0.60:10`；`≥0.50:7`；`≥0.40:4`；`<0.40:0` |
| KRR | 10 | `≥0.95:10`；`≥0.90:7`；`≥0.80:4`；`<0.80:0` |
| Ascend 310B 端到端 FPS | 10 | `≥30:10`；`≥20:7`；`≥10:4`；`<10:0` |

本目录的实现候选只有在保持前三项精度的同时，按 `batch=1` 完整处理输入、三模型、后处理和结果生成，才可形成第四项证据。CUDA 代理、单 OM、已解码 Agent 核心和批量吞吐都不计入310B官方端到端FPS；`30 FPS` 是10分满分线，不是最低得分线。

## 后续 C++ 实现约束

- 三个 OM、输入输出 dataset 和工作内存必须在启动阶段一次性创建，并使用固定地址的 2 至 3 槽缓冲区；请求关键路径不得反复分配设备内存。
- 模型加载、缓冲分配和完整链路预热必须先串行完成，确认输出和事件状态稳定后才把 handle 标记为 Ready。
- 同一设备上分别测量三模型串行、两检测器并发和三模型并发；以完整链路 P95 为第一选择依据，不能假设多 stream 必然更快。
- 正式 release 目前使用 CPU 解码/预处理。DVPP/VPC 编码输入仅是实验候选；完整89张精度门禁通过且端到端性能确有收益后，才允许把它设为默认。
- NMS 前可删除所有类别分数都低于当前检测下限的 anchor。新增专家还可在 NMS 前应用 profile 冻结的最低激活阈值，但场景软惩罚和最终门禁仍须在融合阶段执行。
- 上述预筛选只减少必然会在最终阶段被拒绝的候选，不得跳过基础、增量或场景 owner，也不得读取文件名、清单、标签或真实类别。
- 当前 PyACL 异步/AIPP 候选已在 staging 实测；未来 C++、DVPP 和更多 ACL stream 仍须独立复测，不能直接继承 Python 候选结论。
- 精度顺序固定为 FP32 基线、ATC `mixed_float16`、必要时才做受控 INT8/FP16 PTQ；Detect head、DFL、Softmax、Sigmoid 和输出优先保留 FP16。

固定缓冲与预热原则来自本机 CUDA Graph 代理验证，但真实实现不得依赖 ONNX Runtime 或 CUDA Graph。AscendCL 需要用自己的 stream、event、模型 dataset 和内存生命周期复现同一契约。

## 板前构建与 smoke

```bash
cmake -S native_ascend -B build/native_ascend_stub -DCMAKE_BUILD_TYPE=Release
cmake --build build/native_ascend_stub --config Release -j
python tools/91_smoke_ascend_contract.py \
  build/native_ascend_stub/libagile_agent_ascend_contract_stub.so
```

预期结果是 ABI 版本为1、handle 可析构、Ready 为 false、warmup/predict 明确失败并返回 CANN 不可用错误。若 stub 返回推理结果或尝试 CPU 模型执行，测试必须失败。

如决定从当前 PyACL 后端迁移到 C++，应在保持头文件 ABI 不变的前提下实现真实动态库，并接入 `tools/90_ascend_preflight.py` 已生成的 golden bundle 与性能报告格式。迁移候选必须重新通过三模型逐图正确性、完整89张精度、真实 PNG API 性能和稳定性门禁，不能因为 stub ABI 已通过就直接替换正式服务。

本机候选复核命令为：

```bash
python tools/90_ascend_preflight.py optimize --shape-mode rect --device 0 \
  --samples 6 --warmup 20 --rounds 100
```

当前 RTX 4060 CUDA 代理的稳定候选为平均 `30.337 ms`、P95 `31.918 ms`、按平均值折算 `32.96 FPS`；完整89张混合集上的输入、原始输出、预筛选、阈值前移和最终检测均为零不一致。该结果只证明优化语义和测试框架可用，不证明310B性能，也不表示 production 已切换到这些候选。

板前混合 FP16 敏感性复核可用独立输出目录执行：

```bash
python tools/90_ascend_preflight.py convert-fp16 \
  --source-root runs/ascend310b \
  --output-root runs/ascend310b_mixed_fp16 --shape-mode rect --overwrite
python tools/90_ascend_preflight.py metric-align \
  --output-root runs/ascend310b_mixed_fp16 --shape-mode rect --device 0 --provider cuda
```

当前候选的前三项官方精度满分线和两项内部质量门禁全部通过，但严格保持逐图输出一致的 CUDA 代理没有快于 FP32；带早筛优化的最快 FP16 候选还改变了 `1/89` 张结果。因此它只用于暴露精度敏感点，不是待部署 OM，其 CUDA FPS 也不能获得第四项分数；板后的 ATC、golden、完整指标和 P95 复核均不可跳过。Scene-SensorNet 可通过 `convert-fp16` 的 `--context-fp32` 选项单独保留 FP32。
