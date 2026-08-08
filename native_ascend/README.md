# Ascend 原生后端契约

该目录固定 AgileAgent 原生 AscendCL 整体管线的 C ABI。当前只提供不依赖 CANN 的 contract stub，用于在板卡到达前验证构建、符号、Not Ready 状态、错误传播以及“禁止 CPU 模型回退”的约束；stub 不执行推理，也不提供任何性能结论。

真实实现必须让一个 handle 同时持有：

- 固定 `736×896` 的基础检测 OM；
- 固定 `512×640` 的全部活动增量检测 OM；
- 固定 `160×160` 的 Scene-SensorNet OM；
- DVPP/VPC/AIPP 状态、独立 streams 和 2 至 3 槽复用缓冲区；
- 冻结的类别映射、阈值、NMS 与框级融合规则。

每个请求都必须运行基础 owner 和所有活动增量 owner。文件名、标签、测试清单和场景硬路由不属于 ABI 输入。

## 板前构建与 smoke

```bash
cmake -S native_ascend -B build/native_ascend_stub -DCMAKE_BUILD_TYPE=Release
cmake --build build/native_ascend_stub --config Release -j
python tools/91_smoke_ascend_contract.py \
  build/native_ascend_stub/libagile_agent_ascend_contract_stub.so
```

预期结果是 ABI 版本为1、handle 可析构、Ready 为 false、warmup/predict 明确失败并返回 CANN 不可用错误。若 stub 返回推理结果或尝试 CPU 模型执行，测试必须失败。

板卡到达后在保持头文件 ABI 不变的前提下，用 AscendCL、DVPP/VPC 和 AIPP 实现真实动态库，并接入 `tools/90_ascend_preflight.py` 已生成的 golden bundle 与性能报告格式。
