# Ascend C ABI 契约夹具

`native_ascend/` 提供可构建的 C ABI 契约夹具，用于验证 Python 与原生 Ascend 运行库之间的版本、状态、错误和生命周期接口，并为原生后端实现提供稳定集成边界。

## 构建

```bash
cmake -S native_ascend -B build/native_ascend_stub \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native_ascend_stub --config Release
```

生成产物：

```text
build/native_ascend_stub/libagile_agent_ascend_contract_stub.so
```

## ABI

| 函数 | 作用 |
| --- | --- |
| `agile_agent_ascend_backend_version` | 返回 ABI 版本 |
| `agile_agent_ascend_create` | 创建后端句柄 |
| `agile_agent_ascend_destroy` | 释放后端句柄 |
| `agile_agent_ascend_ready` | 返回就绪状态 |
| `agile_agent_ascend_warmup` | 执行完整链路预热 |
| `agile_agent_ascend_predict` | 执行推理调用契约 |
| `agile_agent_ascend_free_result` | 释放推理结果 |
| `agile_agent_ascend_last_error` | 返回最近错误信息 |

## 契约冒烟

```bash
python tools/91_smoke_ascend_contract.py \
  build/native_ascend_stub/libagile_agent_ascend_contract_stub.so
```

冒烟工具加载动态库，核对 ABI 版本、句柄生命周期、状态传播和错误文本。当前 production 的 PyACL/AscendCL 实现位于 `fair_agent/backends/ascend_acl.py`，两条路径共享同一模型身份与请求语义。
