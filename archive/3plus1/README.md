<!-- generated-by: gsd-doc-writer -->
# 3+1 统一归档

本目录保存 AgileAgent 3+1 阶段的源码、配置、数据划分、模型、测试与板端发布证据。活动工程以 strict 4+2 x86/CUDA 流程和 Ascend310B v2 正式发布为准。

## 目录

| 目录 | 文件数 | 内容 |
| --- | ---: | --- |
| `snapshot/` | 139 | 从活动区移出的纯 3+1 文件；目录结构与原路径一致 |
| `mixed-source-original/` | 54 | 同时包含 3+1 与 4+2 内容的文件在清理前的完整版本 |

`ARCHIVE_MANIFEST.json` 记录分组、数量、原路径映射和核验结果。归档不依赖额外摘要文件；原路径由归档根目录后的相对路径直接确定。

## 路径映射

例如：

- `archive/3plus1/snapshot/tools/70_run_strict_3plus1.py` 的原路径是 `tools/70_run_strict_3plus1.py`；
- `archive/3plus1/mixed-source-original/fair_agent/cli.py` 的原路径是 `fair_agent/cli.py`。

## 恢复

恢复纯 3+1 文件时，将 `snapshot/` 下所需文件按相同相对路径复制回仓库根目录。恢复混合文件时，应先从 `mixed-source-original/` 取出修改前版本，再按需要移植对应的 3+1 段落；直接覆盖会同时撤销该文件当前的 4+2 整理结果。

归档内容只用于历史复现和代码追溯，不参与当前配置加载、模型选择、测试发现或发布打包。
