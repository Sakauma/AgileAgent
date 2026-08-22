# 3+1 production 可恢复归档

本目录保存 4+2 production 切换前的完整 3+1 x86/CUDA 资产。归档操作只移动和复制，未删除旧权重；原始清单、代际注册表、profile、Scene-SensorNet、推理配置和 SHA256 清单均保留。

目录内容：

- `production/incremental_detection/`：旧三类 Base、单类专家及校准/评测证据；
- `context/`：旧 Scene-SensorNet 权重与指标；
- `profiles/`：旧活动 profile 和注册表；
- `metadata/`：切换前的 `manifest.json`、`generations.json` 与校验清单；
- `configs/`：切换前的主要运行、训练和提交配置快照。

本目录不是活动 production。恢复时应把对应文件复制回原路径，并同步恢复 `metadata/manifest.json`、`metadata/generations.json`、profiles、配置快照和 `metadata/SHA256SUMS.txt`，再执行发布校验。
