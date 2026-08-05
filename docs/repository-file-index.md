# 仓库文件索引

本文列出公开仓库中全部受版本控制文件及其职责。竞赛原始数据、实验输出、缓存和本地笔记不逐文件展开，它们不属于可发布源码。

## 目录职责

| 目录 | 职责 |
| --- | --- |
| `fair_agent/` | Agent 核心代码、推理编排、增量生命周期、CLI 与 Web 服务。 |
| `configs/` | 可复现实验、运行时、模型、数据及部署参数。 |
| `models/` | 随仓库发布的冻结模型、代际注册表和验收元数据。 |
| `scripts/` | 环境配置、一键启动、发布检查和模型冒烟入口。 |
| `tools/` | 数据准备、训练实验、推理、TensorRT 和压力测试工具。 |
| `tests/` | 无 GPU 单元测试和集成回归测试。 |
| `docs/` | 操作、实验、部署和审计说明。 |
| `native/` | C++/CUDA TensorRT 原生推理后端。 |
| `splits/` | 固定且公开的数据划分清单。 |
| `demo_artifacts/` | 不含原始样本的脱敏演示状态。 |

## 根目录与自动化

| 文件 | 功能 |
| --- | --- |
| `.gitattributes` | 固定文本换行规则，并将模型权重标记为二进制文件。 |
| `.github/workflows/tests.yml` | 在 Python 3.10/3.12 上运行 Pytest 和发布校验。 |
| `.gitignore` | 隔离数据集、运行产物、凭据、缓存及非发布模型。 |
| `README.md` | 项目首页、安装、启动、功能、指标和使用入口。 |
| `constraints-agent.txt` | 记录本项目验证过的依赖版本组合。 |
| `pyproject.toml` | Python 包元数据、依赖组、`agile-agent` 命令和测试配置。 |
| `requirements-agent.txt` | Web 工作台和基础 Agent 的最小依赖。 |
| `requirements-agent-inference.txt` | NVIDIA GPU 推理所需的 Ultralytics 与 OpenCV 依赖。 |
| `requirements-agent-dev.txt` | 开发和测试依赖聚合入口。 |

## 配置文件

| 文件 | 功能 |
| --- | --- |
| `configs/agent_pipeline.yaml` | Agent 唯一主配置，管理服务、推理、路由、增量、门禁、日志和部署参数。 |
| `configs/base_dataset.yaml` | 四类 IR/SAR 数据集及 train/dev/lock 划分声明。 |
| `configs/functional_models.yaml` | 三种功能模型、产物和模型间协作关系注册表。 |
| `configs/incremental/multibatch_small_sample.yaml` | 四轮小样本连续增量回归配置。 |
| `configs/incremental/multibatch_stress_balanced.yaml` | 类别与样本相对均衡的连续增量压力场景。 |
| `configs/incremental/multibatch_stress_diminishing.yaml` | 每轮样本逐步减少的压力场景。 |
| `configs/incremental/multibatch_stress_matrix.yaml` | 汇总并调度多组压力场景的矩阵配置。 |
| `configs/incremental/multibatch_stress_sensor_shift.yaml` | IR/SAR 传感器分布漂移压力场景。 |
| `configs/incremental/warship_3plus1.yaml` | 可复现舰船 3+1 基础集、增量轮次及验收定义。 |
| `configs/incremental_detection_policy.yaml` | 类别增量和目标增量的数据访问与指标规则。 |
| `configs/local_infer_gpu.yaml` | 本地 GPU 单模型推理配置。 |
| `configs/scene_sensor_model.yaml` | IR/SAR 与四场景多任务认知模型训练配置。 |
| `configs/strict_class_incremental_3plus1.yaml` | strict-p01/p02 双折 3+1 类别增量实验配置。 |
| `configs/submission_infer_yolo11s_imgsz640.yaml` | YOLO11s 640 正式推理和结果输出配置。 |

## 文档与演示证据

| 文件 | 功能 |
| --- | --- |
| `demo_artifacts/agent_demo_state.json` | 不包含原始样本的脱敏数据概况，用于缺少私有数据时展示基础统计。 |
| `docs/agent-audit-logging.md` | 全生命周期结构化事件、关联标识和日志查询说明。 |
| `docs/agent-operations.md` | Web、CLI、审计、TensorRT 与 310B 操作指南。 |
| `docs/compliant-incremental-learning.md` | 仅使用增量数据的合规边界和指标口径。 |
| `docs/functional-models.md` | 场景认知、基础检测和增量检测三类功能模型说明。 |
| `docs/incremental-workbench.md` | 增量数据上传、训练、校准、复核、注册和上线说明。 |
| `docs/multibatch-small-sample-validation.md` | 四批次小样本回归设计与本机结果。 |
| `docs/multibatch-stress-matrix.md` | 多组连续小样本压力测试结果。 |
| `docs/repository-file-index.md` | 当前公开仓库的完整文件职责索引。 |
| `docs/tensorrt-deployment.md` | 目标设备 TensorRT 导出、校准和精度性能验收指南。 |
| `docs/warship-3plus1-reproducibility.md` | 舰船 3+1 数据审计、运行和复现证据说明。 |

## Agent Python 包

| 文件 | 功能 |
| --- | --- |
| `fair_agent/__init__.py` | 顶层 Python 包标记。 |
| `fair_agent/cli.py` | 所有 `agile-agent` 子命令、参数解析和命令调度。 |
| `fair_agent/dataset_utils.py` | 数据扫描、文件名解析、YOLO 标签校验和 metadata 工具。 |
| `fair_agent/backends/__init__.py` | 推理后端子包标记。 |
| `fair_agent/backends/inference.py` | Ultralytics CUDA、Python TensorRT 和原生 TensorRT 后端适配器。 |
| `fair_agent/core/__init__.py` | 核心基础设施子包标记。 |
| `fair_agent/core/audit.py` | 独立 run 目录及 pipeline 计划、manifest 和报告生成。 |
| `fair_agent/core/blackboard.py` | 汇总数据、模型、增量与提交状态并生成 Agent 黑板。 |
| `fair_agent/core/config.py` | YAML 加载、校验、环境变量展开、CLI 覆盖和原子写回。 |
| `fair_agent/core/hashes.py` | 发布资产和离线审计文件的 SHA256 工具，不参与在线图像路由。 |
| `fair_agent/core/runtime_log.py` | 结构化 JSONL 事件日志、轮转、脱敏和状态镜像。 |
| `fair_agent/executors/__init__.py` | 执行器子包标记。 |
| `fair_agent/executors/local.py` | 带超时和审计记录的本地命令执行器。 |
| `fair_agent/models/__init__.py` | 模型子包标记。 |
| `fair_agent/models/context.py` | Scene-SensorNet 定义、加载、批推理和准确率评测。 |
| `fair_agent/modules/__init__.py` | Agent 功能模块子包标记。 |
| `fair_agent/modules/api_benchmark.py` | 检测 API 平均延迟、P95、吞吐和并发基准。 |
| `fair_agent/modules/configuration.py` | `config get/set/unset/diff/migrate/show` 的实现。 |
| `fair_agent/modules/functional_models.py` | 三种功能模型注册表的结构和资产校验。 |
| `fair_agent/modules/generation_management.py` | 候选代际注册、完整 production 复核、shadow load、晋升与回滚。 |
| `fair_agent/modules/incremental_experiment.py` | 通用 YAML 增量实验状态机、快照、运行和复现。 |
| `fair_agent/modules/incremental_guardian.py` | 官方硬门禁、质量告警、失败诊断和动态混淆图。 |
| `fair_agent/modules/incremental_lifecycle.py` | 校准、量化、lock 复核、注册、上线和回滚的生命周期编排。 |
| `fair_agent/modules/incremental_lineage.py` | 仅用于离线训练隔离的基础/增量文件目录审计。 |
| `fair_agent/modules/incremental_methods.py` | strict 3+1 适配训练、参数保护和漂移计算组件。 |
| `fair_agent/modules/incremental_workbench.py` | ZIP 批次导入、类别命名、视图生成和后台训练任务管理。 |
| `fair_agent/modules/model_generations.py` | 动态读取代际、类别所有权和 Web 推理设置。 |
| `fair_agent/modules/operator_view.py` | CLI 运维快照及文本/JSON 渲染。 |
| `fair_agent/modules/release_verification.py` | 配置、模型、代际、功能模型和发布边界静态验收。 |
| `fair_agent/modules/status.py` | 生成黑板输入文件的状态记录。 |
| `fair_agent/modules/strict_incremental.py` | 3+1 数据构建、AP50、阈值、NMS、KRR 和 bootstrap 评测。 |
| `fair_agent/modules/tensorrt_export.py` | FP16/INT8 engine 导出、校准样本选择和配置登记。 |
| `fair_agent/modules/tensorrt_validation.py` | TensorRT 与 CUDA 精度对齐、阈值复核和 API 性能验收。 |
| `fair_agent/modules/web_inference.py` | 无标签图像解码、production 模型编排、软路由、冲突仲裁和框融合。 |
| `fair_agent/policies/__init__.py` | 策略子包标记。 |
| `fair_agent/policies/decision.py` | 根据黑板和 YAML 动作定义生成可审计决策。 |
| `fair_agent/ui/__init__.py` | 运维 UI 子包标记。 |
| `fair_agent/ui/console.py` | 服务器和未来端侧使用的交互式 CLI 前端。 |
| `fair_agent/web/__init__.py` | Web 服务子包标记。 |
| `fair_agent/web/app.py` | Starlette API、单图/批量检测、历史结果及增量工作台路由。 |
| `fair_agent/web/static/index.html` | 面向评委的 Web 页面结构。 |
| `fair_agent/web/static/assets/app.css` | 浅色圆角视觉系统和响应式布局。 |
| `fair_agent/web/static/assets/app.js` | Web 交互、Canvas 绘框、历史跳转、批量预览和增量任务控制。 |
| `fair_agent/web/static/assets/icons.svg` | Web 按钮和状态所用的 SVG 图标集合。 |

## 模型与注册信息

| 文件 | 功能 |
| --- | --- |
| `models/SHA256SUMS.txt` | 发布模型资产完整性清单。 |
| `models/base/yolo11s_ir_sar_imgsz640.pt` | 四类统一 YOLO11s 基准权重，不作为 3+1 production 类别所有者。 |
| `models/context/scene_sensor_metrics.json` | Scene-SensorNet 的传感器、场景和联合准确率证据。 |
| `models/context/scene_sensor_net.pt` | production 场景与传感器认知权重。 |
| `models/generations.json` | 模型、类别所有权、代际关系及 production/candidate/benchmark 通道。 |
| `models/manifest.json` | 发布模型、类别、指标和增量协议总清单。 |
| `models/production/incremental_detection/calibration.json` | 增量检测器 dev 阈值曲线和选择结果。 |
| `models/production/incremental_detection/incremental_detector.pt` | 当前 production 新类别增量检测权重。 |
| `models/production/incremental_detection/metrics.json` | 3+1 基础、新类、KRR 和部署复核指标。 |
| `models/production/incremental_detection/profile.json` | 当前增量检测 profile 的类别映射、阈值和证据入口。 |
| `models/production/incremental_detection/three_class_base_detector.pt` | 未使用舰船竞赛样本训练的三类冻结基础检测器。 |
| `models/profiles/incremental-detection/active.json` | CLI strict profile 使用的活动增量实验快照。 |
| `models/profiles/registry.json` | 已通过 strict 类别增量 profile 的索引。 |

## 原生后端

| 文件 | 功能 |
| --- | --- |
| `native/CMakeLists.txt` | CUDA、TensorRT、OpenCV 和 nlohmann-json 的 CMake 构建定义。 |
| `native/README.md` | 原生后端 ABI、构建和上线验收说明。 |
| `native/src/backend.cpp` | C ABI、图像解码、letterbox、动态 batch、TensorRT 前向和 NMS 实现。 |

## 脚本

| 文件 | 功能 |
| --- | --- |
| `scripts/bootstrap_x86.sh` | 首次安装时选择兼容 CUDA Python 环境并注册项目命令。 |
| `scripts/export_tensorrt_engines.sh` | 在当前 GPU 上一键导出并登记 TensorRT engine。 |
| `scripts/smoke_models.py` | 加载发布模型并执行 GPU 冒烟推理。 |
| `scripts/start_agent.sh` | 环境就绪后的 Web/CLI 一键启动入口。 |
| `scripts/verify_release.py` | 调用静态发布验收并输出结构化结果。 |

## 固定数据划分

| 文件 | 功能 |
| --- | --- |
| `splits/README.md` | 说明划分规模、相对路径约定和私有数据准备方式。 |
| `splits/train.txt` | 560 张基础训练图像清单。 |
| `splits/dev_val.txt` | 95 张开发验证图像清单。 |
| `splits/lock_val.txt` | 95 张冻结复核图像清单。 |
| `splits/lock_val_base_3plus1.txt` | 舰船 3+1 中 74 张不含新增类别的旧类复核清单。 |
| `splits/lock_val_increment_3plus1.txt` | 舰船 3+1 中 21 张包含新增类别的复核清单。 |
| `splits/train_ir.txt` | train 中的 403 张 IR 图像。 |
| `splits/train_sar.txt` | train 中的 157 张 SAR 图像。 |
| `splits/dev_val_ir.txt` | dev 中的 69 张 IR 图像。 |
| `splits/dev_val_sar.txt` | dev 中的 26 张 SAR 图像。 |
| `splits/lock_val_ir.txt` | lock 中的 68 张 IR 图像。 |
| `splits/lock_val_sar.txt` | lock 中的 27 张 SAR 图像。 |

## 自动化测试

| 文件 | 功能 |
| --- | --- |
| `tests/test_agent_workbench.py` | doctor、黑板、决策、pipeline 和低风险执行器测试。 |
| `tests/test_configuration_runtime.py` | 配置 schema、覆盖、写回、TensorRT 参数和受保护字段测试。 |
| `tests/test_functional_models.py` | 三种不同功能模型及协作证据注册测试。 |
| `tests/test_incremental_experiment.py` | 通用实验状态机、快照和复现测试。 |
| `tests/test_incremental_guardian.py` | 官方硬门禁、告警、恢复动作和冲突图测试。 |
| `tests/test_incremental_lifecycle_v2.py` | 多类、多轮、自动 lock、校准、上线、回滚和量化生命周期测试。 |
| `tests/test_incremental_methods.py` | strict 3+1 参数合并、类别映射和保护逻辑测试。 |
| `tests/test_incremental_workbench.py` | ZIP 上传、类别命名、注入、后台任务和 Web API 测试。 |
| `tests/test_multibatch_stress_configs.py` | 多组连续小样本压力配置及结果判定测试。 |
| `tests/test_public_splits.py` | 公开 split 数量、互斥性和传感器子集测试。 |
| `tests/test_runtime_maturity.py` | 启动脚本、CLI、配置、发布和运维前端成熟度测试。 |
| `tests/test_strict_incremental.py` | 3+1 数据隔离、映射、AP50、校准和 bootstrap 测试。 |
| `tests/test_submission_safety.py` | 正式推理输出路径、结果数量和 GPU 设备约束测试。 |
| `tests/test_unlabeled_inference.py` | 确认在线推理不使用图像身份或来源分流，并完整执行 production 复核。 |
| `tests/test_web_inference.py` | 解码、软路由、specialist 激活、冲突仲裁、NMS 和批量结果测试。 |
| `tests/test_web_ui_flow.py` | Web API、无标签输入、Canvas、历史、批量预览和前端契约测试。 |

## 工具程序

| 文件 | 功能 |
| --- | --- |
| `tools/00_check_dataset.py` | 校验图像/标签对应、类别、YOLO 框和分布并生成审计报告。 |
| `tools/01_build_metadata.py` | 生成逐图 metadata.csv 和目标面积统计。 |
| `tools/02_split_dataset.py` | 按传感器、场景和类别存在性生成固定分层划分。 |
| `tools/42_predict_submission.py` | 按 YAML 对官方无标签图像推理并生成提交结果。 |
| `tools/60_train_scene_sensor.py` | 训练并评测 IR/SAR 与四场景认知模型。 |
| `tools/70_run_strict_3plus1.py` | 运行 strict 3+1 双折数据、训练、校准和评测流水线。 |
| `tools/80_export_tensorrt_engines.py` | TensorRT 导出模块的命令行入口。 |
| `tools/81_validate_multibatch_incremental.py` | 验证连续多批次小样本增量机制。 |
| `tools/82_run_multibatch_stress_matrix.py` | 调度多组压力场景并汇总守护器判定。 |

## 本地私有目录

以下内容存在于当前工作区但被 `.gitignore` 排除，不属于公开仓库文件：

- `datasets_r1_base_train/`、`20260701基础训练数据/`：竞赛原始图像与标签。
- `reports/`、`runs/`、`data/`：报告、训练输出、缓存和增量上传批次。
- `external_repos/`：论文复现所需第三方仓库。
- `incremental_protocols*/`、`incremental_strict_3plus1/`：历史增量实验产物。
- `final_submission_assets/`：本地冻结提交材料。
- `.venv`、`.agent-python`、`.streamlit/`：本机环境选择和服务配置。
- 根目录中文旧指南、竞赛 PDF、ZIP 备份和临时权重：仅供本地研究留档。
