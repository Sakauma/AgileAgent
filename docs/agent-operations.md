# Agent 操作指南

## Web 工作台

环境配置完成后运行：

```bash
./scripts/start_agent.sh
```

浏览器打开 `http://127.0.0.1:8501`。工作台包含运行总览、模型与协同、数据与诊断、策略运行、增量学习、审计与部署六个工作区。页面指标、动作和门禁全部来自 `reports/agent_blackboard/blackboard_state.json`，界面不维护独立状态。

## 终端工作台

SSH、无桌面服务器和未来板端环境使用：

```bash
./scripts/start_agent.sh --cli
```

该命令依次执行环境诊断、重建黑板并进入交互式 CLI 前端。它不会启动 Streamlit，也不会安装或修改依赖。

CLI 前端菜单：

```text
[1] 总览  [2] 模型  [3] 数据  [4] 增量  [5] 部署
[r] 刷新  [d] 决策  [p] Dry-run  [x] 执行  [q] 退出
```

`x` 只允许执行配置白名单中的低风险动作，并要求再次输入 `EXECUTE`。训练、正式推理和提交不会从 CLI 前端自动触发。

单独读取状态：

```bash
agile-agent status
agile-agent status --format json --refresh
agile-agent console --once
```

文本格式用于人工查看，JSON 格式用于外部守护进程、评测脚本或 AscendCL 服务集成。

## 审计运行

仅生成计划：

```bash
agile-agent pipeline --mode dryrun
```

执行允许列表中的低风险动作：

```bash
agile-agent pipeline --mode execute
```

每次运行写入独立的 `reports/agent_runs/<run_id>/`，包含计划、manifest、动作日志和 Markdown 报告。训练、正式推理与提交不由 v1 自动执行。

## Ascend 310B 边界

当前发布版本只验证 x86 NVIDIA GPU。310B 状态固定为 `waiting_for_hardware`，后续必须依次完成：

1. 使用 ATC 将冻结模型转换为 OM。
2. 实现 AscendCL 输入预处理、推理和后处理。
3. 对齐 x86 与 310B 精度。
4. 在真实板卡上记录 FPS、时延和内存。

未完成上述门禁前，不得把 x86 GPU 指标作为 310B 性能证据。
