# Ascend 310B SSH 与运行环境

本文记录 Atlas 200I DK A2 的当前连接方式、Python 环境、CANN、release 路径和服务操作。

## 环境快照

| 项目 | 当前值 |
| --- | --- |
| SSH 地址 | `192.168.137.100:22` |
| 日常账户 | `HwHiAiUser` |
| 设备 | Atlas 200I DK A2，`aarch64` |
| SoC | Ascend310B1 |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1` |
| Conda | Miniconda `23.5.0` |
| Python 环境 | `/usr/local/miniconda3/envs/agileagent` |
| Python 版本 | `3.9.2` |
| 正式 release | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` |
| 服务 | `127.0.0.1:8501` |

## Windows 连接

检查 SSH 端口：

```powershell
Test-NetConnection -ComputerName 192.168.137.100 -Port 22
```

OpenSSH 登录：

```powershell
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  HwHiAiUser@192.168.137.100
```

MobaXterm 使用相同地址、端口和账户建立 SSH 会话，并通过 SFTP 浏览 release 文件。

## 本地端口转发

```powershell
ssh -N -L 8501:127.0.0.1:8501 HwHiAiUser@192.168.137.100
```

隧道建立后，Windows 访问：

```powershell
Invoke-RestMethod http://127.0.0.1:8501/api/health
```

## Python 环境

```bash
/usr/local/miniconda3/envs/agileagent/bin/python --version
/usr/local/miniconda3/envs/agileagent/bin/python -c \
  "import acl, numpy, cv2, PIL; print('imports ok')"
```

命名环境由已验证 release 环境离线克隆并完成文件级复核。当前记录包含 181 个 Conda 条目和 176 个 `pip freeze --all` 条目。

激活环境：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate agileagent
```

## CANN 与设备状态

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import acl; print(acl.__file__)"
```

环境迁移后的服务驻留快照为 NPU 内存 `9479 / 11577 MB`、温度 `61°C`、Health `OK`。

## Release 目录

```text
/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2/
├── src/
├── om/
├── validation/
└── agent-web.pid
```

正式解释器：

```text
/usr/local/miniconda3/envs/agileagent/bin/python
```

正式配置：

```text
<release>/src/configs/agent_pipeline_ascend310b.yaml
```

## 启动服务

```bash
cd /home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2/src
./scripts/start_agent_ascend310b.sh
```

脚本加载 CANN 环境，使用命名环境 Python 启动 Uvicorn，并将 PID 写入 `<release>/agent-web.pid`。

## 运行状态

```bash
curl -fsS http://127.0.0.1:8501/api/health
ss -lntp | grep ':8501'
ps -ef | grep 'uvicorn fair_agent.web.app:app'
npu-smi info
```

健康响应包含 `status: ready`、`backend: ascend_acl` 和 `device: ascend:0`。

## 真实图像冒烟

```bash
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

环境迁移使用固定 PNG 完成切换前后语义对照，检测数量、类别、框和置信度保持一致，三模型均完成执行。
