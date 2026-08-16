# Ascend 310B SSH 连接、账户与运行环境

本文记录如何从 Windows 笔记本连接 Atlas 200I DK A2，以及板端官方系统账户、Miniconda、命名环境 `agileagent`、CANN、正式 release 和服务操作。文中的账户密码是华为官方系统镜像的出厂默认值；若设备已经改密，应以实际密码为准，不得把改密后的密码写入仓库。

## 环境快照

| 项目 | 当前值 |
| --- | --- |
| SSH 地址 | `192.168.137.100:22` |
| 官方系统账户 | `root`、`HwHiAiUser` |
| 官方出厂默认密码 | 两个账户均为 `Mind@123` |
| 设备 | Atlas 200I DK A2，`aarch64` |
| SoC | Ascend310B1 |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1` |
| Miniconda 安装根目录 / base | `/usr/local/miniconda3`，Conda `23.5.0` |
| 正式命名环境 | 名称 `agileagent`，路径 `/usr/local/miniconda3/envs/agileagent` |
| 正式 Python 解释器 | `/usr/local/miniconda3/envs/agileagent/bin/python` |
| Python 版本 | `3.9.2` |
| 已删除的旧环境 | `<release>/conda-env`；不得继续使用或写入配置 |
| 正式满分 release | `/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04` |
| 旧三 OM 回滚 release | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` |
| 公共服务 | `127.0.0.1:8501` |
| 满分主实例（双实例拓扑） | `127.0.0.1:18501` |
| 隔离候选 | `127.0.0.1:8502` |

## 官方系统账户

官方系统镜像具有以下两个账户，出厂默认密码相同：

| 账户 | 出厂默认密码 | 用途 |
| --- | --- | --- |
| `HwHiAiUser` | `Mind@123` | 日常 SSH 登录、项目部署、推理服务和普通运维 |
| `root` | `Mind@123` | 仅用于必须修改系统目录、权限或系统配置的管理操作 |

也就是：*root* / `Mind@123`，*HwHiAiUser* / `Mind@123`；两个账户的出厂默认密码相同。日常操作优先使用 `HwHiAiUser`，不要因为密码相同而长期使用 `root`。设备部署稳定后应分别修改两个账户的密码，也不要把修改后的密码保存到 Git、脚本、MobaXterm 宏或命令行参数中。

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

需要执行受控系统管理操作时才使用：

```powershell
ssh root@192.168.137.100
```

MobaXterm 中选择 **Session → SSH**，`Remote host` 填写 `192.168.137.100`，端口保持 `22`，日常会话的 `Specify username` 填写 `HwHiAiUser`。按提示交互输入对应账户密码；MobaXterm 左侧 SFTP 浏览器可以查看该账户有权限访问的 release 文件。不要把密码写进会话名称、宏或脚本。

## 本地端口转发

```powershell
ssh -N -L 8501:127.0.0.1:8501 HwHiAiUser@192.168.137.100
```

隧道建立后，Windows 访问：

```powershell
Invoke-RestMethod http://127.0.0.1:8501/api/health
```

## Miniconda 与命名环境 `agileagent`

必须区分以下三个路径，它们不是一回事：

| 含义 | 正确值 |
| --- | --- |
| 板端 Miniconda 安装根目录，也是 base 环境 | `/usr/local/miniconda3` |
| AgileAgent 正式命名环境的 Conda prefix | `/usr/local/miniconda3/envs/agileagent` |
| 正式服务实际调用的 Python | `/usr/local/miniconda3/envs/agileagent/bin/python` |

原三 OM 回滚 release 曾使用的 release-local 环境 `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2/conda-env` 已在命名环境迁移验收后删除。配置、脚本和人工命令都不得再引用该旧环境路径。

登录板端后先核对 Miniconda 和环境注册信息：

```bash
/usr/local/miniconda3/bin/conda --version
/usr/local/miniconda3/bin/conda info --base
/usr/local/miniconda3/bin/conda env list
test -x /usr/local/miniconda3/envs/agileagent/bin/python
```

预期 `conda info --base` 返回 `/usr/local/miniconda3`，环境列表中存在名为 `agileagent` 的 `/usr/local/miniconda3/envs/agileagent`。

不依赖 shell 激活也可以直接调用正式解释器：

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
printf 'CONDA_DEFAULT_ENV=%s\nCONDA_PREFIX=%s\n' \
  "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX"
readlink -f "$(command -v python)"
```

预期 `CONDA_DEFAULT_ENV=agileagent`、`CONDA_PREFIX=/usr/local/miniconda3/envs/agileagent`，Python 的真实路径位于该 prefix 下。正式启动脚本使用绝对解释器路径，因此不依赖交互式 `conda activate`，也不会使用 base 环境。

## CANN 与设备状态

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
/usr/local/miniconda3/envs/agileagent/bin/python -c \
  "import acl; print(acl.__file__)"
```

环境迁移后的服务驻留快照为 NPU 内存 `9479 / 11577 MB`、温度 `61°C`、Health `OK`。

## Release 目录

```text
/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04/
├── src/
├── om/shared_backbone_dual_head.om
├── om/scene_sensor_net.om
├── provenance/
├── configs/agent_pipeline_ascend310b.yaml
├── validation/
├── release.json
└── agent-web.pid
```

该 release 的可版本化源包已放入仓库 `models/ascend310b/full-score/20260816-full-score-1493b04/`。在已配置好 CANN/Python 的 310B 上，克隆后运行：

```bash
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh
```

脚本核对所有 SHA256 并执行正式 release 验证，不训练、不运行 ATC、不安装依赖，也不启动或停止服务。目标已存在时使用 `--verify-existing` 只读复核。旧三 OM release 仍保留在 `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2`，只承担即时回滚。

正式解释器：

```text
/usr/local/miniconda3/envs/agileagent/bin/python
```

正式配置：

```text
<release>/src/configs/agent_pipeline_ascend310b.yaml
```

## 启动服务

新板未安装旧回滚 listener 时，直接把满分 release 启动到公共 `8501`：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04
AGILE_AGENT_ASCEND_ENV=/usr/local/miniconda3/envs/agileagent \
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

`AGILE_AGENT_ASCEND_ENV` 可以省略，因为脚本默认值就是 `/usr/local/miniconda3/envs/agileagent`；这里显式写出是为了让人工复核时没有路径歧义。脚本加载 CANN 环境，使用命名环境 Python 启动 Uvicorn，并将 PID 写入 `<release>/agent-web.pid`。

已有旧三 OM 回滚服务的正式板使用双实例拓扑，不执行上面的直接启动命令：满分实例由 `agileagent-ascend310b-main.service` 监听 `18501`，旧 listener 由 `agileagent-ascend310b-rollback.service` 继续物理监听 `8501`，`agileagent-ascend310b-route.service` 将公共 `8501` 精确路由到主实例。`8502` 只留给下一轮候选。安装、检查和原子回滚命令见 [`ascend-310b-deployment.md`](ascend-310b-deployment.md)。

## 运行状态

```bash
curl -fsS http://127.0.0.1:8501/api/health
ss -lntp | grep ':8501'
ps -ef | grep 'uvicorn fair_agent.web.app:app'
npu-smi info
```

健康响应包含 `status: ready`、`backend: ascend_acl`、`device: ascend:0`、`validated: true`、`model_layout: shared_backbone_dual_head_v1` 和 `context_mode: fixed_neutral_v1`。

## 真实图像冒烟

```bash
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

当前满分 release 已用固定 PNG 完成服务冒烟；其正式评分和性能原始报告随仓库模型包一同交付。没有竞赛数据集时可核对哈希、release、health 和历史报告；重新测量 FPS 需要 20 张契约 PNG，重新计算 Base/New/KRR 需要合法取得的 89 图和标签。
