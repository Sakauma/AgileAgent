<!-- generated-by: gsd-doc-writer -->
# Ascend 310B SSH 连接与板端环境

本文说明如何从当前 Windows 笔记本通过直连网线登录 Atlas 200I DK A2（Ascend 310B），以及如何查看 AgileAgent 正式 release 使用的 Python 虚拟环境、依赖、CANN、OM 配置和服务状态。文中的固定路径与配置以仓库当前版本为准；温度、内存、软件小版本和服务健康属于实时状态，应在每次交付或复验时重新查询。

本文按设备所有者明确要求记录官方系统镜像的出厂默认账户和密码，便于开发板首次登录；这些公开默认值不等同于设备改密后的当前凭据。除下文这组官方出厂默认值外，严禁把当前登录密码、私钥、代理口令或设备日志中的凭据写入仓库。

## 当前连接与环境快照

截至 2026-08-14，已确认的连接和板端信息如下。

| 项目 | 当前值 | 证据边界 |
|---|---|---|
| SSH 目标 | `192.168.137.100:22` | 已通过 MobaXterm 建立直接 SSH；设备重新刷机或网络重配后可能变化 |
| SSH 账户 | 日常账户 `HwHiAiUser`；管理员账户 `root` | 官方系统提供两个账户；用户名区分大小写 |
| 本机客户端 | MobaXterm Personal Edition `26.3`；Windows OpenSSH `9.5.5.1` | 当前笔记本快照，不是项目运行依赖 |
| 开发套件 / 架构 | Atlas 200I DK A2，`aarch64` | 既有 SSH 只读复核 |
| 芯片 | Ascend310B1，NPU Health `OK` | 需用 `npu-smi info` 实时复核 |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` | 需用 `/etc/os-release` 和 `uname` 实时复核 |
| CANN | `7.0.RC1` | 正式配置和既有板端复核；当前策略是不升级、不替换、不混装 |
| Conda | `23.5.0`；base 为 `/usr/local/miniconda3`；正式命名环境为 `agileagent` | 正式解释器为 `/usr/local/miniconda3/envs/agileagent/bin/python` |
| Python | `3.9.2` | 迁移时已确认源 prefix 与命名环境版本一致；源 prefix 现已删除；不代表 x86 开发环境版本 |
| NPU 内存 / 温度 | 迁移后完成真实 PNG 烟测的服务驻留快照为 `9479 / 11577 MB`、`61°C` | 单次观测，不代表峰值或长期稳定状态 |
| 正式 release | `/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2` | 启停脚本与 Ascend 配置共同固定 |
| 正式服务 | `127.0.0.1:8501`，既有复核时 health 为 `ready` | 只监听板端回环地址，实时状态必须重新查询 |

<!-- VERIFY: 交付或复验时应在目标板重新确认 SSH 地址、NPU Health、CANN/驱动/固件版本、温度、内存和 /api/health，并归档同一 release 的原始输出。 -->

## 连接前的网络检查

笔记本和开发板已通过网线直连时，先在 Windows PowerShell 中检查链路。`ping` 可能被防火墙丢弃，因此 22 端口测试比 ping 更可靠。

```powershell
ipconfig
ping -n 4 192.168.137.100
Test-NetConnection -ComputerName 192.168.137.100 -Port 22
```

成功连接至少应满足：

- 笔记本有线网卡处于“已连接”状态，地址与开发板位于可路由的子网；
- `TcpTestSucceeded` 为 `True`；
- 没有 VPN、安全软件或 Windows 防火墙规则阻断本地 22 端口流量。

Windows Internet Connection Sharing（ICS）通常会把共享出口对应的有线网卡设为 `192.168.137.1/24`，但不要在 SSH 已经可用时为了匹配示例再次强制改地址。登录开发板后可用以下只读命令确认板端地址、默认路由和 DNS：

```bash
ip -br address
ip route
cat /etc/resolv.conf
```

### 笔记本代理的影响

到 `192.168.137.100:22` 的局域网 SSH 通常不经过 HTTP/HTTPS 代理，所以笔记本代理一般不会影响直接登录。ICS 提供的是路由和 NAT，不会自动把 Clash、V2Ray、浏览器代理或企业代理配置传给开发板。

如果开发板能访问笔记本网关但 HTTPS 请求仍失败，先区分 DNS、NAT 和代理问题。只有在确实需要时，才在当前板端终端临时设置代理，并把本地服务与直连子网排除：

```bash
export http_proxy='http://<proxy-host>:<proxy-port>'
export https_proxy="$http_proxy"
export no_proxy='127.0.0.1,localhost,192.168.137.0/24'
```

代理若只监听笔记本的 `127.0.0.1`，开发板无法直接使用；允许代理监听局域网地址及放行防火墙会扩大暴露面，应由设备所有者明确配置。不要把带用户名或密码的代理 URL 写入 shell 启动文件、YAML 或仓库。

## 官方系统出厂默认账户

官方系统镜像具有两个账户：*root* / `Mind@123`，*HwHiAiUser* / `Mind@123`；两个账户的出厂默认密码相同。本开发板已实际确认两个账户均可通过 SSH 认证。

| 账户 | 出厂默认密码 | 建议用途 |
|---|---|---|
| `HwHiAiUser` | `Mind@123` | 日常登录、项目部署和服务运维 |
| `root` | `Mind@123` | 仅用于必须修改系统所有目录或系统配置的受控管理操作 |

这些值只适用于尚未改密的官方系统镜像。由于 `root` 拥有完整系统权限，设备首次部署稳定后应分别执行 `passwd` 修改两个账户的密码，不要继续让它们共用密码，也不要把新密码写入 Git、MobaXterm 宏、Shell 脚本或命令行参数。改密后应以实际新密码为准，本文不得同步记录新密码。

## 使用 MobaXterm 登录

1. 打开 MobaXterm，选择 **Session → SSH**。
2. `Remote host` 填写 `192.168.137.100`，勾选 `Specify username` 并填写日常账户 `HwHiAiUser`；只有受控系统管理操作才使用 `root`。
3. `Port` 保持 `22`。SSH compression 可以开启；只使用终端和文件浏览器时不需要 X11 forwarding。
4. 第一次连接时核对主机密钥指纹，再接受并输入密码。除本文明确记录的官方出厂默认值外，当前密码只在交互提示中输入，不要写进会话名称、宏、脚本或仓库文件。
5. 登录后执行以下命令确认连接到了正确的主机和架构：

```bash
whoami
hostnamectl
uname -m
```

MobaXterm 左侧 SSH browser 可通过 SFTP 浏览用户有权限访问的文件。正式 release 应视为不可变目录；不要直接用文件浏览器覆盖其中的源码、配置、OM 或虚拟环境。

X11 forwarding 只用于转发单个 X11 应用窗口，不等同于显示完整 LXQt 桌面。完整桌面远控需要另行部署并加固 RDP/VNC，本部署手册不要求也不自动修改该服务。

### 主机密钥核对

若可以接触开发板本地终端，可在板端读取公钥指纹，再与 MobaXterm 或 OpenSSH 提示对照：

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

出现 `REMOTE HOST IDENTIFICATION HAS CHANGED` 时，先确认设备是否重新刷机、IP 是否被其他主机占用。只有在通过本地终端或其他可信渠道确认新指纹后，才在 Windows 删除旧记录：

```powershell
ssh-keygen -R 192.168.137.100
```

## 使用 Windows OpenSSH 登录

PowerShell 中可直接运行：

```powershell
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 HwHiAiUser@192.168.137.100
```

当前若尚未部署公钥，OpenSSH 的 key-only 测试失败并不表示 SSH 服务异常；按提示交互输入密码即可。需要长期使用时可创建一把设备专用密钥：

```powershell
ssh-keygen -t ed25519 -a 100 -f "$env:USERPROFILE\.ssh\id_ed25519_ascend310b"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_ascend310b.pub" | ssh HwHiAiUser@192.168.137.100 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys'
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_ascend310b" HwHiAiUser@192.168.137.100
```

确认密钥登录成功前，不要关闭现有密码登录会话，也不要修改板端 `sshd_config`。私钥只留在笔记本，公钥可以写入板端 `authorized_keys`，两者都不应复制进项目目录。

## 通过 SSH 访问板端本地服务

正式 Uvicorn 服务只绑定 `127.0.0.1:8501`，因此笔记本不能直接打开 `http://192.168.137.100:8501`。使用本地端口转发可在不改变板端监听范围的情况下访问：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 8501:127.0.0.1:8501 HwHiAiUser@192.168.137.100
```

保持该终端运行，在另一个 PowerShell 中检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8501/api/health
```

MobaXterm 也可在 **Tunneling / MobaSSHTunnel** 中建立 `Local port forwarding`：本地端口 `8501`，远端主机 `127.0.0.1`，远端端口 `8501`，SSH 服务器为 `192.168.137.100:22`。如果本机 8501 已占用，可将左侧本地端口改成其他值，例如 `18501`，远端端口仍保持 `8501`。

## 板端正式 release 目录

后续命令统一使用一个任务专用变量，避免误操作其他目录：

```bash
RELEASE_ROOT=/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2
printf '%s\n' "$RELEASE_ROOT"
test -d "$RELEASE_ROOT" && find "$RELEASE_ROOT" -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
```

仓库记录的目录职责为：

```text
releases/212705a26d4414eff4e00604ce37c54d2ae729b2/
├── om/          # Base、Incremental 和 Scene 三个正式 OM
├── src/         # 正式服务源码、配置和启停脚本
└── validation/  # smoke、89 图预测与评分记录

/usr/local/miniconda3/envs/agileagent/
└── bin/python   # 正式服务和人工运维共同使用的命名 Python 3.9 环境
```

目录名与旧提交哈希相同，但目录名本身不能证明板端文件与任一 Git checkout 逐文件一致。2026-08-14 的环境迁移对该 release 的启动脚本和 `runtime.local_python` 做了受控修改。迁移成功后，旧 prefix 和板端 `migration-backup` 已按设备所有者要求删除，因此当前没有板上环境或配置回滚副本。正式复现还需要同时归档源码、命名环境包清单、配置、OM、CANN/ATC 版本、SHA256 和原始报告。

## Miniconda 与命名环境 `agileagent`

板端镜像自带 Miniconda，base 位于 `/usr/local/miniconda3`。正式服务与人工运维统一使用其下名为 `agileagent` 的环境，不在 base 中安装项目依赖。确切 Conda 和 Python 小版本以板端输出为准：

```bash
command -v conda || true
conda --version
conda info --base
conda info --envs
```

人工登录后进入同一环境：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate agileagent
printf 'CONDA_DEFAULT_ENV=%s\nCONDA_PREFIX=%s\n' "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX"
python --version
python -m pip --version
python -m pip list --format=columns
```

预期 `CONDA_DEFAULT_ENV=agileagent` 且 `CONDA_PREFIX=/usr/local/miniconda3/envs/agileagent`。自动化脚本仍显式调用该环境的绝对解释器，避免服务依赖交互式 `conda activate`。2026-08-14 已为 `HwHiAiUser` 设置 `auto_activate_base: false`，新 SSH 会话不再默认停留在 `(base)`；需要工作时再显式激活 `agileagent`。

该命名环境由原 release-prefix 环境离线 clone 得到，迁移过程没有重新解析、下载、安装或升级 Python 包。实测双方均为 Python `3.9.2`、181 个 Conda 记录和 176 个 `pip freeze --all` 条目；修复下述 clone 覆盖问题后，双方 28,180 个非 `.pyc` 的 `site-packages` 文件在归一化 prefix 后逐文件一致。验证完成后，原 prefix 已删除且不可恢复。

`/usr/local/miniconda3/envs` 是 `root:root 755`，而 `HwHiAiUser` 的 sudo 白名单不允许执行任意 `conda` 或 `chown`。本次迁移采用“管理员只创建目标目录并授权，普通用户使用现有双缓存离线 clone”的方式；不要放宽整个 Miniconda 目录的权限。该迁移已经完成，原 clone 源和备份均已删除，不能重复执行。若未来必须重建，应先在独立候选目录依据已审核依赖重新构建，并重新执行文件哈希、核心导入、三 OM、真实 PNG、精度和性能门禁，再切换正式环境。

不要以 `conda list` 和 `pip freeze` 相同替代文件级校验。旧 prefix 曾由 pip 把 Conda 的 `pluggy 1.0.0` 覆盖为 `1.6.0`，其 Conda 元数据没有同步；`conda --clone` 因此一度恢复了 1.0.0 源码却保留 1.6.0 元数据。迁移时从旧 prefix 恢复实际 `pluggy 1.6.0` 文件后，`pytest 8.3.5` 导入和完整 `site-packages` 哈希比较才通过。未来重建必须执行同样的真实文件与导入校验，不能盲目复制本次修复命令。

### 依赖边界

[`configs/requirements-ascend310b.txt`](../configs/requirements-ascend310b.txt) 只固定命名环境 `agileagent` 中 Web 与验收所需的补充包：

| 包 | 固定版本 | 用途 |
|---|---:|---|
| `starlette` | `0.37.2` | Web 应用与路由 |
| `uvicorn` | `0.30.6` | 正式 HTTP 服务进程 |
| `python-multipart` | `0.0.9` | PNG multipart 上传解析 |
| `pluggy` | `1.6.0` | `pytest 8.3.5` 的实际运行依赖；显式记录迁移源曾存在的 pip 覆盖状态 |
| `pytest` | `8.3.5` | 板端范围验证，不是在线推理主链 |

CANN/PyACL、NumPy、OpenCV、Pillow 和 PyTorch 由板端环境提供，仓库没有在该文件中固定它们的版本。PyTorch 即使存在，也不参与正式三 OM 在线模型推理；正式后端只允许 CANN/PyACL/AscendCL，不允许 CPU、CUDA、PyTorch、ONNX Runtime 或 TensorRT 模型回退。

在加载 CANN 环境后，可只读查询模块位置和版本。`acl` 没有公开 `__version__` 时，应以 CANN 安装信息为准，不能把“无版本属性”误判为导入失败。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHON=/usr/local/miniconda3/envs/agileagent/bin/python
"$PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata

targets = {
    "PyACL": ("acl", None),
    "NumPy": ("numpy", "numpy"),
    "OpenCV": ("cv2", "opencv-python"),
    "Pillow": ("PIL", "Pillow"),
    "PyTorch": ("torch", "torch"),
    "Starlette": ("starlette", "starlette"),
    "Uvicorn": ("uvicorn", "uvicorn"),
    "python-multipart": ("multipart", "python-multipart"),
    "pluggy": ("pluggy", "pluggy"),
    "pytest": ("pytest", "pytest"),
}

for label, (module_name, distribution) in targets.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is None and distribution is not None:
            version = metadata.version(distribution)
        print(f"{label:18} version={version or '<由 CANN 提供>'} file={getattr(module, '__file__', '<内建>')}")
    except Exception as exc:
        print(f"{label:18} unavailable: {type(exc).__name__}: {exc}")
PY
```

## CANN、驱动、固件与 NPU 状态

当前启停脚本只加载已经安装的 CANN 环境脚本，不安装或升级任何系统组件：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
atc --version
find /usr/local/Ascend -maxdepth 4 -type f -name version.cfg -print
env | grep -E '^(ASCEND|PYTHONPATH|LD_LIBRARY_PATH)='
```

驱动和固件版本文件在不同镜像中可能略有差异，可只读取实际存在的文件：

```bash
for version_file in \
  /usr/local/Ascend/driver/version.info \
  /usr/local/Ascend/firmware/version.info \
  /usr/local/Ascend/ascend-toolkit/latest/version.cfg
do
  if [ -r "$version_file" ]; then
    printf '\n== %s ==\n' "$version_file"
    sed -n '1,120p' "$version_file"
  fi
done
```

`npu-smi info` 用于确认 NPU 数量、芯片名、Health、温度、功耗、AI Core 和 NPU 内存占用。部分 CANN/驱动版本不支持 `npu-smi info -m`；参数报错不等于芯片故障，应以当前版本帮助信息和基础 `npu-smi info` 为准。

本项目冻结现有 CANN `7.0.RC1`、驱动和固件。不要为了查看版本执行 `apt upgrade`、`conda update`、Toolkit 安装器或固件升级；任何 CANN/OM 变化都会使既有精度和性能证据失效。

## Ascend 正式配置与 OM

正式配置位于 release 的 `src` 目录内；它在源码树中的相对路径为 [`configs/agent_pipeline_ascend310b.yaml`](../configs/agent_pipeline_ascend310b.yaml)。关键固定值如下：

| 配置项 | 当前值 |
|---|---|
| `inference.backend` | `ascend_acl` |
| `inference.batch_size` | `1` |
| `runtime.local_python` | `/usr/local/miniconda3/envs/agileagent/bin/python` |
| `runtime.server_host` / `server_port` | `127.0.0.1` / `8501` |
| `ascend_backend.device_id` | `0` |
| `ascend_backend.soc_version` | `Ascend310B1` |
| `ascend_backend.cann_version` | `7.0.RC1` |
| `ascend_backend.precision` | `mixed_float16` |

仓库当前 [`configs/agent_pipeline_ascend310b.yaml`](../configs/agent_pipeline_ascend310b.yaml) 还包含 `execution_mode: async_stream`、`encoded_preprocessing: cpu` 和 `decoding.opencv_threads`，但板端正式 release 使用较早的配置 schema，不接受这些字段。本次迁移没有把仓库 YAML 整份覆盖到板端，只在 release 原配置上替换了一处 `runtime.local_python`。向旧 release 部署配置前必须使用该 release 自己的 `load_config()` 校验；不能假设 main 的 YAML 可直接向后兼容。

迁移后板端两份受控运维文件的身份如下；第二项是“旧 release 原 YAML + 单行解释器路径修改”的哈希，不应与当前仓库完整 YAML 的哈希混用：

| 板端文件 | SHA256 |
|---|---|
| `<release>/src/scripts/start_agent_ascend310b.sh` | `317f19ef82175d0dff0d29ad7698e661e9faf2f009ca1c31a80fe9b4a5a43010` |
| `<release>/src/configs/agent_pipeline_ascend310b.yaml` | `adca54330bdd07bc35207e88e71f75830d4a121f7cfcec3aa7317826380aa3dc` |

| OM | 正式路径 | SHA256 |
|---|---|---|
| Base | `<release>/om/base_detector.om` | `b78b7fa086436f1c10010f879bc4d836d23e0679fb2e189736ab337c1beb1f4d` |
| Incremental | `<release>/om/incremental_detector.om` | `ec8a321a966ddba0cdbb37d2223b8150a8dd39cd3c10752e85155d1f9ef54b27` |
| Scene | `<release>/om/scene_sensor_net.om` | `debdb9c3f550ef24d0d14d736a700d65960abcd67899e00f493e208e6d90e725` |

只读复核文件身份：

```bash
RELEASE_ROOT=/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2
sha256sum \
  "$RELEASE_ROOT/om/base_detector.om" \
  "$RELEASE_ROOT/om/incremental_detector.om" \
  "$RELEASE_ROOT/om/scene_sensor_net.om"
```

哈希不一致时停止部署检查，不要就地重新编译或替换正式 OM。

## 服务状态、启停与健康检查

启动脚本先加载 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，再设置：

```text
AGILE_AGENT_CONFIG=<release>/src/configs/agent_pipeline_ascend310b.yaml
```

随后使用 `/usr/local/miniconda3/envs/agileagent/bin/python -m uvicorn` 启动服务。可用 `AGILE_AGENT_ASCEND_ENV` 显式覆盖环境路径，默认值就是 `/usr/local/miniconda3/envs/agileagent`。PID 文件为 `<release>/agent-web.pid`；停止脚本会核对 PID 对应命令确实是 AgileAgent Uvicorn 后再发送终止信号。

以下检查均为只读：

```bash
RELEASE_ROOT=/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2
PID_FILE="$RELEASE_ROOT/agent-web.pid"

if [ -r "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  ps -fp "$PID"
  tr '\0' ' ' < "/proc/$PID/cmdline"
  printf '\n'
fi

ss -lnt | grep ':8501'
curl -fsS http://127.0.0.1:8501/api/health
```

`/api/health` 正常时应至少返回 `status: ready`、`backend: ascend_acl` 和 `device: ascend:0`。它会触发模型提供器初始化，因此比只检查端口更有意义。

2026-08-14 的迁移验收确认正式进程解释器为 `/usr/local/miniconda3/envs/agileagent/bin/python3.9`；同一真实 PNG 在切换前后均返回 6 个检测，去除耗时字段后的响应语义完全一致，三个 OM 的 SHA256 未变化。切换后单次响应记录为 `inference_ms=29.9`、`system_total_ms=79.4`，只用于证明链路可运行，不是 FPS 或稳定性基准。

切换后的独立复核还确认 `pip check` 返回 `No broken requirements found`，PyACL 与核心模块均能导入；`npu-smi info` 显示 Ascend310B1 `Health: OK`、AI Core 空闲观测 `0%`、NPU 内存 `9479 / 11577 MB`、温度 `61°C`。这些仍是单次快照，不能替代长时间压力测试。

需要人工启停时，在确认 `RELEASE_ROOT` 后使用 release 自带脚本：

```bash
RELEASE_ROOT=/home/HwHiAiUser/agileagent/releases/212705a26d4414eff4e00604ce37c54d2ae729b2
AGILE_AGENT_ASCEND_RELEASE="$RELEASE_ROOT" bash "$RELEASE_ROOT/src/scripts/start_agent_ascend310b.sh"
AGILE_AGENT_ASCEND_RELEASE="$RELEASE_ROOT" bash "$RELEASE_ROOT/src/scripts/stop_agent_ascend310b.sh"
```

启停会改变运行状态，不属于日常只读查询。当前正式方式不是 systemd 服务，不要假定 `systemctl restart agile-agent` 有效。

## 系统资源、公网与 LXQt 检查

### 内存、存储和温度

```bash
free -h
df -hT /
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
npu-smi info
for zone in /sys/class/thermal/thermal_zone*/temp; do
  [ -r "$zone" ] && printf '%s: %s\n' "$zone" "$(cat "$zone")"
done
```

thermal zone 通常以千分之一摄氏度输出，具体传感器名称和量纲以同目录的 `type` 文件及设备文档为准。单次读数不能替代一小时稳定性、峰值 NPU 内存和热降频检查。

### 公网访问

```bash
ip route
getent hosts www.hiascend.com
curl -I --connect-timeout 5 --max-time 15 https://www.hiascend.com/
env | grep -iE '^(http|https|no)_proxy='
```

判断顺序为：先确认默认路由，再确认 DNS，最后确认 HTTPS。若只有设置代理后可访问，记录代理是否由当前 shell 临时提供，但不要记录认证信息。访问 `127.0.0.1:8501` 和 `192.168.137.0/24` 时应绕过代理。

### LXQt 桌面

SSH 登录并不证明显示器上的 LXQt 会话正常，可用以下只读命令查看显示管理器和会话进程：

```bash
systemctl get-default
systemctl status display-manager --no-pager
loginctl list-sessions
ps -u HwHiAiUser -o pid,comm,args | grep -E 'lxqt-session|lxqt-panel|pcmanfm-qt' | grep -v grep
```

`graphical.target`、显示管理器 active 且存在 LXQt 会话进程，说明桌面组件正在运行；仍无画面时应继续检查 HDMI、分辨率、显示管理器日志和物理显示器，而不是修改 CANN 或 Python 环境。

## 常见故障

| 现象 | 优先检查 | 处理原则 |
|---|---|---|
| `Connection timed out` / 22 端口失败 | 网线、网卡地址、板端 IP、VPN、防火墙、ICS | 先恢复同子网和 TCP 22，不要改 CANN |
| `No route to host` | `ipconfig`、板端 `ip route`、掩码 | 清除错误静态路由，确认 `192.168.137.0/24` 可达 |
| `Permission denied (publickey,password)` | 用户名大小写、密码输入、公钥是否已部署 | 用交互密码验证；不要把密码写入命令行 |
| 主机密钥变化 | 是否重刷系统或 IP 冲突 | 先可信核对新指纹，再删除旧 known_hosts 项 |
| `import acl` 失败 | 是否 source CANN、是否激活/调用了 `agileagent` | 加载现有 CANN 环境脚本并显式调用 `/usr/local/miniconda3/envs/agileagent/bin/python` |
| `Invalid agent config` 且提示未知字段 | 是否把当前 main YAML 整份复制给旧 release | 恢复 release 自己的配置，只迁移被该 schema 支持的字段，并用 release 的 `load_config()` 预检 |
| `/api/health` 连接失败 | PID、8501 监听、启动终端输出 | 先查进程和端口，不要重复启动多个实例 |
| health 返回 `503` | OM 路径/哈希、CANN 环境、NPU Health、配置路径 | 保留错误原文并停止；禁止 CPU/CUDA 回退 |
| 板端无公网 | 默认路由、DNS、ICS、企业代理 | 分层排查；ICS 不会自动复制应用层代理 |
| X11 可用但 LXQt 屏幕异常 | X11 转发与本地桌面是不同会话 | 查 display-manager、HDMI 和桌面会话，不改推理栈 |

SSH 详细诊断可以使用：

```powershell
ssh -vvv HwHiAiUser@192.168.137.100
```

`-vvv` 输出可能包含本机用户名、IP、密钥文件路径和主机指纹，分享前应脱敏；它不会要求把密码作为命令行参数。

## 安全与变更边界

- 不升级或替换 CANN、驱动、固件，也不混用其他 310B 设备的 OM。
- 不在 Miniconda base、系统 Python 或正式命名环境 `agileagent` 中临时 `pip install`、`conda update` 或 `apt upgrade`。
- 新依赖先进入独立候选环境验证；通过并固定版本后，才允许按受控迁移流程重建 `agileagent`，并重新执行 OM 哈希、真实 PNG 推理、精度、性能和稳定性门禁。
- 除本文按设备所有者要求记录的官方出厂默认账户信息外，不把当前密码、私钥、代理认证、竞赛数据、板端日志、ONNX 或 OM 提交到 Git。
- 日常运维使用 `HwHiAiUser`；`root` 只用于必须修改系统所有目录的受控步骤。设备仍使用出厂凭据时应在部署稳定后按官方流程分别改密，文档和脚本不得记录修改后的明文密码。
- 不直接覆盖当前正式 release；候选与正式目录保持隔离，验收失败时正式目录不变。

更完整的编译、精度与性能验收流程见 [Ascend 310B 稳定加速设计](ascend-310b-deployment.md)，当前性能、风险和证据边界见 [Ascend 310B 当前工程状态](ascend-310b-current-status.md)。
