<!-- generated-by: gsd-doc-writer -->
# Ascend 310B SSH 连接与运行环境

本文记录 Atlas 200I DK A2 的连接方式、CANN/Python 环境、正式 release 和服务拓扑。连接凭据由设备交互认证或 SSH key 管理，仓库文档只记录非敏感运行参数。

## 环境快照

| 项目 | 当前值 |
| --- | --- |
| SSH 地址 | `192.168.137.100:22` |
| 日常账户 | `HwHiAiUser` |
| 管理账户 | `root`，仅用于 systemd、路由和系统级操作 |
| 设备 | Atlas 200I DK A2，`aarch64` |
| SoC | Ascend310B1 |
| 操作系统 | Ubuntu 22.04 LTS，Linux `5.10.0+` |
| CANN | `7.0.RC1` |
| Conda base | `/usr/local/miniconda3` |
| 正式环境 | `/usr/local/miniconda3/envs/agileagent` |
| 正式 Python | `/usr/local/miniconda3/envs/agileagent/bin/python` |
| 独立训练环境 | `~/agileagent/envs/agileagent_train`，`torch + torch_npu` |
| 正式 release | `/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-replica-pool-v1` |
| 回滚 release | `/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1` |
| 公共入口 | `127.0.0.1:8501` |
| 主实例 | `127.0.0.1:18501` |
| 候选端口 | `127.0.0.1:8502`，正式状态下空闲 |

## Windows 连接

检查端口：

```powershell
Test-NetConnection -ComputerName 192.168.137.100 -Port 22
```

日常登录：

```powershell
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  HwHiAiUser@192.168.137.100
```

仅在需要管理系统服务或路由时使用 root：

```powershell
ssh root@192.168.137.100
```

密码必须在 SSH 提示中交互输入，不写入仓库、命令行参数、终端日志、MobaXterm 宏或 VS Code 配置。优先使用 SSH key，并限制私钥文件权限。

## 本地端口转发

```powershell
ssh -N -L 8501:127.0.0.1:8501 HwHiAiUser@192.168.137.100
```

隧道建立后：

```powershell
Invoke-RestMethod http://127.0.0.1:8501/api/health
```

## Conda 环境

不依赖 shell 激活的检查：

```bash
/usr/local/miniconda3/bin/conda --version
/usr/local/miniconda3/bin/conda info --base
/usr/local/miniconda3/bin/conda env list
test -x /usr/local/miniconda3/envs/agileagent/bin/python

/usr/local/miniconda3/envs/agileagent/bin/python --version
/usr/local/miniconda3/envs/agileagent/bin/python -c \
  "import acl, numpy, cv2, PIL; print('imports ok')"
```

需要交互激活时：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate agileagent
printf 'CONDA_DEFAULT_ENV=%s\nCONDA_PREFIX=%s\n' \
  "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX"
readlink -f "$(command -v python)"
```

预期 `CONDA_DEFAULT_ENV=agileagent`，`CONDA_PREFIX=/usr/local/miniconda3/envs/agileagent`。正式启动脚本使用绝对解释器路径，不依赖交互式激活。

## CANN 与设备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
atc --version
/usr/local/miniconda3/envs/agileagent/bin/python -c \
  "import acl; print(acl.__file__)"
```

不要在部署过程中升级 CANN、驱动、固件、PyTorch 或 CUDA 类关键依赖。当前 release 的构建身份固定为 Ascend310B1、CANN `7.0.RC1`、`mixed_float16`。

## 当前 release 目录

```text
/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1/
├── src/
├── configs/agent_pipeline_ascend310b.yaml
├── om/
│   ├── base_detector.om
│   ├── incremental_detector.om
│   └── scene_sensor_net.om
├── provenance/
├── validation/
├── release.json
└── agent-web.pid
```

仓库副本：

```text
models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/
```

正式 release 物化：

```bash
cd /home/HwHiAiUser/agileagent/repo
./scripts/materialize_ascend310b_full_score_release.sh
```

已有目标时只读复核：

```bash
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

物化复原已构建的三个 OM、配置、来源与验收证据；训练、候选构建和服务切换分别由独立入口编排。

## 服务拓扑

正式设备已登记三个 systemd unit：

```text
agileagent-ascend310b-main.service       18501 的 4+2 主实例
agileagent-ascend310b-rollback.service   8501 的回滚 listener
agileagent-ascend310b-route.service      8501 -> 18501 精确路由
```

状态检查：

```bash
systemctl is-active agileagent-ascend310b-main.service
systemctl is-active agileagent-ascend310b-rollback.service
systemctl is-active agileagent-ascend310b-route.service

curl -fsS http://127.0.0.1:8501/api/health
curl -fsS http://127.0.0.1:18501/api/health
ss -H -ltn 'sport = :8501 or sport = :18501 or sport = :8502'
sudo /usr/local/sbin/agileagent-ascend310b-primary-route status 18501
```

正式运行态下三个 unit 均返回 `active`，`8501/18501` 均有物理 listener，`8502` 保持候选专用。2026-08-26 验收收尾后，三个服务已停止；按安装器或 systemd unit 再次启动即可恢复该拓扑。

公共健康响应必须包含：

```json
{
  "status": "ready",
  "backend": "ascend_acl",
  "device": "ascend:0",
  "validated": true,
  "validation_candidate": false,
  "model_layout": "independent_yolo26_e2e_v1",
  "context_mode": "model",
  "inference_replicas": 3,
  "generation_id": "incremental_detection_generation_4plus2"
}
```

## 新板直接启动

没有旧回滚 listener 的新板可以直接使用公共端口：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
AGILE_AGENT_ASCEND_ENV=/usr/local/miniconda3/envs/agileagent \
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

已有回滚 listener 的设备不要使用该命令占用 `8501`，应按 `docs/ascend-310b-deployment.md` 安装或更新双实例服务。

## 冒烟与性能复核

单图：

```bash
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

2026-08-29 正式 release 已再次按官方口径完成公共入口 `30 + 3×20` 复核：三轮为 `34.4918 / 33.6139 / 33.6015 FPS`、60 帧总耗时 `1770.051105 ms`、aggregate `33.897326 FPS`。计时包含图像解码、Scene、决策、Base/Incremental 检测、后处理和六列 TXT 写出；89 图重新评分也确认 Base/New/KRR 三项门禁全部通过。旧计时报告已移入 legacy engine-only 档案。

板端断网增量演示复用正式 Python 和独立训练环境：

```bash
cd /home/HwHiAiUser/agileagent/repo
./scripts/run_ascend310b_incremental_demo.sh /path/to/datasets_r2_inc_train
```

该入口已完成 `4→4+1→4+2` 实机精度、训练隔离和 OM 数值验收；其旧性能报告未包含正式结果落盘，现已归档。详细环境变量、训练/导出耗时和隔离配置见 `docs/ascend-310b-offline-incremental-demo.md`。

## 回滚

```bash
sudo /usr/local/sbin/agileagent-ascend310b-primary-route remove 18501
curl -fsS http://127.0.0.1:8501/api/health
```

重新提升：

```bash
sudo /usr/local/sbin/agileagent-ascend310b-primary-route apply 18501
curl -fsS http://127.0.0.1:8501/api/health
```

完整安装顺序、失败回退和候选构建命令见 `docs/ascend-310b-deployment.md`。
