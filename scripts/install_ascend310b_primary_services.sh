#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  install_ascend310b_primary_services.sh MAIN_RELEASE_ROOT ROLLBACK_RELEASE_ROOT [MAIN_INTERNAL_PORT]

在已物化并通过release校验的目录上安装两个服务：
  - 满分主实例：内部端口18501；
  - 原三OM回滚实例：继续监听8501。
随后以精确iptables loopback规则原子切换新连接，8502继续保留为候选端口。
EOF
}

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi
if (( EUID != 0 )); then
  printf '服务安装与原子切换必须由root执行。\n' >&2
  exit 1
fi

APP_USER="${AGILE_AGENT_ASCEND_USER:-HwHiAiUser}"
MAIN_ROOT="$(readlink -f "$1")"
ROLLBACK_ROOT="$(readlink -f "$2")"
MAIN_PORT="${3:-18501}"
PUBLIC_PORT=8501
PYTHON=/usr/local/miniconda3/envs/agileagent/bin/python
MAIN_CONFIG="$MAIN_ROOT/configs/agent_pipeline_ascend310b.yaml"
ROLLBACK_CONFIG="$ROLLBACK_ROOT/src/configs/agent_pipeline_ascend310b.yaml"
ROUTE_SOURCE="$MAIN_ROOT/src/scripts/manage_ascend310b_primary_route.sh"
ROUTE_INSTALL=/usr/local/sbin/agileagent-ascend310b-primary-route
MAIN_UNIT=/etc/systemd/system/agileagent-ascend310b-main.service
ROLLBACK_UNIT=/etc/systemd/system/agileagent-ascend310b-rollback.service
ROUTE_UNIT=/etc/systemd/system/agileagent-ascend310b-route.service
CANONICAL_LINK=/home/HwHiAiUser/agileagent/releases/ascend310b-main
ROUTE_APPLIED=0
SUCCESS=0
CURRENT_STEP="preflight"

step() {
  CURRENT_STEP="$1"
  printf '[ascend310b-primary] %s\n' "$CURRENT_STEP"
}

fail() {
  printf '[ascend310b-primary] %s：%s\n' "$CURRENT_STEP" "$1" >&2
  return 1
}

require_active() {
  local unit="$1"
  systemctl is-active --quiet "$unit" || fail "systemd unit未active：${unit}"
}

health_has_field() {
  local port="$1"
  local expected="$2"
  local payload
  payload="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/api/health")" || return 1
  [[ "$payload" == *"$expected"* ]]
}

require_health_field() {
  local port="$1"
  local expected="$2"
  health_has_field "$port" "$expected" || \
    fail "健康检查未包含 ${expected}：http://127.0.0.1:${port}/api/health"
}

require_loopback_listener() {
  local port="$1"
  local sockets
  # 不使用`ss | grep -q`：pipefail下grep提前退出会使ss因SIGPIPE误报失败。
  sockets="$(ss -H -ltn "sport = :${port}")" || fail "无法读取端口${port}的监听状态"
  [[ "$sockets" == *"127.0.0.1:${port}"* ]] || fail "127.0.0.1:${port}没有监听器"
}

step "检查发布输入"
for path in \
  "$MAIN_ROOT/src/scripts/start_agent_ascend310b.sh" \
  "$MAIN_ROOT/src/scripts/stop_agent_ascend310b.sh" \
  "$MAIN_ROOT/src/tools/95_verify_ascend_release.py" \
  "$MAIN_CONFIG" \
  "$ROLLBACK_ROOT/src/scripts/start_agent_ascend310b.sh" \
  "$ROLLBACK_ROOT/src/scripts/stop_agent_ascend310b.sh" \
  "$ROLLBACK_CONFIG" \
  "$ROUTE_SOURCE"; do
  test -e "$path" || { printf '缺少发布输入：%s\n' "$path" >&2; exit 1; }
done
if [[ ! "$MAIN_PORT" =~ ^[0-9]+$ ]] || (( MAIN_PORT == 8501 || MAIN_PORT == 8502 )); then
  printf '主实例内部端口必须避开8501/8502：%s\n' "$MAIN_PORT" >&2
  exit 2
fi
id "$APP_USER" >/dev/null
test -x "$PYTHON" || { printf '板端Python不可执行：%s\n' "$PYTHON" >&2; exit 1; }
MAIN_LAYOUT="$($PYTHON - "$MAIN_CONFIG" <<'PY'
import sys
import yaml

payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
layout = str((payload.get("ascend_backend") or {}).get("model_layout") or "")
if layout not in {"shared_backbone_dual_head_v1", "independent_yolo26_e2e_v1"}:
    raise SystemExit(f"正式主实例布局非法：{layout}")
print(layout)
PY
)"
require_health_field "$PUBLIC_PORT" '"status":"ready"'

runuser -u "$APP_USER" -- env AGILE_AGENT_CONFIG="$MAIN_CONFIG" \
  "$PYTHON" "$MAIN_ROOT/src/tools/95_verify_ascend_release.py" \
  --config "$MAIN_CONFIG" --require-validation >/dev/null

cleanup_failure() {
  local status="$?"
  if (( SUCCESS == 0 )); then
    if (( ROUTE_APPLIED == 1 )); then
      "$ROUTE_INSTALL" remove "$MAIN_PORT" >/dev/null 2>&1 || true
    fi
    systemctl stop agileagent-ascend310b-main.service >/dev/null 2>&1 || true
    if ! curl -fsS --max-time 5 "http://127.0.0.1:${PUBLIC_PORT}/api/health" >/dev/null 2>&1; then
      systemctl start agileagent-ascend310b-rollback.service >/dev/null 2>&1 || \
        env AGILE_AGENT_ASCEND_RELEASE="$ROLLBACK_ROOT" \
          AGILE_AGENT_CONFIG="$ROLLBACK_CONFIG" \
          AGILE_AGENT_ASCEND_PORT="$PUBLIC_PORT" \
          /usr/bin/bash "$ROLLBACK_ROOT/src/scripts/start_agent_ascend310b.sh" \
          >/dev/null 2>&1 &
    fi
    printf '正式提升失败，已尝试恢复三OM回滚服务；step=%s status=%s\n' \
      "$CURRENT_STEP" "$status" >&2
  fi
  exit "$status"
}
trap cleanup_failure EXIT

step "安装路由管理器和systemd units"
install -o root -g root -m 0755 "$ROUTE_SOURCE" "$ROUTE_INSTALL"

main_tmp="$(mktemp)"
rollback_tmp="$(mktemp)"
route_tmp="$(mktemp)"
cat >"$main_tmp" <<EOF
[Unit]
Description=AgileAgent Ascend310B full-score primary
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$MAIN_ROOT/src
Environment=AGILE_AGENT_ASCEND_RELEASE=$MAIN_ROOT
Environment=AGILE_AGENT_CONFIG=$MAIN_CONFIG
Environment=AGILE_AGENT_ASCEND_PORT=$MAIN_PORT
Environment=AGILE_AGENT_ASCEND_PID_FILE=$MAIN_ROOT/agent-web.pid
ExecStart=/usr/bin/bash $MAIN_ROOT/src/scripts/start_agent_ascend310b.sh
ExecStop=/usr/bin/bash $MAIN_ROOT/src/scripts/stop_agent_ascend310b.sh
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
cat >"$rollback_tmp" <<EOF
[Unit]
Description=AgileAgent Ascend310B three-OM rollback listener
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$ROLLBACK_ROOT/src
Environment=AGILE_AGENT_ASCEND_RELEASE=$ROLLBACK_ROOT
Environment=AGILE_AGENT_CONFIG=$ROLLBACK_CONFIG
Environment=AGILE_AGENT_ASCEND_PORT=$PUBLIC_PORT
Environment=AGILE_AGENT_ASCEND_PID_FILE=$ROLLBACK_ROOT/agent-web.pid
ExecStart=/usr/bin/bash $ROLLBACK_ROOT/src/scripts/start_agent_ascend310b.sh
ExecStop=/usr/bin/bash $ROLLBACK_ROOT/src/scripts/stop_agent_ascend310b.sh
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
cat >"$route_tmp" <<EOF
[Unit]
Description=AgileAgent Ascend310B atomic primary route
Requires=agileagent-ascend310b-main.service agileagent-ascend310b-rollback.service
After=agileagent-ascend310b-main.service agileagent-ascend310b-rollback.service

[Service]
Type=oneshot
ExecStart=$ROUTE_INSTALL apply $MAIN_PORT
ExecStop=$ROUTE_INSTALL remove $MAIN_PORT
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
install -o root -g root -m 0644 "$main_tmp" "$MAIN_UNIT"
install -o root -g root -m 0644 "$rollback_tmp" "$ROLLBACK_UNIT"
install -o root -g root -m 0644 "$route_tmp" "$ROUTE_UNIT"
rm -f -- "$main_tmp" "$rollback_tmp" "$route_tmp"
systemctl daemon-reload

step "启动并验证满分主实例"
systemctl enable --now agileagent-ascend310b-main.service
for _ in $(seq 1 180); do
  if health_has_field "$MAIN_PORT" "\"model_layout\":\"$MAIN_LAYOUT\""; then
    break
  fi
  sleep 1
done
require_health_field "$MAIN_PORT" "\"model_layout\":\"$MAIN_LAYOUT\""

step "原子切换公共8501新连接"
"$ROUTE_INSTALL" apply "$MAIN_PORT"
ROUTE_APPLIED=1

# 新连接已进入满分主实例；此后重管旧进程不会影响公共8501。
step "将三OM监听器纳入回滚service"
env AGILE_AGENT_ASCEND_RELEASE="$ROLLBACK_ROOT" \
  AGILE_AGENT_ASCEND_PID_FILE="$ROLLBACK_ROOT/agent-web.pid" \
  /usr/bin/bash "$ROLLBACK_ROOT/src/scripts/stop_agent_ascend310b.sh"
systemctl enable --now agileagent-ascend310b-rollback.service
systemctl enable agileagent-ascend310b-route.service
systemctl start agileagent-ascend310b-route.service

step "更新正式release链接"
link_tmp="${CANONICAL_LINK}.tmp.$$"
ln -s "$MAIN_ROOT" "$link_tmp"
mv -Tf "$link_tmp" "$CANONICAL_LINK"

step "执行正式发布收尾验收"
require_active agileagent-ascend310b-main.service
require_active agileagent-ascend310b-rollback.service
require_active agileagent-ascend310b-route.service
require_health_field "$MAIN_PORT" "\"model_layout\":\"$MAIN_LAYOUT\""
require_health_field "$PUBLIC_PORT" "\"model_layout\":\"$MAIN_LAYOUT\""
require_loopback_listener "$PUBLIC_PORT"
require_loopback_listener "$MAIN_PORT"

SUCCESS=1
trap - EXIT
printf 'Ascend310B满分主线已原子提升：public=%s primary=%s rollback_listener=%s candidate=%s release=%s\n' \
  "$PUBLIC_PORT" "$MAIN_PORT" "$PUBLIC_PORT" 8502 "$MAIN_ROOT"
