#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  install_ascend310b_primary_services.sh MAIN_RELEASE_ROOT ROLLBACK_RELEASE_ROOT [MAIN_INTERNAL_PORT]

在已物化并通过release校验的目录上安装两个服务：
  - 满分共享双头主实例：内部端口18501；
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
curl -fsS --max-time 5 "http://127.0.0.1:${PUBLIC_PORT}/api/health" | grep -q '"status":"ready"'

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
    printf '正式提升失败，已尝试恢复三OM回滚服务；status=%s\n' "$status" >&2
  fi
  exit "$status"
}
trap cleanup_failure EXIT

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

systemctl enable --now agileagent-ascend310b-main.service
for _ in $(seq 1 180); do
  if curl -fsS --max-time 5 "http://127.0.0.1:${MAIN_PORT}/api/health" \
    | grep -q '"model_layout":"shared_backbone_dual_head_v1"'; then
    break
  fi
  sleep 1
done
curl -fsS --max-time 5 "http://127.0.0.1:${MAIN_PORT}/api/health" \
  | grep -q '"model_layout":"shared_backbone_dual_head_v1"'

"$ROUTE_INSTALL" apply "$MAIN_PORT"
ROUTE_APPLIED=1

# 新连接已进入满分主实例；此后重管旧进程不会影响公共8501。
env AGILE_AGENT_ASCEND_RELEASE="$ROLLBACK_ROOT" \
  AGILE_AGENT_ASCEND_PID_FILE="$ROLLBACK_ROOT/agent-web.pid" \
  /usr/bin/bash "$ROLLBACK_ROOT/src/scripts/stop_agent_ascend310b.sh"
systemctl enable --now agileagent-ascend310b-rollback.service
systemctl enable agileagent-ascend310b-route.service
systemctl start agileagent-ascend310b-route.service

link_tmp="${CANONICAL_LINK}.tmp.$$"
ln -s "$MAIN_ROOT" "$link_tmp"
mv -Tf "$link_tmp" "$CANONICAL_LINK"

systemctl is-active --quiet agileagent-ascend310b-main.service
systemctl is-active --quiet agileagent-ascend310b-rollback.service
systemctl is-active --quiet agileagent-ascend310b-route.service
curl -fsS --max-time 5 "http://127.0.0.1:${PUBLIC_PORT}/api/health" \
  | grep -q '"model_layout":"shared_backbone_dual_head_v1"'
ss -ltn | grep -q "127.0.0.1:${PUBLIC_PORT}"
ss -ltn | grep -q "127.0.0.1:${MAIN_PORT}"

SUCCESS=1
trap - EXIT
printf 'Ascend310B满分主线已原子提升：public=%s primary=%s rollback_listener=%s candidate=%s release=%s\n' \
  "$PUBLIC_PORT" "$MAIN_PORT" "$PUBLIC_PORT" 8502 "$MAIN_ROOT"
