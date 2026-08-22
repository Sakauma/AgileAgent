#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  manage_ascend310b_primary_route.sh apply [MAIN_INTERNAL_PORT]
  manage_ascend310b_primary_route.sh remove [MAIN_INTERNAL_PORT]
  manage_ascend310b_primary_route.sh status [MAIN_INTERNAL_PORT]

通过精确的loopback NAT规则，把板端公共8501的新连接原子转发到满分主实例。
旧三OM进程继续监听8501；删除规则即可立即回滚。8502不参与正式路由。
EOF
}

ACTION="${1:-}"
MAIN_PORT="${2:-18501}"
PUBLIC_PORT=8501
COMMENT="AGILE_AGENT_ASCEND310B_PRIMARY"
IPTABLES="${AGILE_AGENT_IPTABLES:-/usr/sbin/iptables-legacy}"
SUPPORTED_PRIMARY_LAYOUTS=(
  shared_backbone_dual_head_v1
  independent_yolo26_e2e_v1
)

if [[ ! "$ACTION" =~ ^(apply|remove|status)$ ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$MAIN_PORT" =~ ^[0-9]+$ ]] || (( MAIN_PORT < 1 || MAIN_PORT > 65535 )); then
  printf '主实例内部端口非法：%s\n' "$MAIN_PORT" >&2
  exit 2
fi
if (( MAIN_PORT == PUBLIC_PORT || MAIN_PORT == 8502 )); then
  printf '主实例内部端口必须避开8501/8502：%s\n' "$MAIN_PORT" >&2
  exit 2
fi
if (( EUID != 0 )); then
  printf '端口原子路由必须由root执行。\n' >&2
  exit 1
fi
test -x "$IPTABLES" || { printf 'iptables不可执行：%s\n' "$IPTABLES" >&2; exit 1; }

RULE=(
  -p tcp -d 127.0.0.1 --dport "$PUBLIC_PORT"
  -m comment --comment "$COMMENT"
  -j REDIRECT --to-ports "$MAIN_PORT"
)

has_rule() {
  "$IPTABLES" -t nat -C OUTPUT "${RULE[@]}" >/dev/null 2>&1
}

health_ready() {
  local port="$1"
  local expected_layout="${2:-}"
  local payload
  payload="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/api/health")" || return 1
  [[ "$payload" == *'"status":"ready"'* ]] || return 1
  if [[ -n "$expected_layout" ]]; then
    [[ "$payload" == *"\"model_layout\":\"${expected_layout}\""* ]] || return 1
  fi
}

primary_layout() {
  local port="$1"
  local payload
  local layout
  payload="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/api/health")" || return 1
  [[ "$payload" == *'"status":"ready"'* ]] || return 1
  for layout in "${SUPPORTED_PRIMARY_LAYOUTS[@]}"; do
    if [[ "$payload" == *"\"model_layout\":\"${layout}\""* ]]; then
      printf '%s\n' "$layout"
      return 0
    fi
  done
  return 1
}

remove_all() {
  while has_rule; do
    "$IPTABLES" -t nat -D OUTPUT "${RULE[@]}"
  done
}

case "$ACTION" in
  apply)
    layout="$(primary_layout "$MAIN_PORT")" || {
      printf '满分主实例未ready或布局不受支持：http://127.0.0.1:%s\n' "$MAIN_PORT" >&2
      exit 1
    }
    if ! has_rule; then
      "$IPTABLES" -t nat -I OUTPUT 1 "${RULE[@]}"
    fi
    if ! health_ready "$PUBLIC_PORT" "$layout"; then
      remove_all
      printf '8501原子切换后的健康检查失败，已自动删除路由规则。\n' >&2
      exit 1
    fi
    printf 'public=%s primary=%s layout=%s route=applied rollback_listener=preserved candidate=8502\n' \
      "$PUBLIC_PORT" "$MAIN_PORT" "$layout"
    ;;
  remove)
    remove_all
    if health_ready "$PUBLIC_PORT"; then
      printf 'public=%s route=removed rollback=ready\n' "$PUBLIC_PORT"
    else
      # 规则删除是remove的原子职责。回滚监听器可能正在systemd重启，
      # 这不应让ExecStop被标记为failed；严格健康验收由status或安装器执行。
      printf '已删除主线路由，但三OM回滚服务尚未ready，请检查rollback service。\n' >&2
      printf 'public=%s route=removed rollback=not_ready\n' "$PUBLIC_PORT"
    fi
    ;;
  status)
    if has_rule; then
      layout="$(primary_layout "$MAIN_PORT")" || {
        printf 'route=primary public=%s target=%s health=invalid\n' \
          "$PUBLIC_PORT" "$MAIN_PORT" >&2
        exit 1
      }
      printf 'route=primary public=%s target=%s layout=%s\n' \
        "$PUBLIC_PORT" "$MAIN_PORT" "$layout"
      health_ready "$PUBLIC_PORT" "$layout"
    else
      printf 'route=rollback public=%s\n' "$PUBLIC_PORT"
      health_ready "$PUBLIC_PORT"
    fi
    ;;
esac
