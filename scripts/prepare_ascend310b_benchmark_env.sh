#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-check}"
USER_CONFIG_ROOT="${XDG_CONFIG_HOME:-${HOME}/.config}"
AUTOSTART_DIR="${USER_CONFIG_ROOT}/autostart"
SYSTEM_SERVICE="/etc/systemd/system/agileagent-benchmark-governor.service"

AUTOSTART_FILES=(
  "lxqt-xscreensaver-autostart.desktop"
  "xscreensaver.desktop"
  "xfce4-screensaver.desktop"
)
SCREENSAVER_PROCESSES=(
  "xscreensaver"
  "xscreensaver-systemd"
  "xfce4-screensaver"
)
USER_UNITS=(
  "xscreensaver.service"
  "app-xscreensaver@autostart.service"
  'app-lxqt\x2dxscreensaver\x2dautostart@autostart.service'
  'app-xfce4\x2dscreensaver@autostart.service'
)

usage() {
  printf 'Usage: %s apply-user|apply-system|check\n' "$0" >&2
  exit 2
}

write_autostart_override() {
  local name="$1"
  local target="${AUTOSTART_DIR}/${name}"
  local temporary
  temporary="$(mktemp "${AUTOSTART_DIR}/.${name}.XXXXXX")"
  {
    printf '[Desktop Entry]\n'
    printf 'Type=Application\n'
    printf 'Name=AgileAgent benchmark screensaver suppression\n'
    printf 'Hidden=true\n'
  } >"${temporary}"
  chmod 0644 "${temporary}"
  mv -f -- "${temporary}" "${target}"
}

apply_user() {
  if (( EUID == 0 )); then
    printf 'apply-user must run as the desktop/benchmark user, not root.\n' >&2
    exit 1
  fi
  mkdir -p -- "${AUTOSTART_DIR}"
  local name process unit
  for name in "${AUTOSTART_FILES[@]}"; do
    write_autostart_override "${name}"
  done
  for process in "${SCREENSAVER_PROCESSES[@]}"; do
    pkill -x -- "${process}" 2>/dev/null || true
  done
  if command -v systemctl >/dev/null 2>&1; then
    for unit in "${USER_UNITS[@]}"; do
      if systemctl --user cat "${unit}" >/dev/null 2>&1; then
        systemctl --user disable --now "${unit}" >/dev/null 2>&1 || true
        systemctl --user mask --now "${unit}" >/dev/null 2>&1 || true
      fi
    done
    systemctl --user daemon-reload >/dev/null 2>&1 || true
  fi
  sleep 1
  check_user
}

governor_policies() {
  compgen -G '/sys/devices/system/cpu/cpufreq/policy*/scaling_governor' || true
}

apply_system() {
  if (( EUID != 0 )); then
    printf 'apply-system requires root.\n' >&2
    exit 1
  fi
  mapfile -t policies < <(governor_policies)
  if (( ${#policies[@]} == 0 )); then
    printf 'cpu_governor=unsupported\n'
    exit 0
  fi
  local policy available
  for policy in "${policies[@]}"; do
    available="$(<"${policy%/*}/scaling_available_governors")"
    if [[ " ${available} " != *' performance '* ]]; then
      printf 'performance governor unavailable for %s\n' "${policy}" >&2
      exit 1
    fi
  done
  local temporary
  temporary="$(mktemp /etc/systemd/system/.agileagent-benchmark-governor.XXXXXX)"
  {
    printf '[Unit]\n'
    printf 'Description=AgileAgent Ascend benchmark CPU governor\n'
    printf 'After=multi-user.target\n\n'
    printf '[Service]\n'
    printf 'Type=oneshot\n'
    printf "ExecStart=/bin/sh -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do printf performance > \"\$f\"; done'\n"
    printf 'RemainAfterExit=yes\n\n'
    printf '[Install]\n'
    printf 'WantedBy=multi-user.target\n'
  } >"${temporary}"
  chmod 0644 "${temporary}"
  mv -f -- "${temporary}" "${SYSTEM_SERVICE}"
  systemctl daemon-reload
  systemctl enable --now agileagent-benchmark-governor.service
  check_governor
}

check_user() {
  local failed=0 name process target
  for name in "${AUTOSTART_FILES[@]}"; do
    target="${AUTOSTART_DIR}/${name}"
    if [[ ! -f "${target}" ]] || ! grep -Fxq 'Hidden=true' "${target}"; then
      printf 'autostart override missing: %s\n' "${target}" >&2
      failed=1
    fi
  done
  for process in "${SCREENSAVER_PROCESSES[@]}"; do
    if pgrep -x -- "${process}" >/dev/null 2>&1; then
      printf 'screensaver process still running: %s\n' "${process}" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
  printf 'screensaver_guard=passed\n'
}

check_governor() {
  mapfile -t policies < <(governor_policies)
  if (( ${#policies[@]} == 0 )); then
    printf 'cpu_governor=unsupported\n'
    return 0
  fi
  local policy governor available
  for policy in "${policies[@]}"; do
    governor="$(<"${policy}")"
    available="$(<"${policy%/*}/scaling_available_governors")"
    if [[ " ${available} " == *' performance '* ]] && [[ "${governor}" != 'performance' ]]; then
      printf 'cpu governor is not performance: %s=%s\n' "${policy}" "${governor}" >&2
      return 1
    fi
    printf '%s=%s\n' "${policy}" "${governor}"
  done
}

case "${ACTION}" in
  apply-user)
    apply_user
    check_governor
    ;;
  apply-system)
    apply_system
    ;;
  check)
    check_user
    check_governor
    ;;
  *)
    usage
    ;;
esac
