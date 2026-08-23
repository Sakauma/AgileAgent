#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  printf '用法: %s CONFIG IMAGE_ROOT OUTPUT_ROOT REQUEST_COUNT [WARMUP_COUNT] [PYTHON]\n' "$0" >&2
  exit 2
fi

config="$1"
image_root="$2"
output_root="$3"
request_count="$4"
warmup_count="${5:-0}"
python="${6:-/usr/local/miniconda3/envs/agileagent/bin/python}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -e "$output_root" ]; then
  printf 'profile输出目录已存在，拒绝覆盖：%s\n' "$output_root" >&2
  exit 20
fi
test -f "$config"
test -d "$image_root"
test -x "$python"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
command -v msprof >/dev/null

mkdir -p "$output_root"
application=(
  "$python" "$repo/tools/100_profile_ascend_request.py"
  --config "$config"
  --image-root "$image_root"
  --output "$output_root/application-report.json"
  --request-count "$request_count"
  --warmup-count "$warmup_count"
  --confidence 0.5
)
printf -v application_command '%q ' "${application[@]}"
printf '%s\n' "$application_command" > "$output_root/application-command.txt"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\nexec'
  printf ' %q' "${application[@]}"
  printf '\n'
} > "$output_root/application.sh"
chmod 700 "$output_root/application.sh"

export AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1
cd "$repo"
msprof \
  --output="$output_root/raw" \
  --application="$output_root/application.sh" \
  --model-execution=on \
  --runtime-api=on \
  --task-time=on \
  --ai-core=on \
  --aic-mode=task-based \
  --dvpp-profiling=on \
  --sys-profiling=on \
  --sys-pid-profiling=on \
  --sys-hardware-mem=on 2>&1 | tee "$output_root/msprof.log"

sha256sum "$output_root/application-report.json"
printf 'raw_file_count='; find "$output_root/raw" -type f | wc -l
du -sh "$output_root/raw"
