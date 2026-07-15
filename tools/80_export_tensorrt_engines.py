from __future__ import annotations

import argparse
import json

from fair_agent.core.config import load_config
from fair_agent.modules.tensorrt_export import export_or_verify_engines, write_export_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="按Agent主YAML导出或校验TensorRT engine。")
    parser.add_argument("--config", default="configs/agent_pipeline.yaml")
    parser.add_argument("--verify-only", action="store_true", help="只校验现有engine，不执行导出。")
    args = parser.parse_args()
    config = load_config(
        args.config,
        allow_unverified_tensorrt_hashes=not args.verify_only,
    )
    result = export_or_verify_engines(config, verify_only=args.verify_only)
    if not args.verify_only:
        result["config_update"] = write_export_hashes(args.config, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
