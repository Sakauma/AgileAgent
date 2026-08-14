#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import load_config
from fair_agent.modules.ascend_release import verify_ascend_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="验证Ascend310B构建清单、OM与验收报告。")
    parser.add_argument("--config", default="configs/agent_pipeline_ascend310b.yaml")
    parser.add_argument(
        "--require-validation",
        action="store_true",
        help="即使配置尚未设置validated，也强制检查golden/精度/性能报告。",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = verify_ascend_artifacts(
        config["ascend_backend"],
        require_validation=args.require_validation or config["ascend_backend"]["validated"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
