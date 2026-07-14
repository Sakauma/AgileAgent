#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.release_verification import verify_release


def main() -> int:
    parser = argparse.ArgumentParser(description="执行灵动Agent静态发布验收。")
    parser.add_argument("--config", default="configs/agent_pipeline.yaml")
    args = parser.parse_args()
    result = verify_release(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
