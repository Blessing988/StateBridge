from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--add", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    add = Path(args.add)
    data = json.load(base.open("r", encoding="utf-8"))
    extra = json.load(add.open("r", encoding="utf-8"))
    data.update(extra)
    json.dump(data, base.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("added:", list(extra))


if __name__ == "__main__":
    main()
