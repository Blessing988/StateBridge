"""Merge a generated LLaMA-Factory dataset-info JSON into dataset_info.json."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to LLaMA-Factory data/dataset_info.json")
    parser.add_argument("--add", required=True, help="Path to generated dataset-info JSON")
    parser.add_argument("--backup-prefix", default="bak")
    args = parser.parse_args()

    base = Path(args.base)
    add = Path(args.add)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = base.with_name(f"{base.name}.{args.backup_prefix}_{timestamp}")
    shutil.copy2(base, backup)

    with base.open("r", encoding="utf-8") as handle:
        base_data = json.load(handle)
    with add.open("r", encoding="utf-8") as handle:
        add_data = json.load(handle)

    base_data.update(add_data)
    with base.open("w", encoding="utf-8") as handle:
        json.dump(base_data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("added:", list(add_data))
    print("backup:", backup)


if __name__ == "__main__":
    main()
