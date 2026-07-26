#!/usr/bin/env python
"""Build option-shuffled and question-type-balanced VQA SFT datasets."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.exporters import write_llamafactory_dataset_info
from synwts.io import write_json
from synwts.vqa_fusion import classify_vqa_question


LETTERS = ("a", "b", "c", "d", "e")
DEFAULT_HARD_TYPES = (
    "vehicle_action",
    "vehicle_position",
    "vehicle_distance",
    "pedestrian_action",
    "pedestrian_attention",
    "pedestrian_line_of_sight",
    "pedestrian_position",
    "pedestrian_distance",
    "pedestrian_orientation",
    "pedestrian_direction",
    "pedestrian_speed",
)


def build_augmented_dataset(
    *,
    datasets: list[Path],
    output: Path,
    dataset_info_output: Path | None,
    dataset_name: str | None,
    include_types: set[str] | None,
    exclude_types: set[str],
    shuffle_copies: int,
    hard_types: set[str],
    hard_extra_copies: int,
    hard_min_count: int,
    max_rows: int | None,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    source_rows = _load_rows(datasets)
    source_rows = [_prepare_row(row, source_idx=idx) for idx, row in enumerate(source_rows)]
    source_rows = [row for row in source_rows if row is not None]
    if include_types is not None:
        source_rows = [row for row in source_rows if str(row["metadata"].get("question_type", "")) in include_types]
    if exclude_types:
        source_rows = [row for row in source_rows if str(row["metadata"].get("question_type", "")) not in exclude_types]

    out_rows: list[dict[str, Any]] = []
    for row in source_rows:
        out_rows.append(_mark(row, augmentation="original", copy_idx=0))
        for copy_idx in range(1, shuffle_copies + 1):
            out_rows.append(_shuffle_options(row, rng=rng, augmentation="option_shuffle", copy_idx=copy_idx))

    if hard_extra_copies > 0:
        hard_rows = [row for row in source_rows if row["metadata"].get("question_type") in hard_types]
        for row in hard_rows:
            for copy_idx in range(1, hard_extra_copies + 1):
                out_rows.append(_shuffle_options(row, rng=rng, augmentation="hard_extra_shuffle", copy_idx=copy_idx))

    if hard_min_count > 0:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in source_rows:
            qtype = str(row["metadata"].get("question_type", ""))
            if qtype in hard_types:
                by_type.setdefault(qtype, []).append(row)
        counts = Counter(str(row["metadata"].get("question_type", "")) for row in out_rows)
        for qtype in sorted(hard_types):
            candidates = by_type.get(qtype, [])
            if not candidates:
                continue
            copy_idx = 0
            while counts[qtype] < hard_min_count:
                base = rng.choice(candidates)
                out_rows.append(
                    _shuffle_options(
                        base,
                        rng=rng,
                        augmentation="hard_min_shuffle",
                        copy_idx=copy_idx,
                    )
                )
                counts[qtype] += 1
                copy_idx += 1

    rng.shuffle(out_rows)
    if max_rows is not None and max_rows > 0:
        out_rows = out_rows[:max_rows]

    write_json(output, out_rows)
    if dataset_info_output and dataset_name:
        write_llamafactory_dataset_info(dataset_info_output, dataset_name=dataset_name, file_name=output.name)

    report = {
        "input_rows": len(source_rows),
        "output_rows": len(out_rows),
        "shuffle_copies": shuffle_copies,
        "hard_extra_copies": hard_extra_copies,
        "hard_min_count": hard_min_count,
        "hard_types": sorted(hard_types),
        "by_type": dict(Counter(str(row["metadata"].get("question_type", "")) for row in out_rows).most_common()),
        "by_augmentation": dict(Counter(str(row["metadata"].get("augmentation", "")) for row in out_rows).most_common()),
        "by_correct": dict(Counter(str(row.get("output", "")).strip().lower() for row in out_rows).most_common()),
    }
    return report


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Dataset must be a JSON list: {path}")
        for row in data:
            if isinstance(row, dict) and row.get("metadata", {}).get("task") == "vqa":
                copied = copy.deepcopy(row)
                copied.setdefault("metadata", {})["source_dataset"] = path.name
                rows.append(copied)
    return rows


def _prepare_row(row: dict[str, Any], *, source_idx: int) -> dict[str, Any] | None:
    instruction = str(row.get("instruction", ""))
    parsed = _parse_instruction_options(instruction)
    if parsed is None:
        return None
    options, _start, _end = parsed
    correct = str(row.get("output", "")).strip().lower()[:1]
    if correct not in options:
        return None
    metadata = row.setdefault("metadata", {})
    question = _extract_question(instruction) or str(metadata.get("question", ""))
    scope = str(metadata.get("scope", ""))
    metadata.setdefault("question", question)
    metadata["question_type"] = classify_vqa_question(question, scope=scope)
    metadata["source_row_index"] = source_idx
    metadata["answer_text"] = options.get(correct, metadata.get("answer_text", ""))
    return row


def _mark(row: dict[str, Any], *, augmentation: str, copy_idx: int) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out.setdefault("metadata", {})["augmentation"] = augmentation
    out["metadata"]["augmentation_copy_idx"] = copy_idx
    return out


def _shuffle_options(
    row: dict[str, Any],
    *,
    rng: random.Random,
    augmentation: str,
    copy_idx: int,
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    parsed = _parse_instruction_options(str(out["instruction"]))
    if parsed is None:
        return _mark(out, augmentation=augmentation, copy_idx=copy_idx)
    options, start, end = parsed
    old_correct = str(out.get("output", "")).strip().lower()[:1]
    entries = list(options.items())
    for _ in range(10):
        shuffled = entries[:]
        rng.shuffle(shuffled)
        if [old for old, _ in shuffled] != [old for old, _ in entries]:
            break
    new_options: dict[str, str] = {}
    new_correct = old_correct
    for new_letter, (old_letter, text) in zip(LETTERS, shuffled):
        new_options[new_letter] = text
        if old_letter == old_correct:
            new_correct = new_letter
    option_block = "Options:\n" + "\n".join(f"{letter}. {text}" for letter, text in new_options.items()) + "\n\n"
    out["instruction"] = str(out["instruction"])[:start] + option_block + str(out["instruction"])[end:]
    out["output"] = new_correct
    out.setdefault("metadata", {})["correct"] = new_correct
    out["metadata"]["answer_text"] = new_options.get(new_correct, "")
    out["metadata"]["augmentation"] = augmentation
    out["metadata"]["augmentation_copy_idx"] = copy_idx
    out["metadata"]["option_shuffle_map"] = {
        new_letter: old_letter for new_letter, (old_letter, _text) in zip(LETTERS, shuffled)
    }
    return out


def _parse_instruction_options(instruction: str) -> tuple[dict[str, str], int, int] | None:
    match = re.search(r"(?ms)^Options:\n(?P<body>.*?)(?:\n\nReturn only the correct option letter\.)", instruction)
    if not match:
        return None
    options: dict[str, str] = {}
    for line in match.group("body").splitlines():
        item = re.match(r"^([a-e])\.\s*(.+?)\s*$", line.strip(), flags=re.I)
        if item:
            options[item.group(1).lower()] = item.group(2).strip()
    if len(options) < 2:
        return None
    return options, match.start(), match.end() - len("Return only the correct option letter.")


def _extract_question(instruction: str) -> str:
    match = re.search(r"(?m)^Question:\s*(.+?)\s*$", instruction)
    return match.group(1).strip() if match else ""


def _parse_hard_types(value: str) -> set[str]:
    if value.strip().lower() in {"default", "hard"}:
        return set(DEFAULT_HARD_TYPES)
    if value.strip().lower() in {"none", ""}:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_optional_types(value: str) -> set[str] | None:
    if not value.strip():
        return None
    if value.strip().lower() in {"all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-info-output", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--include-types", default="")
    parser.add_argument("--exclude-types", default="")
    parser.add_argument("--shuffle-copies", type=int, default=2)
    parser.add_argument("--hard-types", default="default")
    parser.add_argument("--hard-extra-copies", type=int, default=1)
    parser.add_argument("--hard-min-count", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    report = build_augmented_dataset(
        datasets=args.dataset,
        output=args.output,
        dataset_info_output=args.dataset_info_output,
        dataset_name=args.dataset_name,
        include_types=_parse_optional_types(args.include_types),
        exclude_types=_parse_optional_types(args.exclude_types) or set(),
        shuffle_copies=max(args.shuffle_copies, 0),
        hard_types=_parse_hard_types(args.hard_types),
        hard_extra_copies=max(args.hard_extra_copies, 0),
        hard_min_count=max(args.hard_min_count, 0),
        max_rows=args.max_rows or None,
        seed=args.seed,
    )
    if args.report_output:
        write_json(args.report_output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
