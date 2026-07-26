#!/usr/bin/env python
"""Repair VQA predictions using high-confidence train/val answer priors."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_vqa_distribution import _norm_answer, _norm_question, load_public_rows, load_train_rows
from synwts.io import read_json, write_json
from synwts.submission import _parse_vqa_letter
from synwts.validators import validate_caption_submission, validate_vqa_submission


DEFAULT_STABLE_TYPES = {
    "weather_lighting",
    "road_surface",
    "obstacle",
    "pedestrian_attribute",
    "pedestrian_clothing",
    "road_context",
}


def _load_vqa(path: Path) -> dict[str, str]:
    return {
        str(row.get("id", "")).strip(): _parse_vqa_letter(str(row.get("correct", "")))
        for row in read_json(path)
        if str(row.get("id", "")).strip()
    }


def _answer_priors(train_rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    priors: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        priors[row["question_norm"]][_norm_answer(row["answer_text"])] += 1
    return priors


def _option_for_answer(row: dict[str, Any], answer: str) -> str:
    for letter, option in row["option_norms"].items():
        if option == answer:
            return letter
    return ""


def repair(
    *,
    index: Path,
    public_vqa_json: Path,
    base_vqa: Path,
    output: Path,
    report_output: Path,
    splits: set[str] | None,
    min_count: int,
    min_share: float,
    allowed_types: set[str] | None,
    max_changes: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    train_rows = load_train_rows(index, splits)
    public_rows = load_public_rows(public_vqa_json)
    priors = _answer_priors(train_rows)
    base = _load_vqa(base_vqa)
    repaired = dict(base)

    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for row in public_rows:
        qtype = row["question_type"]
        if allowed_types is not None and qtype not in allowed_types:
            skipped["type_block"] += 1
            continue
        counter = priors.get(row["question_norm"])
        if not counter:
            skipped["unseen_question"] += 1
            continue
        total = sum(counter.values())
        top_answer, top_count = counter.most_common(1)[0]
        share = top_count / max(total, 1)
        if total < min_count:
            skipped["low_count"] += 1
            continue
        if share < min_share:
            skipped["low_share"] += 1
            continue
        prior_letter = _option_for_answer(row, top_answer)
        if not prior_letter:
            skipped["answer_not_option"] += 1
            continue
        current = base.get(row["id"], "")
        if not current:
            skipped["missing_base"] += 1
            continue
        if current == prior_letter:
            skipped["already_same"] += 1
            continue
        candidates.append(
            {
                "id": row["id"],
                "question": row["question"],
                "question_type": qtype,
                "scenario_id": row["scenario_id"],
                "phase": row["phase"],
                "from": current,
                "from_text": row["options"].get(current, ""),
                "to": prior_letter,
                "to_text": row["options"].get(prior_letter, ""),
                "train_count": total,
                "train_top_count": top_count,
                "train_top_share": round(share, 4),
            }
        )

    candidates.sort(key=lambda x: (-x["train_top_share"], -x["train_count"], x["question_type"], x["id"]))
    selected = candidates[:max_changes] if max_changes is not None else candidates
    for item in selected:
        repaired[item["id"]] = item["to"]

    rows = [{"id": row["id"], "correct": repaired[row["id"]]} for row in public_rows if row["id"] in repaired]
    write_json(output, rows)
    report = {
        "min_count": min_count,
        "min_share": min_share,
        "allowed_types": sorted(allowed_types) if allowed_types is not None else "all",
        "max_changes": max_changes,
        "total_public": len(public_rows),
        "candidate_changes": len(candidates),
        "applied_changes": len(selected),
        "skipped": dict(skipped),
        "changed_by_type": dict(Counter(item["question_type"] for item in selected).most_common()),
        "selected": selected[:300],
    }
    write_json(report_output, report)
    return rows, report


def _make_zip(caption_path: Path, vqa_path: Path, zip_path: Path) -> None:
    caption_validation = validate_caption_submission(caption_path)
    vqa_validation = validate_vqa_submission(vqa_path)
    if not caption_validation["ok"]:
        raise ValueError(f"Caption validation failed: {caption_validation['errors'][:3]}")
    if not vqa_validation["ok"]:
        raise ValueError(f"VQA validation failed: {vqa_validation['errors'][:3]}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def _parse_types(value: str) -> set[str] | None:
    if not value or value == "all":
        return None
    if value == "stable":
        return set(DEFAULT_STABLE_TYPES)
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--public-vqa-json", type=Path, required=True)
    parser.add_argument("--base-vqa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--min-share", type=float, default=0.95)
    parser.add_argument("--types", default="stable")
    parser.add_argument("--max-changes", type=int)
    parser.add_argument("--caption-submission", type=Path)
    parser.add_argument("--zip-output", type=Path)
    args = parser.parse_args()

    splits = {item.strip() for item in args.splits.split(",") if item.strip()} if args.splits else None
    rows, report = repair(
        index=args.index,
        public_vqa_json=args.public_vqa_json,
        base_vqa=args.base_vqa,
        output=args.output,
        report_output=args.report_output,
        splits=splits,
        min_count=args.min_count,
        min_share=args.min_share,
        allowed_types=_parse_types(args.types),
        max_changes=args.max_changes,
    )
    if args.zip_output:
        if not args.caption_submission:
            raise ValueError("--caption-submission is required with --zip-output")
        _make_zip(args.caption_submission, args.output, args.zip_output)
    print(json.dumps({"rows": len(rows), **{k: report[k] for k in ("candidate_changes", "applied_changes", "changed_by_type")}}, indent=2))


if __name__ == "__main__":
    main()
