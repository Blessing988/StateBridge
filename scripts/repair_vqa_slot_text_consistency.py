#!/usr/bin/env python
"""Repair VQA by scenario slot answer text, then remap to option letters."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.caption_facts import _scenario_from_video_names
from synwts.submission import _parse_vqa_letter
from synwts.validators import validate_caption_submission, validate_vqa_submission
from synwts.vqa_consistency import (
    STABLE_QUESTION_PATTERNS,
    _has_dynamic_pattern,
    _normalize_question,
)
from synwts.vqa_fusion import classify_vqa_question


LETTERS = ("a", "b", "c", "d", "e")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("grey", "gray")
    text = re.sub(r"\bkm / h\b", "km/h", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[.:-]+|[.:-]+$", "", text)
    return text


def _scenario_id(item: dict[str, Any], item_idx: int) -> str:
    return _scenario_from_video_names(
        [str(name) for name in item.get("videos", [])],
        fallback=f"item_{item_idx:05d}",
    )


def _infer_scope(item: dict[str, Any]) -> str:
    for phase in item.get("event_phase", []):
        for question in phase.get("conversations", []):
            q = str(question.get("question", "")).lower()
            if "vehicle" in q and (
                "field of view" in q
                or "action taken by vehicle" in q
                or "position of the vehicle relative" in q
                or "relative distance of vehicle" in q
            ):
                return "vehicle_view"
    return "overhead_view"


def _question_options(question: dict[str, Any]) -> dict[str, str]:
    return {
        letter: str(question.get(letter, "")).strip()
        for letter in LETTERS
        if str(question.get(letter, "")).strip()
    }


def _rows_from_wts_vqa(vqa_json: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(_read_json(vqa_json)):
        sid = _scenario_id(item, item_idx)
        if "event_phase" in item:
            scope = _infer_scope(item)
            for phase in item.get("event_phase", []):
                labels = phase.get("labels") or []
                phase_label = str(labels[0]).strip() if labels else ""
                for question in phase.get("conversations", []):
                    _add_question_row(rows, sid, scope, phase_label, question)
        else:
            for question in item.get("conversations", []):
                _add_question_row(rows, sid, "environment", "", question)
    return rows


def _add_question_row(
    rows: list[dict[str, Any]],
    scenario_id: str,
    scope: str,
    phase: str,
    question: dict[str, Any],
) -> None:
    qid = str(question.get("id", "")).strip()
    if not qid:
        return
    qtext = str(question.get("question", "")).strip()
    rows.append(
        {
            "id": qid,
            "scenario_id": scenario_id,
            "scope": scope,
            "phase": phase,
            "question": qtext,
            "question_type": classify_vqa_question(qtext, scope=scope),
            "options": _question_options(question),
        }
    )


def _is_stable(row: dict[str, Any]) -> bool:
    q = _normalize_question(str(row["question"]))
    if _has_dynamic_pattern(q):
        return False
    if row["scope"] == "environment":
        return True
    return any(pattern in q for pattern in STABLE_QUESTION_PATTERNS)


def _family_id(scenario_id: str) -> str:
    if "_normal_" not in scenario_id:
        return scenario_id
    parts = scenario_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return scenario_id


def _group_key(
    row: dict[str, Any],
    *,
    include_scope: bool,
    family_types: set[str],
) -> tuple[str, ...] | None:
    if not _is_stable(row):
        return None
    question_type = str(row["question_type"])
    scenario_key = _family_id(str(row["scenario_id"])) if question_type in family_types else str(row["scenario_id"])
    parts = [scenario_key, question_type, _normalize_question(str(row["question"]))]
    if include_scope:
        parts.insert(1, str(row["scope"]))
    return tuple(parts)


def _option_letter_for_text(options: dict[str, str], answer_text: str) -> str | None:
    target = _norm_answer(answer_text)
    for letter, text in options.items():
        if _norm_answer(text) == target:
            return letter
    return None


def repair(
    *,
    vqa_json: Path,
    submission: Path,
    caption: Path,
    output_dir: Path,
    name: str,
    min_group_size: int,
    min_top_count: int,
    min_top_share: float,
    max_changes: int,
    include_scope: bool,
    family_types: set[str],
) -> dict[str, Any]:
    rows = _rows_from_wts_vqa(vqa_json)
    pred_rows = _read_json(submission)
    pred = {
        str(row.get("id", "")).strip(): _parse_vqa_letter(str(row.get("correct", "")))
        for row in pred_rows
        if str(row.get("id", "")).strip()
    }
    row_by_id = {row["id"]: row for row in rows}

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _group_key(row, include_scope=include_scope, family_types=family_types)
        if key is not None and row["id"] in pred:
            groups[key].append(row)

    repaired = dict(pred)
    selected_groups: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    for key, group in sorted(groups.items()):
        if len(group) < min_group_size:
            continue
        answer_texts = []
        for row in group:
            letter = pred[row["id"]]
            text = row["options"].get(letter)
            if text:
                answer_texts.append(_norm_answer(text))
        if len(answer_texts) < min_group_size:
            continue
        counts = Counter(answer_texts)
        top_text, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        top_share = top_count / len(answer_texts)
        if top_count < min_top_count or top_share < min_top_share:
            continue

        group_changes = []
        for row in group:
            if max_changes and len(changes) >= max_changes:
                break
            target_letter = _option_letter_for_text(row["options"], top_text)
            if target_letter is None or repaired[row["id"]] == target_letter:
                continue
            old_letter = repaired[row["id"]]
            old_text = row["options"].get(old_letter, "")
            repaired[row["id"]] = target_letter
            item = {
                "id": row["id"],
                "scenario_id": row["scenario_id"],
                "phase": row["phase"],
                "scope": row["scope"],
                "question_type": row["question_type"],
                "question": row["question"],
                "from": old_letter,
                "from_text": old_text,
                "to": target_letter,
                "to_text": row["options"].get(target_letter, ""),
            }
            changes.append(item)
            group_changes.append(item)
        if group_changes:
            selected_groups.append(
                {
                    "key": list(key),
                    "size": len(group),
                    "top_text": top_text,
                    "top_count": top_count,
                    "top_share": round(top_share, 4),
                    "counts": dict(counts),
                    "changes": group_changes,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_rows = [{"id": str(row.get("id", "")).strip(), "correct": repaired[str(row.get("id", "")).strip()]} for row in pred_rows]
    vqa_out = output_dir / f"vqa_submission_{name}.json"
    report_out = output_dir / f"vqa_{name}_slot_text_repair_report.json"
    zip_out = output_dir / f"submission_{name}.zip"
    _write_json(vqa_out, repaired_rows)
    report = {
        "name": name,
        "total_questions": len(repaired_rows),
        "groups": len(groups),
        "changed": len(changes),
        "min_group_size": min_group_size,
        "min_top_count": min_top_count,
        "min_top_share": min_top_share,
        "max_changes": max_changes,
        "include_scope": include_scope,
        "family_types": sorted(family_types),
        "changed_by_type": dict(Counter(item["question_type"] for item in changes)),
        "changes": changes[:300],
        "selected_groups": selected_groups[:200],
    }
    _write_json(report_out, report)

    cap_validation = validate_caption_submission(caption)
    vqa_validation = validate_vqa_submission(vqa_out)
    if not cap_validation["ok"]:
        raise ValueError(f"Caption validation failed: {cap_validation['errors'][:3]}")
    if not vqa_validation["ok"]:
        raise ValueError(f"VQA validation failed: {vqa_validation['errors'][:3]}")
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption, "caption_submission.json")
        zf.write(vqa_out, "vqa_submission.json")
    report["vqa"] = str(vqa_out)
    report["report"] = str(report_out)
    report["zip"] = str(zip_out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa-json", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--caption", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--min-top-count", type=int, default=2)
    parser.add_argument("--min-top-share", type=float, default=0.75)
    parser.add_argument("--max-changes", type=int, default=0)
    parser.add_argument("--include-scope", action="store_true")
    parser.add_argument(
        "--family-types",
        default="",
        help="Comma-separated question types to group by normal-session family instead of exact scenario.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            repair(
                vqa_json=args.vqa_json,
                submission=args.submission,
                caption=args.caption,
                output_dir=args.output_dir,
                name=args.name,
                min_group_size=args.min_group_size,
                min_top_count=args.min_top_count,
                min_top_share=args.min_top_share,
                max_changes=args.max_changes,
                include_scope=args.include_scope,
                family_types={item.strip() for item in args.family_types.split(",") if item.strip()},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
