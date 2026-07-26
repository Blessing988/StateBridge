#!/usr/bin/env python
"""Audit SynWTS/WTS VQA question ontology, answer priors, and submission behavior."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.exporters import load_records
from synwts.io import read_json, write_json
from synwts.parsers import load_vqa_questions
from synwts.submission import _parse_vqa_letter
from synwts.vqa_fusion import classify_vqa_question


def _norm_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower().strip())
    return text.replace("wearning", "wearing").replace("waling cane", "walking cane")


def _norm_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).lower().strip())
    aliases = {
        "none": "no",
        "n/a": "no",
        "not visible": "not visible",
        "near": "near",
        "close": "close",
        "usual": "usual",
        "normal": "usual",
    }
    return aliases.get(text, text)


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return round(value, 4)


def _scenario_from_videos(video_names: list[str], fallback: str) -> str:
    if not video_names:
        return fallback
    stem = Path(str(video_names[0])).stem
    if "_normal_" in stem:
        return stem
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return stem


def _infer_public_scope(item: dict[str, Any]) -> str:
    if "event_phase" not in item:
        return "environment"
    for phase in item.get("event_phase", []):
        for question in phase.get("conversations", []):
            q = str(question.get("question", "")).lower()
            if any(
                needle in q
                for needle in (
                    "vehicle's field of view",
                    "action taken by vehicle",
                    "position of the vehicle relative",
                    "relative distance of vehicle",
                )
            ):
                return "vehicle_view"
    return "overhead_view"


def _options(raw: dict[str, Any]) -> dict[str, str]:
    return {
        letter: str(raw[letter]).strip()
        for letter in ("a", "b", "c", "d", "e")
        if letter in raw and str(raw[letter]).strip()
    }


def load_train_rows(index_path: Path, splits: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records = load_records(index_path)
    if splits:
        records = [record for record in records if record.split in splits]
    for record in records:
        for scope, path in record.vqa_files.items():
            for q in load_vqa_questions(path, scope=scope, scenario_id=record.scenario_id):
                answer_text = q["options"].get(q["correct"], q["correct"])
                qtype = classify_vqa_question(q["question"], scope=scope)
                rows.append(
                    {
                        "split": record.split,
                        "scenario_id": record.scenario_id,
                        "scenario_type": record.scenario_type,
                        "scope": scope,
                        "phase": q["phase"],
                        "question": q["question"],
                        "question_norm": _norm_question(q["question"]),
                        "question_type": qtype,
                        "options": q["options"],
                        "correct": q["correct"],
                        "answer_text": answer_text,
                        "answer_norm": _norm_answer(answer_text),
                    }
                )
    return rows


def load_public_rows(vqa_json: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(read_json(vqa_json)):
        scenario_id = _scenario_from_videos(
            [str(name) for name in item.get("videos", [])],
            fallback=f"item_{item_idx:05d}",
        )
        scenario_type = "normal_trimmed" if "normal" in scenario_id else "event"
        scope = _infer_public_scope(item)
        if "event_phase" not in item:
            for q_idx, question in enumerate(item.get("conversations", [])):
                opts = _options(question)
                rows.append(_public_row(question, opts, item_idx, scenario_id, scenario_type, "environment", None, q_idx))
            continue
        for phase_idx, phase in enumerate(item.get("event_phase", [])):
            labels = phase.get("labels") or []
            phase_label = str(labels[0]) if labels else ""
            for q_idx, question in enumerate(phase.get("conversations", [])):
                opts = _options(question)
                rows.append(_public_row(question, opts, item_idx, scenario_id, scenario_type, scope, phase_label, q_idx))
    return rows


def _public_row(
    question: dict[str, Any],
    opts: dict[str, str],
    item_idx: int,
    scenario_id: str,
    scenario_type: str,
    scope: str,
    phase: str | None,
    q_idx: int,
) -> dict[str, Any]:
    text = str(question.get("question", "")).strip()
    qid = str(question.get("id", "")).strip() or f"item_{item_idx:05d}_{phase or 'env'}_{q_idx:03d}"
    return {
        "id": qid,
        "item_idx": item_idx,
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "scope": scope,
        "phase": phase,
        "question": text,
        "question_norm": _norm_question(text),
        "question_type": classify_vqa_question(text, scope=scope),
        "options": opts,
        "option_norms": {letter: _norm_answer(value) for letter, value in opts.items()},
    }


def _top(counter: Counter[str], n: int = 8) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(n)]


def _question_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_q: dict[str, Counter[str]] = defaultdict(Counter)
    q_examples: dict[str, str] = {}
    q_types: dict[str, str] = {}
    for row in rows:
        key = row["question_norm"]
        q_examples.setdefault(key, row["question"])
        q_types.setdefault(key, row["question_type"])
        if "answer_norm" in row:
            by_q[key][row["answer_norm"]] += 1
    out = {}
    for key, counter in by_q.items():
        total = sum(counter.values())
        top_value, top_count = counter.most_common(1)[0]
        out[key] = {
            "question": q_examples[key],
            "question_type": q_types[key],
            "count": total,
            "num_answers": len(counter),
            "entropy": _entropy(counter),
            "top_answer": top_value,
            "top_share": round(top_count / total, 4),
            "answers": _top(counter, 10),
        }
    return out


def _type_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(row["question_type"] for row in rows)
    scope_counts = Counter(row["scope"] for row in rows)
    phase_counts = Counter(str(row["phase"]) for row in rows)
    answer_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, str] = {}
    for row in rows:
        qtype = row["question_type"]
        examples.setdefault(qtype, row["question"])
        if "answer_norm" in row:
            answer_by_type[qtype][row["answer_norm"]] += 1
    return {
        "total": len(rows),
        "type_counts": dict(type_counts.most_common()),
        "scope_counts": dict(scope_counts.most_common()),
        "phase_counts": dict(phase_counts.most_common()),
        "type_answer_priors": {
            qtype: {
                "example": examples.get(qtype, ""),
                "entropy": _entropy(counter),
                "top": _top(counter, 10),
            }
            for qtype, counter in sorted(answer_by_type.items())
        },
        "examples": dict(sorted(examples.items())),
    }


def _submission_map(path: Path) -> dict[str, str]:
    return {
        str(row.get("id", "")).strip(): _parse_vqa_letter(str(row.get("correct", "")))
        for row in read_json(path)
        if str(row.get("id", "")).strip()
    }


def _submission_audit(public_rows: list[dict[str, Any]], submissions: dict[str, Path]) -> dict[str, Any]:
    if not submissions:
        return {}
    loaded = {name: _submission_map(path) for name, path in submissions.items()}
    by_name: dict[str, Any] = {}
    for name, pred in loaded.items():
        missing = [row["id"] for row in public_rows if row["id"] not in pred]
        selected_by_type: dict[str, Counter[str]] = defaultdict(Counter)
        invalid = []
        for row in public_rows:
            letter = pred.get(row["id"], "")
            if letter not in row["options"]:
                invalid.append(row["id"])
                continue
            selected_by_type[row["question_type"]][_norm_answer(row["options"][letter])] += 1
        by_name[name] = {
            "missing": len(missing),
            "invalid": len(invalid),
            "selected_by_type": {
                qtype: _top(counter, 8)
                for qtype, counter in sorted(selected_by_type.items())
            },
        }
    disagreements = {}
    for left, right in combinations(sorted(loaded), 2):
        diff = 0
        diff_by_type = Counter()
        for row in public_rows:
            qid = row["id"]
            if loaded[left].get(qid) != loaded[right].get(qid):
                diff += 1
                diff_by_type[row["question_type"]] += 1
        disagreements[f"{left}__vs__{right}"] = {
            "different": diff,
            "different_share": round(diff / max(len(public_rows), 1), 4),
            "by_type": dict(diff_by_type.most_common()),
        }
    return {"submissions": by_name, "pairwise_disagreements": disagreements}


def build_report(
    *,
    index: Path,
    public_vqa_json: Path,
    output: Path,
    splits: set[str] | None,
    submissions: dict[str, Path],
) -> dict[str, Any]:
    train_rows = load_train_rows(index, splits)
    public_rows = load_public_rows(public_vqa_json)
    train_questions = _question_stats(train_rows)
    public_question_counts = Counter(row["question_norm"] for row in public_rows)
    unseen_public_questions = [
        {
            "question": next(row["question"] for row in public_rows if row["question_norm"] == q),
            "count": count,
            "question_type": next(row["question_type"] for row in public_rows if row["question_norm"] == q),
        }
        for q, count in public_question_counts.most_common()
        if q not in train_questions
    ]
    public_question_with_priors = []
    for q, count in public_question_counts.most_common():
        if q not in train_questions:
            continue
        prior = train_questions[q]
        public_question_with_priors.append(
            {
                "question": prior["question"],
                "question_type": prior["question_type"],
                "public_count": count,
                "train_count": prior["count"],
                "train_top_answer": prior["top_answer"],
                "train_top_share": prior["top_share"],
                "train_entropy": prior["entropy"],
            }
        )
    high_prior = [
        row
        for row in public_question_with_priors
        if row["train_count"] >= 10 and row["train_top_share"] >= 0.9
    ]
    high_entropy = [
        row
        for row in public_question_with_priors
        if row["train_count"] >= 10 and row["train_entropy"] >= 1.0
    ]
    report = {
        "train": _type_stats(train_rows),
        "public": _type_stats(public_rows),
        "question_overlap": {
            "train_unique_questions": len(train_questions),
            "public_unique_questions": len(public_question_counts),
            "public_unique_seen_in_train": sum(1 for q in public_question_counts if q in train_questions),
            "public_rows_seen_question": sum(count for q, count in public_question_counts.items() if q in train_questions),
            "public_rows_unseen_question": sum(count for q, count in public_question_counts.items() if q not in train_questions),
            "unseen_public_questions": unseen_public_questions[:100],
            "high_prior_public_questions": high_prior[:100],
            "high_entropy_public_questions": high_entropy[:100],
        },
        "submission_audit": _submission_audit(public_rows, submissions),
    }
    write_json(output, report)
    return report


def _parse_submissions(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Submission must be name=path: {value}")
        name, path = value.split("=", 1)
        parsed[name.strip()] = Path(path.strip())
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--public-vqa-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--submission", action="append", default=[], help="name=path")
    args = parser.parse_args()

    splits = {item.strip() for item in args.splits.split(",") if item.strip()} if args.splits else None
    report = build_report(
        index=args.index,
        public_vqa_json=args.public_vqa_json,
        output=args.output,
        splits=splits,
        submissions=_parse_submissions(args.submission),
    )
    summary = {
        "train_total": report["train"]["total"],
        "public_total": report["public"]["total"],
        "public_rows_seen_question": report["question_overlap"]["public_rows_seen_question"],
        "public_rows_unseen_question": report["question_overlap"]["public_rows_unseen_question"],
        "high_prior_questions": len(report["question_overlap"]["high_prior_public_questions"]),
        "high_entropy_questions": len(report["question_overlap"]["high_entropy_public_questions"]),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
