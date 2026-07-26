"""Conservative scenario-level consistency repair for VQA submissions."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from pathlib import Path
from typing import Any, Iterable

from .io import read_json, write_json
from .submission import _parse_vqa_letter
from .vqa_fusion import classify_vqa_question


DEFAULT_STABLE_TYPES = {
    "weather_lighting",
    "road_surface",
    "obstacle",
    "pedestrian_attribute",
    "pedestrian_clothing",
    "road_context",
    "environment_other",
}

STABLE_QUESTION_PATTERNS = (
    "age group",
    "gender",
    "height of the pedestrian",
    "hat",
    "glasses",
    "walking cane",
    "waling cane",
    "wearing",
    "wearning",
    "clothing",
    "clothes",
    "color of the vehicle",
    "vehicle color",
    "type of vehicle",
    "vehicle type",
    "weather",
    "brightness",
    "road surface",
    "road inclination",
    "surface type",
    "traffic light",
    "sidewalk",
    "street lights",
    "lanes",
    "guardrail",
    "roadside strip",
    "formation of the road",
    "setting",
)

DYNAMIC_QUESTION_PATTERNS = (
    "orientation",
    "position",
    "relative distance",
    "line of sight",
    "visual status",
    "awareness",
    "direction of travel",
    "speed",
    "action",
    "field of view",
)


def repair_vqa_scenario_consistency(
    *,
    candidates_path: str | Path,
    submission: str | Path,
    output: str | Path,
    report_output: str | Path | None = None,
    stable_types: Iterable[str] | None = None,
    min_group_size: int = 2,
    min_top_count: int = 2,
    min_top_share: float = 0.75,
    max_changes: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Repair only stable scenario facts by majority vote within a scenario.

    The function intentionally avoids dynamic phase questions such as position,
    distance, speed, action, and awareness. It only changes a prediction when
    repeated stable questions in the same scenario strongly agree.
    """

    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2.")
    if min_top_count < 2:
        raise ValueError("min_top_count must be at least 2.")
    if not (0.0 < min_top_share <= 1.0):
        raise ValueError("min_top_share must satisfy 0 < value <= 1.")

    stable_type_set = set(stable_types or DEFAULT_STABLE_TYPES)
    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("Candidate file must be a JSON list.")
    submission_rows = read_json(submission)
    if not isinstance(submission_rows, list):
        raise ValueError("VQA submission must be a JSON list.")

    predictions = {
        str(row.get("id", "")).strip(): _parse_vqa_letter(str(row.get("correct", "")))
        for row in submission_rows
        if str(row.get("id", "")).strip()
    }
    candidate_by_id = {
        str(row.get("metadata", {}).get("vqa_id", "")).strip(): row
        for row in candidates
        if str(row.get("metadata", {}).get("vqa_id", "")).strip()
    }

    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    skipped_types: Counter[str] = Counter()
    stable_type_counts: Counter[str] = Counter()
    for qid, candidate in candidate_by_id.items():
        key = _stable_group_key(candidate, stable_type_set=stable_type_set)
        metadata = candidate.get("metadata", {})
        qtype = classify_vqa_question(
            str(metadata.get("question", "")),
            scope=str(metadata.get("scope", "")),
        )
        if key is None:
            skipped_types[qtype] += 1
            continue
        groups[key].append(qid)
        stable_type_counts[qtype] += 1

    repaired = dict(predictions)
    group_reports: list[dict[str, Any]] = []
    changed = 0
    changed_by_type: Counter[str] = Counter()
    eligible_groups = 0
    applied_groups = 0

    for key, qids in sorted(groups.items()):
        if len(qids) < min_group_size:
            continue
        eligible_groups += 1
        answers = [predictions[qid] for qid in qids if qid in predictions]
        if len(answers) < min_group_size:
            continue
        counts = Counter(answers)
        top_answer, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        top_share = top_count / len(answers)
        if top_count < min_top_count or top_share < min_top_share:
            continue
        if not _answer_is_valid_for_all(top_answer, qids, candidate_by_id):
            continue

        group_changed = []
        for qid in qids:
            if qid not in repaired or repaired[qid] == top_answer:
                continue
            if max_changes is not None and changed >= max_changes:
                continue
            original = repaired[qid]
            repaired[qid] = top_answer
            changed += 1
            metadata = candidate_by_id[qid].get("metadata", {})
            qtype = classify_vqa_question(
                str(metadata.get("question", "")),
                scope=str(metadata.get("scope", "")),
            )
            changed_by_type[qtype] += 1
            group_changed.append({"id": qid, "from": original, "to": top_answer})
        if group_changed:
            applied_groups += 1
            group_reports.append(
                {
                    "key": list(key),
                    "size": len(qids),
                    "counts": dict(sorted(counts.items())),
                    "top_answer": top_answer,
                    "top_share": round(top_share, 4),
                    "changed": group_changed,
                }
            )

    output_rows = [
        {"id": str(row.get("id", "")).strip(), "correct": repaired[str(row.get("id", "")).strip()]}
        for row in submission_rows
        if str(row.get("id", "")).strip()
    ]
    write_json(output, output_rows)

    report = {
        "total_predictions": len(output_rows),
        "changed": changed,
        "changed_by_type": dict(sorted(changed_by_type.items())),
        "stable_type_counts": dict(sorted(stable_type_counts.items())),
        "skipped_type_counts": dict(sorted(skipped_types.items())),
        "num_groups": len(groups),
        "eligible_groups": eligible_groups,
        "applied_groups": applied_groups,
        "min_group_size": min_group_size,
        "min_top_count": min_top_count,
        "min_top_share": min_top_share,
        "groups": group_reports[:200],
    }
    if report_output:
        write_json(report_output, report)
    return output_rows, report


def _stable_group_key(
    candidate: dict[str, Any],
    *,
    stable_type_set: set[str],
) -> tuple[str, str, str] | None:
    metadata = candidate.get("metadata", {})
    scenario_id = str(metadata.get("scenario_id", "")).strip()
    question = str(metadata.get("question", "")).strip()
    scope = str(metadata.get("scope", "")).strip()
    if not scenario_id or not question:
        return None

    normalized = _normalize_question(question)
    qtype = classify_vqa_question(question, scope=scope)
    if qtype not in stable_type_set and not _has_stable_pattern(normalized):
        return None
    if _has_dynamic_pattern(normalized):
        return None
    return scenario_id, qtype, normalized


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())


def _has_stable_pattern(question: str) -> bool:
    return any(pattern in question for pattern in STABLE_QUESTION_PATTERNS)


def _has_dynamic_pattern(question: str) -> bool:
    return any(pattern in question for pattern in DYNAMIC_QUESTION_PATTERNS)


def _answer_is_valid_for_all(answer: str, qids: list[str], candidate_by_id: dict[str, dict[str, Any]]) -> bool:
    for qid in qids:
        options = {
            str(letter).strip().lower()
            for letter in candidate_by_id[qid].get("options", {})
        }
        if answer not in options:
            return False
    return True
