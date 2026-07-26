"""Question-type-aware VQA fusion utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

from .io import read_json, write_json
from .submission import _load_vqa_submission, _parse_vqa_letter


def summarize_vqa_question_types(
    *,
    vqa_json: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    rows = load_vqa_question_rows(vqa_json)
    by_type = Counter(row["question_type"] for row in rows)
    by_scope = Counter(row["scope"] for row in rows)
    by_type_scope = Counter((row["question_type"], row["scope"]) for row in rows)
    report = {
        "total": len(rows),
        "by_type": dict(sorted(by_type.items())),
        "by_scope": dict(sorted(by_scope.items())),
        "by_type_scope": {
            f"{question_type}|{scope}": count
            for (question_type, scope), count in sorted(by_type_scope.items())
        },
        "examples": _type_examples(rows),
    }
    if output:
        write_json(output, report)
    return report


def fuse_vqa_by_question_type(
    *,
    vqa_json: str | Path,
    submissions: dict[str, str | Path],
    rules_path: str | Path,
    output: str | Path,
    report_output: str | Path | None = None,
) -> list[dict[str, str]]:
    rows = load_vqa_question_rows(vqa_json)
    rules = read_json(rules_path)
    named = {name: _load_vqa_submission(path) for name, path in submissions.items()}
    fallback_name = str(rules.get("fallback", next(iter(named)))).strip()
    if fallback_name not in named:
        raise ValueError(f"Fallback submission is not available: {fallback_name}")
    fallback = named[fallback_name]

    missing_ids = [row["id"] for row in rows if row["id"] not in fallback]
    if missing_ids:
        raise ValueError(f"Fallback submission misses VQA id: {missing_ids[0]}")

    output_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    by_type = Counter()
    changed_by_type = Counter()
    rule_inputs_by_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        qid = row["id"]
        question_type = row["question_type"]
        inputs, rule_fallback = _resolve_rule(
            rules=rules,
            question_type=question_type,
            available=set(named),
            global_fallback=fallback_name,
        )
        answer, winners, scores = _vote(
            qid=qid,
            input_names=inputs,
            named=named,
            fallback=named[rule_fallback],
        )
        fallback_answer = _parse_vqa_letter(fallback[qid])
        output_rows.append({"id": qid, "correct": answer})
        by_type[question_type] += 1
        if answer != fallback_answer:
            changed_by_type[question_type] += 1
        rule_inputs_by_type[question_type].update(inputs)
        report_rows.append(
            {
                "id": qid,
                "question_type": question_type,
                "scope": row["scope"],
                "phase": row["phase"],
                "question": row["question"],
                "selected": answer,
                "fallback": fallback_answer,
                "changed_from_fallback": answer != fallback_answer,
                "winners": winners,
                "scores": scores,
                "inputs": inputs,
            }
        )

    write_json(output, output_rows)
    if report_output:
        report = {
            "total": len(output_rows),
            "fallback": fallback_name,
            "changed_from_fallback": sum(changed_by_type.values()),
            "changed_by_type": dict(sorted(changed_by_type.items())),
            "by_type": dict(sorted(by_type.items())),
            "rule_inputs_by_type": {
                question_type: dict(counter)
                for question_type, counter in sorted(rule_inputs_by_type.items())
            },
            "rows": report_rows,
        }
        write_json(report_output, report)
    return output_rows


def load_vqa_question_rows(vqa_json: str | Path) -> list[dict[str, Any]]:
    items = read_json(vqa_json)
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        if "event_phase" in item:
            scope = _infer_scope(item)
            for phase_idx, phase in enumerate(item.get("event_phase", [])):
                labels = phase.get("labels") or []
                phase_label = str(labels[0]) if labels else ""
                for q_idx, question in enumerate(phase.get("conversations", [])):
                    rows.append(
                        _question_row(
                            question,
                            item_idx=item_idx,
                            phase_idx=phase_idx,
                            q_idx=q_idx,
                            scope=scope,
                            phase=phase_label,
                        )
                    )
        else:
            for q_idx, question in enumerate(item.get("conversations", [])):
                rows.append(
                    _question_row(
                        question,
                        item_idx=item_idx,
                        phase_idx=None,
                        q_idx=q_idx,
                        scope="environment",
                        phase=None,
                    )
                )
    return rows


def classify_vqa_question(question: str, *, scope: str = "") -> str:
    q = " ".join(question.lower().strip().split())

    if scope == "environment":
        if "weather" in q or "brightness" in q:
            return "weather_lighting"
        if any(term in q for term in ("road surface", "road inclination", "surface type")):
            return "road_surface"
        if "obstacle" in q:
            return "obstacle"
        if (
            "age group" in q
            or "height of the pedestrian" in q
            or _has_word(q, "hat")
            or _has_word(q, "glasses")
            or "walking cane" in q
            or "waling cane" in q
        ):
            return "pedestrian_attribute"
        if "wearing" in q or "wearning" in q or "clothing" in q:
            return "pedestrian_clothing"
        if any(
            term in q
            for term in (
                "traffic",
                "type of the road",
                "lanes",
                "sidewalk",
                "street lights",
                "roadside strip",
                "guardrail",
                "formation of the road",
                "setting",
            )
        ):
            return "road_context"
        return "environment_other"

    if "vehicle's field of view" in q:
        return "vehicle_fov"
    if "action taken by vehicle" in q:
        return "vehicle_action"
    if "position of the vehicle relative" in q:
        return "vehicle_position"
    if "relative distance of vehicle" in q:
        return "vehicle_distance"

    if "orientation of the pedestrian" in q:
        return "pedestrian_orientation"
    if "position of the pedestrian relative" in q:
        return "pedestrian_position"
    if "relative distance of pedestrian" in q:
        return "pedestrian_distance"
    if "line of sight" in q:
        return "pedestrian_line_of_sight"
    if "visual status" in q or "awareness" in q:
        return "pedestrian_attention"
    if "direction of travel" in q:
        return "pedestrian_direction"
    if "pedestrian's speed" in q:
        return "pedestrian_speed"
    if "pedestrian's action" in q or "fine-grained action" in q:
        return "pedestrian_action"
    return "phase_other"


def _question_row(
    question: dict[str, Any],
    *,
    item_idx: int,
    phase_idx: int | None,
    q_idx: int,
    scope: str,
    phase: str | None,
) -> dict[str, Any]:
    text = str(question.get("question", "")).strip()
    qid = str(question.get("id", "")).strip()
    if not qid:
        qid = f"item_{item_idx:05d}_phase_{phase_idx if phase_idx is not None else 'env'}_q_{q_idx:03d}"
    return {
        "id": qid,
        "question": text,
        "question_type": classify_vqa_question(text, scope=scope),
        "scope": scope,
        "phase": phase,
        "item_idx": item_idx,
        "phase_idx": phase_idx,
        "q_idx": q_idx,
    }


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


def _resolve_rule(
    *,
    rules: dict[str, Any],
    question_type: str,
    available: set[str],
    global_fallback: str,
) -> tuple[list[str], str]:
    raw_rule = rules.get("types", {}).get(question_type, rules.get("default"))
    if raw_rule is None:
        raw_rule = [global_fallback]
    fallback = global_fallback
    if isinstance(raw_rule, dict):
        fallback = str(raw_rule.get("fallback", global_fallback))
        raw_inputs = raw_rule.get("inputs", [fallback])
    else:
        raw_inputs = raw_rule
    if not isinstance(raw_inputs, list):
        raise ValueError(f"Rule for {question_type} must be a list or object with inputs.")
    inputs: list[str] = []
    for raw_name in raw_inputs:
        name = str(raw_name).strip()
        optional = name.endswith("?")
        if optional:
            name = name[:-1]
        if name in available:
            inputs.append(name)
        elif not optional:
            raise ValueError(f"Rule for {question_type} references missing submission: {name}")
    if fallback not in available:
        raise ValueError(f"Rule for {question_type} references missing fallback: {fallback}")
    if not inputs:
        inputs = [fallback]
    return inputs, fallback


def _vote(
    *,
    qid: str,
    input_names: list[str],
    named: dict[str, dict[str, str]],
    fallback: dict[str, str],
) -> tuple[str, list[str], dict[str, float]]:
    scores: dict[str, float] = {}
    for name in input_names:
        submission = named[name]
        if qid not in submission:
            raise ValueError(f"Submission {name} misses VQA id: {qid}")
        answer = _parse_vqa_letter(submission[qid])
        scores[answer] = scores.get(answer, 0.0) + 1.0
    best_score = max(scores.values())
    winners = sorted(answer for answer, score in scores.items() if score == best_score)
    fallback_answer = _parse_vqa_letter(fallback.get(qid, "a"))
    answer = fallback_answer if fallback_answer in winners else winners[0]
    return answer, winners, scores


def _type_examples(rows: list[dict[str, Any]]) -> dict[str, str]:
    examples: dict[str, str] = {}
    for row in rows:
        examples.setdefault(row["question_type"], row["question"])
    return dict(sorted(examples.items()))


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None
