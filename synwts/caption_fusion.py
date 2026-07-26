"""Caption candidate fusion and lightweight fact consistency scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from pathlib import Path
from typing import Any

from .caption_facts import _flatten_facts, _predicted_wts_fact_map
from .io import read_json, write_json


ARTIFACT_TERMS = (
    "caption_pedestrian",
    "caption_vehicle",
    "retrieved synthetic",
    "retrieved annotation",
    "vqa fact context",
    "return json",
    "bbox context",
    "yellow",
    "orange",
    "cyan",
    "lime",
)

PHASE_CUES = {
    "4": (
        "collid",
        "avoid",
        "brak",
        "emergency",
        "stationary",
        "stopped",
        "impact",
        "speed",
    ),
    "3": ("action", "speed", "turn", "straight", "brak", "moving", "slowing"),
    "2": ("aware", "unaware", "notice", "judg", "closely watched", "line of sight"),
    "1": ("recogn", "notice", "line of sight", "visual", "attention"),
    "0": ("approach", "pre", "prior", "before", "position", "distance"),
}

COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "gray",
    "grey",
    "brown",
    "beige",
    "cyan",
    "purple",
    "pink",
}

DECADE_PATTERN = re.compile(r"\b([1-9]0s)\b", flags=re.IGNORECASE)
SPEED_PATTERN = re.compile(r"\b([0-9]+)\s*km/h\b", flags=re.IGNORECASE)


def fuse_caption_submissions(
    *,
    submissions: dict[str, str | Path],
    output: str | Path,
    fallback_name: str,
    wts_vqa_json: str | Path | None = None,
    vqa_submission: str | Path | None = None,
    report_output: str | Path | None = None,
    target_words: int = 250,
    min_switch_margin: float = 2.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if fallback_name not in submissions:
        raise ValueError(f"Fallback caption source is not available: {fallback_name}")
    if bool(wts_vqa_json) != bool(vqa_submission):
        raise ValueError("--wts-vqa-json and --vqa-submission must be provided together.")

    loaded = {name: _load_caption_submission(path) for name, path in submissions.items()}
    fallback = loaded[fallback_name]
    facts = _predicted_wts_fact_map(wts_vqa_json, vqa_submission) if wts_vqa_json else {}

    fused: dict[str, list[dict[str, Any]]] = {}
    source_counts: Counter[str] = Counter()
    changed = 0
    total = 0
    score_sums: defaultdict[str, float] = defaultdict(float)
    score_counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []

    for scenario_id, fallback_rows in fallback.items():
        fused_rows: list[dict[str, Any]] = []
        by_source = {
            name: _rows_by_phase(rows.get(scenario_id, []))
            for name, rows in loaded.items()
        }
        for fallback_idx, fallback_row in enumerate(fallback_rows):
            phase = _phase_label(fallback_row, fallback_idx)
            candidates: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
            for name in sorted(loaded):
                row = by_source[name].get(phase)
                if row is None:
                    continue
                score, parts = score_caption_candidate(
                    row,
                    phase=phase,
                    facts=facts.get(scenario_id, {}),
                    target_words=target_words,
                )
                # Prefer the known best fallback in exact ties, but allow real
                # fact/quality advantages from other candidates to win.
                if name == fallback_name:
                    score += 0.05
                    parts["fallback_tiebreak"] = 0.05
                candidates.append((score, name, row, parts))
                score_sums[name] += score
                score_counts[name] += 1

            if not candidates:
                selected_name = fallback_name
                selected = fallback_row
                selected_score = 0.0
                selected_parts = {}
            else:
                ranked = sorted(
                    candidates,
                    key=lambda item: (-item[0], item[1]),
                )
                selected_score, selected_name, selected, selected_parts = ranked[0]
                fallback_candidate = next((item for item in candidates if item[1] == fallback_name), None)
                if (
                    fallback_candidate is not None
                    and selected_name != fallback_name
                    and selected_score < fallback_candidate[0] + min_switch_margin
                ):
                    selected_score, selected_name, selected, selected_parts = fallback_candidate
            fused_rows.append(_normalize_caption_row(selected, phase=phase))
            source_counts[selected_name] += 1
            total += 1
            if selected_name != fallback_name:
                changed += 1
            decisions.append(
                {
                    "scenario_id": scenario_id,
                    "phase": phase,
                    "selected": selected_name,
                    "score": round(selected_score, 4),
                    "score_parts": selected_parts,
                }
            )
        fused[scenario_id] = fused_rows

    report = {
        "total_rows": total,
        "changed_from_fallback": changed,
        "source_counts": dict(source_counts),
        "mean_source_scores": {
            name: round(score_sums[name] / max(score_counts[name], 1), 4)
            for name in sorted(score_counts)
        },
        "target_words": target_words,
        "min_switch_margin": min_switch_margin,
        "fallback_name": fallback_name,
        "decisions_sample": decisions[:50],
    }
    write_json(output, fused)
    if report_output:
        write_json(report_output, report)
    return fused, report


def score_caption_candidate(
    row: dict[str, Any],
    *,
    phase: str,
    facts: dict[str, Any],
    target_words: int = 250,
) -> tuple[float, dict[str, float]]:
    ped = str(row.get("caption_pedestrian", "")).strip()
    veh = str(row.get("caption_vehicle", "")).strip()
    text = f"{ped} {veh}".strip()
    lower = text.lower()
    words = _word_count(text)
    parts: dict[str, float] = {}

    parts["non_empty"] = 3.0 if ped and veh else -20.0
    parts["length"] = _length_score(words, target_words=target_words)
    parts["artifact_penalty"] = -4.0 * sum(1 for term in ARTIFACT_TERMS if term in lower)
    parts["phase_cues"] = _phase_cue_score(lower, phase)

    fact_score, fact_parts = _fact_consistency_score(lower, facts, phase=phase)
    parts["fact_consistency"] = fact_score
    parts.update(fact_parts)

    if _looks_repetitive(text):
        parts["repetition_penalty"] = -4.0
    return sum(parts.values()), parts


def _fact_consistency_score(
    lower_text: str,
    facts: dict[str, Any],
    *,
    phase: str,
) -> tuple[float, dict[str, float]]:
    flat = _flatten_facts(facts.get("global", {}))
    flat.extend(_flatten_facts(facts.get("phases", {}).get(phase, {})))
    score = 0.0
    parts: dict[str, float] = {}
    matched = 0
    contradictions = 0
    for key, value in flat:
        delta, is_match, is_contradiction = _score_one_fact(lower_text, key.lower(), value.lower())
        score += delta
        matched += int(is_match)
        contradictions += int(is_contradiction)
    if matched:
        parts["fact_matches"] = min(8.0, matched * 0.6)
        score += parts["fact_matches"]
    if contradictions:
        parts["fact_contradictions"] = -3.0 * contradictions
        score += parts["fact_contradictions"]
    return score, parts


def _score_one_fact(text: str, key: str, value: str) -> tuple[float, bool, bool]:
    value = _normalize_value(value)
    if not value or value in {"unknown", "none", "n/a"}:
        return 0.0, False, False

    match = _value_present(text, value)
    contradiction = False
    score = 0.0
    if match:
        score += 0.7

    if "age_group" in key:
        contradiction = _age_contradiction(text, value)
        if match:
            score += 1.0
    elif "speed" in key:
        contradiction = _speed_contradiction(text, value)
        if match:
            score += 0.8
    elif "gender" in key or value in {"male", "female"}:
        contradiction = _gender_contradiction(text, value)
        if match:
            score += 0.8
    elif "color" in key:
        contradiction = _color_contradiction(text, value)
        if match:
            score += 0.5
    elif any(token in key for token in ("position", "distance", "action", "awareness", "line_of_sight")):
        if match:
            score += 0.8

    if contradiction:
        score -= 2.5
    return score, match, contradiction


def _load_caption_submission(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Caption submission must be an object: {path}")
    return data


def _rows_by_phase(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_phase_label(row, idx): row for idx, row in enumerate(rows)}


def _phase_label(row: dict[str, Any], idx: int) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return str(idx)


def _normalize_caption_row(row: dict[str, Any], *, phase: str) -> dict[str, Any]:
    return {
        "labels": [phase],
        "caption_pedestrian": str(row.get("caption_pedestrian", "")).strip(),
        "caption_vehicle": str(row.get("caption_vehicle", "")).strip(),
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _length_score(words: int, *, target_words: int) -> float:
    if words <= 0:
        return -10.0
    return max(-3.0, 4.0 - abs(words - target_words) / 18.0)


def _phase_cue_score(lower_text: str, phase: str) -> float:
    cues = PHASE_CUES.get(phase, ())
    hits = sum(1 for cue in cues if cue in lower_text)
    return min(3.0, hits * 0.6)


def _looks_repetitive(text: str) -> bool:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 4:
        return False
    counts = Counter(sentences)
    return max(counts.values()) >= 3


def _normalize_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value.lower()).strip()
    return value.replace("grey", "gray")


def _value_present(text: str, value: str) -> bool:
    if value in text:
        return True
    if value.endswith(" km/h"):
        return value.replace(" ", "") in text.replace(" ", "")
    return False


def _age_contradiction(text: str, value: str) -> bool:
    expected = {match.group(1).lower() for match in DECADE_PATTERN.finditer(value)}
    if not expected:
        return False
    present = {match.group(1).lower() for match in DECADE_PATTERN.finditer(text)}
    return bool(present - expected)


def _speed_contradiction(text: str, value: str) -> bool:
    expected = {match.group(1) for match in SPEED_PATTERN.finditer(value)}
    if not expected:
        return False
    present = {match.group(1) for match in SPEED_PATTERN.finditer(text)}
    return bool(present - expected)


def _gender_contradiction(text: str, value: str) -> bool:
    if "male" in value and "female" not in value:
        return bool(re.search(r"\b(female|woman|girl)\b", text))
    if "female" in value:
        return bool(re.search(r"\b(male|man|boy)\b", text))
    return False


def _color_contradiction(text: str, value: str) -> bool:
    expected = {color for color in COLOR_WORDS if color in value}
    if not expected:
        return False
    present = {color for color in COLOR_WORDS if re.search(rf"\b{re.escape(color)}\b", text)}
    # Captions may mention several objects, so make color contradictions weak:
    # only flag when no expected color appears and a different color does.
    return bool(present and not (present & expected))
