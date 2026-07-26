"""Phase-slot caption rewriting for metric-aligned WTS submissions."""

from __future__ import annotations

from collections import Counter
import math
import re
from pathlib import Path
from typing import Any

from .caption_facts import _predicted_wts_fact_map
from .io import read_json, write_json


CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")

PHASE_ALIASES = {
    "0": ("prerecognition", "pre-recognition", "pre_recognition"),
    "1": ("recognition",),
    "2": ("judgement", "judgment"),
    "3": ("action",),
    "4": ("avoidance",),
    "pre-recognition": ("prerecognition", "pre-recognition", "pre_recognition", "0"),
    "prerecognition": ("prerecognition", "pre-recognition", "pre_recognition", "0"),
    "recognition": ("recognition", "1"),
    "judgement": ("judgement", "judgment", "2"),
    "judgment": ("judgement", "judgment", "2"),
    "action": ("action", "3"),
    "avoidance": ("avoidance", "4"),
}

SLOT_ORDER = ("location", "attention", "behavior", "context", "other")

PEDESTRIAN_SLOT_PATHS = {
    "location": (
        "global.pedestrian.age_group",
        "global.pedestrian.height",
        "phase.pedestrian.body_orientation",
        "phase.pedestrian.position_relative_to_vehicle",
        "phase.pedestrian.distance_to_vehicle",
    ),
    "attention": (
        "phase.pedestrian.line_of_sight",
        "phase.pedestrian.visual_status",
        "phase.pedestrian.awareness_of_vehicle",
    ),
    "behavior": (
        "phase.pedestrian.action",
        "phase.pedestrian.fine_grained_action",
        "phase.pedestrian.direction_of_travel",
        "phase.pedestrian.speed",
    ),
    "context": (
        "global.environment.weather",
        "global.environment.brightness",
        "global.road.surface_condition",
        "global.road.inclination",
        "global.road.surface_type",
        "global.road.traffic_volume",
        "global.road.road_type",
        "global.road.lane_count",
    ),
}

VEHICLE_SLOT_PATHS = {
    "location": (
        "phase.vehicle.position_relative_to_pedestrian",
        "phase.vehicle.distance_to_pedestrian",
        "phase.vehicle.field_of_view",
    ),
    "behavior": (
        "phase.vehicle.action",
        "phase.vehicle.speed",
    ),
    "context": (
        "global.environment.weather",
        "global.environment.brightness",
        "global.road.surface_condition",
        "global.road.inclination",
        "global.road.surface_type",
        "global.road.road_type",
        "global.road.lane_count",
    ),
}

COLOR_RE = re.compile(
    r"\b(black|white|red|blue|green|yellow|orange|gray|grey|brown|beige|navy|purple|pink|cyan)\b",
    flags=re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def rewrite_caption_submission_slots(
    *,
    caption: str | Path,
    output: str | Path,
    wts_vqa_json: str | Path,
    vqa_submission: str | Path,
    report_output: str | Path | None = None,
    mode: str = "balanced",
    max_added_sentences: int = 2,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rewrite caption rows into location/attention/behavior/context order.

    Modes:
    - reorder: preserve generated sentences, only reorder into slots.
    - balanced: reorder and add short missing high-value fact sentences.
    - fill: preserve sentence order and append missing high-value facts.
    - template: prefer canonical slot templates from predicted VQA facts.
    """
    if mode not in {"reorder", "balanced", "fill", "template"}:
        raise ValueError(f"Unsupported caption slot mode: {mode}")
    source = read_json(caption)
    if not isinstance(source, dict):
        raise ValueError("Caption submission must be a JSON object.")
    fact_map = _predicted_wts_fact_map(wts_vqa_json, vqa_submission)

    rewritten: dict[str, list[dict[str, Any]]] = {}
    report = {
        "mode": mode,
        "total_rows": 0,
        "changed_rows": 0,
        "changed_fields": 0,
        "added_sentences": 0,
        "slot_counts": {},
    }
    slot_counts: Counter[str] = Counter()

    for scenario_id, rows in source.items():
        if not isinstance(rows, list):
            rewritten[scenario_id] = rows
            continue
        scenario_facts = fact_map.get(str(scenario_id), {})
        out_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                out_rows.append(row)
                continue
            phase = _phase_label(row, idx)
            bundle = _fact_bundle(scenario_facts, phase)
            new_row = dict(row)
            row_changed = False
            for key in CAPTION_KEYS:
                original = str(row.get(key, "")).strip()
                if not original:
                    continue
                role = "pedestrian" if key == "caption_pedestrian" else "vehicle"
                text, stats = _rewrite_one_caption(
                    original,
                    role=role,
                    facts=bundle,
                    mode=mode,
                    max_added_sentences=max_added_sentences,
                )
                slot_counts.update(stats.get("slot_counts", {}))
                report["added_sentences"] += stats.get("added_sentences", 0)
                if text != original:
                    new_row[key] = text
                    row_changed = True
                    report["changed_fields"] += 1
            report["total_rows"] += 1
            if row_changed:
                report["changed_rows"] += 1
            out_rows.append(new_row)
        rewritten[scenario_id] = out_rows

    report["slot_counts"] = dict(slot_counts)
    write_json(output, rewritten)
    if report_output:
        write_json(report_output, report)
    return rewritten, report


def rerank_caption_slot_variants(
    *,
    base_caption: str | Path,
    candidates: dict[str, str | Path],
    output: str | Path,
    wts_vqa_json: str | Path,
    vqa_submission: str | Path,
    report_output: str | Path | None = None,
    min_switch_margin: float = 1.0,
    min_source_overlap: float = 0.72,
    max_changed_rows: int | None = None,
    preserve_base_order: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select slot-structured rows only when proxy metric score improves.

    The base caption is treated as the safe fallback. Candidate rows are scored
    against the base row, not references, because public test references are
    unavailable. This intentionally favors small, source-preserving repairs.
    """
    base = read_json(base_caption)
    if not isinstance(base, dict):
        raise ValueError("Base caption submission must be a JSON object.")
    loaded = {name: read_json(path) for name, path in candidates.items()}
    for name, data in loaded.items():
        if not isinstance(data, dict):
            raise ValueError(f"Candidate caption submission must be an object: {name}")
    fact_map = _predicted_wts_fact_map(wts_vqa_json, vqa_submission)

    decisions: list[dict[str, Any]] = []
    proposed: list[tuple[float, str, int, dict[str, Any], str, dict[str, float]]] = []
    kept = _copy_caption_submission(base)

    for scenario_id, rows in base.items():
        if not isinstance(rows, list):
            continue
        scenario_facts = fact_map.get(str(scenario_id), {})
        for idx, base_row in enumerate(rows):
            if not isinstance(base_row, dict):
                continue
            phase = _phase_label(base_row, idx)
            bundle = _fact_bundle(scenario_facts, phase)
            base_score, base_parts = _score_row_against_base(
                base_row,
                base_row,
                facts=bundle,
                is_base=True,
            )
            best = (base_score, "__base__", base_row, base_parts)
            for name, data in loaded.items():
                row = _candidate_row(data, str(scenario_id), phase, idx)
                if row is None:
                    continue
                if _row_text(row).strip() == _row_text(base_row).strip():
                    continue
                score, parts = _score_row_against_base(
                    row,
                    base_row,
                    facts=bundle,
                    preserve_base_order=preserve_base_order,
                )
                if parts.get("source_overlap", 0.0) < min_source_overlap:
                    parts["low_overlap_block"] = -100.0
                    score -= 100.0
                if score > best[0]:
                    best = (score, name, row, parts)
            margin = best[0] - base_score
            decisions.append(
                {
                    "scenario_id": scenario_id,
                    "row_index": idx,
                    "phase": phase,
                    "best": best[1],
                    "margin": round(margin, 4),
                    "base_score": round(base_score, 4),
                    "best_score": round(best[0], 4),
                    "base_parts": base_parts,
                    "best_parts": best[3],
                }
            )
            if best[1] != "__base__" and margin >= min_switch_margin:
                proposed.append((margin, str(scenario_id), idx, best[2], best[1], best[3]))

    proposed.sort(key=lambda item: item[0], reverse=True)
    if max_changed_rows is not None:
        proposed = proposed[:max_changed_rows]

    changed = 0
    source_counts: Counter[str] = Counter({"__base__": sum(len(v) for v in base.values() if isinstance(v, list))})
    for _margin, scenario_id, idx, row, source_name, _parts in proposed:
        if scenario_id not in kept or idx >= len(kept[scenario_id]):
            continue
        kept[scenario_id][idx] = _normalize_output_row(row, fallback=kept[scenario_id][idx])
        changed += 1
        source_counts["__base__"] -= 1
        source_counts[source_name] += 1

    report = {
        "total_rows": sum(len(v) for v in base.values() if isinstance(v, list)),
        "changed_rows": changed,
        "candidate_names": sorted(candidates),
        "min_switch_margin": min_switch_margin,
        "min_source_overlap": min_source_overlap,
        "max_changed_rows": max_changed_rows,
        "preserve_base_order": preserve_base_order,
        "source_counts": dict(source_counts),
        "proposed_rows": len(proposed),
        "decisions_sample": decisions[:100],
    }
    write_json(output, kept)
    if report_output:
        write_json(report_output, report)
    return kept, report


def _rewrite_one_caption(
    text: str,
    *,
    role: str,
    facts: dict[str, Any],
    mode: str,
    max_added_sentences: int,
) -> tuple[str, dict[str, Any]]:
    if mode == "template":
        template = _template_caption(role, facts)
        if template:
            return _clean_caption_text(template), {"slot_counts": Counter({"template": 1}), "added_sentences": 0}
        return _clean_caption_text(text), {"slot_counts": Counter({"template_missing": 1}), "added_sentences": 0}

    sentences = _split_sentences(text)
    slot_sentences: dict[str, list[str]] = {slot: [] for slot in SLOT_ORDER}
    for sentence in sentences:
        slot = _classify_sentence(sentence, role=role)
        slot_sentences[slot].append(sentence)

    added: list[str] = []
    if mode in {"balanced", "fill"}:
        occupied_slots = {slot for slot, values in slot_sentences.items() if values}
        added = _missing_fact_sentences(
            text,
            role=role,
            facts=facts,
            limit=max_added_sentences,
            occupied_slots=occupied_slots,
        )
        for sentence in added:
            slot_sentences[_classify_sentence(sentence, role=role)].append(sentence)

    if mode == "fill":
        rewritten = _clean_caption_text(" ".join(_dedupe_sentences(sentences + added)))
        return rewritten, {
            "slot_counts": Counter(
                slot for slot, values in slot_sentences.items() for _ in values
            ),
            "added_sentences": len(added),
        }

    ordered = []
    for slot in SLOT_ORDER:
        ordered.extend(slot_sentences[slot])
    rewritten = _clean_caption_text(" ".join(_dedupe_sentences(ordered)))
    return rewritten, {
        "slot_counts": Counter(
            slot for slot, values in slot_sentences.items() for _ in values
        ),
        "added_sentences": len(added),
    }


def _score_row_against_base(
    row: dict[str, Any],
    base_row: dict[str, Any],
    *,
    facts: dict[str, Any],
    is_base: bool = False,
    preserve_base_order: bool = True,
) -> tuple[float, dict[str, float]]:
    row_text = _row_text(row)
    base_text = _row_text(base_row)
    parts: dict[str, float] = {}
    parts["non_empty"] = 5.0 if all(str(row.get(key, "")).strip() for key in CAPTION_KEYS) else -30.0
    parts["source_overlap"] = _token_f1(row_text, base_text)
    parts["ngram_preservation"] = _ngram_f1(row_text, base_text, n=4)
    parts["slot_order"] = 0.0 if preserve_base_order else _slot_order_score(row)
    parts["fact_match"] = _fact_match_score(row_text, facts)
    parts["length"] = _relative_length_score(row_text, base_text)
    parts["artifact_penalty"] = -8.0 if _has_artifacts(row_text) else 0.0
    parts["repetition_penalty"] = -5.0 if _looks_repetitive(row_text) else 0.0
    parts["added_fact_penalty"] = _added_fact_penalty(row_text, base_text, facts)
    parts["base_tiebreak"] = 0.35 if is_base else 0.0
    score = (
        8.0 * parts["source_overlap"]
        + 7.0 * parts["ngram_preservation"]
        + (0.0 if preserve_base_order else 1.8 * parts["slot_order"])
        + 1.2 * parts["fact_match"]
        + parts["length"]
        + parts["artifact_penalty"]
        + parts["repetition_penalty"]
        + parts["added_fact_penalty"]
        + parts["base_tiebreak"]
    )
    return score, {key: round(value, 4) for key, value in parts.items()}


def _slot_order_score(row: dict[str, Any]) -> float:
    scores = []
    for key in CAPTION_KEYS:
        role = "pedestrian" if key == "caption_pedestrian" else "vehicle"
        slots = [_classify_sentence(sentence, role=role) for sentence in _split_sentences(str(row.get(key, "")))]
        numeric = [SLOT_ORDER.index(slot) for slot in slots if slot in SLOT_ORDER]
        if len(numeric) < 2:
            scores.append(0.0)
            continue
        inversions = 0
        pairs = 0
        for i, left in enumerate(numeric):
            for right in numeric[i + 1 :]:
                pairs += 1
                inversions += int(left > right)
        scores.append(1.0 - inversions / max(pairs, 1))
    return sum(scores) / max(len(scores), 1)


def _fact_match_score(text: str, facts: dict[str, Any]) -> float:
    flat = _flatten_fact_values(facts)
    if not flat:
        return 0.0
    lower = text.lower()
    matches = sum(1 for value in flat if value and value.lower() in lower)
    return min(1.0, matches / min(len(flat), 12))


def _added_fact_penalty(text: str, base_text: str, facts: dict[str, Any]) -> float:
    lower = text.lower()
    base_lower = base_text.lower()
    penalty = 0.0
    for value in _flatten_fact_values(facts):
        value = value.lower()
        if not value or len(value) < 4:
            continue
        if value in lower and value not in base_lower:
            penalty -= 0.25
    return max(-3.0, penalty)


def _relative_length_score(text: str, base_text: str) -> float:
    row_len = max(len(re.findall(r"\b\w+\b", text)), 1)
    base_len = max(len(re.findall(r"\b\w+\b", base_text)), 1)
    ratio = row_len / base_len
    return max(-4.0, 2.0 - abs(math.log(ratio)) * 8.0)


def _token_f1(text: str, reference: str) -> float:
    return _counter_f1(Counter(_token_list(text)), Counter(_token_list(reference)))


def _ngram_f1(text: str, reference: str, *, n: int) -> float:
    return _counter_f1(_ngrams(_token_list(text), n), _ngrams(_token_list(reference), n))


def _counter_f1(left: Counter[Any], right: Counter[Any]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right[item]) for item, count in left.items())
    precision = overlap / max(sum(left.values()), 1)
    recall = overlap / max(sum(right.values()), 1)
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _token_list(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "")).strip() for key in CAPTION_KEYS)


def _has_artifacts(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            "caption_pedestrian",
            "caption_vehicle",
            "return json",
            "bbox context",
            "visual grounding note",
        )
    )


def _looks_repetitive(text: str) -> bool:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 4:
        return False
    counts = Counter(sentences)
    return max(counts.values()) >= 3


def _flatten_fact_values(facts: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif str(value).strip():
            values.append(str(value).strip())

    visit(facts)
    return values


def _copy_caption_submission(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    copied: dict[str, list[dict[str, Any]]] = {}
    for scenario_id, rows in data.items():
        copied[str(scenario_id)] = [dict(row) if isinstance(row, dict) else row for row in rows]
    return copied


def _candidate_row(
    data: dict[str, Any],
    scenario_id: str,
    phase: str,
    idx: int,
) -> dict[str, Any] | None:
    rows = data.get(scenario_id)
    if not isinstance(rows, list):
        return None
    for row_idx, row in enumerate(rows):
        if isinstance(row, dict) and _phase_label(row, row_idx) == phase:
            return row
    if idx < len(rows) and isinstance(rows[idx], dict):
        return rows[idx]
    return None


def _normalize_output_row(row: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
    return {
        "labels": row.get("labels", fallback.get("labels", [])),
        "caption_pedestrian": str(row.get("caption_pedestrian", fallback.get("caption_pedestrian", ""))).strip(),
        "caption_vehicle": str(row.get("caption_vehicle", fallback.get("caption_vehicle", ""))).strip(),
    }


def _template_caption(role: str, facts: dict[str, Any]) -> str:
    if role == "pedestrian":
        return " ".join(
            sentence
            for sentence in (
                _pedestrian_identity_sentence(facts),
                _pedestrian_location_sentence(facts),
                _pedestrian_attention_sentence(facts),
                _pedestrian_behavior_sentence(facts),
                _context_sentence(facts),
            )
            if sentence
        )
    return " ".join(
        sentence
        for sentence in (
            _vehicle_location_sentence(facts),
            _vehicle_behavior_sentence(facts),
            _context_sentence(facts),
        )
        if sentence
    )


def _missing_fact_sentences(
    text: str,
    *,
    role: str,
    facts: dict[str, Any],
    limit: int,
    occupied_slots: set[str],
) -> list[str]:
    if limit <= 0:
        return []
    lower = text.lower()
    candidates = (
        [
            _pedestrian_location_sentence(facts),
            _pedestrian_attention_sentence(facts),
            _pedestrian_behavior_sentence(facts),
            _context_sentence(facts),
        ]
        if role == "pedestrian"
        else [
            _vehicle_location_sentence(facts),
            _vehicle_behavior_sentence(facts),
            _context_sentence(facts),
        ]
    )
    selected: list[str] = []
    for sentence in candidates:
        if not sentence:
            continue
        if _classify_sentence(sentence, role=role) in occupied_slots:
            continue
        if _sentence_supported_by_source(sentence, lower):
            continue
        if _would_add_color_conflict(sentence, lower):
            continue
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def _sentence_supported_by_source(sentence: str, lower_text: str) -> bool:
    stop = {
        "pedestrian",
        "vehicle",
        "road",
        "environment",
        "context",
        "distance",
        "positioned",
        "surface",
        "with",
        "from",
        "that",
        "this",
        "were",
        "around",
    }
    values = [
        token
        for token in re.findall(r"\b[a-z0-9]+(?:/[a-z0-9]+)?\b", sentence.lower())
        if len(token) > 3 and token not in stop
    ]
    if not values:
        return False
    hits = sum(1 for token in values if token in lower_text)
    return hits >= max(2, int(len(values) * 0.6))


def _would_add_color_conflict(sentence: str, lower_text: str) -> bool:
    source_colors = set(COLOR_RE.findall(lower_text))
    sentence_colors = set(COLOR_RE.findall(sentence.lower()))
    return bool(source_colors and sentence_colors and not (source_colors & sentence_colors))


def _pedestrian_identity_sentence(facts: dict[str, Any]) -> str:
    age = _get(facts, "global.pedestrian.age_group")
    height = _get(facts, "global.pedestrian.height")
    upper_type = _get(facts, "global.pedestrian.upper_body_type")
    upper_color = _get(facts, "global.pedestrian.upper_body_color")
    lower_type = _get(facts, "global.pedestrian.lower_body_type")
    lower_color = _get(facts, "global.pedestrian.lower_body_color")
    parts = []
    if age:
        parts.append(f"in his {age}")
    if height:
        parts.append(f"approximately {height} tall")
    clothing = _clothing_phrase(upper_color, upper_type, lower_color, lower_type)
    if clothing:
        parts.append(f"was wearing {clothing}")
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].startswith("was wearing"):
        return "The pedestrian " + parts[0] + "."
    return "The pedestrian, " + ", ".join(parts) + "."


def _pedestrian_location_sentence(facts: dict[str, Any]) -> str:
    body = _get(facts, "phase.pedestrian.body_orientation")
    position = _get(facts, "phase.pedestrian.position_relative_to_vehicle")
    distance = _get(facts, "phase.pedestrian.distance_to_vehicle")
    chunks = []
    if position:
        chunks.append(f"was positioned {_relative_position_phrase(position, 'vehicle')}")
    if distance:
        chunks.append(f"at a {_lower_first(distance)} distance from the vehicle")
    if body:
        chunks.append(f"with his body {_body_orientation_phrase(body)}")
    if not chunks:
        return ""
    return "The pedestrian " + ", ".join(chunks) + "."


def _pedestrian_attention_sentence(facts: dict[str, Any]) -> str:
    line = _get(facts, "phase.pedestrian.line_of_sight")
    visual = _get(facts, "phase.pedestrian.visual_status")
    awareness = _get(facts, "phase.pedestrian.awareness_of_vehicle")
    chunks = []
    if line:
        chunks.append(_line_of_sight_phrase(line))
    if visual:
        chunks.append(_visual_status_phrase(visual))
    if awareness:
        chunks.append(_awareness_phrase(awareness))
    if not chunks:
        return ""
    return " ".join(chunks)


def _pedestrian_behavior_sentence(facts: dict[str, Any]) -> str:
    action = _get(facts, "phase.pedestrian.action")
    fine = _get(facts, "phase.pedestrian.fine_grained_action")
    direction = _get(facts, "phase.pedestrian.direction_of_travel")
    speed = _get(facts, "phase.pedestrian.speed")
    chunks = []
    if action:
        chunks.append(_pedestrian_action_clause(action))
    if fine:
        chunks.append(f"with fine-grained behavior of {_lower_first(fine)}")
    if direction:
        chunks.append(f"traveling {_lower_first(direction)}")
    if speed:
        chunks.append(f"at a {_lower_first(speed)} speed")
    if not chunks:
        return ""
    return _capitalize("; ".join(chunks)) + "."


def _vehicle_location_sentence(facts: dict[str, Any]) -> str:
    position = _get(facts, "phase.vehicle.position_relative_to_pedestrian")
    distance = _get(facts, "phase.vehicle.distance_to_pedestrian")
    fov = _get(facts, "phase.vehicle.field_of_view")
    chunks = []
    if position:
        chunks.append(f"positioned {_relative_position_phrase(position, 'pedestrian')}")
    if distance:
        chunks.append(f"at a {_lower_first(distance)} distance from the pedestrian")
    if fov:
        chunks.append(_vehicle_fov_phrase(fov))
    if not chunks:
        return ""
    return "The vehicle was " + ", ".join(chunks) + "."


def _vehicle_behavior_sentence(facts: dict[str, Any]) -> str:
    action = _get(facts, "phase.vehicle.action")
    speed = _get(facts, "phase.vehicle.speed")
    chunks = []
    if action:
        chunks.append(_vehicle_action_clause(action))
    if speed:
        chunks.append(f"with a speed of {speed}")
    if not chunks:
        return ""
    return _capitalize("; ".join(chunks)) + "."


def _context_sentence(facts: dict[str, Any]) -> str:
    weather = _get(facts, "global.environment.weather")
    brightness = _get(facts, "global.environment.brightness")
    surface = _get(facts, "global.road.surface_condition")
    inclination = _get(facts, "global.road.inclination")
    material = _get(facts, "global.road.surface_type")
    road_type = _get(facts, "global.road.road_type")
    lanes = _get(facts, "global.road.lane_count")
    traffic = _get(facts, "global.road.traffic_volume")
    chunks = []
    if weather or brightness:
        chunks.append(
            "the weather was "
            + _lower_first(weather or "unknown")
            + ", and the brightness of the surroundings was "
            + _lower_first(brightness or "unknown")
        )
    road_bits = [value for value in (surface, inclination, material) if value]
    if road_bits:
        chunks.append(
            "the road surface conditions were "
            + " on a ".join(
                [
                    " ".join(_lower_first(value) for value in (surface,) if value),
                    " ".join(_lower_first(value) for value in (inclination, material) if value) + " road",
                ]
            ).strip()
        )
    if road_type or lanes:
        chunks.append(
            "the road was classified as "
            + " with ".join(
                part
                for part in (
                    _article_phrase(road_type),
                    _lower_first(lanes),
                )
                if part
            )
        )
    if traffic:
        chunks.append(f"the traffic volume was {_lower_first(traffic)}")
    if not chunks:
        return ""
    return "For the environment, " + ". ".join(_capitalize(chunk) for chunk in chunks) + "."


def _clothing_phrase(
    upper_color: str,
    upper_type: str,
    lower_color: str,
    lower_type: str,
) -> str:
    upper = " ".join(_lower_first(part) for part in (upper_color, upper_type) if part).strip()
    lower = " ".join(_lower_first(part) for part in (lower_color, lower_type) if part).strip()
    if upper and lower:
        return f"{upper} and {lower}"
    return upper or lower


def _pedestrian_action_clause(value: str) -> str:
    lower = value.lower()
    if "collision" in lower or "collid" in lower:
        return "the pedestrian was involved in a collision"
    if "standing still" in lower:
        return "the pedestrian was standing still"
    if lower.startswith("going"):
        return f"the pedestrian was {_lower_first(value)}"
    return f"the pedestrian was {_lower_first(value)}"


def _vehicle_action_clause(value: str) -> str:
    lower = value.lower()
    if "collided" in lower or "collision" in lower:
        return "the vehicle collided with the pedestrian"
    if "emergency braking" in lower or "emergency brake" in lower:
        return "the vehicle avoided the pedestrian by emergency braking"
    if lower.startswith("started"):
        return f"the vehicle {_lower_first(value)}"
    if lower.startswith("going") or lower.startswith("turn"):
        return f"the vehicle was {_lower_first(value)}"
    if "stopped" in lower:
        return "the vehicle was stopped"
    return f"the vehicle was {_lower_first(value)}"


def _line_of_sight_phrase(value: str) -> str:
    lower = value.lower()
    if "vehicle" in lower:
        return "The pedestrian's line of sight was fixed on the vehicle."
    if "direction of travel" in lower or "front" in lower:
        return "The pedestrian's line of sight was in front, aligned with the direction of travel."
    return f"The pedestrian's line of sight was {_lower_first(value)}."


def _visual_status_phrase(value: str) -> str:
    lower = value.lower()
    if "closely" in lower:
        return "The pedestrian was closely watching the surroundings."
    if "slowly" in lower and "looking" in lower:
        return "The pedestrian was slowly looking around."
    if "constant" in lower:
        return "The pedestrian was constantly looking around intently."
    return f"The pedestrian's visual status was {_lower_first(value)}."


def _awareness_phrase(value: str) -> str:
    lower = value.lower()
    if "unaware" in lower:
        return "The pedestrian appeared unaware of the vehicle."
    if "notice" in lower or "aware" in lower:
        return "The pedestrian appeared to notice the vehicle and was aware of its presence."
    return f"The pedestrian's awareness was {_lower_first(value)}."


def _relative_position_phrase(value: str, target: str) -> str:
    text = _lower_first(value)
    if target not in text:
        if "front" in text:
            text = text.replace("front", f"front of the {target}")
        elif "side" in text:
            text = f"{text} of the {target}"
        elif "left" in text or "right" in text or "behind" in text:
            text = f"{text} of the {target}"
    return text


def _body_orientation_phrase(value: str) -> str:
    lower = value.lower()
    if "opposite" in lower:
        return "facing the opposite direction from the vehicle"
    if "same direction" in lower:
        return "facing the same direction as the vehicle"
    if "perpendicular" in lower:
        return "perpendicular to the vehicle"
    return _lower_first(value)


def _vehicle_fov_phrase(value: str) -> str:
    lower = value.lower()
    if "not visible" in lower:
        return "the pedestrian was not visible within the vehicle's field of view"
    if "visible" in lower:
        return "the pedestrian was visible within the vehicle's field of view"
    return _lower_first(value)


def _article_phrase(value: str) -> str:
    value = _lower_first(value)
    if not value:
        return ""
    if value.startswith(("a ", "an ", "the ")):
        return value
    return ("an " if value[0] in "aeiou" else "a ") + value


def _classify_sentence(sentence: str, *, role: str) -> str:
    lower = sentence.lower()
    if any(
        token in lower
        for token in (
            "weather",
            "brightness",
            "road",
            "surface",
            "asphalt",
            "traffic",
            "sidewalk",
            "roadside",
            "street light",
            "environment",
            "lane",
        )
    ):
        return "context"
    if role == "pedestrian" and any(
        token in lower
        for token in (
            "line of sight",
            "watch",
            "aware",
            "notice",
            "visual",
            "look",
            "gaze",
            "attention",
        )
    ):
        return "attention"
    has_location = any(
        token in lower
        for token in (
            "position",
            "distance",
            "near",
            "close",
            "far",
            "front",
            "left",
            "right",
            "diagonal",
            "perpendicular",
            "opposite",
            "height",
            "wearing",
            "t-shirt",
            "shirt",
            "slacks",
            "pants",
            "vehicle is visible",
            "field of view",
            "age group",
        )
    )
    has_strong_behavior = any(
        token in lower
        for token in (
            "speed",
            "collision",
            "collided",
            "brak",
            "turn",
            "stopped",
            "action",
            "behavior",
        )
    )
    if has_location and not has_strong_behavior:
        return "location"
    if has_strong_behavior or any(
        token in lower
        for token in (
            "going",
            "travel",
            "walking",
            "moving",
            "cross",
            "standing",
        )
    ):
        return "behavior"
    if has_location:
        return "location"
    return "other"


def _fact_bundle(scenario_facts: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_facts = _phase_facts(scenario_facts, phase)
    return {
        "global": scenario_facts.get("global", {}) if isinstance(scenario_facts, dict) else {},
        "phase": phase_facts,
    }


def _phase_facts(scenario_facts: dict[str, Any], phase: str) -> dict[str, Any]:
    if not isinstance(scenario_facts, dict):
        return {}
    phases = scenario_facts.get("phases", {})
    if not isinstance(phases, dict):
        return {}
    keys = [str(phase)]
    keys.extend(PHASE_ALIASES.get(str(phase).lower(), ()))
    normalized = {_normalize_phase_key(key): key for key in phases}
    for key in keys:
        if key in phases and isinstance(phases[key], dict):
            return phases[key]
        actual = normalized.get(_normalize_phase_key(key))
        if actual and isinstance(phases[actual], dict):
            return phases[actual]
    return {}


def _phase_label(row: dict[str, Any], idx: int) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return str(idx)


def _get(facts: dict[str, Any], dotted_key: str) -> str:
    cursor: Any = facts
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return ""
        cursor = cursor[part]
    return str(cursor).strip()


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            if sentence[-1] not in ".!?":
                sentence += "."
            sentences.append(sentence)
    return sentences or [text.strip()]


def _dedupe_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def _clean_caption_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _normalize_phase_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _lower_first(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    return value[:1].lower() + value[1:]


def _capitalize(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    return value[:1].upper() + value[1:]
