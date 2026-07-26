"""Segment-expert caption datasets and assembly.

This follows the AIO_ISC divide-and-conquer idea: learn smaller caption
segments first, then compose pedestrian/vehicle captions in a stable order.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any

from .caption_facts import _flatten_facts, _gold_fact_map, _predicted_wts_fact_map
from .io import read_json, write_json


PEDESTRIAN_SEGMENTS = ("appearance", "environment", "location", "attention", "action")
VEHICLE_SEGMENTS = ("appearance", "environment", "location", "attention", "action")
SEGMENT_ORDER = {
    "pedestrian": PEDESTRIAN_SEGMENTS,
    "vehicle": VEHICLE_SEGMENTS,
}
PHASE_ORDER = ["4", "3", "2", "1", "0"]
VIEW_PRIORITY = {"overhead_view": 0, "vehicle_view": 1, "environment": 2}
CAPTION_KEYS = {
    "pedestrian": "caption_pedestrian",
    "vehicle": "caption_vehicle",
}


def export_caption_segment_train(
    *,
    dataset: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    max_rows: int = 0,
    extraction: str = "rules",
    role_filter: str | None = None,
    segment_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Export segment-level SFT rows from a caption LLaMA-Factory dataset."""
    rows = _segment_rows(
        read_json(dataset),
        train=True,
        max_rows=max_rows,
        extraction=extraction,
        role_filter=role_filter,
        segment_filter=segment_filter,
    )
    write_json(output, rows)
    _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def export_caption_segment_test(
    *,
    dataset: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    max_rows: int = 0,
    role_filter: str | None = None,
    segment_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Export segment-level inference rows from a caption LLaMA-Factory dataset."""
    rows = _segment_rows(
        read_json(dataset),
        train=False,
        max_rows=max_rows,
        extraction="rules",
        role_filter=role_filter,
        segment_filter=segment_filter,
    )
    write_json(output, rows)
    _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def assemble_caption_segments(
    *,
    inference_dataset: str | Path,
    predictions: str | Path,
    output: str | Path,
    fallback_caption: str | Path | None = None,
    report_output: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_json(inference_dataset)
    pred_texts = _read_prediction_texts(predictions)
    if len(rows) != len(pred_texts):
        raise ValueError(f"Prediction count mismatch: dataset={len(rows)} predictions={len(pred_texts)}")

    fallback = read_json(fallback_caption) if fallback_caption else {}
    grouped: dict[tuple[str, str], dict[str, dict[str, tuple[int, str]]]] = {}
    parse_failed = 0
    for row, pred in zip(rows, pred_texts):
        meta = row.get("metadata", {})
        scenario_id = str(meta.get("scenario_id", ""))
        phase = str(meta.get("phase", meta.get("fallback_idx", "")))
        view = str(meta.get("view", ""))
        priority = VIEW_PRIORITY.get(view, 99)
        role = str(meta.get("role", ""))
        segment = str(meta.get("segment", ""))
        if not scenario_id or role not in SEGMENT_ORDER or segment not in SEGMENT_ORDER[role]:
            parse_failed += 1
            continue
        text = _clean_segment_text(pred)
        if not text:
            parse_failed += 1
            continue
        key = (scenario_id, phase)
        role_bucket = grouped.setdefault(key, {"pedestrian": {}, "vehicle": {}})[role]
        old = role_bucket.get(segment)
        if old is None or priority < old[0]:
            role_bucket[segment] = (priority, text)

    submission: dict[str, list[dict[str, Any]]] = {}
    if isinstance(fallback, dict) and fallback:
        submission = {
            str(sid): [dict(row) if isinstance(row, dict) else row for row in phase_rows]
            for sid, phase_rows in fallback.items()
        }

    changed = 0
    for key in sorted(grouped, key=lambda item: (item[0], _phase_sort_key(item[1]))):
        scenario_id, phase = key
        idx = _phase_index_for_submission(submission, scenario_id, phase)
        existing = _fallback_row(submission, scenario_id, idx)
        new_row = dict(existing) if existing else {"labels": [phase]}
        for role in ("pedestrian", "vehicle"):
            caption = _compose_caption(role, {seg: value for seg, (_priority, value) in grouped[key][role].items()})
            if caption:
                new_row[CAPTION_KEYS[role]] = caption
        if scenario_id not in submission:
            submission[scenario_id] = []
        while len(submission[scenario_id]) <= idx:
            label = PHASE_ORDER[len(submission[scenario_id])] if len(submission[scenario_id]) < len(PHASE_ORDER) else str(len(submission[scenario_id]))
            submission[scenario_id].append({"labels": [label]})
        before = submission[scenario_id][idx]
        submission[scenario_id][idx] = _normalize_row(new_row, before)
        if any(
            str(before.get(field, "")).strip() != str(submission[scenario_id][idx].get(field, "")).strip()
            for field in CAPTION_KEYS.values()
        ):
            changed += 1

    report = {
        "segment_rows": len(rows),
        "prediction_rows": len(pred_texts),
        "assembled_phase_rows": sum(len(v) for v in submission.values() if isinstance(v, list)),
        "changed_from_fallback": changed,
        "parse_failed": parse_failed,
        "segments_per_role": {role: list(segments) for role, segments in SEGMENT_ORDER.items()},
    }
    write_json(output, submission)
    if report_output:
        write_json(report_output, report)
    return submission, report


def export_caption_rewrite_train(
    *,
    dataset: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    extraction: str = "remap",
    max_rows: int = 0,
    lock_target_caption: bool = False,
    index_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    fact_map = _gold_fact_map(index_path) if index_path else {}
    rows = _rewrite_rows(
        read_json(dataset),
        train=True,
        extraction=extraction,
        max_rows=max_rows,
        lock_target_caption=lock_target_caption,
        fact_map=fact_map,
    )
    write_json(output, rows)
    _maybe_write_text_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def export_caption_rewrite_test(
    *,
    segment_submission: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    fallback_caption: str | Path | None = None,
    max_rows: int = 0,
    trusted_caption: str | Path | None = None,
    wts_vqa_json: str | Path | None = None,
    vqa_submission: str | Path | None = None,
) -> list[dict[str, Any]]:
    if bool(wts_vqa_json) != bool(vqa_submission):
        raise ValueError("--wts-vqa-json and --vqa-submission must be provided together.")
    segment_caps = read_json(segment_submission)
    fallback = read_json(fallback_caption) if fallback_caption else {}
    trusted = read_json(trusted_caption) if trusted_caption else fallback
    fact_map = _predicted_wts_fact_map(wts_vqa_json, vqa_submission) if wts_vqa_json else {}
    rows: list[dict[str, Any]] = []
    for scenario_id, phase_rows in sorted(segment_caps.items()):
        if not isinstance(phase_rows, list):
            continue
        for phase_idx, row in enumerate(phase_rows):
            if not isinstance(row, dict):
                continue
            labels = row.get("labels") if isinstance(row.get("labels"), list) else []
            phase = str(labels[0]) if labels else str(phase_idx)
            fb_row = _fallback_row(fallback, str(scenario_id), phase_idx) if isinstance(fallback, dict) else {}
            trusted_row = _fallback_row(trusted, str(scenario_id), phase_idx) if isinstance(trusted, dict) else {}
            rows.append(
                {
                    "instruction": _rewrite_instruction(
                        pedestrian=str(row.get("caption_pedestrian", "")),
                        vehicle=str(row.get("caption_vehicle", "")),
                        lock_context=_format_lock_context(
                            trusted_row=trusted_row,
                            facts=fact_map.get(str(scenario_id), {}),
                            phase=phase,
                        ),
                    ),
                    "input": "",
                    "output": "",
                    "metadata": {
                        "task": "caption_rewrite_inference",
                        "scenario_id": str(scenario_id),
                        "phase": phase,
                        "phase_index": phase_idx,
                        "fallback": fb_row,
                    },
                }
            )
            if max_rows and len(rows) >= max_rows:
                write_json(output, rows)
                _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
                return rows
    write_json(output, rows)
    _maybe_write_text_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def assemble_caption_rewrite(
    *,
    inference_dataset: str | Path,
    predictions: str | Path,
    output: str | Path,
    fallback_caption: str | Path | None = None,
    report_output: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_json(inference_dataset)
    pred_texts = _read_prediction_texts(predictions)
    if len(rows) != len(pred_texts):
        raise ValueError(f"Prediction count mismatch: dataset={len(rows)} predictions={len(pred_texts)}")
    fallback = read_json(fallback_caption) if fallback_caption else {}
    submission: dict[str, list[dict[str, Any]]] = {}
    if isinstance(fallback, dict):
        submission = {
            str(sid): [dict(row) if isinstance(row, dict) else row for row in phase_rows]
            for sid, phase_rows in fallback.items()
        }
    changed = 0
    parse_failed = 0
    for row, pred in zip(rows, pred_texts):
        meta = row.get("metadata", {})
        sid = str(meta.get("scenario_id", ""))
        idx = int(meta.get("phase_index", 0) or 0)
        if not sid:
            parse_failed += 1
            continue
        parsed = _parse_output(pred)
        if not parsed:
            parsed = _split_rewrite_prediction(pred)
        ped = _clean_caption_text(str(parsed.get("caption_pedestrian", "")))
        veh = _clean_caption_text(str(parsed.get("caption_vehicle", "")))
        if not ped and not veh:
            parse_failed += 1
            continue
        if sid not in submission:
            submission[sid] = []
        while len(submission[sid]) <= idx:
            label = PHASE_ORDER[len(submission[sid])] if len(submission[sid]) < len(PHASE_ORDER) else str(len(submission[sid]))
            submission[sid].append({"labels": [label]})
        before = dict(submission[sid][idx])
        if ped:
            submission[sid][idx]["caption_pedestrian"] = ped
        if veh:
            submission[sid][idx]["caption_vehicle"] = veh
        if before != submission[sid][idx]:
            changed += 1
    report = {
        "rows": len(rows),
        "prediction_rows": len(pred_texts),
        "changed_from_fallback": changed,
        "parse_failed": parse_failed,
    }
    write_json(output, submission)
    if report_output:
        write_json(report_output, report)
    return submission, report


def _segment_rows(
    data: list[dict[str, Any]],
    *,
    train: bool,
    max_rows: int,
    extraction: str,
    role_filter: str | None,
    segment_filter: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        meta = row.get("metadata", {})
        if meta.get("task") != "caption":
            continue
        target = _parse_output(row.get("output", ""))
        for role, caption_key in CAPTION_KEYS.items():
            if role_filter and role != role_filter:
                continue
            caption = str(target.get(caption_key, "")).strip()
            segments = _extract_segments(caption, role=role, extraction=extraction) if train else {seg: "" for seg in SEGMENT_ORDER[role]}
            for segment in SEGMENT_ORDER[role]:
                if segment_filter and segment != segment_filter:
                    continue
                if train and not segments.get(segment):
                    continue
                instruction = _segment_instruction(row, role=role, segment=segment)
                out.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": segments.get(segment, "") if train else "",
                        "videos": list(row.get("videos", [])),
                        "images": list(row.get("images", [])) if isinstance(row.get("images"), list) else row.get("images", []),
                        "metadata": {
                            "task": "caption_segment" if train else "caption_segment_inference",
                            "source_task": meta.get("task"),
                            "split": meta.get("split"),
                            "scenario_id": meta.get("scenario_id"),
                            "scenario_type": meta.get("scenario_type"),
                            "view": meta.get("view"),
                            "phase": str(meta.get("phase", "")),
                            "source_row_index": idx,
                            "role": role,
                            "segment": segment,
                        },
                    }
                )
                if max_rows and len(out) >= max_rows:
                    return out
    return out


def _rewrite_rows(
    data: list[dict[str, Any]],
    *,
    train: bool,
    extraction: str,
    max_rows: int,
    lock_target_caption: bool = False,
    fact_map: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fact_map = fact_map or {}
    for idx, row in enumerate(data):
        meta = row.get("metadata", {})
        if meta.get("task") != "caption":
            continue
        target = _parse_output(row.get("output", ""))
        ped_target = str(target.get("caption_pedestrian", "")).strip()
        veh_target = str(target.get("caption_vehicle", "")).strip()
        if not ped_target and not veh_target:
            continue
        ped_segments = _extract_segments(ped_target, role="pedestrian", extraction=extraction)
        veh_segments = _extract_segments(veh_target, role="vehicle", extraction=extraction)
        rows.append(
            {
                "instruction": _rewrite_instruction(
                    pedestrian=_compose_caption("pedestrian", ped_segments),
                    vehicle=_compose_caption("vehicle", veh_segments),
                    lock_context=_format_lock_context(
                        trusted_row={
                            "caption_pedestrian": ped_target,
                            "caption_vehicle": veh_target,
                        }
                        if lock_target_caption
                        else {},
                        facts=fact_map.get(str(meta.get("scenario_id", "")), {}),
                        phase=str(meta.get("phase", "")),
                    ),
                ),
                "input": "",
                "output": json.dumps(
                    {
                        "caption_pedestrian": ped_target,
                        "caption_vehicle": veh_target,
                    },
                    ensure_ascii=False,
                )
                if train
                else "",
                "metadata": {
                    "task": "caption_rewrite" if train else "caption_rewrite_inference",
                    "source_task": meta.get("task"),
                    "split": meta.get("split"),
                    "scenario_id": meta.get("scenario_id"),
                    "scenario_type": meta.get("scenario_type"),
                    "view": meta.get("view"),
                    "phase": str(meta.get("phase", "")),
                    "source_row_index": idx,
                },
            }
        )
        if max_rows and len(rows) >= max_rows:
            return rows
    return rows


def _segment_instruction(row: dict[str, Any], *, role: str, segment: str) -> str:
    base = str(row.get("instruction", "")).strip()
    boundaries = _segment_boundary_text(role=role, segment=segment)
    return (
        "You are a traffic safety segment captioner.\n"
        f"Generate only the {segment} segment for the {role} caption.\n"
        "Use one or two official WTS-style sentences. Preserve concrete visual facts.\n"
        "Do not output JSON. Do not mention colored overlays, bbox text, prompts, or segment names.\n\n"
        f"{boundaries}\n\n"
        f"{base}\n\n"
        f"Target role: {role}\n"
        f"Target segment: {segment}\n"
        "Return only the segment text."
    )


def _rewrite_instruction(*, pedestrian: str, vehicle: str, lock_context: str = "") -> str:
    lock_block = f"\nLocked fact context:\n{lock_context.strip()}\n" if lock_context.strip() else ""
    return (
        "You are a WTS traffic safety caption rewriter.\n"
        "Rewrite the segment notes into the official caption style.\n"
        "Preserve all concrete facts. Keep pedestrian and vehicle captions separate.\n"
        "If segment notes conflict with locked facts, keep the locked facts and rewrite around them.\n"
        "Use complete sentences. Do not add unsupported facts.\n"
        "Return strict JSON with keys caption_pedestrian and caption_vehicle.\n\n"
        "Pedestrian caption order: appearance, environment, location, attention/action.\n"
        "Vehicle caption order: location, visibility, vehicle action, pedestrian appearance, environment.\n\n"
        f"{lock_block}"
        f"Pedestrian segment notes:\n{pedestrian.strip()}\n\n"
        f"Vehicle segment notes:\n{vehicle.strip()}\n"
    )


def _format_lock_context(*, trusted_row: dict[str, Any], facts: dict[str, Any], phase: str) -> str:
    lines: list[str] = []
    ped = str(trusted_row.get("caption_pedestrian", "")).strip()
    veh = str(trusted_row.get("caption_vehicle", "")).strip()
    if ped or veh:
        lines.append("Trusted baseline caption facts. Prefer these over noisy segment notes.")
        if ped:
            lines.append(f"- trusted_caption_pedestrian: {_shorten_lock_text(ped)}")
        if veh:
            lines.append(f"- trusted_caption_vehicle: {_shorten_lock_text(veh)}")
    fact_lines = _fact_lock_lines(facts, phase=phase)
    if fact_lines:
        lines.append("Predicted VQA facts. Treat these as attribute locks unless impossible.")
        lines.extend(fact_lines)
    if lines:
        lines.append("Do not introduce new age, gender, clothing, weather, road, position, speed, or action facts that contradict these locks.")
    return "\n".join(lines)


def _fact_lock_lines(facts: dict[str, Any], *, phase: str, max_items: int = 20) -> list[str]:
    selected: list[tuple[str, str]] = []
    if isinstance(facts.get("global"), dict):
        selected.extend((f"scenario.{key}", value) for key, value in _flatten_facts(facts["global"]))
    phase_facts = facts.get("phases", {}).get(str(phase), {}) if isinstance(facts.get("phases"), dict) else {}
    if isinstance(phase_facts, dict):
        selected.extend((f"phase_{phase}.{key}", value) for key, value in _flatten_facts(phase_facts))
    lock_keywords = (
        "age",
        "gender",
        "clothing",
        "upper",
        "lower",
        "weather",
        "brightness",
        "road",
        "traffic",
        "speed",
        "action",
        "position",
        "distance",
        "direction",
        "line_of_sight",
        "aware",
    )
    out = []
    for key, value in selected:
        if any(word in key.lower() for word in lock_keywords) or len(out) < 8:
            out.append(f"- {key}: {value}")
        if len(out) >= max_items:
            break
    return out


def _shorten_lock_text(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _extract_segments(caption: str, *, role: str, extraction: str = "rules") -> dict[str, str]:
    if extraction == "remap":
        return _extract_segments_remap(caption, role=role)
    buckets: dict[str, list[str]] = {seg: [] for seg in SEGMENT_ORDER[role]}
    for sentence in _split_sentences(caption):
        for clause in _split_clauses(sentence):
            segment = _classify_sentence(clause, role=role)
            if segment in buckets and _clause_allowed(clause, role=role, segment=segment):
                buckets[segment].append(clause)
    return _sanitize_segments(buckets, role=role)


def _extract_segments_remap(caption: str, *, role: str) -> dict[str, str]:
    sentences = _split_sentences(caption)
    if not sentences:
        return {}
    anchors = [_segment_anchor(role, segment) for segment in SEGMENT_ORDER[role]]
    vectors = _embed_texts(sentences + anchors)
    sent_vecs = vectors[: len(sentences)]
    anchor_vecs = vectors[len(sentences) :]
    buckets: dict[str, list[str]] = {seg: [] for seg in SEGMENT_ORDER[role]}
    for sentence, sent_vec in zip(sentences, sent_vecs):
        rule_segment = _classify_sentence(sentence, role=role)
        forced_segment = _forced_segment(sentence, role=role)
        scores = []
        for segment, anchor_vec in zip(SEGMENT_ORDER[role], anchor_vecs):
            score = _cosine(sent_vec, anchor_vec)
            if segment == rule_segment:
                score += 0.08
            if forced_segment and segment == forced_segment:
                score += 0.18
            if _segment_forbidden(sentence, role=role, segment=segment):
                score -= 0.35
            scores.append((score, segment))
        _score, best = max(scores, key=lambda item: item[0])
        for clause in _split_clauses(sentence):
            segment = _forced_segment(clause, role=role) or _classify_sentence(clause, role=role)
            if (
                best in SEGMENT_ORDER[role]
                and _compatible_segment_override(clause, role=role, source=segment, target=best)
                and _clause_allowed(clause, role=role, segment=best)
            ):
                segment = best
            if segment in buckets and _clause_allowed(clause, role=role, segment=segment):
                buckets[segment].append(clause)
    return _sanitize_segments(buckets, role=role)


def _compatible_segment_override(clause: str, *, role: str, source: str, target: str) -> bool:
    if source == target:
        return True
    lower = clause.lower()
    if target == "appearance":
        return (_has_appearance_terms(lower) or _has_attribute_terms(lower)) and not (
            _has_action_terms(lower) or _has_attention_terms(lower) or _has_environment_terms(lower)
        )
    if target == "environment":
        return _has_environment_terms(lower) and not (_has_action_terms(lower) or _has_attention_terms(lower))
    if target == "location":
        if _has_action_terms(lower) or _has_appearance_terms(lower) or _has_attribute_terms(lower) or _has_environment_terms(lower):
            return False
        if role == "vehicle":
            return _has_location_terms(lower) or _has_vehicle_visibility_terms(lower)
        return _has_location_terms(lower)
    if target == "attention":
        return role == "pedestrian" and (_has_attention_terms(lower) or _has_action_terms(lower)) and not (
            _has_appearance_terms(lower) or _has_environment_terms(lower)
        )
    if target == "action":
        return _has_action_terms(lower) and not (_has_appearance_terms(lower) or _has_attribute_terms(lower))
    return False


def _segment_anchor(role: str, segment: str) -> str:
    anchors = {
        "appearance": "pedestrian age gender height clothing shirt pants wearing male female appearance",
        "location": "relative position distance front behind left right diagonal close far orientation road lane vehicle pedestrian",
        "attention": "line of sight looking watching gaze aware unaware notice visibility field of view attention",
        "action": "speed moving walking standing stopped braking going straight turning crossing planned action behavior avoidance",
        "environment": "weather brightness road surface asphalt traffic volume sidewalk roadside street lights urban residential lanes context",
    }
    return f"{role} {segment}: {anchors[segment]}"


def _embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = _get_minilm_model()
        arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in arr]
    except Exception:
        return _tfidf_vectors(texts)


def _get_minilm_model() -> Any:
    global _MINILM_MODEL
    try:
        return _MINILM_MODEL
    except NameError:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _MINILM_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _MINILM_MODEL


def _tfidf_vectors(texts: list[str]) -> list[list[float]]:
    tokenized = [_tokens(text) for text in texts]
    df = Counter(token for toks in tokenized for token in set(toks))
    n = max(1, len(texts))
    vocab = sorted(df)
    index = {token: idx for idx, token in enumerate(vocab)}
    vectors: list[list[float]] = []
    for toks in tokenized:
        counts = Counter(toks)
        vec = [0.0] * len(vocab)
        for token, count in counts.items():
            vec[index[token]] = float(count) * math.log((n + 1) / (df[token] + 1)) + 1.0
        vectors.append(vec)
    return vectors


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _classify_sentence(sentence: str, *, role: str) -> str:
    lower = sentence.lower()
    if _has_appearance_terms(lower):
        return "appearance"
    if _has_attribute_terms(lower):
        return "appearance"
    if role == "pedestrian" and _has_appearance_terms(lower):
        return "appearance"
    if any(t in lower for t in ("line of sight", "watch", "looking", "gaze", "aware", "unaware", "notice", "visual", "field of view", "visible")):
        return "attention"
    if any(t in lower for t in ("speed", "going", "travel", "moving", "moved", "walking", "walked", "cross", "standing", "stood", "stopped", "brak", "turn", "collision", "collid", "action", "behavior", "planned", "intended", "resulting")):
        return "action"
    if any(t in lower for t in ("weather", "brightness", "road", "surface", "asphalt", "traffic", "sidewalk", "roadside", "street light", "lane", "residential", "urban", "environment")):
        return "environment"
    if any(t in lower for t in ("position", "distance", "near", "close", "far", "front", "behind", "left", "right", "diagonal", "perpendicular", "opposite", "same direction", "vehicle", "pedestrian")):
        return "location"
    return "action" if role == "vehicle" else "location"


def _forced_segment(sentence: str, *, role: str) -> str | None:
    lower = sentence.lower()
    if _has_appearance_terms(lower) or _has_attribute_terms(lower):
        return "appearance"
    if role == "pedestrian" and _has_action_terms(lower):
        return "action"
    if role == "vehicle" and _has_vehicle_visibility_terms(lower):
        return "attention"
    return None


def _segment_forbidden(sentence: str, *, role: str, segment: str) -> bool:
    lower = sentence.lower()
    if segment == "action" and (_has_appearance_terms(lower) or _has_attribute_terms(lower)):
        motion_terms = ("speed", "going", "travel", "moving", "walking", "cross", "standing", "stopped", "brak", "turn", "collision", "collid")
        return not any(term in lower for term in motion_terms)
    if role == "vehicle" and segment in {"location", "action"} and _has_appearance_terms(lower):
        return True
    return False


def _clause_allowed(clause: str, *, role: str, segment: str) -> bool:
    lower = clause.lower().strip()
    if not lower:
        return False
    if _is_vague_context(lower):
        return False
    if segment == "appearance":
        if _has_obstacle_dimension_terms(lower):
            return False
        if _has_attention_terms(lower) or _has_action_terms(lower) or _has_environment_terms(lower):
            return False
        return _has_appearance_terms(lower) or _has_attribute_terms(lower)
    if segment == "location":
        if _has_environment_terms(lower) or _has_appearance_terms(lower) or _has_attribute_terms(lower):
            return False
        if _has_action_terms(lower):
            return False
        if role == "vehicle" and _has_vehicle_awareness_terms(lower):
            return False
        return _has_location_terms(lower)
    if segment == "attention":
        if _has_environment_terms(lower) or _has_appearance_terms(lower) or _has_action_terms(lower):
            return False
        return _has_attention_terms(lower) or (role == "vehicle" and _has_vehicle_visibility_terms(lower))
    if segment == "action":
        if _has_appearance_terms(lower) or _has_attribute_terms(lower):
            return False
        return _has_action_terms(lower)
    if segment == "environment":
        if _has_attention_terms(lower):
            return False
        if _has_action_terms(lower):
            return False
        if (_has_appearance_terms(lower) or _has_attribute_terms(lower)) and not _has_environment_terms(lower):
            return False
        return _has_environment_terms(lower)
    return True


def _sanitize_segments(buckets: dict[str, list[str]], *, role: str) -> dict[str, str]:
    clean: dict[str, list[str]] = {seg: [] for seg in SEGMENT_ORDER[role]}
    for segment, clauses in buckets.items():
        for clause in clauses:
            target = _forced_segment(clause, role=role) or segment
            clause = _sanitize_clause_for_segment(clause, role=role, segment=target)
            if not clause:
                continue
            if target in clean and _clause_allowed(clause, role=role, segment=target):
                clean[target].append(clause)
    out: dict[str, str] = {}
    for segment, parts in clean.items():
        filtered = [
            sentence
            for sentence in _dedupe(parts)
            if _final_sentence_allowed(sentence, role=role, segment=segment)
        ]
        if filtered:
            out[segment] = " ".join(filtered)
    return out


def _final_sentence_allowed(sentence: str, *, role: str, segment: str) -> bool:
    lower = sentence.lower().strip()
    if not lower:
        return False
    if lower.startswith(("which ", "while ", "and ", "but ", "although ", "was ")):
        return False
    if lower.startswith(("in front of him", "in front of her")):
        return False
    if "victim" in lower or "obstacle" in lower:
        return False
    if segment == "appearance":
        if re.search(r"\b(positioned|diagonal|front|behind|left|right|distance|field of view|visible|environment conditions|surrounding the event|were as follows)\b", lower):
            return False
        return _has_appearance_terms(lower) or _has_attribute_terms(lower)
    if segment == "environment":
        if re.search(r"\b(comfortable|hustle|safety|seemed|event being narrated)\b", lower):
            return False
        return _has_environment_terms(lower) and not (_has_attention_terms(lower) or _has_action_terms(lower))
    if segment == "location":
        if re.search(r"\b(notice|noticed|aware|unaware|moving|moved|walking|speed|cross|turn|road|outlook|bright day|risk|danger|stationary)\b", lower):
            return False
        return _has_location_terms(lower)
    if segment == "attention":
        if _has_location_terms(lower) and not _has_attention_terms(lower):
            return False
        if _has_environment_terms(lower) or _has_appearance_terms(lower):
            return False
        return _has_attention_terms(lower) or (role == "vehicle" and _has_vehicle_visibility_terms(lower))
    if segment == "action":
        if _has_environment_terms(lower) or _has_appearance_terms(lower) or _has_attribute_terms(lower):
            return False
        return _has_action_terms(lower)
    return True


def _sanitize_clause_for_segment(clause: str, *, role: str, segment: str) -> str:
    text = str(clause).strip()
    lower = text.lower()
    if not text:
        return ""
    if segment == "appearance":
        if _has_obstacle_dimension_terms(lower):
            return ""
        text = re.sub(
            r"(?i)^(the environment conditions?|the environment surrounding the event|the individual|the driver sees|this event occurred involved)\s+(indicate|indicates|include|included|present|presents|consisted of|involved|sees)?\s*(that\s+)?",
            "",
            text,
        ).strip(" ,.")
        if re.search(r"(?i)\b(positioned|diagonally|directly|in front|behind|left|right|distance|field of view|visible)\b", text):
            # Keep only the appearance tail when a sentence mixes position + appearance.
            parts = re.split(r"(?i)\b(?:the pedestrian is wearing|he is wearing|she is wearing|wearing)\b", text, maxsplit=1)
            attr = _extract_attribute_phrase(text)
            if len(parts) > 1:
                wear = "The pedestrian is wearing " + parts[1].strip(" .")
                text = f"{attr} {wear}".strip() if attr else wear
            elif attr:
                text = attr
            else:
                return ""
        if re.search(r"(?i)\b(width|meters?\s+in\s+width|obstacle|victim)\b", text):
            text = re.sub(r"(?i)\b(?:and\s+)?(?:wearing\s+glasses\s+and\s+)?there is an obstacle.*$", "", text).strip(" ,.")
            text = re.sub(r"(?i)\b(?:an obstacle|with an obstacle|obstacle measuring).*$", "", text).strip(" ,.")
        if not (_has_appearance_terms(text.lower()) or _has_attribute_terms(text.lower())):
            return ""
    elif segment == "environment":
        if _has_location_terms(lower) and not _has_environment_terms(lower):
            return ""
        text = re.sub(r"(?i)^stands?\s+.*?\bon\s+", "The event occurred on ", text).strip()
        text = re.sub(r"(?i)^in\s+the\s+bustling\s+urban\s+setting\.?\s*", "The surroundings were urban. ", text).strip()
        text = re.sub(r"(?i)\bensuring .*?$", "", text).strip(" ,.")
        text = re.sub(r"(?i)\bamidst .*?$", "", text).strip(" ,.")
        text = re.sub(r"(?i)\bdaily urban hustle\.?$", "", text).strip(" ,.")
        if not _has_environment_terms(text.lower()):
            return ""
    elif segment == "location":
        text = re.sub(r"(?i)^as the pedestrian noticed .*?\.?\s*", "", text).strip()
        text = re.sub(r"(?i)^he noticed .*?\.?\s*", "", text).strip()
        text = re.sub(r"(?i)^she noticed .*?\.?\s*", "", text).strip()
        text = re.sub(r"(?i)^bright weekday\.?\s*", "", text).strip()
        if _has_action_terms(text.lower()) or _has_attention_terms(text.lower()):
            # Visibility is allowed only for vehicle location.
            if not (role == "vehicle" and _has_vehicle_visibility_terms(text.lower()) and not _has_vehicle_awareness_terms(text.lower())):
                return ""
        if not (_has_location_terms(text.lower()) or (role == "vehicle" and _has_vehicle_visibility_terms(text.lower()))):
            return ""
    elif segment == "attention":
        text = re.sub(r"(?i)^it is a .*?road.*?\.?\s*", "", text).strip()
        text = re.sub(r"(?i)^the vehicle was traveling in a car lane\.?\s*", "", text).strip()
        if _has_environment_terms(text.lower()) or _has_appearance_terms(text.lower()):
            return ""
        if not (_has_attention_terms(text.lower()) or _has_action_terms(text.lower())):
            return ""
    elif segment == "action":
        text = re.sub(r"(?i)\baction\.?\s*speed\.?$", "", text).strip(" ,.")
        if re.search(r"(?i)\broad\b", text) and not re.search(r"(?i)\b(speed|km/h|moving|stopped|brak|turn|travel|going|start)\b", text):
            return ""
        text = re.sub(r"(?i)\btraveling on a .*?road.*?$", "", text).strip(" ,.")
        text = re.sub(r"(?i)\bnavigating the road.*?$", "", text).strip(" ,.")
        if _has_appearance_terms(text.lower()) or _has_attribute_terms(text.lower()) or _has_environment_terms(text.lower()):
            return ""
        if not _has_action_terms(text.lower()):
            return ""
    return text


def _extract_attribute_phrase(text: str) -> str:
    bits: list[str] = []
    gender_age = re.search(r"(?i)\b(?:a\s+)?(?:male|female)(?:\s+pedestrian)?(?:\s+in\s+(?:his|her|their)\s+\d+0s)?", text)
    if gender_age:
        phrase = gender_age.group(0).strip()
        if phrase.lower().startswith("a "):
            bits.append("The pedestrian is " + phrase + ".")
        else:
            bits.append("The pedestrian is a " + phrase + ".")
    height = re.search(r"(?i)\b(?:approximately\s+)?(?:1[4-9]0|2[0-2]0)\s*cm\s+tall\b|\bheight\s+of\s+(?:1[4-9]0|2[0-2]0)\s*cm\b", text)
    if height:
        bits.append("The pedestrian is " + height.group(0).strip().replace("height of ", "approximately ") + ".")
    return " ".join(bits)


def _has_appearance_terms(lower: str) -> bool:
    return re.search(
        r"\b(wear|wearing|shirt|t-shirt|pants|slacks|jacket|hat|cap|clothing|upper body|lower body)\b",
        lower,
    ) is not None


def _has_attribute_terms(lower: str) -> bool:
    return re.search(r"\b(male|female|gender|age|20s|30s|40s|50s|60s|70s|height|cm)\b", lower) is not None


def _has_obstacle_dimension_terms(lower: str) -> bool:
    return (
        "obstacle" in lower
        or "victim" in lower
        or re.search(r"\b[1-9]\s*meters?\s+in\s+(?:height|width)\b", lower) is not None
        or re.search(r"\bmeasuring\s+[1-9]\s*meters?\b", lower) is not None
    )


def _has_environment_terms(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "weather",
            "brightness",
            "lighting",
            "road surface",
            "surface",
            "asphalt",
            "traffic volume",
            "traffic",
            "sidewalk",
            "roadside",
            "street light",
            "street lights",
            "residential",
            "urban",
            "two-way",
            "one-way",
            "level",
            "dry",
            "wet",
            "cloudy",
            "clear",
            "rain",
        )
    )


def _has_location_terms(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "position",
            "positioned",
            "located",
            "distance",
            "near",
            "close",
            "far",
            "front",
            "behind",
            "left",
            "right",
            "diagonal",
            "perpendicular",
            "opposite direction",
            "same direction",
            "body",
            "orientation",
        )
    )


def _has_attention_terms(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "line of sight",
            "watch",
            "watching",
            "looking",
            "gaze",
            "aware",
            "unaware",
            "notice",
            "noticed",
            "visual",
            "field of view",
            "visible",
        )
    )


def _has_vehicle_visibility_terms(lower: str) -> bool:
    return any(term in lower for term in ("field of view", "visible", "not visible", "visibility"))


def _has_vehicle_awareness_terms(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "driver is aware",
            "driver was aware",
            "driver is unaware",
            "driver was unaware",
            "vehicle is aware",
            "vehicle was aware",
            "aware of",
            "unaware of",
            "taking necessary action",
            "taking appropriate action",
            "proceeding cautiously",
            "maintaining a safe distance",
            "maintaining its course",
        )
    )


def _has_action_terms(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "speed",
            "going",
            "travel",
            "moving",
            "moved",
            "walking",
            "cross",
            "standing",
            "stood",
            "stopped",
            "brak",
            "turn",
            "collision",
            "collid",
            "action",
            "behavior",
            "planned",
            "intended",
            "about to",
            "continue",
        )
    )


def _is_vague_context(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "overall situation",
            "seems calm",
            "ordinary",
            "momentarily",
            "respective positions",
            "despite this",
            "meanwhile",
            "usual day",
            "despite",
        )
    )


def _segment_boundary_text(*, role: str, segment: str) -> str:
    if role == "vehicle" and segment == "appearance":
        return "Slot boundary: include only the pedestrian's gender, age, height, and clothing as described from the vehicle caption. Exclude vehicle movement, position, visibility, weather, and road."
    if role == "pedestrian" and segment == "attention":
        return "Slot boundary: include only line of sight, looking, awareness, noticing, and visibility. Exclude clothing, road/weather, spatial-only position, and movement/intent."
    rules = {
        "appearance": "Slot boundary: include only pedestrian gender, age, height, and clothing. Exclude road, weather, position, gaze, and motion.",
        "location": "Slot boundary: include only spatial position, distance, side, front/behind, and body orientation. Exclude crossing, moving, speed, clothing, weather, and road surface.",
        "attention": "Slot boundary: include only line of sight, looking, awareness, noticing, visibility, and field of view. Exclude position, motion, clothing, and road context.",
        "action": "Slot boundary: include only motion, speed, stopping, braking, turning, crossing, collision, or intended movement. Exclude clothing, age, height, weather, road, and spatial-only position.",
        "environment": "Slot boundary: include only weather, brightness, road surface, road type, traffic volume/direction, sidewalk, roadside, and street lights. Exclude motion, intent, gaze, and vague situation summaries.",
    }
    return rules.get(segment, f"Slot boundary: include only facts for {role} {segment}.")


def _compose_caption(role: str, segments: dict[str, str]) -> str:
    parts = [segments.get(segment, "").strip() for segment in SEGMENT_ORDER[role]]
    return _clean_caption_text(" ".join(part for part in parts if part))


def _parse_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value if value is not None else {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _split_rewrite_prediction(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    cleaned = str(text).strip()
    ped_match = re.search(r"caption[_\s-]*pedestrian\s*[:=]\s*(.*?)(?:caption[_\s-]*vehicle\s*[:=]|$)", cleaned, flags=re.I | re.S)
    veh_match = re.search(r"caption[_\s-]*vehicle\s*[:=]\s*(.*)$", cleaned, flags=re.I | re.S)
    if ped_match:
        result["caption_pedestrian"] = ped_match.group(1).strip(" \n\r\t\"',{}")
    if veh_match:
        result["caption_vehicle"] = veh_match.group(1).strip(" \n\r\t\"',{}")
    return result


def _read_prediction_texts(path: str | Path) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(str(obj.get("predict", obj.get("prediction", obj.get("output", "")))))
            else:
                rows.append(str(obj))
    return rows


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?]?", text.strip()) if s.strip()]


def _split_clauses(text: str) -> list[str]:
    pieces: list[str] = []
    normalized = re.sub(r"(?i)\balthough\b", ".", text)
    normalized = re.sub(r"(?i)\bwhile\b", ".", normalized)
    normalized = re.sub(r"(?i)\bwhich was\b", ". The vehicle was", normalized)
    normalized = re.sub(r"(?i)\bwhich has\b", ". The road has", normalized)
    normalized = re.sub(r"(?i)\bwho is\b", ". The pedestrian is", normalized)
    normalized = re.sub(r"(?i)\bdue to\b", ".", normalized)
    normalized = re.sub(r"(?i)\bas they are\b", ". They are", normalized)
    normalized = re.sub(r"(?i)\bindicating\b", ". This indicates", normalized)
    for part in re.split(r"(?i)\b(?:despite this|meanwhile|notably|moving on to|in addition|as for)\b|;|,", normalized):
        part = part.strip(" \t\r\n")
        if not part:
            continue
        pieces.append(part if part[-1] in ".!?" else part + ".")
    return pieces


def _dedupe(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if key and key not in seen:
            cleaned = _clean_segment_text(sentence)
            if cleaned:
                out.append(cleaned)
            seen.add(key)
    return out


def _clean_segment_text(text: str) -> str:
    text = re.sub(r"^\s*[-*]?\s*(?:appearance|location|environment|attention|action|behavior|segment)\s*:\s*", "", str(text).strip(), flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = _drop_cross_slot_tail(text)
    text = _repair_fragment(text)
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


def _repair_fragment(text: str) -> str:
    text = re.sub(r"^(and|but)\s+", "", text, flags=re.I)
    text = re.sub(r"^and\.\s*", "", text, flags=re.I)
    text = re.sub(r"^who\s+was\s+wearing\b", "The pedestrian was wearing", text, flags=re.I)
    text = re.sub(r"^who\s+is\s+wearing\b", "The pedestrian is wearing", text, flags=re.I)
    text = re.sub(r"^is\s+wearing\b", "The pedestrian is wearing", text, flags=re.I)
    text = re.sub(r"^is\s+a\s+(male|female)\b", r"The pedestrian is a \1", text, flags=re.I)
    text = re.sub(r"^has\s+a\s+height\b", "The pedestrian has a height", text, flags=re.I)
    text = re.sub(r"^along\s+with\b", "The pedestrian is wearing", text, flags=re.I)
    text = re.sub(r"^\.\s*", "", text)
    text = re.sub(r"^looking\b", "The pedestrian is looking", text, flags=re.I)
    text = re.sub(r"^obliviously\s+unaware\b", "The pedestrian was unaware", text, flags=re.I)
    text = re.sub(r"^showing\b", "This shows", text, flags=re.I)
    text = re.sub(r"^maintaining\b", "The pedestrian is maintaining", text, flags=re.I)
    text = re.sub(r"^which is parked on\b", "The vehicle is parked on", text, flags=re.I)
    text = re.sub(r"^which is\b", "This is", text, flags=re.I)
    text = re.sub(r"^which was\b", "The vehicle was", text, flags=re.I)
    text = re.sub(r"^which has\b", "The road has", text, flags=re.I)
    text = re.sub(r"^was\s+(diagonally|directly|perpendicularly|positioned|located)\b", r"The pedestrian was \1", text, flags=re.I)
    text = re.sub(r"^was\s+observed\b", "The pedestrian was observed", text, flags=re.I)
    text = re.sub(r"^was\s+(standing|moving|walking|stopped|traveling|travelling)\b", r"The pedestrian was \1", text, flags=re.I)
    text = re.sub(r"^was\s+(aware|unaware)\b", r"The pedestrian was \1", text, flags=re.I)
    text = re.sub(r"^positioned\b", "The pedestrian was positioned", text, flags=re.I)
    text = re.sub(r"^situated\b", "The pedestrian was situated", text, flags=re.I)
    text = re.sub(r"^stands?\s+directly\b", "The pedestrian stood directly", text, flags=re.I)
    text = re.sub(r"^stands?\s+diagonally\b", "The pedestrian stood diagonally", text, flags=re.I)
    text = re.sub(r"^facing\b", "The pedestrian was facing", text, flags=re.I)
    text = re.sub(r"^traveling\b", "The vehicle was traveling", text, flags=re.I)
    text = re.sub(r"^travelling\b", "The vehicle was travelling", text, flags=re.I)
    text = re.sub(r"^with\s+a\s+speed\s+of\b", "The vehicle speed was", text, flags=re.I)
    text = re.sub(r"^with\s+a\s+line\s+of\s+sight\b", "The pedestrian's line of sight", text, flags=re.I)
    text = re.sub(r"^with\s+clear weather\b", "The weather was clear", text, flags=re.I)
    text = re.sub(r"^with\s+cloudy weather\b", "The weather was cloudy", text, flags=re.I)
    text = re.sub(r"^with\s+bright brightness\b", "The brightness was bright", text, flags=re.I)
    text = re.sub(r"^with\s+bright visibility\b", "The visibility was bright", text, flags=re.I)
    text = re.sub(r"^with\s+dry road surfaces\b", "The road surfaces were dry", text, flags=re.I)
    text = re.sub(r"^with\s+a\s+level\s+incline\b", "The road had a level incline", text, flags=re.I)
    text = re.sub(r"^with\s+dark brightness\b", "The brightness was dark", text, flags=re.I)
    text = re.sub(r"^with\s+dim brightness\b", "The brightness was dim", text, flags=re.I)
    text = re.sub(r"^with\s+dry road surface conditions\b", "The road surface conditions were dry", text, flags=re.I)
    text = re.sub(r"^with\s+a\s+dry asphalt road surface\b", "The road surface was dry asphalt", text, flags=re.I)
    text = re.sub(r"^with\s+(a\s+)?(light|usual|heavy)\s+traffic volume\b", r"The traffic volume was \2", text, flags=re.I)
    text = re.sub(r"^with\s+the pedestrian positioned\b", "The pedestrian was positioned", text, flags=re.I)
    text = re.sub(r"^with\s+his\s+line of sight\b", "The pedestrian's line of sight", text, flags=re.I)
    text = re.sub(r"^with\s+her\s+line of sight\b", "The pedestrian's line of sight", text, flags=re.I)
    text = re.sub(r"^with\s+a\s+height\s+of\b", "The pedestrian has a height of", text, flags=re.I)
    text = re.sub(r"^with\s+(?:an?\s+)?(?:black|white|blue|brown|gray|grey|beige|silver|orange|purplish red)\b", "The pedestrian is wearing", text, flags=re.I)
    text = re.sub(r"^were\s+as\s+follows:\s*", "", text, flags=re.I)
    text = re.sub(r"^surrounding\s+the\s+event\s+were\s+as\s+follows:\s*", "", text, flags=re.I)
    text = re.sub(r"^surrounding\s+this\s+event\s+include\b", "The event included", text, flags=re.I)
    text = re.sub(r"^reveals\s+that\b", "The scene reveals that", text, flags=re.I)
    text = re.sub(r"^describe\s+a\b", "The pedestrian is a", text, flags=re.I)
    text = re.sub(r"^although\s+the road surface\b", "The road surface", text, flags=re.I)
    text = re.sub(r"^although\s+the weather\b", "The weather", text, flags=re.I)
    text = re.sub(r"^although\s+he was\b", "The pedestrian was", text, flags=re.I)
    text = re.sub(r"^although\s+she was\b", "The pedestrian was", text, flags=re.I)
    text = re.sub(r"^in the direction of travel\.?$", "The pedestrian's line of sight is in the direction of travel.", text, flags=re.I)
    text = re.sub(r"^with (his|her|their) body\b", r"The pedestrian's body", text, flags=re.I)
    text = re.sub(r"^(The pedestrian's body) facing\b", r"\1 was facing", text, flags=re.I)
    text = re.sub(r"^with a bright level of brightness\b", "The brightness is bright", text, flags=re.I)
    text = re.sub(r"^(approximately \d+\s*cm tall)\.?$", r"The pedestrian is \1.", text, flags=re.I)
    text = re.sub(r"^(in his|in her) (\d+0s)\.?$", r"The pedestrian is \1 \2.", text, flags=re.I)
    text = re.sub(r"^(wearing\b)", "The pedestrian is wearing", text, flags=re.I)
    text = re.sub(r"^(dressed in\b)", "The pedestrian is dressed in", text, flags=re.I)
    text = re.sub(r"^(moving\b)", "The pedestrian is moving", text, flags=re.I)
    text = re.sub(r"^(resulting in\b)", "This resulted in", text, flags=re.I)
    text = re.sub(r"^(providing\b)", "This provided", text, flags=re.I)
    text = re.sub(r"^which covered\b", "This covered", text, flags=re.I)
    text = re.sub(r"^(with a dry asphalt surface\b)", "The road has a dry asphalt surface", text, flags=re.I)
    text = re.sub(r"^(he|she) stood on\b", "The pedestrian stood on", text, flags=re.I)
    text = re.sub(r"\b([0-9]+)km/h\b", r"\1 km/h", text, flags=re.I)
    return text


def _drop_cross_slot_tail(text: str) -> str:
    text = re.sub(
        r"\b(?:these details paint a picture|this information presents|the pedestrian's actions and the environment conditions suggest|the overall situation seems).*$",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"\b(?:an obstacle measuring|there is an obstacle|with an obstacle|and there is an obstacle).*$",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(r"\b\d+\s*meters?\s+in\s+width.*$", "", text, flags=re.I).strip()
    text = re.sub(r"\bbright day\.?$", "", text, flags=re.I).strip()
    text = re.sub(r"\borientation indicated.*$", "", text, flags=re.I).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip(" ,;")
    return text


def _clean_caption_text(text: str) -> str:
    sentences = _dedupe(_split_sentences(text))
    return " ".join(sentence for sentence in sentences if sentence)


def _fallback_row(submission: dict[str, list[dict[str, Any]]], scenario_id: str, idx: int) -> dict[str, Any]:
    rows = submission.get(scenario_id, [])
    if isinstance(rows, list) and 0 <= idx < len(rows) and isinstance(rows[idx], dict):
        return rows[idx]
    return {}


def _phase_index_for_submission(
    submission: dict[str, list[dict[str, Any]]],
    scenario_id: str,
    phase: str,
) -> int:
    rows = submission.get(scenario_id, [])
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                labels = row.get("labels")
                if isinstance(labels, list) and labels and str(labels[0]) == str(phase):
                    return idx
    if str(phase) in PHASE_ORDER:
        return PHASE_ORDER.index(str(phase))
    return len(rows) if isinstance(rows, list) else 0


def _phase_sort_key(phase: str) -> tuple[int, str]:
    if str(phase) in PHASE_ORDER:
        return (PHASE_ORDER.index(str(phase)), str(phase))
    return (len(PHASE_ORDER), str(phase))


def _normalize_row(row: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(fallback)
    out["labels"] = row.get("labels", fallback.get("labels", []))
    for field in CAPTION_KEYS.values():
        text = str(row.get(field, fallback.get(field, ""))).strip()
        if text:
            out[field] = text
    return out


def _maybe_write_dataset_info(
    output: str | Path,
    dataset_info_output: str | Path | None,
    dataset_name: str | None,
) -> None:
    if dataset_info_output and dataset_name:
        # Include both media columns. LLaMA-Factory ignores absent columns in
        # rows, and this lets the same exporter support video and frame data.
        write_json(
            dataset_info_output,
            {
                dataset_name: {
                    "file_name": Path(output).name,
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                        "videos": "videos",
                        "images": "images",
                    },
                }
            },
        )
    elif dataset_info_output or dataset_name:
        raise ValueError("--dataset-info-output and --dataset-name must be provided together.")


def _maybe_write_text_dataset_info(
    output: str | Path,
    dataset_info_output: str | Path | None,
    dataset_name: str | None,
) -> None:
    if dataset_info_output and dataset_name:
        write_json(
            dataset_info_output,
            {
                dataset_name: {
                    "file_name": Path(output).name,
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                    },
                }
            },
        )
    elif dataset_info_output or dataset_name:
        raise ValueError("--dataset-info-output and --dataset-name must be provided together.")
