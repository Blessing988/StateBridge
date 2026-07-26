"""Parsers for captions, VQA, and bounding-box annotations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import read_json
from .schema import normalize_phase_label


def load_caption_phases(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    phases = data.get("event_phase", [])
    parsed: list[dict[str, Any]] = []
    for phase in phases:
        labels = phase.get("labels") or []
        label = normalize_phase_label(labels[0] if labels else None)
        parsed.append(
            {
                "label": label,
                "labels": [label] if label else labels,
                "caption_pedestrian": phase.get("caption_pedestrian", "").strip(),
                "caption_vehicle": phase.get("caption_vehicle", "").strip(),
                "start_time": phase.get("start_time"),
                "end_time": phase.get("end_time"),
            }
        )
    return parsed


def _stable_vqa_id(
    *,
    scenario_id: str,
    scope: str,
    phase: str | None,
    question: str,
    options: dict[str, str],
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "scope": scope,
        "phase": phase,
        "question": question,
        "options": options,
    }
    raw = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_vqa_questions(path: str | Path, *, scope: str, scenario_id: str) -> list[dict[str, Any]]:
    data = read_json(path)
    items = data if isinstance(data, list) else [data]
    questions: list[dict[str, Any]] = []
    for item in items:
        if scope == "environment":
            for idx, question in enumerate(item.get("environment", [])):
                questions.append(_parse_question(question, scenario_id, scope, None, idx))
            continue

        for phase in item.get("event_phase", []):
            labels = phase.get("labels") or []
            phase_label = normalize_phase_label(labels[0] if labels else None)
            for idx, question in enumerate(phase.get("conversations", [])):
                questions.append(_parse_question(question, scenario_id, scope, phase_label, idx))
    return questions


def _parse_question(
    raw: dict[str, Any],
    scenario_id: str,
    scope: str,
    phase: str | None,
    idx: int,
) -> dict[str, Any]:
    options = {
        letter: str(raw[letter]).strip()
        for letter in ("a", "b", "c", "d", "e")
        if letter in raw and str(raw[letter]).strip()
    }
    qid = str(raw.get("id") or "")
    if not qid:
        qid = _stable_vqa_id(
            scenario_id=scenario_id,
            scope=scope,
            phase=phase,
            question=str(raw.get("question", "")).strip(),
            options=options,
        )
    return {
        "id": qid,
        "scenario_id": scenario_id,
        "scope": scope,
        "phase": phase,
        "index": idx,
        "question": str(raw.get("question", "")).strip(),
        "options": options,
        "correct": str(raw.get("correct", "")).strip().lower(),
    }


def load_bbox_summary(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    annotations = data.get("annotations", [])
    by_phase: dict[str, dict[str, Any]] = {}
    for ann in annotations:
        phase = normalize_phase_label(ann.get("phase_number"))
        phase_row = by_phase.setdefault(
            phase,
            {
                "count": 0,
                "first_image_id": None,
                "last_image_id": None,
                "mean_bbox": [0.0, 0.0, 0.0, 0.0],
            },
        )
        bbox = ann.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        phase_row["count"] += 1
        image_id = ann.get("image_id")
        if phase_row["first_image_id"] is None:
            phase_row["first_image_id"] = image_id
        phase_row["last_image_id"] = image_id
        for idx, value in enumerate(bbox[:4]):
            phase_row["mean_bbox"][idx] += float(value)

    for phase_row in by_phase.values():
        count = max(phase_row["count"], 1)
        phase_row["mean_bbox"] = [round(v / count, 3) for v in phase_row["mean_bbox"]]
    return {"path": str(path), "phases": by_phase, "total_annotations": len(annotations)}

