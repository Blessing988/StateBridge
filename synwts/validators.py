"""Submission validators for AI City Track 2 style outputs."""

from __future__ import annotations

from pathlib import Path

from .io import read_json


def validate_caption_submission(path: str | Path) -> dict:
    data = read_json(path)
    errors: list[str] = []
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["Caption submission must be a JSON object."]}
    for scenario_id, rows in data.items():
        if not isinstance(rows, list):
            errors.append(f"{scenario_id}: value must be a list of phase rows.")
            continue
        seen = set()
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{scenario_id}[{idx}]: row must be an object.")
                continue
            labels = row.get("labels")
            if not isinstance(labels, list) or not labels:
                errors.append(f"{scenario_id}[{idx}]: labels must be a non-empty list.")
                continue
            label = str(labels[0])
            if label in seen:
                errors.append(f"{scenario_id}: duplicate label {label}.")
            seen.add(label)
            for key in ("caption_pedestrian", "caption_vehicle"):
                if key not in row:
                    errors.append(f"{scenario_id}[{idx}]: missing {key}.")
                elif not isinstance(row[key], str):
                    errors.append(f"{scenario_id}[{idx}]: {key} must be a string.")
    return {"ok": not errors, "errors": errors}


def validate_vqa_submission(path: str | Path) -> dict:
    data = read_json(path)
    errors: list[str] = []
    if not isinstance(data, list):
        return {"ok": False, "errors": ["VQA submission must be a JSON list."]}
    seen = set()
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            errors.append(f"row {idx}: must be an object.")
            continue
        qid = str(row.get("id", "")).strip()
        answer = str(row.get("correct", "")).strip().lower()
        if not qid:
            errors.append(f"row {idx}: missing id.")
        if qid in seen:
            errors.append(f"row {idx}: duplicate id {qid}.")
        seen.add(qid)
        if answer not in {"a", "b", "c", "d", "e"}:
            errors.append(f"row {idx}: correct must be one of a/b/c/d/e.")
    return {"ok": not errors, "errors": errors}
