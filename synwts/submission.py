"""Assemble AI City Track 2 submission files from model predictions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io import read_json, write_json


PHASE_ORDER = ["4", "3", "2", "1", "0"]
VIEW_PRIORITY = {"overhead_view": 0, "vehicle_view": 1, "environment": 2}


def assemble_caption_submission(
    *,
    inference_dataset: str | Path,
    predictions: str | Path,
    output: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    rows = read_json(inference_dataset)
    prediction_texts = _load_prediction_texts(predictions)
    if len(prediction_texts) != len(rows):
        raise ValueError(
            f"Prediction count ({len(prediction_texts)}) does not match dataset rows ({len(rows)})."
        )

    candidates: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for idx, (row, pred_text) in enumerate(zip(rows, prediction_texts)):
        metadata = row.get("metadata", {})
        if metadata.get("task") != "caption":
            continue
        scenario_id = str(metadata["scenario_id"])
        phase = str(metadata["phase"])
        parsed = _parse_caption_prediction(pred_text, phase)
        _attach_caption_timestamps(parsed, metadata)
        view = str(metadata.get("view", ""))
        priority = VIEW_PRIORITY.get(view, 99)
        candidates.setdefault((scenario_id, phase), []).append((priority, parsed))

    submission: dict[str, list[dict[str, Any]]] = {}
    scenario_ids = sorted({scenario_id for scenario_id, _ in candidates})
    for scenario_id in scenario_ids:
        rows_out: list[dict[str, Any]] = []
        phases = [phase for phase in PHASE_ORDER if (scenario_id, phase) in candidates]
        phases.extend(
            sorted(
                phase
                for sid, phase in candidates
                if sid == scenario_id and phase not in PHASE_ORDER
            )
        )
        for phase in phases:
            selected = sorted(candidates[(scenario_id, phase)], key=lambda item: item[0])[0][1]
            rows_out.append(selected)
        submission[scenario_id] = rows_out

    write_json(output, submission)
    return submission


def _attach_caption_timestamps(row: dict[str, Any], metadata: dict[str, Any]) -> None:
    start_time = str(metadata.get("start_time", "")).strip()
    end_time = str(metadata.get("end_time", "")).strip()
    if start_time:
        row["start_time"] = start_time
    if end_time:
        row["end_time"] = end_time


def assemble_vqa_submission(
    *,
    inference_dataset: str | Path,
    predictions: str | Path,
    output: str | Path,
) -> list[dict[str, str]]:
    rows = read_json(inference_dataset)
    prediction_texts = _load_prediction_texts(predictions)
    if len(prediction_texts) != len(rows):
        raise ValueError(
            f"Prediction count ({len(prediction_texts)}) does not match dataset rows ({len(rows)})."
        )

    submission: list[dict[str, str]] = []
    for row, pred_text in zip(rows, prediction_texts):
        metadata = row.get("metadata", {})
        if metadata.get("task") != "vqa":
            continue
        qid = str(metadata.get("vqa_id", "")).strip()
        if not qid:
            continue
        submission.append({"id": qid, "correct": _parse_vqa_letter(pred_text)})

    write_json(output, submission)
    return submission


def ensemble_vqa_submissions(
    *,
    inputs: list[str | Path],
    output: str | Path,
    fallback: str | Path | None = None,
    weights: list[float] | None = None,
) -> list[dict[str, str]]:
    if not inputs:
        raise ValueError("At least one VQA submission is required.")
    if weights is not None and len(weights) != len(inputs):
        raise ValueError("weights must have the same length as inputs.")

    submissions = [_load_vqa_submission(path) for path in inputs]
    fallback_map = _load_vqa_submission(fallback) if fallback else submissions[0]
    ids = list(fallback_map.keys())
    missing_ids = [
        qid
        for qid in ids
        for submission in submissions
        if qid not in submission
    ]
    if missing_ids:
        raise ValueError(f"Input submissions do not share all IDs. First missing ID: {missing_ids[0]}")

    weights = weights or [1.0] * len(inputs)
    rows: list[dict[str, str]] = []
    for qid in ids:
        scores: dict[str, float] = {}
        for submission, weight in zip(submissions, weights):
            answer = _parse_vqa_letter(submission[qid])
            scores[answer] = scores.get(answer, 0.0) + float(weight)

        best_score = max(scores.values())
        winners = sorted(answer for answer, score in scores.items() if score == best_score)
        fallback_answer = _parse_vqa_letter(fallback_map.get(qid, "a"))
        answer = fallback_answer if fallback_answer in winners else winners[0]
        rows.append({"id": qid, "correct": answer})

    write_json(output, rows)
    return rows


def _load_prediction_texts(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        texts: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    texts.append(_prediction_text_from_obj(json.loads(line)))
        return texts

    data = read_json(path)
    if isinstance(data, list):
        return [_prediction_text_from_obj(item) for item in data]
    if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], list):
        return [_prediction_text_from_obj(item) for item in data["predictions"]]
    raise ValueError(f"Unsupported predictions format: {path}")


def _prediction_text_from_obj(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return str(obj)
    for key in ("predict", "prediction", "response", "output", "generated_text", "text"):
        if key in obj:
            return str(obj[key])
    if "messages" in obj:
        return str(obj["messages"][-1].get("content", "")) if obj["messages"] else ""
    return json.dumps(obj, ensure_ascii=False)


def _load_vqa_submission(path: str | Path) -> dict[str, str]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"VQA submission must be a list: {path}")
    rows: dict[str, str] = {}
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {idx} must be an object.")
        qid = str(row.get("id", "")).strip()
        if not qid:
            raise ValueError(f"{path}: row {idx} is missing id.")
        rows[qid] = _parse_vqa_letter(str(row.get("correct", "")))
    return rows


def _parse_caption_prediction(text: str, phase: str) -> dict[str, Any]:
    obj = _extract_json_object(text)
    if isinstance(obj, dict):
        ped = str(obj.get("caption_pedestrian", "")).strip()
        veh = str(obj.get("caption_vehicle", "")).strip()
        labels = obj.get("labels") if isinstance(obj.get("labels"), list) else [phase]
        if ped or veh:
            return {
                "labels": [str(labels[0]) if labels else phase],
                "caption_pedestrian": ped,
                "caption_vehicle": veh,
            }

    # Fallback: split plain text if the model did not obey JSON.
    ped, veh = _split_plain_caption(text)
    return {
        "labels": [phase],
        "caption_pedestrian": ped,
        "caption_vehicle": veh,
    }


def _extract_json_object(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _split_plain_caption(text: str) -> tuple[str, str]:
    clean = text.strip()
    lower = clean.lower()
    vehicle_idx = lower.find("caption_vehicle")
    pedestrian_idx = lower.find("caption_pedestrian")
    if pedestrian_idx >= 0 and vehicle_idx > pedestrian_idx:
        ped = clean[pedestrian_idx:vehicle_idx]
        veh = clean[vehicle_idx:]
        return _strip_label(ped), _strip_label(veh)
    vehicle_marker = lower.find("vehicle:")
    if vehicle_marker > 0:
        return clean[:vehicle_marker].strip(), _strip_label(clean[vehicle_marker:])
    return clean, clean


def _strip_label(text: str) -> str:
    clean = text.strip()
    clean = re.sub(
        r"""^\s*[\{\[,]*\s*["']?\s*(?:caption_)?(?:pedestrian|vehicle|caption_pedestrian|caption_vehicle)\s*["']?\s*:\s*["']?""",
        "",
        clean,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    clean = re.sub(r"""[\s"',}\]]+$""", "", clean).strip()
    return clean


def _parse_vqa_letter(text: str) -> str:
    cleaned = text.strip().lower()
    match = re.search(r"\b([abcde])\b", cleaned)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([abcde])[\).:\s-]", cleaned)
    if match:
        return match.group(1)
    return "a"
