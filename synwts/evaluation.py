"""Local evaluation utilities for caption and VQA predictions."""

from __future__ import annotations

from pathlib import Path

from .index import build_index
from .io import read_json, write_json
from .metrics import caption_metric_bundle, vqa_accuracy
from .parsers import load_caption_phases, load_vqa_questions
from .schema import ScenarioRecord


def evaluate_caption_predictions(
    dataset_root: str | Path,
    prediction_path: str | Path,
    *,
    split: str = "val",
) -> dict:
    records = build_index(dataset_root, splits=(split,))
    predictions = read_json(prediction_path)
    references = _caption_references(records)
    pred_lookup = _caption_prediction_lookup(predictions)
    candidates: list[str] = []
    refs: list[str] = []
    missing: list[str] = []
    for key, ref_text in references.items():
        pred_text = pred_lookup.get(key, "")
        if not pred_text:
            missing.append("|".join(key))
        candidates.append(pred_text)
        refs.append(ref_text)
    metrics = caption_metric_bundle(candidates, refs)
    return {
        "split": split,
        "num_caption_units": len(refs),
        "missing": missing[:50],
        "num_missing": len(missing),
        "metrics": metrics,
        "note": "Dependency-light local metrics; official leaderboard is authoritative.",
    }


def evaluate_vqa_predictions(
    dataset_root: str | Path,
    prediction_path: str | Path,
    *,
    split: str = "val",
) -> dict:
    records = build_index(dataset_root, splits=(split,))
    predictions_raw = read_json(prediction_path)
    predictions = {
        str(row.get("id", "")): str(row.get("correct", "")).strip().lower()
        for row in predictions_raw
    }
    references = _vqa_references(records)
    return {
        "split": split,
        "metrics": vqa_accuracy(predictions, references),
        "num_predictions": len(predictions),
        "num_references": len(references),
    }


def write_eval_report(report: dict, output: str | Path) -> None:
    write_json(output, report)


def _caption_references(records: list[ScenarioRecord]) -> dict[tuple[str, str, str], str]:
    refs: dict[tuple[str, str, str], str] = {}
    for record in records:
        caption_path = _preferred_caption_file(record)
        if not caption_path:
            continue
        for phase in load_caption_phases(caption_path):
            refs[(record.scenario_id, phase["label"], "caption_pedestrian")] = phase["caption_pedestrian"]
            refs[(record.scenario_id, phase["label"], "caption_vehicle")] = phase["caption_vehicle"]
    return refs


def _preferred_caption_file(record: ScenarioRecord) -> str | None:
    for view in ("overhead_view", "vehicle_view"):
        if view in record.caption_files:
            return record.caption_files[view]
    return next(iter(record.caption_files.values()), None)


def _caption_prediction_lookup(predictions: dict) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for scenario_id, rows in predictions.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            labels = row.get("labels") or []
            label = str(labels[0]) if labels else ""
            for role_key in ("caption_pedestrian", "caption_vehicle"):
                lookup[(str(scenario_id), label, role_key)] = str(row.get(role_key, "")).strip()
    return lookup


def _vqa_references(records: list[ScenarioRecord]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for record in records:
        for scope, vqa_path in record.vqa_files.items():
            for question in load_vqa_questions(vqa_path, scope=scope, scenario_id=record.scenario_id):
                refs[question["id"]] = question["correct"]
    return refs

