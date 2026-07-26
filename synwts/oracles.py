"""Reference-prediction generators for smoke tests only."""

from __future__ import annotations

from pathlib import Path

from .index import build_index
from .parsers import load_caption_phases, load_vqa_questions
from .schema import ScenarioRecord


def make_oracle_caption_predictions(dataset_root: str | Path, *, split: str = "val") -> dict:
    records = build_index(dataset_root, splits=(split,))
    predictions: dict[str, list[dict]] = {}
    for record in records:
        caption_path = _preferred_caption_file(record)
        if not caption_path:
            continue
        predictions[record.scenario_id] = [
            {
                "labels": [phase["label"]],
                "caption_pedestrian": phase["caption_pedestrian"],
                "caption_vehicle": phase["caption_vehicle"],
            }
            for phase in load_caption_phases(caption_path)
        ]
    return predictions


def make_oracle_vqa_predictions(dataset_root: str | Path, *, split: str = "val") -> list[dict]:
    records = build_index(dataset_root, splits=(split,))
    predictions: list[dict] = []
    for record in records:
        for scope, vqa_path in record.vqa_files.items():
            for question in load_vqa_questions(vqa_path, scope=scope, scenario_id=record.scenario_id):
                predictions.append({"id": question["id"], "correct": question["correct"]})
    return predictions


def _preferred_caption_file(record: ScenarioRecord) -> str | None:
    for view in ("overhead_view", "vehicle_view"):
        if view in record.caption_files:
            return record.caption_files[view]
    return next(iter(record.caption_files.values()), None)
