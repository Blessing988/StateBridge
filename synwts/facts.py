"""Build canonical traffic-fact records from VQA supervision."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .exporters import load_records
from .io import write_jsonl
from .parsers import load_vqa_questions
from .schema import ScenarioRecord


def export_fact_jsonl(index_path: str | Path, output: str | Path) -> list[dict[str, Any]]:
    records = load_records(index_path)
    rows = build_fact_rows(records)
    write_jsonl(output, rows)
    return rows


def build_fact_rows(records: list[ScenarioRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        global_facts: dict[str, Any] = {}
        phase_facts: dict[str, dict[str, Any]] = {}
        for scope, vqa_path in record.vqa_files.items():
            for question in load_vqa_questions(vqa_path, scope=scope, scenario_id=record.scenario_id):
                fact_key = question_to_fact_key(question["question"], scope)
                answer_text = question["options"].get(question["correct"], question["correct"])
                if question["phase"] is None:
                    _nested_set(global_facts, fact_key, answer_text)
                else:
                    phase_row = phase_facts.setdefault(question["phase"], {})
                    _nested_set(phase_row, fact_key, answer_text)
        rows.append(
            {
                "split": record.split,
                "scenario_id": record.scenario_id,
                "scenario_type": record.scenario_type,
                "global": global_facts,
                "phases": phase_facts,
            }
        )
    return rows


def question_to_fact_key(question: str, scope: str) -> str:
    q = question.lower().strip()
    if scope == "environment":
        return _environment_key(q)
    if "position of the pedestrian relative" in q:
        return "pedestrian.position_relative_to_vehicle"
    if "orientation of the pedestrian" in q:
        return "pedestrian.body_orientation"
    if "relative distance of pedestrian" in q:
        return "pedestrian.distance_to_vehicle"
    if "pedestrian's line of sight" in q:
        return "pedestrian.line_of_sight"
    if "pedestrian's visual status" in q:
        return "pedestrian.visual_status"
    if "pedestrian's direction of travel" in q:
        return "pedestrian.direction_of_travel"
    if "pedestrian's awareness" in q:
        return "pedestrian.awareness_of_vehicle"
    if "fine-grained action taken by the pedestrian" in q:
        return "pedestrian.fine_grained_action"
    if "pedestrian's action" in q:
        return "pedestrian.action"
    if "pedestrian's speed" in q or "pedestrian speed" in q:
        return "pedestrian.speed"
    if "position of the vehicle relative" in q:
        return "vehicle.position_relative_to_pedestrian"
    if "relative distance of vehicle" in q:
        return "vehicle.distance_to_pedestrian"
    if "vehicle's field of view" in q:
        return "vehicle.field_of_view"
    if "action taken by vehicle" in q or "vehicle action" in q:
        return "vehicle.action"
    if "vehicle's speed" in q or "speed of the vehicle" in q or "vehicle speed" in q:
        return "vehicle.speed"
    return f"{scope}.{_slug(q)}"


def _environment_key(q: str) -> str:
    rules = [
        ("age group", "pedestrian.age_group"),
        ("height of the pedestrian", "pedestrian.height"),
        ("wearing on upper body", "pedestrian.upper_body_type"),
        ("upper body clothing", "pedestrian.upper_body_color"),
        ("wearing on lower body", "pedestrian.lower_body_type"),
        ("lower body clothing", "pedestrian.lower_body_color"),
        ("weather", "environment.weather"),
        ("brightness", "environment.brightness"),
        ("road surface conditions", "road.surface_condition"),
        ("road inclination", "road.inclination"),
        ("surface type of the road", "road.surface_type"),
        ("volume of the traffic", "road.traffic_volume"),
        ("type of the road", "road.road_type"),
        ("how many lanes", "road.lane_count"),
        ("formation of the road", "road.formation"),
    ]
    for needle, key in rules:
        if needle in q:
            return key
    return f"environment.{_slug(q)}"


def _nested_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:80] or "unknown"

