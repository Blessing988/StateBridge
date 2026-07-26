"""Typed records used by the SynWTS scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PHASE_NAME_TO_NUMBER = {
    "prerecognition": "0",
    "pre-recognition": "0",
    "recognition": "1",
    "judgement": "2",
    "judgment": "2",
    "action": "3",
    "avoidance": "4",
}

PHASE_NUMBER_TO_NAME = {
    "0": "prerecognition",
    "1": "recognition",
    "2": "judgement",
    "3": "action",
    "4": "avoidance",
}

VIEWS = ("overhead_view", "vehicle_view")
VQA_SCOPES = ("environment", "overhead_view", "vehicle_view")
ROLES = ("pedestrian", "vehicle")


def normalize_phase_label(label: str | int | None) -> str:
    value = "" if label is None else str(label).strip().lower()
    return PHASE_NAME_TO_NUMBER.get(value, value)


@dataclass(slots=True)
class ScenarioRecord:
    split: str
    scenario_id: str
    scenario_type: str
    videos: dict[str, list[str]] = field(default_factory=dict)
    caption_files: dict[str, str] = field(default_factory=dict)
    vqa_files: dict[str, str] = field(default_factory=dict)
    bbox_files: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def is_normal(self) -> bool:
        return self.scenario_type == "normal_trimmed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "videos": self.videos,
            "caption_files": self.caption_files,
            "vqa_files": self.vqa_files,
            "bbox_files": self.bbox_files,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScenarioRecord":
        return cls(
            split=row["split"],
            scenario_id=row["scenario_id"],
            scenario_type=row["scenario_type"],
            videos=row.get("videos", {}),
            caption_files=row.get("caption_files", {}),
            vqa_files=row.get("vqa_files", {}),
            bbox_files=row.get("bbox_files", {}),
        )


def to_path_list(values: list[str]) -> list[Path]:
    return [Path(v) for v in values]

