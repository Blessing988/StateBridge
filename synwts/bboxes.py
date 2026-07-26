"""Bounding-box utilities for prompt conditioning."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from .io import read_json
from .schema import normalize_phase_label


def xywh_to_xyxy_1000(
    bbox: list[float],
    *,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> list[int]:
    """Convert SynWTS xywh pixels into Qwen-style normalized xyxy coordinates."""
    x, y, w, h = [float(v) for v in bbox[:4]]
    x1 = _clip(round((x / frame_width) * 1000))
    y1 = _clip(round((y / frame_height) * 1000))
    x2 = _clip(round(((x + w) / frame_width) * 1000))
    y2 = _clip(round(((y + h) / frame_height) * 1000))
    return [x1, y1, x2, y2]


def bbox_track_summary(
    path: str | Path,
    *,
    phase: str,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> dict[str, Any] | None:
    data = read_json(path)
    annotations = [
        ann
        for ann in data.get("annotations", [])
        if normalize_phase_label(ann.get("phase_number")) == phase and ann.get("bbox")
    ]
    if not annotations:
        return None
    annotations = sorted(annotations, key=lambda ann: ann.get("image_id", 0))
    boxes = [
        xywh_to_xyxy_1000(ann["bbox"], frame_width=frame_width, frame_height=frame_height)
        for ann in annotations
    ]
    first = boxes[0]
    mid = boxes[len(boxes) // 2]
    last = boxes[-1]
    mean_box = [round(mean(box[idx] for box in boxes)) for idx in range(4)]
    return {
        "track": Path(path).stem,
        "count": len(boxes),
        "first": first,
        "mid": mid,
        "last": last,
        "mean": mean_box,
    }


def make_bbox_context(
    bbox_files: dict[str, dict[str, list[str]]],
    *,
    view: str,
    phase: str | None,
    frame_width: int = 1920,
    frame_height: int = 1080,
    max_tracks_per_role: int = 4,
) -> str:
    if phase is None:
        return ""
    lines = [
        "BBox context: coordinates are [x1, y1, x2, y2] normalized to a 0-1000 full-frame grid."
    ]
    added = 0
    for role in ("pedestrian", "vehicle"):
        paths = bbox_files.get(role, {}).get(view, [])
        if not paths and role == "vehicle" and view == "vehicle_view":
            # Vehicle boxes are usually only provided from overhead cameras.
            paths = bbox_files.get(role, {}).get("overhead_view", [])
        for path in paths[:max_tracks_per_role]:
            summary = bbox_track_summary(
                path,
                phase=phase,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if summary is None:
                continue
            lines.append(
                f"- {role} {view} {summary['track']}: "
                f"first={summary['first']} mid={summary['mid']} "
                f"last={summary['last']} mean={summary['mean']} frames={summary['count']}"
            )
            added += 1
    return "\n".join(lines) if added else ""


def _clip(value: int) -> int:
    return max(0, min(1000, value))

