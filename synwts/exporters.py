"""Export SynWTS annotations into training formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .bboxes import make_bbox_context
from .clips import load_phase_clip_map
from .io import read_jsonl, write_json
from .parsers import load_caption_phases, load_vqa_questions
from .prompts import make_visual_media_context, phase_caption_instruction, vqa_instruction
from .schema import ScenarioRecord


def load_records(path: str | Path) -> list[ScenarioRecord]:
    return [ScenarioRecord.from_dict(row) for row in read_jsonl(path)]


def export_llamafactory(
    records: Iterable[ScenarioRecord],
    output: str | Path,
    *,
    tasks: set[str] | None = None,
    media_policy: str = "all",
    include_missing_media: bool = False,
    bbox_mode: str = "none",
    frame_width: int = 1920,
    frame_height: int = 1080,
    phase_clip_manifest: str | Path | None = None,
) -> list[dict]:
    tasks = tasks or {"caption", "vqa"}
    rows: list[dict] = []
    phase_clip_map = load_phase_clip_map(phase_clip_manifest) if phase_clip_manifest else {}
    for record in records:
        if "caption" in tasks:
            rows.extend(
                _caption_rows(
                    record,
                    media_policy,
                    include_missing_media,
                    bbox_mode,
                    frame_width,
                    frame_height,
                    phase_clip_map,
                )
            )
        if "vqa" in tasks:
            rows.extend(
                _vqa_rows(
                    record,
                    media_policy,
                    include_missing_media,
                    bbox_mode,
                    frame_width,
                    frame_height,
                    phase_clip_map,
                )
            )
    write_json(output, rows)
    return rows


def write_llamafactory_dataset_info(
    output: str | Path,
    *,
    dataset_name: str,
    file_name: str,
) -> None:
    data = {
        dataset_name: {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "videos": "videos",
            },
        }
    }
    write_json(output, data)


def write_llamafactory_preference_dataset_info(
    output: str | Path,
    *,
    dataset_name: str,
    file_name: str,
) -> None:
    data = {
        dataset_name: {
            "file_name": file_name,
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
                "videos": "videos",
            },
        }
    }
    write_json(output, data)


def _select_videos(record: ScenarioRecord, view: str, media_policy: str) -> list[str]:
    return apply_media_policy(record.videos.get(view, []), media_policy)


def apply_media_policy(videos: list[str], media_policy: str) -> list[str]:
    if media_policy == "first":
        return list(videos[:1])
    if media_policy == "none":
        return []
    if media_policy != "all":
        raise ValueError(f"Unsupported media_policy: {media_policy}")
    return list(videos)


def _caption_rows(
    record: ScenarioRecord,
    media_policy: str,
    include_missing_media: bool,
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
    phase_clip_map: dict[tuple[str, str, str], list[str]],
) -> list[dict]:
    rows: list[dict] = []
    for view, caption_path in sorted(record.caption_files.items()):
        videos = _select_videos(record, view, media_policy)
        if not videos and not include_missing_media:
            continue
        for phase in load_caption_phases(caption_path):
            phase_videos = apply_media_policy(
                phase_clip_map.get((record.scenario_id, view, phase["label"]), videos),
                media_policy,
            )
            bbox_context = _bbox_context_for(
                record,
                view=view,
                phase=phase["label"],
                bbox_mode=bbox_mode,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            output = {
                "labels": [phase["label"]],
                "caption_pedestrian": phase["caption_pedestrian"],
                "caption_vehicle": phase["caption_vehicle"],
            }
            rows.append(
                {
                    "instruction": phase_caption_instruction(
                        view=view,
                        scenario_type=record.scenario_type,
                        phase=phase["label"],
                        media_count=len(phase_videos),
                        bbox_context=bbox_context,
                        visual_context=make_visual_media_context(phase_videos),
                    ),
                    "input": "",
                    "output": json.dumps(output, ensure_ascii=False),
                    "videos": phase_videos,
                    "metadata": {
                        "task": "caption",
                        "split": record.split,
                        "scenario_id": record.scenario_id,
                        "scenario_type": record.scenario_type,
                        "view": view,
                        "phase": phase["label"],
                        "start_time": str(phase.get("start_time", "")),
                        "end_time": str(phase.get("end_time", "")),
                    },
                }
            )
    return rows


def _vqa_rows(
    record: ScenarioRecord,
    media_policy: str,
    include_missing_media: bool,
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
    phase_clip_map: dict[tuple[str, str, str], list[str]],
) -> list[dict]:
    rows: list[dict] = []
    for scope, vqa_path in sorted(record.vqa_files.items()):
        view_for_media = "overhead_view" if scope == "environment" else scope
        videos = _select_videos(record, view_for_media, media_policy)
        if not videos and scope == "environment":
            videos = _select_videos(record, "vehicle_view", media_policy)
        if not videos and not include_missing_media:
            continue
        for question in load_vqa_questions(vqa_path, scope=scope, scenario_id=record.scenario_id):
            phase_videos = videos
            if question["phase"] is not None:
                phase_videos = apply_media_policy(
                    phase_clip_map.get((record.scenario_id, view_for_media, question["phase"]), videos),
                    media_policy,
                )
            bbox_context = _bbox_context_for(
                record,
                view=view_for_media,
                phase=question["phase"],
                bbox_mode=bbox_mode,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            rows.append(
                {
                    "instruction": vqa_instruction(
                        scope=scope,
                        scenario_type=record.scenario_type,
                        phase=question["phase"],
                        question=question["question"],
                        options=question["options"],
                        media_count=len(phase_videos),
                        bbox_context=bbox_context,
                        visual_context=make_visual_media_context(phase_videos),
                    ),
                    "input": "",
                    "output": question["correct"],
                    "videos": phase_videos,
                    "metadata": {
                        "task": "vqa",
                        "vqa_id": question["id"],
                        "split": record.split,
                        "scenario_id": record.scenario_id,
                        "scenario_type": record.scenario_type,
                        "scope": scope,
                        "phase": question["phase"],
                    },
                }
            )
    return rows


def _bbox_context_for(
    record: ScenarioRecord,
    *,
    view: str,
    phase: str | None,
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
) -> str:
    if bbox_mode == "none":
        return ""
    if bbox_mode != "summary":
        raise ValueError(f"Unsupported bbox_mode: {bbox_mode}")
    return make_bbox_context(
        record.bbox_files,
        view=view,
        phase=phase,
        frame_width=frame_width,
        frame_height=frame_height,
    )
