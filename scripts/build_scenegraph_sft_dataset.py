#!/usr/bin/env python
"""Export phase-conditioned scene-graph SFT rows for LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.clips import load_phase_clip_map
from synwts.exporters import (
    _bbox_context_for,
    apply_media_policy,
    load_records,
    write_llamafactory_dataset_info,
)
from synwts.facts import build_fact_rows
from synwts.io import write_json
from synwts.schema import PHASE_NUMBER_TO_NAME


def _media_tokens(count: int) -> str:
    return "\n".join("<video>" for _ in range(count))


def _compact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _compact(v) for k, v in obj.items() if _compact(v) not in ({}, [], "", None)}
    if isinstance(obj, list):
        return [_compact(v) for v in obj if _compact(v) not in ({}, [], "", None)]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def _scenegraph_output(facts: dict[str, Any], phase: str) -> dict[str, Any]:
    global_facts = facts.get("global", {})
    phase_facts = facts.get("phases", {}).get(phase, {})
    return _compact(
        {
            "phase_label": phase,
            "phase_name": PHASE_NUMBER_TO_NAME.get(phase, phase),
            "environment": {
                "pedestrian": global_facts.get("pedestrian", {}),
                "environment": global_facts.get("environment", {}),
                "road": global_facts.get("road", {}),
            },
            "pedestrian": phase_facts.get("pedestrian", {}),
            "vehicle": phase_facts.get("vehicle", {}),
        }
    )


def _instruction(
    *,
    scenario_type: str,
    phase: str,
    views: list[str],
    media_count: int,
    bbox_context: str,
) -> str:
    phase_name = PHASE_NUMBER_TO_NAME.get(phase, phase)
    parts = [
        "You are a traffic safety scene-graph extractor.",
        "Predict only structured attributes that are visually supported by the videos, bboxes, and phase label.",
        "Return strict JSON. Do not write prose.",
        "",
        f"Scenario type: {scenario_type}",
        f"Phase label: {phase} ({phase_name})",
        f"Views: {', '.join(views)}",
        "",
        _media_tokens(media_count),
    ]
    if bbox_context.strip():
        parts.extend(["", bbox_context.strip()])
    parts.extend(
        [
            "",
            "Return JSON with keys:",
            "- phase_label",
            "- phase_name",
            "- environment: pedestrian profile, weather, lighting, and road context",
            "- pedestrian: phase-specific location, attention, behavior, direction, speed",
            "- vehicle: phase-specific location, field of view, action, speed",
        ]
    )
    return "\n".join(part for part in parts if part is not None)


def _videos_for_phase(
    record: Any,
    *,
    phase: str,
    views: list[str],
    media_policy: str,
    phase_clip_map: dict[tuple[str, str, str], list[str]],
    max_videos_per_row: int,
) -> list[str]:
    videos: list[str] = []
    for view in views:
        base = apply_media_policy(record.videos.get(view, []), media_policy)
        phase_videos = apply_media_policy(
            phase_clip_map.get((record.scenario_id, view, phase), base),
            media_policy,
        )
        videos.extend(phase_videos)
    if max_videos_per_row > 0:
        videos = videos[:max_videos_per_row]
    return videos


def _bbox_for_phase(
    record: Any,
    *,
    phase: str,
    views: list[str],
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
) -> str:
    if bbox_mode == "none":
        return ""
    chunks = []
    for view in views:
        text = _bbox_context_for(
            record,
            view=view,
            phase=phase,
            bbox_mode=bbox_mode,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if text.strip():
            chunks.append(f"{view} bbox context:\n{text.strip()}")
    return "\n\n".join(chunks)


def export_scenegraph_sft(
    *,
    index: Path,
    output: Path,
    dataset_info_output: Path | None,
    dataset_name: str | None,
    splits: set[str] | None,
    views: list[str],
    media_policy: str,
    phase_clip_manifest: Path | None,
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
    max_videos_per_row: int,
) -> list[dict[str, Any]]:
    records = load_records(index)
    if splits:
        records = [record for record in records if record.split in splits]
    facts_by_scenario = {row["scenario_id"]: row for row in build_fact_rows(records)}
    phase_clip_map = load_phase_clip_map(phase_clip_manifest) if phase_clip_manifest else {}

    rows: list[dict[str, Any]] = []
    for record in records:
        facts = facts_by_scenario.get(record.scenario_id, {})
        for phase in sorted(facts.get("phases", {}), reverse=True):
            videos = _videos_for_phase(
                record,
                phase=phase,
                views=views,
                media_policy=media_policy,
                phase_clip_map=phase_clip_map,
                max_videos_per_row=max_videos_per_row,
            )
            bbox_context = _bbox_for_phase(
                record,
                phase=phase,
                views=views,
                bbox_mode=bbox_mode,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            target = _scenegraph_output(facts, phase)
            rows.append(
                {
                    "instruction": _instruction(
                        scenario_type=record.scenario_type,
                        phase=phase,
                        views=views,
                        media_count=len(videos),
                        bbox_context=bbox_context,
                    ),
                    "input": "",
                    "output": json.dumps(target, ensure_ascii=False),
                    "videos": videos,
                    "metadata": {
                        "task": "scenegraph",
                        "split": record.split,
                        "scenario_id": record.scenario_id,
                        "scenario_type": record.scenario_type,
                        "phase": phase,
                        "views": views,
                    },
                }
            )
    write_json(output, rows)
    if dataset_info_output and dataset_name:
        write_llamafactory_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=output.name,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-info-output", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--views", default="overhead_view,vehicle_view")
    parser.add_argument("--media-policy", choices=("all", "first", "none"), default="all")
    parser.add_argument("--phase-clip-manifest", type=Path)
    parser.add_argument("--bbox-mode", choices=("none", "summary"), default="summary")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--max-videos-per-row", type=int, default=4)
    args = parser.parse_args()

    rows = export_scenegraph_sft(
        index=args.index,
        output=args.output,
        dataset_info_output=args.dataset_info_output,
        dataset_name=args.dataset_name,
        splits={item.strip() for item in args.splits.split(",") if item.strip()} if args.splits else None,
        views=[item.strip() for item in args.views.split(",") if item.strip()],
        media_policy=args.media_policy,
        phase_clip_manifest=args.phase_clip_manifest,
        bbox_mode=args.bbox_mode,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        max_videos_per_row=args.max_videos_per_row,
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
