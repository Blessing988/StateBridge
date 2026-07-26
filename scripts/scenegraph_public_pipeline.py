#!/usr/bin/env python
"""Export public scene-graph inference rows and assemble caption candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.caption_slots import _clean_caption_text, _template_caption
from synwts.exporters import write_llamafactory_dataset_info
from synwts.io import read_json, write_json
from synwts.schema import PHASE_NUMBER_TO_NAME
from synwts.submission import _load_prediction_texts
from synwts.validators import validate_caption_submission, validate_vqa_submission


PHASE_ORDER = ["4", "3", "2", "1", "0"]


def _media_tokens(count: int) -> str:
    return "\n".join("<video>" for _ in range(count))


def _phase_sort_key(phase: str) -> tuple[int, str]:
    return (PHASE_ORDER.index(phase), phase) if phase in PHASE_ORDER else (len(PHASE_ORDER), phase)


def _grounding_context(instruction: str) -> str:
    markers = ("Visual grounding note:", "BBox context:")
    starts = [instruction.find(marker) for marker in markers if marker in instruction]
    if not starts:
        return ""
    start = min(pos for pos in starts if pos >= 0)
    text = instruction[start:].strip()
    stop = text.find("\n\nReturn JSON")
    if stop >= 0:
        text = text[:stop].strip()
    return text


def _instruction(*, scenario_type: str, phase: str, media_count: int, grounding_context: str) -> str:
    phase_name = PHASE_NUMBER_TO_NAME.get(phase, phase)
    parts = [
        "You are a traffic safety scene-graph extractor.",
        "Predict structured traffic attributes only from the videos, bboxes, and phase label.",
        "Return strict JSON. Do not write prose.",
        "",
        f"Scenario type: {scenario_type}",
        f"Phase label: {phase} ({phase_name})",
        "",
        _media_tokens(media_count),
    ]
    if grounding_context:
        parts.extend(["", grounding_context])
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
    return "\n".join(parts)


def export_public_scenegraph(
    *,
    caption_dataset: Path,
    output: Path,
    dataset_info_output: Path | None,
    dataset_name: str | None,
    max_videos_per_row: int,
) -> list[dict[str, Any]]:
    rows = read_json(caption_dataset)
    if not isinstance(rows, list):
        raise ValueError(f"Caption inference dataset must be a list: {caption_dataset}")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
        if metadata.get("task") != "caption":
            continue
        scenario_id = str(metadata.get("scenario_id", "")).strip()
        phase = str(metadata.get("phase", "")).strip()
        if not scenario_id or not phase:
            continue
        key = (scenario_id, phase)
        item = grouped.setdefault(
            key,
            {
                "scenario_id": scenario_id,
                "phase": phase,
                "scenario_type": str(metadata.get("scenario_type", "")),
                "videos": [],
                "contexts": [],
                "metadata": metadata,
            },
        )
        for video in row.get("videos", []):
            if video not in item["videos"]:
                item["videos"].append(video)
        context = _grounding_context(str(row.get("instruction", "")))
        if context and context not in item["contexts"]:
            item["contexts"].append(context)

    out_rows: list[dict[str, Any]] = []
    for (scenario_id, phase), item in sorted(grouped.items(), key=lambda kv: (kv[0][0], _phase_sort_key(kv[0][1]))):
        videos = item["videos"][:max_videos_per_row] if max_videos_per_row > 0 else item["videos"]
        context = "\n\n".join(item["contexts"][:2])
        out_rows.append(
            {
                "instruction": _instruction(
                    scenario_type=item["scenario_type"],
                    phase=phase,
                    media_count=len(videos),
                    grounding_context=context,
                ),
                "input": "",
                "output": "{}",
                "videos": videos,
                "metadata": {
                    "task": "scenegraph",
                    "split": "test",
                    "scenario_id": scenario_id,
                    "scenario_type": item["scenario_type"],
                    "phase": phase,
                    "source_task": "caption",
                },
            }
        )
    write_json(output, out_rows)
    if dataset_info_output and dataset_name:
        write_llamafactory_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=output.name,
        )
    return out_rows


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _scenegraph_to_facts(obj: dict[str, Any]) -> dict[str, Any]:
    env = obj.get("environment", {}) if isinstance(obj.get("environment"), dict) else {}
    return {
        "global": {
            "pedestrian": env.get("pedestrian", {}) if isinstance(env.get("pedestrian"), dict) else {},
            "environment": env.get("environment", {}) if isinstance(env.get("environment"), dict) else {},
            "road": env.get("road", {}) if isinstance(env.get("road"), dict) else {},
        },
        "phase": {
            "pedestrian": obj.get("pedestrian", {}) if isinstance(obj.get("pedestrian"), dict) else {},
            "vehicle": obj.get("vehicle", {}) if isinstance(obj.get("vehicle"), dict) else {},
        },
    }


def _fallback_rows(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Fallback caption must be an object: {path}")
    return data


def _fallback_row(fallback: dict[str, list[dict[str, Any]]], scenario_id: str, phase: str, idx: int) -> dict[str, Any] | None:
    rows = fallback.get(scenario_id)
    if not isinstance(rows, list):
        return None
    for row in rows:
        labels = row.get("labels") if isinstance(row, dict) else None
        if labels and str(labels[0]) == phase:
            return row
    if idx < len(rows) and isinstance(rows[idx], dict):
        return rows[idx]
    return None


def _caption_from_scenegraph(obj: dict[str, Any], fallback: dict[str, Any] | None) -> tuple[str, str, bool]:
    facts = _scenegraph_to_facts(obj)
    ped = _clean_caption_text(_template_caption("pedestrian", facts))
    veh = _clean_caption_text(_template_caption("vehicle", facts))
    used_fallback = False
    if len(ped.split()) < 20 and fallback:
        ped = str(fallback.get("caption_pedestrian", "")).strip()
        used_fallback = True
    if len(veh.split()) < 12 and fallback:
        veh = str(fallback.get("caption_vehicle", "")).strip()
        used_fallback = True
    return ped, veh, used_fallback


def assemble_scenegraph_caption(
    *,
    inference_dataset: Path,
    predictions: Path,
    output: Path,
    report_output: Path | None,
    fallback_caption: Path | None,
    vqa_submission: Path | None,
    zip_output: Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_json(inference_dataset)
    texts = _load_prediction_texts(predictions)
    if len(rows) != len(texts):
        raise ValueError(f"Prediction count {len(texts)} != dataset rows {len(rows)}")
    fallback = _fallback_rows(fallback_caption)

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report = {
        "rows": 0,
        "parse_failed": 0,
        "fallback_fields": 0,
        "examples": [],
    }
    for idx, (row, text) in enumerate(zip(rows, texts)):
        metadata = row.get("metadata", {})
        scenario_id = str(metadata.get("scenario_id", "")).strip()
        phase = str(metadata.get("phase", "")).strip()
        if not scenario_id or not phase:
            continue
        obj = _extract_json(text)
        fb = _fallback_row(fallback, scenario_id, phase, len(out[scenario_id]))
        if obj is None:
            report["parse_failed"] += 1
            if fb:
                ped = str(fb.get("caption_pedestrian", "")).strip()
                veh = str(fb.get("caption_vehicle", "")).strip()
                used_fallback = True
            else:
                ped = veh = ""
                used_fallback = True
        else:
            ped, veh, used_fallback = _caption_from_scenegraph(obj, fb)
        if used_fallback:
            report["fallback_fields"] += 1
        item = {
            "labels": [phase],
            "caption_pedestrian": ped,
            "caption_vehicle": veh,
        }
        out[scenario_id].append(item)
        report["rows"] += 1
        if len(report["examples"]) < 5:
            report["examples"].append({"scenario_id": scenario_id, "phase": phase, "parsed": obj is not None, "row": item})

    ordered = {
        scenario_id: sorted(items, key=lambda row: _phase_sort_key(str((row.get("labels") or [""])[0])))
        for scenario_id, items in sorted(out.items())
    }
    write_json(output, ordered)
    validation = validate_caption_submission(output)
    report["validation"] = validation
    if report_output:
        write_json(report_output, report)
    if not validation["ok"]:
        raise ValueError(f"Caption validation failed: {validation['errors'][:3]}")
    if zip_output:
        if not vqa_submission:
            raise ValueError("--vqa-submission is required with --zip-output")
        vqa_validation = validate_vqa_submission(vqa_submission)
        if not vqa_validation["ok"]:
            raise ValueError(f"VQA validation failed: {vqa_validation['errors'][:3]}")
        with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output, "caption_submission.json")
            zf.write(vqa_submission, "vqa_submission.json")
        report["zip_output"] = str(zip_output)
    return ordered, report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--caption-dataset", type=Path, required=True)
    p_export.add_argument("--output", type=Path, required=True)
    p_export.add_argument("--dataset-info-output", type=Path)
    p_export.add_argument("--dataset-name")
    p_export.add_argument("--max-videos-per-row", type=int, default=4)

    p_assemble = sub.add_parser("assemble")
    p_assemble.add_argument("--inference-dataset", type=Path, required=True)
    p_assemble.add_argument("--predictions", type=Path, required=True)
    p_assemble.add_argument("--output", type=Path, required=True)
    p_assemble.add_argument("--report-output", type=Path)
    p_assemble.add_argument("--fallback-caption", type=Path)
    p_assemble.add_argument("--vqa-submission", type=Path)
    p_assemble.add_argument("--zip-output", type=Path)

    args = parser.parse_args()
    if args.command == "export":
        rows = export_public_scenegraph(
            caption_dataset=args.caption_dataset,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            max_videos_per_row=args.max_videos_per_row,
        )
        print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))
        return
    _submission, report = assemble_scenegraph_caption(
        inference_dataset=args.inference_dataset,
        predictions=args.predictions,
        output=args.output,
        report_output=args.report_output,
        fallback_caption=args.fallback_caption,
        vqa_submission=args.vqa_submission,
        zip_output=args.zip_output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
