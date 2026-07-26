"""Caption style retrieval for Sim2Real caption prompts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

from .bboxes import make_bbox_context
from .exporters import load_records, write_llamafactory_dataset_info
from .io import read_json, read_jsonl, write_json, write_jsonl
from .parsers import load_caption_phases
from .schema import PHASE_NUMBER_TO_NAME, ScenarioRecord


RETURN_JSON_MARKER = "Return JSON with exactly these keys:"


def build_caption_style_bank(
    index_path: str | Path,
    output: str | Path,
    *,
    splits: set[str] | None = None,
    bbox_mode: str = "summary",
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> list[dict[str, Any]]:
    """Build a searchable bank of ground-truth synthetic caption exemplars."""
    records = load_records(index_path)
    rows = _caption_style_bank_rows(
        records,
        splits=splits,
        bbox_mode=bbox_mode,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    write_jsonl(output, rows)
    return rows


def augment_caption_dataset_with_retrieval(
    dataset_path: str | Path,
    style_bank_path: str | Path,
    output: str | Path,
    *,
    k: int = 3,
    exclude_same_scenario: bool = True,
    max_caption_chars: int = 600,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    """Inject nearest caption exemplars into LLaMA-Factory caption prompts."""
    rows = read_json(dataset_path)
    if not isinstance(rows, list):
        raise ValueError(f"LLaMA-Factory dataset must be a list: {dataset_path}")

    bank = read_jsonl(style_bank_path)
    augmented: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            augmented.append(row)
            continue
        metadata = row.get("metadata", {})
        if metadata.get("task") != "caption":
            augmented.append(row)
            continue

        selected = retrieve_caption_examples(
            row,
            bank,
            k=k,
            exclude_same_scenario=exclude_same_scenario,
        )
        new_row = dict(row)
        new_metadata = dict(metadata)
        new_metadata["retrieval_style_ids"] = [example["style_id"] for example in selected]
        new_row["metadata"] = new_metadata
        new_row["instruction"] = _inject_examples(
            str(row.get("instruction", "")),
            selected,
            max_caption_chars=max_caption_chars,
        )
        augmented.append(new_row)

    write_json(output, augmented)
    if dataset_info_output and dataset_name:
        write_llamafactory_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=Path(output).name,
        )
    elif dataset_info_output or dataset_name:
        raise ValueError("--dataset-info-output and --dataset-name must be provided together.")
    return augmented


def retrieve_caption_examples(
    row: dict[str, Any],
    bank: list[dict[str, Any]],
    *,
    k: int = 3,
    exclude_same_scenario: bool = True,
) -> list[dict[str, Any]]:
    metadata = row.get("metadata", {})
    query = {
        "scenario_id": str(metadata.get("scenario_id", "")),
        "scenario_type": str(metadata.get("scenario_type", "")),
        "view": str(metadata.get("view", "")),
        "phase": str(metadata.get("phase", "")),
        "bbox_features": _bbox_features_from_text(str(row.get("instruction", ""))),
    }
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for example in bank:
        if exclude_same_scenario and example.get("scenario_id") == query["scenario_id"]:
            continue
        score = _caption_similarity(query, example)
        scored.append((score, str(example.get("style_id", "")), example))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [example for _, _, example in scored[: max(k, 0)]]


def _caption_style_bank_rows(
    records: Iterable[ScenarioRecord],
    *,
    splits: set[str] | None,
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted_splits = {split.strip() for split in splits or set() if split.strip()}
    for record in records:
        if wanted_splits and record.split not in wanted_splits:
            continue
        for view, caption_path in sorted(record.caption_files.items()):
            for phase in load_caption_phases(caption_path):
                phase_label = str(phase["label"])
                bbox_context = ""
                if bbox_mode == "summary":
                    bbox_context = make_bbox_context(
                        record.bbox_files,
                        view=view,
                        phase=phase_label,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                elif bbox_mode != "none":
                    raise ValueError(f"Unsupported bbox_mode: {bbox_mode}")
                style_id = "|".join([record.split, record.scenario_id, view, phase_label])
                rows.append(
                    {
                        "style_id": style_id,
                        "split": record.split,
                        "scenario_id": record.scenario_id,
                        "scenario_type": record.scenario_type,
                        "view": view,
                        "phase": phase_label,
                        "phase_name": PHASE_NUMBER_TO_NAME.get(phase_label, phase_label),
                        "caption_pedestrian": _clean_caption(phase["caption_pedestrian"]),
                        "caption_vehicle": _clean_caption(phase["caption_vehicle"]),
                        "bbox_features": _bbox_features_from_text(bbox_context),
                    }
                )
    return rows


def _caption_similarity(query: dict[str, Any], example: dict[str, Any]) -> float:
    score = 0.0
    if query.get("phase") == example.get("phase"):
        score += 100.0
    if query.get("view") == example.get("view"):
        score += 30.0
    if query.get("scenario_type") == example.get("scenario_type"):
        score += 12.0

    query_boxes = query.get("bbox_features") or {}
    example_boxes = example.get("bbox_features") or {}
    for role, role_weight in (("pedestrian", 10.0), ("vehicle", 7.0)):
        q_box = query_boxes.get(role)
        e_box = example_boxes.get(role)
        if not q_box or not e_box:
            continue
        score += role_weight * _box_similarity(q_box, e_box)
    return score


def _box_similarity(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    distance = math.sqrt(
        (box_a["cx"] - box_b["cx"]) ** 2
        + (box_a["cy"] - box_b["cy"]) ** 2
    )
    size_distance = abs(box_a["area"] - box_b["area"])
    # Coordinates are normalized to 0..1000, so these constants keep bbox
    # proximity useful without dominating phase/view matching.
    return max(0.0, 1.0 - (distance / 900.0) - (size_distance / 500000.0))


def _bbox_features_from_text(text: str) -> dict[str, dict[str, float]]:
    features: dict[str, list[dict[str, float]]] = {}
    pattern = re.compile(
        r"-\s+(pedestrian|vehicle)\b.*?mean=\[([0-9,\s]+)\]",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        role = match.group(1).lower()
        values = [int(value.strip()) for value in match.group(2).split(",") if value.strip()]
        if len(values) != 4:
            continue
        x1, y1, x2, y2 = values
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        features.setdefault(role, []).append(
            {
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "area": float(width * height),
            }
        )

    return {
        role: {
            "cx": sum(item["cx"] for item in items) / len(items),
            "cy": sum(item["cy"] for item in items) / len(items),
            "area": sum(item["area"] for item in items) / len(items),
        }
        for role, items in features.items()
        if items
    }


def _inject_examples(
    instruction: str,
    examples: list[dict[str, Any]],
    *,
    max_caption_chars: int,
) -> str:
    if not examples:
        return instruction
    block = _format_retrieval_block(examples, max_caption_chars=max_caption_chars)
    if RETURN_JSON_MARKER in instruction:
        return instruction.replace(RETURN_JSON_MARKER, f"{block}\n\n{RETURN_JSON_MARKER}", 1)
    return f"{instruction.rstrip()}\n\n{block}"


def _format_retrieval_block(
    examples: list[dict[str, Any]],
    *,
    max_caption_chars: int,
) -> str:
    lines = [
        "Retrieved synthetic annotation-style examples for wording only:",
    ]
    for idx, example in enumerate(examples, start=1):
        phase = example.get("phase", "")
        phase_name = example.get("phase_name", phase)
        lines.append(
            f"Example {idx} ({example.get('scenario_type', '')}, "
            f"{example.get('view', '')}, phase {phase} {phase_name}):"
        )
        lines.append(
            "caption_pedestrian: "
            + _truncate(_clean_caption(str(example.get("caption_pedestrian", ""))), max_caption_chars)
        )
        lines.append(
            "caption_vehicle: "
            + _truncate(_clean_caption(str(example.get("caption_vehicle", ""))), max_caption_chars)
        )
    lines.append(
        "Use these examples only for annotation style, vocabulary, and level of detail. "
        "Do not copy facts unless they are visible in the current videos and bbox context."
    )
    return "\n".join(lines)


def _clean_caption(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
