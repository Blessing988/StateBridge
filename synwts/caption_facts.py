"""Fact-conditioned caption prompt augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .facts import build_fact_rows, question_to_fact_key
from .exporters import load_records, write_llamafactory_dataset_info
from .io import read_json, write_json


RETURN_JSON_MARKER = "Return JSON with exactly these keys:"


def augment_caption_dataset_with_facts(
    dataset_path: str | Path,
    output: str | Path,
    *,
    index_path: str | Path | None = None,
    wts_vqa_json: str | Path | None = None,
    vqa_submission: str | Path | None = None,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    max_global_facts: int = 12,
    max_phase_facts: int = 16,
) -> list[dict[str, Any]]:
    """Inject gold or predicted VQA facts into caption rows.

    Training mode uses `index_path` and gold synthetic VQA labels. Public-test
    mode uses the official WTS VQA JSON plus a VQA submission file.
    """
    if index_path and (wts_vqa_json or vqa_submission):
        raise ValueError("Use either --index or --wts-vqa-json/--vqa-submission, not both.")
    if bool(wts_vqa_json) != bool(vqa_submission):
        raise ValueError("--wts-vqa-json and --vqa-submission must be provided together.")
    if not index_path and not wts_vqa_json:
        raise ValueError("Provide --index for gold facts or --wts-vqa-json with --vqa-submission.")

    fact_map = (
        _gold_fact_map(index_path)
        if index_path
        else _predicted_wts_fact_map(wts_vqa_json, vqa_submission)
    )
    rows = read_json(dataset_path)
    if not isinstance(rows, list):
        raise ValueError(f"LLaMA-Factory dataset must be a list: {dataset_path}")

    augmented: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("metadata", {}).get("task") != "caption":
            augmented.append(row)
            continue
        metadata = row.get("metadata", {})
        scenario_id = str(metadata.get("scenario_id", ""))
        phase = str(metadata.get("phase", ""))
        facts = fact_map.get(scenario_id, {})
        context = _format_fact_context(
            facts,
            phase=phase,
            max_global_facts=max_global_facts,
            max_phase_facts=max_phase_facts,
        )
        new_row = dict(row)
        new_metadata = dict(metadata)
        new_metadata["fact_conditioned"] = bool(context)
        new_row["metadata"] = new_metadata
        if context:
            new_row["instruction"] = _inject_fact_context(str(row.get("instruction", "")), context)
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


def _gold_fact_map(index_path: str | Path) -> dict[str, dict[str, Any]]:
    records = load_records(index_path)
    rows = build_fact_rows(records)
    return {str(row["scenario_id"]): row for row in rows}


def _predicted_wts_fact_map(
    wts_vqa_json: str | Path,
    vqa_submission: str | Path,
) -> dict[str, dict[str, Any]]:
    submission = {
        str(row.get("id", "")).strip(): str(row.get("correct", "")).strip().lower()
        for row in read_json(vqa_submission)
    }
    facts: dict[str, dict[str, Any]] = {}
    for item_idx, item in enumerate(read_json(wts_vqa_json)):
        video_names = [str(name) for name in item.get("videos", [])]
        scenario_id = _scenario_from_video_names(video_names, fallback=f"item_{item_idx:05d}")
        scenario_type = "normal_trimmed" if any("normal" in name for name in video_names) else "event"
        row = facts.setdefault(
            scenario_id,
            {
                "split": "test",
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "global": {},
                "phases": {},
            },
        )
        if "event_phase" not in item:
            for question in item.get("conversations", []):
                _add_predicted_fact(row["global"], question, scope="environment", submission=submission)
            continue

        scope = _infer_wts_vqa_scope(item)
        for phase in item.get("event_phase", []):
            labels = phase.get("labels") or []
            phase_label = str(labels[0]).strip() if labels else ""
            phase_row = row["phases"].setdefault(phase_label, {})
            for question in phase.get("conversations", []):
                _add_predicted_fact(phase_row, question, scope=scope, submission=submission)
    return facts


def _add_predicted_fact(
    target: dict[str, Any],
    question: dict[str, Any],
    *,
    scope: str,
    submission: dict[str, str],
) -> None:
    qid = str(question.get("id", "")).strip()
    answer = submission.get(qid)
    if not answer:
        return
    answer_text = str(question.get(answer, "")).strip()
    if not answer_text:
        return
    key = question_to_fact_key(str(question.get("question", "")), scope)
    _nested_set(target, key, answer_text)


def _format_fact_context(
    facts: dict[str, Any],
    *,
    phase: str,
    max_global_facts: int,
    max_phase_facts: int,
) -> str:
    global_items = _flatten_facts(facts.get("global", {}))[:max_global_facts]
    phase_items = _flatten_facts(facts.get("phases", {}).get(phase, {}))[:max_phase_facts]
    if not global_items and not phase_items:
        return ""
    lines = [
        "VQA fact context: use these structured facts as grounding hints when they agree with the videos and bbox context.",
    ]
    if global_items:
        lines.append("Scenario facts:")
        lines.extend(f"- {key}: {value}" for key, value in global_items)
    if phase_items:
        lines.append(f"Phase {phase} facts:")
        lines.extend(f"- {key}: {value}" for key, value in phase_items)
    lines.append("If a fact conflicts with visible evidence, prioritize the current video.")
    return "\n".join(lines)


def _inject_fact_context(instruction: str, context: str) -> str:
    if RETURN_JSON_MARKER in instruction:
        return instruction.replace(RETURN_JSON_MARKER, f"{context}\n\n{RETURN_JSON_MARKER}", 1)
    return f"{instruction.rstrip()}\n\n{context}"


def _flatten_facts(facts: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key in sorted(facts):
        value = facts[key]
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_facts(value, dotted))
        elif str(value).strip():
            rows.append((dotted, str(value).strip()))
    return rows


def _nested_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _scenario_from_video_names(video_names: list[str], *, fallback: str) -> str:
    if not video_names:
        return fallback
    stem = Path(video_names[0]).stem
    if "_normal_" in stem:
        return stem
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return stem


def _infer_wts_vqa_scope(item: dict[str, Any]) -> str:
    for phase in item.get("event_phase", []):
        for question in phase.get("conversations", []):
            q = str(question.get("question", "")).lower()
            if "vehicle's field of view" in q or "action taken by vehicle" in q:
                return "vehicle_view"
            if "position of the vehicle relative" in q or "relative distance of vehicle" in q:
                return "vehicle_view"
    return "overhead_view"
