"""Learned caption realization datasets and assembly helpers."""

from __future__ import annotations

from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Any

from .caption_facts import _format_fact_context, _gold_fact_map, _predicted_wts_fact_map
from .exporters import load_records, write_llamafactory_dataset_info
from .io import read_json, write_json
from .parsers import load_caption_phases
from .submission import _parse_caption_prediction


CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")


def export_caption_realizer_train(
    *,
    index: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    splits: set[str] | None = None,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    """Build text-only SFT data for learning WTS caption realization.

    The model input contains deliberately degraded synthetic captions plus gold
    VQA facts. The target is the original WTS-style caption JSON.
    """
    wanted_splits = splits or {"train"}
    facts = _gold_fact_map(index)
    rows: list[dict[str, Any]] = []
    for record in load_records(index):
        if record.split not in wanted_splits:
            continue
        for view, caption_path in sorted(record.caption_files.items()):
            for phase_idx, phase in enumerate(load_caption_phases(caption_path)):
                target = _target_from_phase(phase)
                candidates = _synthetic_noisy_candidates(target)
                instruction = _realizer_instruction(
                    scenario_type=record.scenario_type,
                    view=view,
                    phase=str(phase["label"]),
                    fact_context=_format_fact_context(
                        facts.get(record.scenario_id, {}),
                        phase=str(phase["label"]),
                        max_global_facts=16,
                        max_phase_facts=24,
                    ),
                    candidate_blocks=candidates,
                )
                rows.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": json.dumps(target, ensure_ascii=False),
                        "videos": [],
                        "metadata": {
                            "task": "caption_realizer",
                            "split": record.split,
                            "scenario_id": record.scenario_id,
                            "scenario_type": record.scenario_type,
                            "view": view,
                            "phase": str(phase["label"]),
                            "phase_idx": phase_idx,
                        },
                    }
                )
                if max_rows and len(rows) >= max_rows:
                    break
            if max_rows and len(rows) >= max_rows:
                break
        if max_rows and len(rows) >= max_rows:
            break

    write_json(output, rows)
    _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def export_caption_realizer_test(
    *,
    captions: dict[str, str | Path],
    fallback_name: str,
    output: str | Path,
    wts_vqa_json: str | Path | None = None,
    vqa_submission: str | Path | None = None,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    """Build public-test realizer inference rows from caption candidates."""
    if fallback_name not in captions:
        raise ValueError(f"Fallback caption source is not available: {fallback_name}")
    if bool(wts_vqa_json) != bool(vqa_submission):
        raise ValueError("--wts-vqa-json and --vqa-submission must be provided together.")

    loaded = {name: _load_caption_submission(path) for name, path in captions.items()}
    fallback = loaded[fallback_name]
    facts = _predicted_wts_fact_map(wts_vqa_json, vqa_submission) if wts_vqa_json else {}
    rows: list[dict[str, Any]] = []
    for scenario_id, fallback_rows in fallback.items():
        by_source = {
            name: _rows_by_phase(source_rows.get(scenario_id, []))
            for name, source_rows in loaded.items()
        }
        for fallback_idx, fallback_row in enumerate(fallback_rows):
            phase = _phase_label(fallback_row, fallback_idx)
            candidates: list[str] = []
            for name in sorted(loaded):
                row = by_source[name].get(phase)
                if row is None:
                    continue
                candidates.append(_format_candidate(name, row))
            instruction = _realizer_instruction(
                scenario_type=str(fallback_row.get("scenario_type", "unknown")),
                view=str(fallback_row.get("view", "unknown")),
                phase=phase,
                fact_context=_format_fact_context(
                    facts.get(scenario_id, {}),
                    phase=phase,
                    max_global_facts=16,
                    max_phase_facts=24,
                ),
                candidate_blocks=candidates,
            )
            rows.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": "",
                    "videos": [],
                    "metadata": {
                        "task": "caption_realizer_inference",
                        "scenario_id": scenario_id,
                        "phase": phase,
                        "fallback_idx": fallback_idx,
                        "fallback_caption": {
                            "caption_pedestrian": str(fallback_row.get("caption_pedestrian", "")).strip(),
                            "caption_vehicle": str(fallback_row.get("caption_vehicle", "")).strip(),
                        },
                    },
                }
            )

    write_json(output, rows)
    _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def assemble_caption_realizer_submission(
    *,
    inference_dataset: str | Path,
    predictions: str | Path,
    output: str | Path,
    fallback_caption: str | Path,
    report_output: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_json(inference_dataset)
    pred_rows = _read_prediction_rows(predictions)
    fallback = _load_caption_submission(fallback_caption)
    if len(rows) != len(pred_rows):
        raise ValueError(f"Prediction count mismatch: dataset={len(rows)} predictions={len(pred_rows)}")

    output_rows: dict[str, list[dict[str, Any]]] = {
        scenario_id: [dict(row) for row in phase_rows]
        for scenario_id, phase_rows in fallback.items()
    }
    changed = 0
    parse_failed = 0
    for row, pred in zip(rows, pred_rows):
        metadata = row.get("metadata", {})
        scenario_id = str(metadata.get("scenario_id", ""))
        idx = int(metadata.get("fallback_idx", 0))
        fallback_obj = metadata.get("fallback_caption", {})
        parsed = _parse_caption_prediction(str(pred), str(metadata.get("phase", "")))
        if not _valid_realized_caption(parsed):
            parse_failed += 1
            parsed = {
                "caption_pedestrian": str(fallback_obj.get("caption_pedestrian", "")).strip(),
                "caption_vehicle": str(fallback_obj.get("caption_vehicle", "")).strip(),
            }
        if scenario_id in output_rows and 0 <= idx < len(output_rows[scenario_id]):
            before = output_rows[scenario_id][idx]
            new_row = dict(before)
            new_row.update(parsed)
            output_rows[scenario_id][idx] = new_row
            if any(str(before.get(key, "")).strip() != str(parsed.get(key, "")).strip() for key in CAPTION_KEYS):
                changed += 1

    report = {
        "rows": len(rows),
        "changed_from_fallback": changed,
        "parse_failed": parse_failed,
    }
    write_json(output, output_rows)
    if report_output:
        write_json(report_output, report)
    return output_rows, report


def _realizer_instruction(
    *,
    scenario_type: str,
    view: str,
    phase: str,
    fact_context: str,
    candidate_blocks: list[str],
) -> str:
    candidate_text = "\n\n".join(candidate_blocks)
    return (
        "You are a traffic safety caption realization model. Rewrite the candidate captions into the official WTS style.\n"
        "Use the candidates as phrase sources. Preserve concrete facts and useful 4-gram phrases when they are consistent.\n"
        "Follow this order for the pedestrian caption: identity, location, attention, behavior, context.\n"
        "Follow this order for the vehicle caption: motion, relative position, field of view, speed/action, context.\n"
        "Do not mention candidate names, prompts, bboxes, overlays, or colored graphics.\n"
        "Do not invent facts that are not supported by candidates or VQA facts.\n"
        "Target length: pedestrian 115-155 words; vehicle 100-140 words.\n\n"
        f"Scenario type: {scenario_type}\n"
        f"View: {view}\n"
        f"Phase label: {phase}\n\n"
        f"{fact_context.strip()}\n\n"
        "Candidate captions:\n"
        f"{candidate_text}\n\n"
        "Return JSON with exactly these keys: caption_pedestrian, caption_vehicle."
    )


def _synthetic_noisy_candidates(target: dict[str, str]) -> list[str]:
    ped = target["caption_pedestrian"]
    veh = target["caption_vehicle"]
    return [
        _format_candidate("source_a_shuffled", {
            "caption_pedestrian": _shuffle_sentences(ped),
            "caption_vehicle": _shuffle_sentences(veh),
        }),
        _format_candidate("source_b_front_trimmed", {
            "caption_pedestrian": _first_sentences(ped, 4),
            "caption_vehicle": _first_sentences(veh, 4),
        }),
        _format_candidate("source_c_context_trimmed", {
            "caption_pedestrian": _drop_context_sentences(ped),
            "caption_vehicle": _drop_context_sentences(veh),
        }),
        _format_candidate("source_d_gold_style", target),
    ]


def _target_from_phase(phase: dict[str, Any]) -> dict[str, str]:
    return {
        "caption_pedestrian": str(phase.get("caption_pedestrian", "")).strip(),
        "caption_vehicle": str(phase.get("caption_vehicle", "")).strip(),
    }


def _format_candidate(name: str, row: dict[str, Any]) -> str:
    return (
        f"[{name}]\n"
        f"caption_pedestrian: {str(row.get('caption_pedestrian', '')).strip()}\n"
        f"caption_vehicle: {str(row.get('caption_vehicle', '')).strip()}"
    )


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _shuffle_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return text
    return " ".join(sentences[1::2] + sentences[::2])


def _first_sentences(text: str, count: int) -> str:
    sentences = _split_sentences(text)
    return " ".join(sentences[:count]) if sentences else text


def _drop_context_sentences(text: str) -> str:
    context_terms = ("weather", "brightness", "road surface", "traffic volume", "sidewalk", "asphalt")
    kept = [s for s in _split_sentences(text) if not any(term in s.lower() for term in context_terms)]
    return " ".join(kept) if kept else _first_sentences(text, 3)


def _load_caption_submission(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Caption submission must be a dict: {path}")
    return data


def _rows_by_phase(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_phase_label(row, idx): row for idx, row in enumerate(rows)}


def _phase_label(row: dict[str, Any], fallback_idx: int) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    for key in ("phase", "phase_label", "event_phase"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return str(fallback_idx)


def _read_prediction_rows(path: str | Path) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(str(obj.get("predict", obj.get("prediction", obj.get("output", "")))))
            else:
                rows.append(str(obj))
    return rows


def _valid_realized_caption(obj: dict[str, str]) -> bool:
    for key in CAPTION_KEYS:
        text = str(obj.get(key, "")).strip()
        if len(text.split()) < 20:
            return False
        if not _looks_complete_sentence(text):
            return False
    return True


def _looks_complete_sentence(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if clean[-1] not in ".!?":
        return False
    tail = clean[-80:].lower()
    bad_tails = (
        "there is",
        "there are",
        "with",
        "and",
        "or",
        "to the",
        "in the",
        "on the",
        "at a",
        "diagonally to the right",
        "diagonally to the left",
    )
    return not any(tail.endswith(term) for term in bad_tails)


def _maybe_write_dataset_info(
    output: str | Path,
    dataset_info_output: str | Path | None,
    dataset_name: str | None,
) -> None:
    if dataset_info_output and dataset_name:
        write_llamafactory_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=Path(output).name,
        )
    elif dataset_info_output or dataset_name:
        raise ValueError("--dataset-info-output and --dataset-name must be provided together.")
