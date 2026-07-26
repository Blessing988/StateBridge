"""Learned caption selector datasets and conservative assembly."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from .caption_facts import _format_fact_context, _gold_fact_map, _predicted_wts_fact_map
from .caption_realizer import (
    _drop_context_sentences,
    _first_sentences,
    _format_candidate,
    _load_caption_submission,
    _phase_label,
    _rows_by_phase,
    _shuffle_sentences,
)
from .caption_slots import _ngram_f1, _token_f1
from .exporters import load_records, write_llamafactory_dataset_info
from .io import read_json, write_json
from .parsers import load_caption_phases


CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")
PHASE_ORDER = ("0", "1", "2", "3", "4")
PHASE_CUES = {
    "0": ("before", "approach", "standing", "walking", "far", "surroundings"),
    "1": ("notice", "recogn", "line of sight", "aware", "unaware", "watch"),
    "2": ("judg", "intend", "plan", "decid", "aware", "notice"),
    "3": ("action", "cross", "move", "straight", "turn", "brak", "slow"),
    "4": ("avoid", "stop", "stopped", "collision", "deceler", "brak", "stationary"),
}


def export_caption_selector_train(
    *,
    index: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    splits: set[str] | None = None,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    wanted_splits = splits or {"train"}
    facts = _gold_fact_map(index)
    records = load_records(index)
    phase_pool: dict[str, list[dict[str, str]]] = {phase: [] for phase in PHASE_ORDER}
    for record in records:
        if record.split not in wanted_splits:
            continue
        for caption_path in record.caption_files.values():
            for phase in load_caption_phases(caption_path):
                phase_pool.setdefault(str(phase["label"]), []).append(_target_from_phase(phase))

    rows: list[dict[str, Any]] = []
    for record in records:
        if record.split not in wanted_splits:
            continue
        for view, caption_path in sorted(record.caption_files.items()):
            for phase in load_caption_phases(caption_path):
                phase_label = str(phase["label"])
                target = _target_from_phase(phase)
                candidates = _training_candidates(target, phase_pool.get(phase_label, []))
                for source_name, candidate, label in candidates:
                    rows.append(
                        {
                            "instruction": _selector_instruction(
                                scenario_type=record.scenario_type,
                                view=view,
                                phase=phase_label,
                                fact_context=_format_fact_context(
                                    facts.get(record.scenario_id, {}),
                                    phase=phase_label,
                                    max_global_facts=16,
                                    max_phase_facts=24,
                                ),
                                candidate_name=source_name,
                                candidate=candidate,
                            ),
                            "input": "",
                            "output": label,
                            "videos": [],
                            "metadata": {
                                "task": "caption_selector",
                                "split": record.split,
                                "scenario_id": record.scenario_id,
                                "scenario_type": record.scenario_type,
                                "view": view,
                                "phase": phase_label,
                                "source": source_name,
                            },
                        }
                    )
                    if max_rows and len(rows) >= max_rows:
                        break
                if max_rows and len(rows) >= max_rows:
                    break
            if max_rows and len(rows) >= max_rows:
                break
        if max_rows and len(rows) >= max_rows:
            break

    write_json(output, rows)
    _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def export_caption_selector_test(
    *,
    captions: dict[str, str | Path],
    fallback_name: str,
    output: str | Path,
    wts_vqa_json: str | Path | None = None,
    vqa_submission: str | Path | None = None,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    if fallback_name not in captions:
        raise ValueError(f"Fallback caption source is not available: {fallback_name}")
    if bool(wts_vqa_json) != bool(vqa_submission):
        raise ValueError("--wts-vqa-json and --vqa-submission must be provided together.")
    loaded = {name: _load_caption_submission(path) for name, path in captions.items()}
    fallback = loaded[fallback_name]
    facts = _predicted_wts_fact_map(wts_vqa_json, vqa_submission) if wts_vqa_json else {}

    rows: list[dict[str, Any]] = []
    for scenario_id, fallback_rows in fallback.items():
        by_source = {name: _rows_by_phase(source.get(scenario_id, [])) for name, source in loaded.items()}
        for fallback_idx, fallback_row in enumerate(fallback_rows):
            phase = _phase_label(fallback_row, fallback_idx)
            for source_name in sorted(loaded):
                candidate = by_source[source_name].get(phase)
                if candidate is None:
                    continue
                rows.append(
                    {
                        "instruction": _selector_instruction(
                            scenario_type=str(fallback_row.get("scenario_type", "unknown")),
                            view=str(fallback_row.get("view", "unknown")),
                            phase=phase,
                            fact_context=_format_fact_context(
                                facts.get(scenario_id, {}),
                                phase=phase,
                                max_global_facts=16,
                                max_phase_facts=24,
                            ),
                            candidate_name=source_name,
                            candidate=candidate,
                        ),
                        "input": "",
                        "output": "",
                        "videos": [],
                        "metadata": {
                            "task": "caption_selector_inference",
                            "scenario_id": scenario_id,
                            "phase": phase,
                            "fallback_idx": fallback_idx,
                            "source": source_name,
                        },
                    }
                )

    write_json(output, rows)
    _maybe_write_dataset_info(output, dataset_info_output, dataset_name)
    return rows


def assemble_caption_selector_submission(
    *,
    inference_dataset: str | Path,
    predictions: str | Path,
    captions: dict[str, str | Path],
    fallback_name: str,
    output: str | Path,
    wts_vqa_json: str | Path | None = None,
    vqa_submission: str | Path | None = None,
    report_output: str | Path | None = None,
    min_good_margin: float = 0.0,
    min_source_overlap: float = 0.70,
    max_changed_rows: int = 40,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if fallback_name not in captions:
        raise ValueError(f"Fallback caption source is not available: {fallback_name}")
    rows = read_json(inference_dataset)
    pred_rows = _read_prediction_rows(predictions)
    if len(rows) != len(pred_rows):
        raise ValueError(f"Prediction count mismatch: dataset={len(rows)} predictions={len(pred_rows)}")
    loaded = {name: _load_caption_submission(path) for name, path in captions.items()}
    fallback = loaded[fallback_name]
    facts = _predicted_wts_fact_map(wts_vqa_json, vqa_submission) if wts_vqa_json and vqa_submission else {}

    scored: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row, pred in zip(rows, pred_rows):
        metadata = row.get("metadata", {})
        scenario_id = str(metadata.get("scenario_id", ""))
        idx = int(metadata.get("fallback_idx", 0))
        source = str(metadata.get("source", ""))
        phase = str(metadata.get("phase", idx))
        candidate = _candidate_by_idx_or_phase(loaded.get(source, {}), scenario_id, phase, idx)
        base_row = _candidate_by_idx_or_phase(fallback, scenario_id, phase, idx)
        if candidate is None or base_row is None:
            continue
        label_score = _label_score(str(pred))
        overlap = _token_f1(_row_text(candidate), _row_text(base_row))
        ngram = _ngram_f1(_row_text(candidate), _row_text(base_row), n=4)
        lock_penalty = _attribute_lock_penalty(_row_text(candidate), facts.get(scenario_id, {}), phase=phase)
        phase_penalty = _phase_transition_penalty(candidate, phase=phase)
        score = label_score + 0.30 * overlap + 0.30 * ngram + lock_penalty + phase_penalty
        if source == fallback_name:
            score += 0.25
        if overlap < min_source_overlap:
            score -= 5.0
        scored.setdefault((scenario_id, idx), []).append(
            {
                "source": source,
                "phase": phase,
                "score": score,
                "label_score": label_score,
                "overlap": overlap,
                "ngram": ngram,
                "lock_penalty": lock_penalty,
                "phase_penalty": phase_penalty,
                "row": candidate,
            }
        )

    selected = {sid: [dict(row) for row in phase_rows] for sid, phase_rows in fallback.items()}
    proposed: list[tuple[float, str, int, dict[str, Any], dict[str, Any]]] = []
    source_counts: Counter[str] = Counter()
    for (scenario_id, idx), options in scored.items():
        ranked = sorted(options, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        base = next((item for item in options if item["source"] == fallback_name), None)
        if base is None or best["source"] == fallback_name:
            source_counts[fallback_name] += 1
            continue
        margin = best["score"] - base["score"]
        if margin >= min_good_margin:
            proposed.append((margin, scenario_id, idx, best["row"], best))
        else:
            source_counts[fallback_name] += 1

    proposed.sort(key=lambda item: item[0], reverse=True)
    if max_changed_rows:
        proposed = proposed[:max_changed_rows]
    for _margin, scenario_id, idx, row, best in proposed:
        if scenario_id in selected and 0 <= idx < len(selected[scenario_id]):
            selected[scenario_id][idx] = _normalize_output_row(row, fallback=selected[scenario_id][idx])
            source_counts[str(best["source"])] += 1
    total = sum(len(rows_) for rows_ in fallback.values())
    source_counts[fallback_name] += total - sum(source_counts.values())

    report = {
        "total_rows": total,
        "changed_rows": len(proposed),
        "source_counts": dict(source_counts),
        "min_good_margin": min_good_margin,
        "min_source_overlap": min_source_overlap,
        "max_changed_rows": max_changed_rows,
        "proposed_sample": [
            {
                "scenario_id": sid,
                "row_index": idx,
                "margin": round(margin, 4),
                "source": best["source"],
                "score": round(best["score"], 4),
                "label_score": round(best["label_score"], 4),
                "overlap": round(best["overlap"], 4),
                "ngram": round(best["ngram"], 4),
                "lock_penalty": round(best["lock_penalty"], 4),
                "phase_penalty": round(best["phase_penalty"], 4),
            }
            for margin, sid, idx, _row, best in proposed[:100]
        ],
    }
    write_json(output, selected)
    if report_output:
        write_json(report_output, report)
    return selected, report


def repair_caption_consistency(
    *,
    caption: str | Path,
    fallback_caption: str | Path,
    output: str | Path,
    wts_vqa_json: str | Path,
    vqa_submission: str | Path,
    report_output: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data = _load_caption_submission(caption)
    fallback = _load_caption_submission(fallback_caption)
    facts = _predicted_wts_fact_map(wts_vqa_json, vqa_submission)
    repaired: dict[str, list[dict[str, Any]]] = {}
    changed = 0
    reasons: Counter[str] = Counter()
    for scenario_id, rows in data.items():
        out_rows = []
        for idx, row in enumerate(rows):
            phase = _phase_label(row, idx)
            text = _row_text(row)
            penalty = _attribute_lock_penalty(text, facts.get(scenario_id, {}), phase=phase)
            penalty += _phase_transition_penalty(row, phase=phase)
            if penalty <= -2.0 and scenario_id in fallback and idx < len(fallback[scenario_id]):
                out_rows.append(dict(fallback[scenario_id][idx]))
                changed += 1
                reasons["fallback_due_to_lock_or_phase"] += 1
            else:
                out_rows.append(dict(row))
        repaired[scenario_id] = out_rows
    report = {"changed_rows": changed, "reasons": dict(reasons)}
    write_json(output, repaired)
    if report_output:
        write_json(report_output, report)
    return repaired, report


def _selector_instruction(
    *,
    scenario_type: str,
    view: str,
    phase: str,
    fact_context: str,
    candidate_name: str,
    candidate: dict[str, Any],
) -> str:
    return (
        "You are a learned reward model for WTS traffic safety captions.\n"
        "Judge whether the candidate caption is official-style, fact-consistent, phase-consistent, and metric-friendly.\n"
        "Answer only good or bad.\n\n"
        f"Scenario type: {scenario_type}\n"
        f"View: {view}\n"
        f"Phase label: {phase}\n\n"
        f"{fact_context.strip()}\n\n"
        f"{_format_candidate(candidate_name, candidate)}\n\n"
        "Decision:"
    )


def _training_candidates(target: dict[str, str], same_phase_pool: list[dict[str, str]]) -> list[tuple[str, dict[str, str], str]]:
    candidates = [
        ("gold", target, "good"),
        ("shuffled", {
            "caption_pedestrian": _shuffle_sentences(target["caption_pedestrian"]),
            "caption_vehicle": _shuffle_sentences(target["caption_vehicle"]),
        }, "bad"),
        ("front_trimmed", {
            "caption_pedestrian": _first_sentences(target["caption_pedestrian"], 3),
            "caption_vehicle": _first_sentences(target["caption_vehicle"], 3),
        }, "bad"),
        ("context_trimmed", {
            "caption_pedestrian": _drop_context_sentences(target["caption_pedestrian"]),
            "caption_vehicle": _drop_context_sentences(target["caption_vehicle"]),
        }, "bad"),
        ("contradicted", _inject_simple_contradiction(target), "bad"),
    ]
    for other in same_phase_pool:
        if other is not target and other != target:
            candidates.append(("wrong_scenario_same_phase", other, "bad"))
            break
    return candidates


def _target_from_phase(phase: dict[str, Any]) -> dict[str, str]:
    return {
        "caption_pedestrian": str(phase.get("caption_pedestrian", "")).strip(),
        "caption_vehicle": str(phase.get("caption_vehicle", "")).strip(),
    }


def _inject_simple_contradiction(row: dict[str, str]) -> dict[str, str]:
    text = json.dumps(row)
    swaps = [
        ("male", "female"),
        ("female", "male"),
        ("20s", "70s"),
        ("30s", "60s"),
        ("clear", "cloudy"),
        ("cloudy", "clear"),
        ("bright", "dim"),
        ("dim", "bright"),
        ("close", "far"),
        ("far", "close"),
    ]
    for src, dst in swaps:
        if re.search(rf"\b{re.escape(src)}\b", text, flags=re.IGNORECASE):
            return {
                key: re.sub(rf"\b{re.escape(src)}\b", dst, value, count=1, flags=re.IGNORECASE)
                for key, value in row.items()
            }
    return {
        "caption_pedestrian": row["caption_pedestrian"] + " The pedestrian was in his 70s.",
        "caption_vehicle": row["caption_vehicle"],
    }


def _read_prediction_rows(path: str | Path) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(str(obj.get("predict", obj.get("prediction", obj.get("output", ""))) if isinstance(obj, dict) else obj))
    return rows


def _label_score(text: str) -> float:
    lower = text.strip().lower()
    if re.search(r"\bgood\b", lower) and not re.search(r"\bbad\b", lower[:20]):
        return 1.0
    if re.search(r"\bbad\b", lower):
        return -1.0
    return 0.0


def _candidate_by_idx_or_phase(data: dict[str, list[dict[str, Any]]], scenario_id: str, phase: str, idx: int) -> dict[str, Any] | None:
    rows = data.get(scenario_id)
    if not isinstance(rows, list):
        return None
    for row_idx, row in enumerate(rows):
        if isinstance(row, dict) and _phase_label(row, row_idx) == phase:
            return row
    if idx < len(rows) and isinstance(rows[idx], dict):
        return rows[idx]
    return None


def _attribute_lock_penalty(text: str, facts: dict[str, Any], *, phase: str) -> float:
    flat = _flatten_fact_map(facts.get("global", {}))
    flat.update(_flatten_fact_map(facts.get("phases", {}).get(phase, {}), prefix="phase"))
    lower = text.lower()
    penalty = 0.0
    if _contradicts_age(lower, flat):
        penalty -= 2.5
    if _contradicts_gender(lower, flat):
        penalty -= 2.5
    if _contradicts_weather_or_brightness(lower, flat):
        penalty -= 1.5
    if _contradicts_distance(lower, flat):
        penalty -= 1.0
    return penalty


def _phase_transition_penalty(row: dict[str, Any], *, phase: str) -> float:
    text = _row_text(row).lower()
    cues = PHASE_CUES.get(str(phase), ())
    if not cues:
        return 0.0
    hits = sum(1 for cue in cues if cue in text)
    penalty = 0.0 if hits else -0.6
    if str(phase) in {"0", "1"} and any(term in text for term in ("collision occurred", "collided with")):
        penalty -= 1.5
    return penalty


def _flatten_fact_map(value: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_fact_map(child, dotted))
    elif str(value).strip():
        out[prefix] = str(value).strip().lower()
    return out


def _contradicts_age(text: str, facts: dict[str, str]) -> bool:
    expected = {v for k, v in facts.items() if "age" in k and re.search(r"\b[1-9]0s\b", v)}
    if not expected:
        return False
    present = set(re.findall(r"\b[1-9]0s\b", text))
    allowed = set().union(*(set(re.findall(r"\b[1-9]0s\b", value)) for value in expected))
    return bool(present and allowed and present - allowed)


def _contradicts_gender(text: str, facts: dict[str, str]) -> bool:
    expected = " ".join(v for k, v in facts.items() if "gender" in k)
    if "male" in expected and "female" not in expected:
        return bool(re.search(r"\b(female|woman|girl)\b", text))
    if "female" in expected:
        return bool(re.search(r"\b(male|man|boy)\b", text))
    return False


def _contradicts_weather_or_brightness(text: str, facts: dict[str, str]) -> bool:
    weather = " ".join(v for k, v in facts.items() if "weather" in k or "brightness" in k)
    checks = [("clear", "cloudy"), ("cloudy", "clear"), ("bright", "dim"), ("dim", "bright")]
    return any(src in weather and dst in text and src not in text for src, dst in checks)


def _contradicts_distance(text: str, facts: dict[str, str]) -> bool:
    distance = " ".join(v for k, v in facts.items() if "distance" in k)
    if "close" in distance or "near" in distance:
        return " far " in f" {text} "
    if "far" in distance:
        return " close " in f" {text} " or " near " in f" {text} "
    return False


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "")).strip() for key in CAPTION_KEYS)


def _normalize_output_row(row: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
    return {
        "labels": row.get("labels", fallback.get("labels", [])),
        "caption_pedestrian": str(row.get("caption_pedestrian", fallback.get("caption_pedestrian", ""))).strip(),
        "caption_vehicle": str(row.get("caption_vehicle", fallback.get("caption_vehicle", ""))).strip(),
    }


def _maybe_write_dataset_info(output: str | Path, dataset_info_output: str | Path | None, dataset_name: str | None) -> None:
    if dataset_info_output and dataset_name:
        write_llamafactory_dataset_info(dataset_info_output, dataset_name=dataset_name, file_name=Path(output).name)
    elif dataset_info_output or dataset_name:
        raise ValueError("--dataset-info-output and --dataset-name must be provided together.")
