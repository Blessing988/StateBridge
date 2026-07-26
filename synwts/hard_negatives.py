"""Hard-negative selection for multimodal VQA preference training."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Any, Iterable

from .exporters import write_llamafactory_preference_dataset_info
from .io import read_json, read_jsonl, write_json


VALID_SELECTIONS = {"all", "errors", "margin", "errors_and_margin"}
VALID_BALANCE_FIELDS = {"question_type", "scope", "phase", "correct", "rejected", "scenario_type"}


def build_hard_negative_preference_dataset(
    *,
    candidates_path: str | Path,
    scores_path: str | Path | Iterable[str | Path],
    output: str | Path,
    report_output: str | Path | None = None,
    dataset_info_output: str | Path | None = None,
    dataset_name: str = "synwts_vqa_hard_negative",
    selection: str = "errors_and_margin",
    max_gold_margin: float | None = 2.0,
    balance_fields: Iterable[str] = ("question_type", "scope", "phase", "correct"),
    max_per_group: int | None = None,
    max_rows: int | None = None,
    response_mode: str = "letter",
    remap_option_letters: bool = False,
    allow_missing_scores: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if selection not in VALID_SELECTIONS:
        raise ValueError(f"Unsupported selection: {selection}")
    if selection in {"margin", "errors_and_margin"} and max_gold_margin is None:
        raise ValueError(f"selection={selection} requires max_gold_margin")
    if response_mode not in {"letter", "letter_text"}:
        raise ValueError(f"Unsupported response_mode: {response_mode}")

    fields = tuple(field.strip() for field in balance_fields if field.strip())
    unknown_fields = sorted(set(fields) - VALID_BALANCE_FIELDS)
    if unknown_fields:
        raise ValueError(f"Unsupported balance fields: {', '.join(unknown_fields)}")

    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("Candidate file must be a JSON list.")
    scores = _load_scores(scores_path)

    evaluated: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    invalid_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        qid = str(candidate.get("metadata", {}).get("vqa_id", "")).strip()
        if not qid:
            invalid_rows.append({"vqa_id": "", "error": "candidate_missing_vqa_id"})
            continue
        if qid in seen_ids:
            raise ValueError(f"Duplicate candidate vqa_id: {qid}")
        seen_ids.add(qid)
        score_row = scores.get(qid)
        if score_row is None:
            missing_ids.append(qid)
            continue
        try:
            evaluated.append(_evaluate_candidate(candidate, score_row))
        except ValueError as exc:
            invalid_rows.append({"vqa_id": qid, "error": str(exc)})

    if (missing_ids or invalid_rows) and not allow_missing_scores:
        detail = missing_ids[0] if missing_ids else invalid_rows[0]
        raise ValueError(
            "Incomplete or invalid option scores. "
            f"missing={len(missing_ids)} invalid={len(invalid_rows)} first={detail}"
        )

    eligible = [
        row
        for row in evaluated
        if _is_selected(row, selection=selection, max_gold_margin=max_gold_margin)
    ]
    eligible.sort(key=_difficulty_sort_key)
    selected = _balanced_select(
        eligible,
        fields=fields,
        max_per_group=max_per_group,
        max_rows=max_rows,
    )
    if remap_option_letters:
        selected = _remap_option_letters(selected)
    preference_rows = [
        _preference_row(row, response_mode=response_mode)
        for row in selected
    ]
    write_json(output, preference_rows)
    if dataset_info_output:
        write_llamafactory_preference_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=Path(output).name,
        )

    report = _build_report(
        candidates=candidates,
        evaluated=evaluated,
        eligible=eligible,
        selected=selected,
        missing_ids=missing_ids,
        invalid_rows=invalid_rows,
        selection=selection,
        max_gold_margin=max_gold_margin,
        balance_fields=fields,
        max_per_group=max_per_group,
        max_rows=max_rows,
        remap_option_letters=remap_option_letters,
    )
    if report_output:
        write_json(report_output, report)
    return preference_rows, report


def select_hard_negative(
    *,
    options: dict[str, str],
    correct: str,
    scores: dict[str, float],
) -> dict[str, Any]:
    letters = sorted(options)
    correct = correct.strip().lower()
    if correct not in options:
        raise ValueError(f"Correct option is absent: {correct}")
    missing = [letter for letter in letters if letter not in scores]
    if missing:
        raise ValueError(f"Missing scores for options: {','.join(missing)}")
    normalized_scores: dict[str, float] = {}
    for letter in letters:
        value = float(scores[letter])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite score for option {letter}")
        normalized_scores[letter] = value

    wrong_letters = [letter for letter in letters if letter != correct]
    if not wrong_letters:
        raise ValueError("Question has no incorrect options.")
    rejected = min(wrong_letters, key=lambda letter: (-normalized_scores[letter], letter))
    predicted = min(letters, key=lambda letter: (-normalized_scores[letter], letter))
    gold_score = normalized_scores[correct]
    rejected_score = normalized_scores[rejected]
    return {
        "correct": correct,
        "rejected": rejected,
        "predicted": predicted,
        "is_model_error": predicted != correct,
        "gold_score": gold_score,
        "hard_negative_score": rejected_score,
        "gold_margin": gold_score - rejected_score,
        "scores": normalized_scores,
    }


def _load_scores(paths: str | Path | Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    by_id: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        rows = read_jsonl(path) if path.suffix.lower() == ".jsonl" else read_json(path)
        if isinstance(rows, dict):
            rows = rows.get("rows", rows.get("scores", []))
        if not isinstance(rows, list):
            raise ValueError("Score file must be a JSON/JSONL list of score rows.")
        for row in rows:
            qid = str(row.get("vqa_id", row.get("id", ""))).strip()
            if not qid:
                raise ValueError("Score row is missing vqa_id.")
            if qid in by_id:
                existing = by_id[qid]
                if "scores" in row and "scores" not in existing:
                    by_id[qid] = row
                    continue
                if "scores" not in row:
                    continue
                raise ValueError(f"Duplicate valid score vqa_id: {qid}")
            by_id[qid] = row
    return by_id


def _evaluate_candidate(candidate: dict[str, Any], score_row: dict[str, Any]) -> dict[str, Any]:
    options = {
        str(letter).strip().lower(): str(text)
        for letter, text in candidate.get("options", {}).items()
    }
    score_values = score_row.get("scores")
    if not isinstance(score_values, dict):
        raise ValueError("score row has no scores object")
    scored_correct = str(score_row.get("correct", "")).strip().lower()
    candidate_correct = str(candidate.get("correct", "")).strip().lower()
    if scored_correct and scored_correct != candidate_correct:
        raise ValueError(
            f"score/candidate correct mismatch: {scored_correct} != {candidate_correct}"
        )
    result = select_hard_negative(
        options=options,
        correct=candidate_correct,
        scores=score_values,
    )
    return {"candidate": candidate, "score_row": score_row, **result}


def _is_selected(
    row: dict[str, Any],
    *,
    selection: str,
    max_gold_margin: float | None,
) -> bool:
    if selection == "all":
        return True
    if selection == "errors":
        return bool(row["is_model_error"])
    within_margin = row["gold_margin"] <= float(max_gold_margin)
    if selection == "margin":
        return within_margin
    return bool(row["is_model_error"]) or within_margin


def _difficulty_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata = row["candidate"].get("metadata", {})
    return (
        float(row["gold_margin"]),
        str(metadata.get("question_type", "")),
        str(metadata.get("vqa_id", "")),
    )


def _balanced_select(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    max_per_group: int | None,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if not fields:
        limit = max_rows if max_rows and max_rows > 0 else len(rows)
        return rows[:limit]

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row["candidate"].get("metadata", {})
        key = tuple(
            str(
                row[field]
                if field in {"correct", "rejected"}
                else metadata.get(field, "")
            )
            for field in fields
        )
        if max_per_group and max_per_group > 0 and len(grouped[key]) >= max_per_group:
            continue
        grouped[key].append(row)

    selected: list[dict[str, Any]] = []
    group_keys = sorted(grouped)
    index = 0
    limit = max_rows if max_rows and max_rows > 0 else sum(len(rows) for rows in grouped.values())
    while len(selected) < limit:
        added = False
        for key in group_keys:
            group = grouped[key]
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


def _preference_row(row: dict[str, Any], *, response_mode: str) -> dict[str, Any]:
    candidate = row["candidate"]
    options = candidate["options"]
    metadata = dict(candidate.get("metadata", {}))
    metadata.update(
        {
            "task": "vqa_hard_negative_preference",
            "correct": row["correct"],
            "rejected": row["rejected"],
            "predicted": row["predicted"],
            "is_model_error": row["is_model_error"],
            "gold_score": row["gold_score"],
            "hard_negative_score": row["hard_negative_score"],
            "gold_margin": row["gold_margin"],
            "option_scores": row["scores"],
            "scoring_model": row["score_row"].get("model_name_or_path"),
            "scoring_adapter": row["score_row"].get("adapter_name_or_path"),
        }
    )
    if row.get("option_letter_remap"):
        metadata["option_letter_remap"] = row["option_letter_remap"]
        metadata["original_correct"] = row.get("original_correct")
        metadata["original_rejected"] = row.get("original_rejected")
    return {
        "instruction": candidate["instruction"],
        "input": candidate.get("input", ""),
        "chosen": _format_answer(row["correct"], options, response_mode),
        "rejected": _format_answer(row["rejected"], options, response_mode),
        "videos": candidate.get("videos", []),
        "metadata": metadata,
    }


def _format_answer(letter: str, options: dict[str, str], response_mode: str) -> str:
    if response_mode == "letter_text":
        return f"{letter}. {options.get(letter, '').strip()}".strip()
    return letter


def _remap_option_letters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Balance answer-letter priors without changing option semantics.

    DPO sees short responses such as "a" and "d", so a skewed chosen/rejected
    letter distribution can dominate the visual preference signal. This function
    keeps the same correct and hard-negative option texts, but rewrites option
    labels in the prompt so chosen/rejected letters are round-robin balanced.
    """

    if not rows:
        return []
    remapped: list[dict[str, Any]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        letters = tuple(sorted(str(letter) for letter in row["candidate"]["options"]))
        target_pairs = [
            (chosen, rejected)
            for chosen in letters
            for rejected in letters
            if rejected != chosen
        ]
        target_chosen, target_rejected = min(
            target_pairs,
            key=lambda pair: (pair_counts[pair], pair[0], pair[1]),
        )
        pair_counts[(target_chosen, target_rejected)] += 1
        remapped.append(
            _remap_row_option_letters(
                row,
                target_chosen=target_chosen,
                target_rejected=target_rejected,
            )
        )
    return remapped


def _remap_row_option_letters(
    row: dict[str, Any],
    *,
    target_chosen: str,
    target_rejected: str,
) -> dict[str, Any]:
    row = copy.deepcopy(row)
    candidate = row["candidate"]
    options = {str(letter): str(text) for letter, text in candidate["options"].items()}
    original_correct = str(row["correct"])
    original_rejected = str(row["rejected"])
    if target_chosen == target_rejected:
        raise ValueError("Target chosen/rejected letters must differ.")
    if original_correct == original_rejected:
        raise ValueError("Original chosen/rejected letters must differ.")
    letters = sorted(options)
    if target_chosen not in options or target_rejected not in options:
        raise ValueError("Target letters must be present in options.")

    old_to_new = {
        original_correct: target_chosen,
        original_rejected: target_rejected,
    }
    remaining_old = [
        letter for letter in letters if letter not in {original_correct, original_rejected}
    ]
    remaining_new = [
        letter for letter in letters if letter not in {target_chosen, target_rejected}
    ]
    for old_letter, new_letter in zip(remaining_old, remaining_new):
        old_to_new[old_letter] = new_letter

    new_options = {new_letter: options[old_letter] for old_letter, new_letter in old_to_new.items()}
    candidate["options"] = dict(sorted(new_options.items()))
    candidate["instruction"] = _replace_options_block(
        str(candidate["instruction"]),
        candidate["options"],
    )
    row["original_correct"] = original_correct
    row["original_rejected"] = original_rejected
    row["correct"] = target_chosen
    row["rejected"] = target_rejected
    row["option_letter_remap"] = dict(sorted(old_to_new.items()))
    return row


def _replace_options_block(instruction: str, options: dict[str, str]) -> str:
    options_block = "Options:\n" + "\n".join(
        f"{letter}. {options[letter]}" for letter in sorted(options)
    )
    updated, count = re.subn(
        r"Options:\n.*?\n\nReturn only",
        options_block + "\n\nReturn only",
        instruction,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Could not locate VQA options block for option-letter remapping.")
    return updated


def _build_report(
    *,
    candidates: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    missing_ids: list[str],
    invalid_rows: list[dict[str, str]],
    selection: str,
    max_gold_margin: float | None,
    balance_fields: tuple[str, ...],
    max_per_group: int | None,
    max_rows: int | None,
    remap_option_letters: bool,
) -> dict[str, Any]:
    margins = [float(row["gold_margin"]) for row in evaluated]
    selected_margins = [float(row["gold_margin"]) for row in selected]
    correct_count = sum(not row["is_model_error"] for row in evaluated)
    return {
        "ok": not missing_ids and not invalid_rows,
        "total_candidates": len(candidates),
        "scored": len(evaluated),
        "missing_scores": len(missing_ids),
        "invalid_scores": len(invalid_rows),
        "model_accuracy": correct_count / len(evaluated) if evaluated else 0.0,
        "model_errors": len(evaluated) - correct_count,
        "eligible": len(eligible),
        "selected": len(selected),
        "selected_model_errors": sum(row["is_model_error"] for row in selected),
        "selection": selection,
        "max_gold_margin": max_gold_margin,
        "balance_fields": list(balance_fields),
        "max_per_group": max_per_group,
        "max_rows": max_rows,
        "remap_option_letters": remap_option_letters,
        "margin_summary": _margin_summary(margins),
        "selected_margin_summary": _margin_summary(selected_margins),
        "evaluated_distribution": _distribution(evaluated),
        "selected_distribution": _distribution(selected),
        "scoring_sources": _scoring_sources(evaluated),
        "missing_vqa_ids": missing_ids[:50],
        "invalid_rows": invalid_rows[:50],
    }


def _margin_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field in ("question_type", "scope", "phase", "correct", "rejected"):
        counts = Counter()
        for row in rows:
            metadata = row["candidate"].get("metadata", {})
            value = row.get(field) if field in {"correct", "rejected"} else metadata.get(field)
            counts[str(value)] += 1
        result[field] = dict(sorted(counts.items()))
    return result


def _scoring_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        score_row = row["score_row"]
        source = (
            str(score_row.get("model_name_or_path", "")),
            str(score_row.get("adapter_name_or_path", "")),
        )
        counts[" | ".join(source)] += 1
    return dict(sorted(counts.items()))
