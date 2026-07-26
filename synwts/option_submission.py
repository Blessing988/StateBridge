"""Public-test VQA option scoring exports and score-based submissions."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from .io import read_json, read_jsonl, write_json
from .submission import _load_vqa_submission, _parse_vqa_letter


OPTION_LINE_RE = re.compile(r"^\s*([a-e])\.\s*(.+?)\s*$", flags=re.IGNORECASE)


def export_vqa_option_candidates_from_inference_dataset(
    *,
    dataset: str | Path,
    output: str | Path,
    prompt_variant: str = "base",
) -> list[dict[str, Any]]:
    """Create option-scoring candidates from a LLaMA-Factory VQA inference file."""

    if prompt_variant not in {"base", "direct", "evidence", "anti-prior"}:
        raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")

    rows = read_json(dataset)
    if not isinstance(rows, list):
        raise ValueError("Inference dataset must be a JSON list.")

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        instruction = _apply_prompt_variant(str(row.get("instruction", "")), prompt_variant)
        metadata = dict(row.get("metadata", {}))
        qid = str(metadata.get("vqa_id", "")).strip()
        if not qid:
            raise ValueError(f"VQA row {idx} is missing metadata.vqa_id.")
        if qid in seen_ids:
            raise ValueError(f"Duplicate VQA id in inference dataset: {qid}")
        seen_ids.add(qid)

        options = _parse_options_from_instruction(instruction)
        candidate_metadata = dict(metadata)
        candidate_metadata["task"] = "vqa_option_scoring"
        candidate_metadata["prompt_variant"] = prompt_variant
        if "question" not in candidate_metadata:
            question = _parse_question_from_instruction(instruction)
            if question:
                candidate_metadata["question"] = question
        candidate = {
            "instruction": instruction,
            "input": str(row.get("input", "")),
            "options": options,
            "videos": list(row.get("videos", [])),
            "metadata": candidate_metadata,
        }
        correct = str(row.get("correct", "")).strip().lower()
        if correct in options:
            candidate["correct"] = correct
        candidates.append(candidate)

    write_json(output, candidates)
    return candidates


def _apply_prompt_variant(instruction: str, variant: str) -> str:
    if variant == "base":
        return instruction

    lines = instruction.splitlines()
    first_video_idx = next((idx for idx, line in enumerate(lines) if "<video>" in line), len(lines))
    first_question_idx = next((idx for idx, line in enumerate(lines) if line.strip().startswith("Question:")), len(lines))
    split_idx = min(first_video_idx, first_question_idx)
    tail = lines[split_idx:]

    scenario_lines: list[str] = []
    for line in lines[:split_idx]:
        stripped = line.strip()
        if stripped.startswith(("Scenario type:", "Scope:", "Phase label:", "View:")):
            scenario_lines.append(line)
        if stripped.endswith("question") or stripped.startswith("Scenario-level"):
            scenario_lines.append(line)

    if variant == "direct":
        header = [
            "Answer the traffic-safety multiple-choice question from the videos.",
            "Return exactly one lowercase option letter.",
        ]
    elif variant == "evidence":
        header = [
            "Choose the answer best supported by visible traffic-scene evidence.",
            "Compare every option against the video evidence before deciding.",
            "Return exactly one lowercase option letter.",
        ]
    else:
        header = [
            "Answer only from visible evidence, not demographic or traffic priors.",
            "If options are visually similar, choose the option with strongest direct evidence.",
            "Return exactly one lowercase option letter.",
        ]

    compact: list[str] = [*header, ""]
    compact.extend(dict.fromkeys(scenario_lines))
    if scenario_lines:
        compact.append("")
    compact.extend(tail)
    text = "\n".join(compact).strip()
    if "Question:" not in text or "Options:" not in text:
        raise ValueError(f"Prompt variant removed required fields: {variant}")
    return text


def assemble_vqa_submission_from_option_scores(
    *,
    candidates_path: str | Path,
    scores_path: Iterable[str | Path],
    output: str | Path,
    weights: Iterable[float] | None = None,
    normalization: str = "center",
    fallback: str | Path | None = None,
    report_output: str | Path | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Combine option-score files into an official VQA submission JSON."""

    if normalization not in {"none", "center", "zscore"}:
        raise ValueError(f"Unsupported normalization: {normalization}")

    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("Candidate file must be a JSON list.")
    score_paths = [Path(path) for path in scores_path]
    if not score_paths:
        raise ValueError("At least one score file is required.")
    raw_weights = list(weights) if weights is not None else [1.0] * len(score_paths)
    if len(raw_weights) != len(score_paths):
        raise ValueError("--weights count must match --scores count.")
    if any(weight < 0 or not math.isfinite(weight) for weight in raw_weights):
        raise ValueError("Weights must be non-negative finite numbers.")
    if not any(weight > 0 for weight in raw_weights):
        raise ValueError("At least one weight must be positive.")

    score_sets = [_load_score_map(path) for path in score_paths]
    fallback_map = _load_vqa_submission(fallback) if fallback else {}

    submission: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    missing_scores: list[str] = []
    invalid_scores: list[dict[str, str]] = []
    option_vote_counts: Counter[str] = Counter()

    for candidate in candidates:
        qid = str(candidate.get("metadata", {}).get("vqa_id", "")).strip()
        if not qid:
            raise ValueError("Candidate is missing metadata.vqa_id.")
        options = {
            str(letter).lower(): str(text)
            for letter, text in candidate.get("options", {}).items()
        }
        if not options:
            raise ValueError(f"Candidate {qid} has no options.")

        combined: defaultdict[str, float] = defaultdict(float)
        used_sources = 0
        for path, score_map, weight in zip(score_paths, score_sets, raw_weights):
            if weight <= 0:
                continue
            row = score_map.get(qid)
            if row is None:
                continue
            scores = row.get("scores")
            if not isinstance(scores, dict):
                invalid_scores.append({"vqa_id": qid, "path": str(path), "error": "missing_scores"})
                continue
            try:
                normalized = _normalize_scores(scores, options=options, mode=normalization)
            except ValueError as exc:
                invalid_scores.append({"vqa_id": qid, "path": str(path), "error": str(exc)})
                continue
            for letter, value in normalized.items():
                combined[letter] += weight * value
            used_sources += 1

        if combined:
            answer = min(combined, key=lambda letter: (-combined[letter], letter))
            source_counts["scores"] += 1
            option_vote_counts[answer] += 1
        elif qid in fallback_map:
            answer = _parse_vqa_letter(fallback_map[qid])
            source_counts["fallback"] += 1
        else:
            missing_scores.append(qid)
            answer = sorted(options)[0]
            source_counts["default_first_option"] += 1
        submission.append({"id": qid, "correct": answer})

    write_json(output, submission)
    report = {
        "total_candidates": len(candidates),
        "score_files": [str(path) for path in score_paths],
        "weights": raw_weights,
        "normalization": normalization,
        "source_counts": dict(source_counts),
        "missing_scores": len(missing_scores),
        "invalid_scores": len(invalid_scores),
        "option_distribution": dict(sorted(option_vote_counts.items())),
        "missing_vqa_ids": missing_scores[:50],
        "invalid_score_rows": invalid_scores[:50],
    }
    if fallback:
        fallback_submission = _load_vqa_submission(fallback)
        changed = sum(
            row["correct"] != _parse_vqa_letter(fallback_submission.get(row["id"], ""))
            for row in submission
            if row["id"] in fallback_submission
        )
        report["fallback"] = str(fallback)
        report["changed_from_fallback"] = changed
    if report_output:
        write_json(report_output, report)
    return submission, report


def _parse_options_from_instruction(instruction: str) -> dict[str, str]:
    options: dict[str, str] = {}
    in_options = False
    for line in instruction.splitlines():
        if line.strip().lower() == "options:":
            in_options = True
            continue
        if in_options and line.strip().lower().startswith("return only"):
            break
        match = OPTION_LINE_RE.match(line)
        if match:
            options[match.group(1).lower()] = match.group(2).strip()
    if len(options) < 2:
        raise ValueError("Could not parse at least two options from instruction.")
    return options


def _parse_question_from_instruction(instruction: str) -> str:
    for line in instruction.splitlines():
        if line.strip().lower().startswith("question:"):
            return line.split(":", 1)[1].strip()
    return ""


def _load_score_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path) if path.suffix.lower() == ".jsonl" else read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("rows", rows.get("scores", []))
    if not isinstance(rows, list):
        raise ValueError(f"Score file must be a JSON/JSONL list: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("vqa_id", row.get("id", ""))).strip()
        if not qid:
            continue
        if qid in by_id and "scores" in row and "scores" in by_id[qid]:
            raise ValueError(f"Duplicate score row for VQA id {qid}: {path}")
        if "scores" in row or qid not in by_id:
            by_id[qid] = row
    return by_id


def _normalize_scores(scores: dict[str, Any], *, options: dict[str, str], mode: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for letter in options:
        if letter not in scores:
            raise ValueError(f"missing option score: {letter}")
        value = float(scores[letter])
        if not math.isfinite(value):
            raise ValueError(f"non-finite option score: {letter}")
        values[letter] = value
    if mode == "none":
        return values
    avg = mean(values.values())
    centered = {letter: value - avg for letter, value in values.items()}
    if mode == "center":
        return centered
    std = pstdev(centered.values())
    if std <= 1e-12:
        return centered
    return {letter: value / std for letter, value in centered.items()}
