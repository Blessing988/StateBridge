"""Preference-data exports for verifier-style VQA training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from .bboxes import make_bbox_context
from .clips import load_phase_clip_map
from .exporters import apply_media_policy, write_llamafactory_preference_dataset_info
from .io import write_json
from .parsers import load_vqa_questions
from .prompts import make_visual_media_context, vqa_instruction
from .schema import ScenarioRecord
from .vqa_fusion import classify_vqa_question


def export_vqa_preference_llamafactory(
    records: Iterable[ScenarioRecord],
    output: str | Path,
    *,
    media_policy: str = "all",
    include_missing_media: bool = False,
    bbox_mode: str = "none",
    frame_width: int = 1920,
    frame_height: int = 1080,
    phase_clip_manifest: str | Path | None = None,
    negative_policy: str = "all",
    max_rejected_per_question: int | None = None,
    response_mode: str = "letter",
    seed: int = 13,
) -> list[dict]:
    """Export VQA labels as chosen/rejected preference pairs.

    Each pair keeps the same visual prompt and uses the ground-truth option as
    `chosen` and one wrong option as `rejected`. This is compatible with
    LLaMA-Factory ranking datasets for DPO/ORPO/SimPO.
    """

    if negative_policy not in {"all", "first", "random"}:
        raise ValueError(f"Unsupported negative_policy: {negative_policy}")
    if response_mode not in {"letter", "letter_text"}:
        raise ValueError(f"Unsupported response_mode: {response_mode}")

    rng = random.Random(seed)
    candidates = build_vqa_option_candidates(
        records,
        media_policy=media_policy,
        include_missing_media=include_missing_media,
        bbox_mode=bbox_mode,
        frame_width=frame_width,
        frame_height=frame_height,
        phase_clip_manifest=phase_clip_manifest,
    )
    rows: list[dict] = []
    for candidate in candidates:
        correct = candidate["correct"]
        options = candidate["options"]
        wrong_letters = [letter for letter in sorted(options) if letter != correct]
        wrong_letters = _select_negative_letters(
            wrong_letters,
            policy=negative_policy,
            max_count=max_rejected_per_question,
            rng=rng,
        )
        chosen = _format_answer(correct, options, response_mode)
        for pair_idx, rejected_letter in enumerate(wrong_letters):
            metadata = dict(candidate["metadata"])
            metadata.update(
                {
                    "task": "vqa_preference",
                    "rejected": rejected_letter,
                    "pair_idx": pair_idx,
                }
            )
            rows.append(
                {
                    "instruction": candidate["instruction"],
                    "input": candidate["input"],
                    "chosen": chosen,
                    "rejected": _format_answer(rejected_letter, options, response_mode),
                    "videos": candidate["videos"],
                    "metadata": metadata,
                }
            )

    write_json(output, rows)
    return rows


def export_vqa_option_candidates(
    records: Iterable[ScenarioRecord],
    output: str | Path,
    **kwargs,
) -> list[dict]:
    """Export one multimodal scoring record per VQA question."""

    rows = build_vqa_option_candidates(records, **kwargs)
    write_json(output, rows)
    return rows


def build_vqa_option_candidates(
    records: Iterable[ScenarioRecord],
    *,
    media_policy: str = "all",
    include_missing_media: bool = False,
    bbox_mode: str = "none",
    frame_width: int = 1920,
    frame_height: int = 1080,
    phase_clip_manifest: str | Path | None = None,
) -> list[dict]:
    """Build prompts and media once so every option can be scored consistently."""

    rows: list[dict] = []
    phase_clip_map = load_phase_clip_map(phase_clip_manifest) if phase_clip_manifest else {}
    for record in records:
        for scope, vqa_path in sorted(record.vqa_files.items()):
            view_for_media = "overhead_view" if scope == "environment" else scope
            videos = _select_videos(record, view_for_media, media_policy)
            if not videos and scope == "environment":
                videos = _select_videos(record, "vehicle_view", media_policy)
            if not videos and not include_missing_media:
                continue

            for question in load_vqa_questions(vqa_path, scope=scope, scenario_id=record.scenario_id):
                correct = str(question["correct"]).strip().lower()
                if correct not in question["options"]:
                    continue

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
                        "options": question["options"],
                        "correct": correct,
                        "videos": phase_videos,
                        "metadata": {
                            "task": "vqa_option_scoring",
                            "vqa_id": question["id"],
                            "split": record.split,
                            "scenario_id": record.scenario_id,
                            "scenario_type": record.scenario_type,
                            "scope": scope,
                            "phase": question["phase"],
                            "question_type": classify_vqa_question(question["question"], scope=scope),
                            "question": question["question"],
                        },
                    }
                )
    return rows


def write_vqa_preference_dataset_info(
    output: str | Path,
    *,
    dataset_name: str,
    file_name: str,
) -> None:
    write_llamafactory_preference_dataset_info(
        output,
        dataset_name=dataset_name,
        file_name=file_name,
    )


def _select_videos(record: ScenarioRecord, view: str, media_policy: str) -> list[str]:
    return apply_media_policy(record.videos.get(view, []), media_policy)


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


def _select_negative_letters(
    letters: list[str],
    *,
    policy: str,
    max_count: int | None,
    rng: random.Random,
) -> list[str]:
    if policy == "first":
        selected = letters[:1]
    elif policy == "random":
        selected = list(letters)
        rng.shuffle(selected)
    else:
        selected = list(letters)

    if max_count is not None and max_count > 0:
        selected = selected[:max_count]
    return selected


def _format_answer(letter: str, options: dict[str, str], response_mode: str) -> str:
    letter = letter.strip().lower()
    if response_mode == "letter_text":
        return f"{letter}. {options.get(letter, '').strip()}".strip()
    return letter
