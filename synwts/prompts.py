"""Prompt templates for SynWTS fine-tuning exports."""

from __future__ import annotations

from .schema import PHASE_NUMBER_TO_NAME


CAPTION_SYSTEM = (
    "You are a traffic safety video-language model. Generate grounded, "
    "fine-grained captions only from the visual evidence. Preserve the "
    "traffic-safety checklist style: location, attention, behavior, and context."
)

VQA_SYSTEM = (
    "You are a traffic safety visual question answering model. Answer the "
    "multiple-choice question from the visual evidence and return only the "
    "option letter."
)


def phase_caption_instruction(
    *,
    view: str,
    scenario_type: str,
    phase: str,
    media_count: int,
    bbox_context: str = "",
    visual_context: str = "",
) -> str:
    phase_name = PHASE_NUMBER_TO_NAME.get(phase, phase)
    tags = "\n".join("<video>" for _ in range(media_count))
    prompt = f"""{CAPTION_SYSTEM}

Scenario type: {scenario_type}
View: {view}
Phase label: {phase} ({phase_name})

{tags}

{visual_context}

{bbox_context}

Return JSON with exactly these keys:
- labels: list containing the phase label
- caption_pedestrian: one detailed sentence group about the pedestrian
- caption_vehicle: one detailed sentence group about the vehicle
"""
    return prompt.strip()


def vqa_instruction(
    *,
    scope: str,
    scenario_type: str,
    phase: str | None,
    question: str,
    options: dict[str, str],
    media_count: int,
    bbox_context: str = "",
    visual_context: str = "",
) -> str:
    tags = "\n".join("<video>" for _ in range(media_count))
    phase_line = "Scenario-level environment question" if phase is None else f"Phase label: {phase}"
    option_lines = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
    prompt = f"""{VQA_SYSTEM}

Scenario type: {scenario_type}
Scope: {scope}
{phase_line}

{tags}

{visual_context}

{bbox_context}

Question: {question}
Options:
{option_lines}

Return only the correct option letter."""
    return prompt.strip()


def make_visual_media_context(videos: list[str]) -> str:
    variants = _visual_variants(videos)
    if not variants:
        return ""
    lines = [
        "Visual grounding note: some videos are derived grounding views, not separate events.",
    ]
    if "overlay" in variants:
        lines.append(
            "Overlay clips draw target tracks: pedestrian boxes use yellow->orange->red for first/mid/last; "
            "vehicle boxes use cyan->blue->lime for first/mid/last; faint white boxes mark each track's phase union."
        )
    if "interaction_crop" in variants:
        lines.append(
            "Interaction crop clips zoom around the target pedestrian-vehicle region for this phase."
        )
    if "pedestrian_crop" in variants:
        lines.append("Pedestrian crop clips zoom around the target pedestrian.")
    if "vehicle_crop" in variants:
        lines.append("Vehicle crop clips zoom around the target vehicle.")
    lines.append(
        "Use these overlays and crops only as grounding aids; describe the real traffic scene, not the colored graphics."
    )
    return "\n".join(lines)


def _visual_variants(videos: list[str]) -> set[str]:
    variants: set[str] = set()
    for video in videos:
        name = str(video)
        if "_overlay." in name:
            variants.add("overlay")
        if "_interaction_crop." in name:
            variants.add("interaction_crop")
        if "_pedestrian_crop." in name:
            variants.add("pedestrian_crop")
        if "_vehicle_crop." in name:
            variants.add("vehicle_crop")
    return variants
