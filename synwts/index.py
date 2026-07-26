"""Build a portable dataset index for SynWTS."""

from __future__ import annotations

from pathlib import Path

from .io import as_path_string, find_files, write_jsonl
from .schema import ROLES, VIEWS, VQA_SCOPES, ScenarioRecord


def build_index(
    dataset_root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    absolute_paths: bool = True,
) -> list[ScenarioRecord]:
    root = Path(dataset_root)
    records: list[ScenarioRecord] = []
    for split in splits:
        scenario_ids = _collect_scenario_ids(root, split)
        for scenario_type, scenario_id in sorted(scenario_ids):
            records.append(_build_record(root, split, scenario_type, scenario_id, absolute_paths))
    return records


def write_index(records: list[ScenarioRecord], output: str | Path) -> None:
    write_jsonl(output, [record.to_dict() for record in records])


def _collect_scenario_ids(root: Path, split: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    base_candidates = [
        root / "videos" / split,
        root / "annotations" / "caption" / split,
        root / "annotations" / "vqa" / split,
    ]
    for base in base_candidates:
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if child.name == "normal_trimmed":
                for event_dir in child.iterdir():
                    if event_dir.is_dir():
                        found.add(("normal_trimmed", event_dir.name))
            else:
                found.add(("event", child.name))

    bbox_base = root / "annotations" / "bbox_annotated"
    for role in ROLES:
        role_split = bbox_base / role / split
        if not role_split.exists():
            continue
        for child in role_split.iterdir():
            if not child.is_dir():
                continue
            if child.name == "normal_trimmed":
                for event_dir in child.iterdir():
                    if event_dir.is_dir():
                        found.add(("normal_trimmed", event_dir.name))
            else:
                found.add(("event", child.name))
    return found


def _scenario_base(root: Path, kind: str, split: str, scenario_type: str, scenario_id: str) -> Path:
    if scenario_type == "normal_trimmed":
        return root / kind / split / "normal_trimmed" / scenario_id
    return root / kind / split / scenario_id


def _annotation_base(root: Path, group: str, split: str, scenario_type: str, scenario_id: str) -> Path:
    if scenario_type == "normal_trimmed":
        return root / "annotations" / group / split / "normal_trimmed" / scenario_id
    return root / "annotations" / group / split / scenario_id


def _bbox_base(root: Path, role: str, split: str, scenario_type: str, scenario_id: str) -> Path:
    if scenario_type == "normal_trimmed":
        return root / "annotations" / "bbox_annotated" / role / split / "normal_trimmed" / scenario_id
    return root / "annotations" / "bbox_annotated" / role / split / scenario_id


def _build_record(
    root: Path,
    split: str,
    scenario_type: str,
    scenario_id: str,
    absolute_paths: bool,
) -> ScenarioRecord:
    videos: dict[str, list[str]] = {}
    video_base = _scenario_base(root, "videos", split, scenario_type, scenario_id)
    for view in VIEWS:
        videos[view] = [
            as_path_string(path, root=root, absolute=absolute_paths)
            for path in find_files(video_base / view, ".mp4")
        ]

    caption_files: dict[str, str] = {}
    caption_base = _annotation_base(root, "caption", split, scenario_type, scenario_id)
    for view in VIEWS:
        files = find_files(caption_base / view, ".json")
        if files:
            caption_files[view] = as_path_string(files[0], root=root, absolute=absolute_paths)

    vqa_files: dict[str, str] = {}
    vqa_base = _annotation_base(root, "vqa", split, scenario_type, scenario_id)
    for scope in VQA_SCOPES:
        files = find_files(vqa_base / scope, ".json")
        if files:
            vqa_files[scope] = as_path_string(files[0], root=root, absolute=absolute_paths)

    bbox_files: dict[str, dict[str, list[str]]] = {}
    for role in ROLES:
        role_files: dict[str, list[str]] = {}
        role_base = _bbox_base(root, role, split, scenario_type, scenario_id)
        for view in VIEWS:
            role_files[view] = [
                as_path_string(path, root=root, absolute=absolute_paths)
                for path in find_files(role_base / view, ".json")
            ]
        bbox_files[role] = role_files

    return ScenarioRecord(
        split=split,
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        videos=videos,
        caption_files=caption_files,
        vqa_files=vqa_files,
        bbox_files=bbox_files,
    )

