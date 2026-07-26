"""Index WTS public test data for Track 2 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .clips import cut_video_clip
from .exporters import write_llamafactory_dataset_info
from .io import as_path_string, find_files, read_json, write_jsonl
from .prompts import make_visual_media_context, vqa_instruction
from .schema import ROLES, VIEWS, ScenarioRecord, normalize_phase_label
from .visual_clips import make_visual_clip_variants


def build_wts_public_test_index(
    test_root: str | Path,
    output: str | Path,
    *,
    absolute_paths: bool = True,
) -> list[ScenarioRecord]:
    """Build a ScenarioRecord index for the official public test layout.

    Expected root:
    test_data/
      SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/
      SubTask1-Caption/WTS_DATASET_PUBLIC_TEST_BBOX/
      SubTask2-VQA/

    The BDD_PC_5K external subset is intentionally ignored.
    """
    root = Path(test_root)
    caption_root = root / "SubTask1-Caption" / "WTS_DATASET_PUBLIC_TEST"
    bbox_root = root / "SubTask1-Caption" / "WTS_DATASET_PUBLIC_TEST_BBOX"

    video_by_name = _video_name_index(caption_root)
    bbox_by_scenario = _bbox_index(bbox_root, absolute_paths=absolute_paths)
    records: dict[str, ScenarioRecord] = {}

    for caption_file in _internal_json_files(caption_root / "annotations" / "caption"):
        parsed = _parse_caption_path(caption_file)
        if parsed is None:
            continue
        scenario_id, scenario_type, view = parsed
        record = records.setdefault(
            scenario_id,
            ScenarioRecord(
                split="test",
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                videos={"overhead_view": [], "vehicle_view": []},
                caption_files={},
                vqa_files={},
                bbox_files={role: {view_name: [] for view_name in VIEWS} for role in ROLES},
            ),
        )
        record.caption_files[view] = as_path_string(caption_file, absolute=absolute_paths)
        for video_name in _caption_video_names(caption_file, view):
            if video_name in video_by_name:
                record.videos.setdefault(view, []).append(as_path_string(video_by_name[video_name], absolute=absolute_paths))

    for scenario_id, record in records.items():
        if scenario_id in bbox_by_scenario:
            record.bbox_files = bbox_by_scenario[scenario_id]
        for view in VIEWS:
            record.videos[view] = sorted(set(record.videos.get(view, [])))

    ordered = [records[key] for key in sorted(records)]
    write_jsonl(output, [record.to_dict() for record in ordered])
    return ordered


def export_wts_public_vqa(
    *,
    test_root: str | Path,
    vqa_json: str | Path,
    output: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str = "wts_public_test_vqa",
    clip_output_root: str | Path | None = None,
    make_clips: bool = False,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    clip_mode: str = "mpeg4",
    bbox_mode: str = "none",
    frame_width: int = 1920,
    frame_height: int = 1080,
    max_videos_per_row: int | None = None,
    env_clip_duration: float | None = None,
    visual_variants: set[str] | None = None,
    visual_output_root: str | Path | None = None,
    visual_max_clips_per_row: int | None = None,
    visual_crop_padding: float = 0.35,
    visual_crop_size: int = 768,
    visual_max_tracks_per_role: int = 2,
) -> list[dict[str, Any]]:
    root = Path(test_root)
    video_by_name = {
        path.name: path
        for path in _internal_mp4_files(root / "SubTask1-Caption" / "WTS_DATASET_PUBLIC_TEST")
    }
    bbox_by_scenario = _bbox_index(
        root / "SubTask1-Caption" / "WTS_DATASET_PUBLIC_TEST_BBOX",
        absolute_paths=True,
    )
    items = read_json(vqa_json)
    rows: list[dict[str, Any]] = []

    for item_idx, item in enumerate(items):
        video_names = [str(name) for name in item.get("videos", [])]
        videos = _limit_paths(
            [video_by_name[name] for name in video_names if name in video_by_name],
            max_videos_per_row,
        )
        scenario_id = _scenario_from_video_names(video_names, fallback=f"item_{item_idx:05d}")
        scenario_type = "normal_trimmed" if any("normal" in name for name in video_names) else "event"

        if "event_phase" in item:
            scope = _infer_vqa_scope(item)
            for phase in item.get("event_phase", []):
                labels = phase.get("labels") or []
                phase_label = normalize_phase_label(labels[0] if labels else None)
                start = float(phase.get("start_time", 0.0))
                end = float(phase.get("end_time", start))
                media_paths = _phase_media_paths(
                    videos,
                    output_root=Path(clip_output_root) if clip_output_root else None,
                    scenario_id=scenario_id,
                    phase=phase_label,
                    start=start,
                    end=end,
                    make_clips=make_clips,
                    overwrite=overwrite,
                    ffmpeg_bin=ffmpeg_bin,
                    clip_mode=clip_mode,
                )
                if visual_variants:
                    bbox_files = bbox_by_scenario.get(scenario_id)
                    if bbox_files is not None:
                        media_paths = _visual_vqa_media_paths(
                            media_paths,
                            output_root=_vqa_visual_root(
                                output=output,
                                clip_output_root=clip_output_root,
                                visual_output_root=visual_output_root,
                            ),
                            bbox_files=bbox_files,
                            view="vehicle_view" if scope == "vehicle_view" else "overhead_view",
                            phase=phase_label,
                            variants=visual_variants,
                            make_clips=make_clips,
                            overwrite=overwrite,
                            ffmpeg_bin=ffmpeg_bin,
                            clip_mode=clip_mode,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            crop_padding=visual_crop_padding,
                            crop_size=visual_crop_size,
                            max_tracks_per_role=visual_max_tracks_per_role,
                            max_clips_per_row=visual_max_clips_per_row,
                        )
                for question in phase.get("conversations", []):
                    rows.append(
                        _vqa_inference_row(
                            question,
                            videos=media_paths,
                            scenario_id=scenario_id,
                            scenario_type=scenario_type,
                            scope=scope,
                            phase=phase_label,
                            bbox_context=_vqa_bbox_context(
                                bbox_by_scenario,
                                scenario_id=scenario_id,
                                scope=scope,
                                phase=phase_label,
                                bbox_mode=bbox_mode,
                                frame_width=frame_width,
                                frame_height=frame_height,
                            ),
                        )
                    )
        else:
            media_paths = _environment_media_paths(
                videos,
                output_root=Path(clip_output_root) if clip_output_root else None,
                scenario_id=scenario_id,
                duration=env_clip_duration,
                make_clips=make_clips,
                overwrite=overwrite,
                ffmpeg_bin=ffmpeg_bin,
                clip_mode=clip_mode,
            )
            for question in item.get("conversations", []):
                rows.append(
                    _vqa_inference_row(
                        question,
                        videos=media_paths,
                        scenario_id=scenario_id,
                        scenario_type=scenario_type,
                        scope="environment",
                        phase=None,
                        bbox_context="",
                    )
                )

    from .io import write_json

    write_json(output, rows)
    if dataset_info_output:
        write_llamafactory_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=Path(output).name,
        )
    return rows


def _limit_paths(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit <= 0:
        return paths
    return paths[:limit]


def _internal_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and "external" not in path.parts and "BDD_PC_5K" not in path.parts
    )


def _internal_mp4_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.mp4")
        if path.is_file() and "external" not in path.parts and "BDD_PC_5K" not in path.parts
    )


def _video_name_index(caption_root: Path) -> dict[str, Path]:
    return {path.name: path for path in _internal_mp4_files(caption_root / "videos")}


def _parse_caption_path(path: Path) -> tuple[str, str, str] | None:
    parts = path.parts
    if path.parent.name not in VIEWS:
        return None
    view = path.parent.name
    scenario_id = path.parent.parent.name
    scenario_type = "normal_trimmed" if "normal_trimmed" in parts else "event"
    return scenario_id, scenario_type, view


def _caption_video_names(caption_file: Path, view: str) -> list[str]:
    data = read_json(caption_file)
    if view == "overhead_view":
        return [str(item) for item in data.get("overhead_videos", [])]
    video = data.get("vehicle_view")
    return [str(video)] if video else []


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


def _infer_vqa_scope(item: dict[str, Any]) -> str:
    for phase in item.get("event_phase", []):
        for question in phase.get("conversations", []):
            q = str(question.get("question", "")).lower()
            if "vehicle's field of view" in q or "action taken by vehicle" in q:
                return "vehicle_view"
            if "position of the vehicle relative" in q or "relative distance of vehicle" in q:
                return "vehicle_view"
    return "overhead_view"


def _phase_media_paths(
    videos: list[Path],
    *,
    output_root: Path | None,
    scenario_id: str,
    phase: str,
    start: float,
    end: float,
    make_clips: bool,
    overwrite: bool,
    ffmpeg_bin: str,
    clip_mode: str,
) -> list[str]:
    if output_root is None:
        return [as_path_string(path, absolute=True) for path in videos]
    duration = max(end - start, 0.001)
    paths: list[str] = []
    for video in videos:
        clip_path = output_root / "vqa_clips" / scenario_id / f"{video.stem}_phase_{phase}.mp4"
        if make_clips:
            cut_video_clip(
                ffmpeg_bin=ffmpeg_bin,
                input_path=video,
                output_path=clip_path,
                start=start,
                duration=duration,
                overwrite=overwrite,
                clip_mode=clip_mode,
            )
        paths.append(as_path_string(clip_path, absolute=True))
    return paths


def _environment_media_paths(
    videos: list[Path],
    *,
    output_root: Path | None,
    scenario_id: str,
    duration: float | None,
    make_clips: bool,
    overwrite: bool,
    ffmpeg_bin: str,
    clip_mode: str,
) -> list[str]:
    if output_root is None or duration is None or duration <= 0:
        return [as_path_string(path, absolute=True) for path in videos]
    paths: list[str] = []
    for video in videos:
        clip_path = output_root / "vqa_env_clips" / scenario_id / f"{video.stem}_env.mp4"
        if make_clips:
            cut_video_clip(
                ffmpeg_bin=ffmpeg_bin,
                input_path=video,
                output_path=clip_path,
                start=0.0,
                duration=duration,
                overwrite=overwrite,
                clip_mode=clip_mode,
            )
        paths.append(as_path_string(clip_path, absolute=True))
    return paths


def _visual_vqa_media_paths(
    media_paths: list[str],
    *,
    output_root: Path,
    bbox_files: dict[str, dict[str, list[str]]],
    view: str,
    phase: str,
    variants: set[str],
    make_clips: bool,
    overwrite: bool,
    ffmpeg_bin: str,
    clip_mode: str,
    frame_width: int,
    frame_height: int,
    crop_padding: float,
    crop_size: int,
    max_tracks_per_role: int,
    max_clips_per_row: int | None,
) -> list[str]:
    visual_paths: list[str] = []
    for media_path in media_paths:
        clip_path = Path(media_path)
        camera_id = _camera_id_from_phase_clip(clip_path, phase=phase)
        media_view = _view_from_camera_id(camera_id, default=view)
        generated = make_visual_clip_variants(
            clip_path=clip_path,
            output_dir=output_root / "vqa_visual_clips" / camera_id,
            camera_id=camera_id,
            bbox_files=bbox_files,
            view=media_view,
            phase=phase,
            variants=variants,
            make_clips=make_clips,
            overwrite=overwrite,
            ffmpeg_bin=ffmpeg_bin,
            clip_mode=clip_mode,
            frame_width=frame_width,
            frame_height=frame_height,
            crop_padding=crop_padding,
            crop_size=crop_size,
            max_tracks_per_role=max_tracks_per_role,
            absolute_paths=True,
        )
        visual_paths.extend(str(row["clip_path"]) for row in generated)
        if max_clips_per_row and max_clips_per_row > 0 and len(visual_paths) >= max_clips_per_row:
            return visual_paths[:max_clips_per_row]
    return visual_paths or media_paths


def _vqa_visual_root(
    *,
    output: str | Path,
    clip_output_root: str | Path | None,
    visual_output_root: str | Path | None,
) -> Path:
    if visual_output_root:
        return Path(visual_output_root)
    if clip_output_root:
        return Path(clip_output_root)
    return Path(output).parent


def _camera_id_from_phase_clip(path: Path, *, phase: str) -> str:
    suffix = f"_phase_{phase}"
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _view_from_camera_id(camera_id: str, *, default: str) -> str:
    if "vehicle_view" in camera_id:
        return "vehicle_view"
    if "overhead_view" in camera_id:
        return "overhead_view"
    # WTS overhead files usually carry camera/IP identifiers, while vehicle-view
    # files explicitly include vehicle_view in the stem.
    if camera_id:
        return "overhead_view"
    return default


def _vqa_inference_row(
    question: dict[str, Any],
    *,
    videos: list[str],
    scenario_id: str,
    scenario_type: str,
    scope: str,
    phase: str | None,
    bbox_context: str,
) -> dict[str, Any]:
    options = {
        letter: str(question[letter]).strip()
        for letter in ("a", "b", "c", "d", "e")
        if letter in question and str(question[letter]).strip()
    }
    return {
        "instruction": vqa_instruction(
            scope=scope,
            scenario_type=scenario_type,
            phase=phase,
            question=str(question.get("question", "")).strip(),
            options=options,
            media_count=len(videos),
            bbox_context=bbox_context,
            visual_context=make_visual_media_context(videos),
        ),
        "input": "",
        "output": "a",
        "videos": videos,
        "metadata": {
            "task": "vqa",
            "vqa_id": str(question.get("id", "")).strip(),
            "split": "test",
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "scope": scope,
            "phase": phase,
        },
    }


def _vqa_bbox_context(
    bbox_by_scenario: dict[str, dict[str, dict[str, list[str]]]],
    *,
    scenario_id: str,
    scope: str,
    phase: str | None,
    bbox_mode: str,
    frame_width: int,
    frame_height: int,
) -> str:
    if bbox_mode == "none" or phase is None:
        return ""
    if bbox_mode != "summary":
        raise ValueError(f"Unsupported bbox_mode: {bbox_mode}")
    from .bboxes import make_bbox_context

    bbox_files = bbox_by_scenario.get(scenario_id)
    if bbox_files is None:
        return ""
    view = "vehicle_view" if scope == "vehicle_view" else "overhead_view"
    return make_bbox_context(
        bbox_files,
        view=view,
        phase=phase,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _bbox_index(
    bbox_root: Path,
    *,
    absolute_paths: bool,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    result: dict[str, dict[str, dict[str, list[str]]]] = {}
    annotated_root = bbox_root / "annotations" / "bbox_annotated"
    if not annotated_root.exists():
        return result

    for role in ROLES:
        role_root = annotated_root / role
        for path in _internal_json_files(role_root):
            if path.name == ".DS_Store":
                continue
            scenario_id = _scenario_id_from_bbox_path(path)
            if scenario_id is None:
                continue
            view = "vehicle_view" if "vehicle_view" in path.parts else "overhead_view"
            row = result.setdefault(
                scenario_id,
                {role_name: {view_name: [] for view_name in VIEWS} for role_name in ROLES},
            )
            row[role][view].append(as_path_string(path, absolute=absolute_paths))

    for row in result.values():
        for role in ROLES:
            for view in VIEWS:
                row[role][view] = sorted(set(row[role][view]))
    return result


def _scenario_id_from_bbox_path(path: Path) -> str | None:
    for part in path.parts:
        if part.endswith("_T1") or part.endswith("_T2") or "_normal_" in part:
            return part
    stem = path.stem.replace("_bbox", "")
    pieces = stem.split("_")
    if len(pieces) >= 4:
        # Typical bbox filename starts with the scenario ID followed by camera ID.
        return "_".join(pieces[:4])
    return None
