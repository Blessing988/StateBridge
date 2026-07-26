"""Create visually grounded bbox-overlay and interaction-crop clips."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import as_path_string, read_json, read_jsonl, write_jsonl
from .schema import ROLES, ScenarioRecord, normalize_phase_label


VISUAL_VARIANTS = ("overlay", "interaction_crop", "pedestrian_crop", "vehicle_crop")


@dataclass(slots=True)
class TrackBoxes:
    role: str
    track: str
    boxes: list[tuple[int, int, int, int]]

    @property
    def first(self) -> tuple[int, int, int, int]:
        return self.boxes[0]

    @property
    def mid(self) -> tuple[int, int, int, int]:
        return self.boxes[len(self.boxes) // 2]

    @property
    def last(self) -> tuple[int, int, int, int]:
        return self.boxes[-1]

    @property
    def union(self) -> tuple[int, int, int, int]:
        return union_boxes(self.boxes)


def build_visual_phase_clip_manifest(
    *,
    index_path: str | Path,
    phase_clip_manifest: str | Path,
    output_manifest: str | Path,
    output_root: str | Path,
    variants: set[str] | None = None,
    include_base: bool = False,
    max_clips_per_key: int | None = None,
    make_clips: bool = False,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    clip_mode: str = "mpeg4",
    frame_width: int = 1920,
    frame_height: int = 1080,
    crop_padding: float = 0.35,
    crop_size: int = 768,
    max_tracks_per_role: int = 2,
    absolute_paths: bool = True,
) -> list[dict[str, Any]]:
    requested = _normalize_variants(variants)
    records = {
        record.scenario_id: record
        for record in (ScenarioRecord.from_dict(row) for row in read_jsonl(index_path))
    }
    base_rows = read_jsonl(phase_clip_manifest)
    output_root = Path(output_root)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    for row in base_rows:
        key = (str(row["scenario_id"]), str(row["view"]), str(row["phase"]))
        record = records.get(str(row["scenario_id"]))
        if record is None:
            continue

        if include_base and _can_add(counts, key, max_clips_per_key):
            rows.append(_copy_row(row, row["clip_path"], "base", absolute_paths=absolute_paths))
            counts[key] += 1

        variants_added = 0
        if _can_add(counts, key, max_clips_per_key):
            generated = make_visual_clip_variants(
                clip_path=row["clip_path"],
                output_dir=_visual_clip_dir(output_root, row),
                camera_id=str(row["camera_id"]),
                bbox_files=record.bbox_files,
                view=str(row["view"]),
                phase=str(row["phase"]),
                variants=requested,
                make_clips=make_clips,
                overwrite=overwrite,
                ffmpeg_bin=ffmpeg_bin,
                clip_mode=clip_mode,
                frame_width=frame_width,
                frame_height=frame_height,
                crop_padding=crop_padding,
                crop_size=crop_size,
                max_tracks_per_role=max_tracks_per_role,
                absolute_paths=absolute_paths,
            )
            for visual in generated:
                if not _can_add(counts, key, max_clips_per_key):
                    break
                rows.append(
                    _copy_row(
                        row,
                        visual["clip_path"],
                        visual["variant"],
                        absolute_paths=absolute_paths,
                        extra={
                            "source_clip": as_path_string(Path(row["clip_path"]), absolute=absolute_paths),
                            "bbox_track_count": visual["bbox_track_count"],
                        },
                    )
                )
                counts[key] += 1
                variants_added += 1

        if variants_added == 0 and not include_base and _can_add(counts, key, max_clips_per_key):
            # Keep the key covered so exporters do not silently fall back to full videos.
            rows.append(_copy_row(row, row["clip_path"], "base_fallback", absolute_paths=absolute_paths))
            counts[key] += 1

    write_jsonl(output_manifest, rows)
    return rows


def make_visual_clip_variants(
    *,
    clip_path: str | Path,
    output_dir: str | Path,
    camera_id: str,
    bbox_files: dict[str, dict[str, list[str]]],
    view: str,
    phase: str,
    variants: set[str] | None = None,
    make_clips: bool = False,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    clip_mode: str = "mpeg4",
    frame_width: int = 1920,
    frame_height: int = 1080,
    crop_padding: float = 0.35,
    crop_size: int = 768,
    max_tracks_per_role: int = 2,
    absolute_paths: bool = True,
) -> list[dict[str, Any]]:
    requested = _normalize_variants(variants)
    clip_path = Path(clip_path)
    output_dir = Path(output_dir)
    tracks = collect_phase_tracks(
        bbox_files,
        view=view,
        camera_id=camera_id,
        phase=phase,
        frame_width=frame_width,
        frame_height=frame_height,
        max_tracks_per_role=max_tracks_per_role,
    )
    if not tracks:
        return []

    generated: list[dict[str, Any]] = []
    stem = clip_path.stem
    if "overlay" in requested:
        overlay_filter = _overlay_filter(tracks)
        if overlay_filter:
            output_path = output_dir / f"{stem}_overlay.mp4"
            if make_clips:
                _run_ffmpeg_filter(
                    ffmpeg_bin=ffmpeg_bin,
                    input_path=clip_path,
                    output_path=output_path,
                    vf=overlay_filter,
                    overwrite=overwrite,
                    clip_mode=clip_mode,
                )
            generated.append(
                {
                    "variant": "overlay",
                    "clip_path": as_path_string(output_path, absolute=absolute_paths),
                    "bbox_track_count": len(tracks),
                }
            )

    role_to_tracks = {
        role: [track for track in tracks if track.role == role]
        for role in ROLES
    }
    crop_specs = {
        "interaction_crop": tracks,
        "pedestrian_crop": role_to_tracks["pedestrian"],
        "vehicle_crop": role_to_tracks["vehicle"],
    }
    for variant, variant_tracks in crop_specs.items():
        if variant not in requested or not variant_tracks:
            continue
        crop_filter = _crop_filter(
            [box for track in variant_tracks for box in track.boxes],
            frame_width=frame_width,
            frame_height=frame_height,
            padding=crop_padding,
            crop_size=crop_size,
        )
        output_path = output_dir / f"{stem}_{variant}.mp4"
        if make_clips:
            _run_ffmpeg_filter(
                ffmpeg_bin=ffmpeg_bin,
                input_path=clip_path,
                output_path=output_path,
                vf=crop_filter,
                overwrite=overwrite,
                clip_mode=clip_mode,
            )
        generated.append(
            {
                "variant": variant,
                "clip_path": as_path_string(output_path, absolute=absolute_paths),
                "bbox_track_count": len(variant_tracks),
            }
        )

    return generated


def collect_phase_tracks(
    bbox_files: dict[str, dict[str, list[str]]],
    *,
    view: str,
    camera_id: str,
    phase: str,
    frame_width: int,
    frame_height: int,
    max_tracks_per_role: int = 2,
) -> list[TrackBoxes]:
    tracks: list[TrackBoxes] = []
    for role in ROLES:
        paths = bbox_files.get(role, {}).get(view, [])
        matched_paths = [path for path in paths if _camera_matches(path, camera_id)]
        for path in matched_paths[:max_tracks_per_role]:
            track = _load_track_boxes(
                path,
                role=role,
                phase=phase,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if track is not None:
                tracks.append(track)
    return tracks


def union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _load_track_boxes(
    path: str | Path,
    *,
    role: str,
    phase: str,
    frame_width: int,
    frame_height: int,
) -> TrackBoxes | None:
    data = read_json(path)
    annotations = [
        ann
        for ann in data.get("annotations", [])
        if normalize_phase_label(ann.get("phase_number")) == phase and ann.get("bbox")
    ]
    if not annotations:
        return None
    annotations = sorted(annotations, key=lambda ann: ann.get("image_id", 0))
    boxes = [
        _xywh_to_xyxy_pixels(ann["bbox"], frame_width=frame_width, frame_height=frame_height)
        for ann in annotations
    ]
    boxes = [box for box in boxes if box[2] > box[0] + 1 and box[3] > box[1] + 1]
    if not boxes:
        return None
    return TrackBoxes(role=role, track=Path(path).stem, boxes=boxes)


def _xywh_to_xyxy_pixels(
    bbox: list[float],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = [float(value) for value in bbox[:4]]
    x1 = _clip(round(x), 0, frame_width - 2)
    y1 = _clip(round(y), 0, frame_height - 2)
    x2 = _clip(round(x + w), x1 + 1, frame_width)
    y2 = _clip(round(y + h), y1 + 1, frame_height)
    return x1, y1, x2, y2


def _overlay_filter(tracks: list[TrackBoxes]) -> str:
    filters: list[str] = []
    role_colors = {
        "pedestrian": ("yellow@0.95", "orange@0.95", "red@0.95"),
        "vehicle": ("cyan@0.95", "dodgerblue@0.95", "lime@0.95"),
    }
    for track in tracks:
        colors = role_colors.get(track.role, ("white@0.95", "white@0.95", "white@0.95"))
        # First/mid/last boxes form a sparse trajectory cue that works for dense
        # synthetic bboxes and sparse keyframe real bboxes.
        for box, color, thickness in (
            (track.first, colors[0], 3),
            (track.mid, colors[1], 4),
            (track.last, colors[2], 5),
        ):
            x1, y1, x2, y2 = box
            filters.append(
                f"drawbox=x={x1}:y={y1}:w={x2 - x1}:h={y2 - y1}:color={color}:t={thickness}"
            )
        ux1, uy1, ux2, uy2 = track.union
        filters.append(
            f"drawbox=x={ux1}:y={uy1}:w={ux2 - ux1}:h={uy2 - uy1}:color=white@0.35:t=2"
        )
    return ",".join(filters)


def _crop_filter(
    boxes: list[tuple[int, int, int, int]],
    *,
    frame_width: int,
    frame_height: int,
    padding: float,
    crop_size: int,
) -> str:
    x1, y1, x2, y2 = union_boxes(boxes)
    width = max(x2 - x1, 2)
    height = max(y2 - y1, 2)
    side = max(width, height)
    side = int(round(side * (1.0 + max(padding, 0.0) * 2.0)))
    side = max(side, 96)
    side = min(side, frame_width, frame_height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = _clip(left, 0, frame_width - side)
    top = _clip(top, 0, frame_height - side)
    left = _even(left)
    top = _even(top)
    side = _even(side)
    if left + side > frame_width:
        left = _even(max(0, frame_width - side))
    if top + side > frame_height:
        top = _even(max(0, frame_height - side))
    return (
        f"crop={side}:{side}:{left}:{top},"
        f"scale={crop_size}:{crop_size}:force_original_aspect_ratio=decrease,"
        f"pad={crop_size}:{crop_size}:(ow-iw)/2:(oh-ih)/2"
    )


def _run_ffmpeg_filter(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    vf: str,
    overwrite: bool,
    clip_mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-an",
    ]
    if clip_mode == "mpeg4":
        command.extend(["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p"])
    elif clip_mode == "h264":
        command.extend(["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"])
    else:
        raise ValueError("Visual clips require re-encoding; use clip_mode='mpeg4' or 'h264'.")
    command.append(str(output_path))
    subprocess.run(command, check=True)


def _copy_row(
    row: dict[str, Any],
    clip_path: str | Path,
    variant: str,
    *,
    absolute_paths: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    copied = dict(row)
    copied["clip_path"] = as_path_string(Path(clip_path), absolute=absolute_paths)
    copied["visual_variant"] = variant
    if extra:
        copied.update(extra)
    return copied


def _visual_clip_dir(output_root: Path, row: dict[str, Any]) -> Path:
    return (
        output_root
        / "visual_clips"
        / str(row["split"])
        / str(row["scenario_type"])
        / str(row["scenario_id"])
        / str(row["view"])
    )


def _normalize_variants(variants: set[str] | None) -> set[str]:
    requested = set(variants or {"overlay", "interaction_crop"})
    unsupported = requested.difference(VISUAL_VARIANTS)
    if unsupported:
        raise ValueError(f"Unsupported visual variants: {sorted(unsupported)}")
    return requested


def _can_add(
    counts: dict[tuple[str, str, str], int],
    key: tuple[str, str, str],
    max_clips_per_key: int | None,
) -> bool:
    return max_clips_per_key is None or max_clips_per_key <= 0 or counts[key] < max_clips_per_key


def _camera_matches(path: str | Path, camera_id: str) -> bool:
    stem = Path(path).stem
    if stem.endswith("_bbox"):
        stem = stem[: -len("_bbox")]
    return stem == camera_id or stem.endswith(camera_id) or camera_id.endswith(stem)


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1
