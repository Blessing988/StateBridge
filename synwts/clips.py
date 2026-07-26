"""Create phase-level video clips from SynWTS timestamps."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .io import as_path_string, read_json, write_jsonl
from .parsers import load_caption_phases
from .schema import ScenarioRecord


def build_phase_clip_manifest(
    index_path: str | Path,
    output_manifest: str | Path,
    *,
    output_root: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    make_clips: bool = False,
    overwrite: bool = False,
    absolute_paths: bool = True,
    clip_mode: str = "h264",
) -> list[dict[str, Any]]:
    records = _load_records(index_path)
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(
            _record_clip_rows(
                record,
                output_root=output_root,
                ffmpeg_bin=ffmpeg_bin,
                make_clips=make_clips,
                overwrite=overwrite,
                absolute_paths=absolute_paths,
                clip_mode=clip_mode,
            )
        )
    write_jsonl(output_manifest, rows)
    return rows


def load_phase_clip_map(manifest_path: str | Path) -> dict[tuple[str, str, str], list[str]]:
    clip_map: dict[tuple[str, str, str], list[str]] = {}
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json

            row = json.loads(line)
            key = (row["scenario_id"], row["view"], row["phase"])
            clip_map.setdefault(key, []).append(row["clip_path"])
    return clip_map


def validate_phase_clips(
    manifest_path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    max_errors: int = 50,
) -> dict[str, Any]:
    rows = _load_manifest_rows(manifest_path)
    errors: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["clip_path"])
        if not path.exists():
            errors.append({"clip_path": str(path), "error": "missing"})
        elif path.stat().st_size == 0:
            errors.append({"clip_path": str(path), "error": "zero_bytes"})
        else:
            try:
                duration, frame_count = _ffprobe_duration_and_frames(ffprobe_bin, path)
                if duration <= 0:
                    errors.append({"clip_path": str(path), "error": "non_positive_duration", "duration": duration})
                elif frame_count <= 0:
                    errors.append(
                        {
                            "clip_path": str(path),
                            "error": "zero_decodable_frames",
                            "duration": duration,
                            "frame_count": frame_count,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - report validation failures compactly.
                errors.append({"clip_path": str(path), "error": "ffprobe_failed", "message": str(exc)})
        if len(errors) >= max_errors:
            break
    return {
        "ok": not errors,
        "total_manifest_rows": len(rows),
        "num_errors_reported": len(errors),
        "errors": errors,
    }


def filter_phase_clip_manifest(
    manifest_path: str | Path,
    output_manifest: str | Path,
    *,
    validation_report: str | Path,
) -> dict[str, Any]:
    rows = _load_manifest_rows(manifest_path)
    report = read_json(validation_report)
    bad_paths = {str(error["clip_path"]) for error in report.get("errors", [])}
    kept = [row for row in rows if str(row.get("clip_path", "")) not in bad_paths]
    write_jsonl(output_manifest, kept)
    return {
        "input_rows": len(rows),
        "removed_rows": len(rows) - len(kept),
        "output_rows": len(kept),
        "bad_paths": sorted(bad_paths),
    }


def _load_records(index_path: str | Path) -> list[ScenarioRecord]:
    import json

    records: list[ScenarioRecord] = []
    with Path(index_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(ScenarioRecord.from_dict(json.loads(line)))
    return records


def _load_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ffprobe_duration_and_frames(ffprobe_bin: str, path: Path) -> tuple[float, int]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=duration,nb_read_frames",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    duration = 0.0
    frame_count = 0
    for line in result.stdout.strip().splitlines():
        key, _, value = line.partition("=")
        if key == "duration":
            try:
                duration = float(value)
            except ValueError:
                duration = 0.0
        elif key == "nb_read_frames":
            try:
                frame_count = int(value)
            except ValueError:
                frame_count = 0
    return duration, frame_count


def _record_clip_rows(
    record: ScenarioRecord,
    *,
    output_root: Path,
    ffmpeg_bin: str,
    make_clips: bool,
    overwrite: bool,
    absolute_paths: bool,
    clip_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view, caption_path in sorted(record.caption_files.items()):
        videos = record.videos.get(view, [])
        if not videos:
            continue
        phases = load_caption_phases(caption_path)
        for video_path in videos:
            video_path_obj = Path(video_path)
            camera_id = video_path_obj.stem
            for phase in phases:
                start = float(phase["start_time"])
                end = float(phase["end_time"])
                duration = max(end - start, 0.001)
                clip_path = _clip_path(
                    output_root,
                    record=record,
                    view=view,
                    camera_id=camera_id,
                    phase=phase["label"],
                )
                if make_clips:
                    _run_ffmpeg_clip(
                        ffmpeg_bin=ffmpeg_bin,
                        input_path=video_path_obj,
                        output_path=clip_path,
                        start=start,
                        duration=duration,
                        overwrite=overwrite,
                        clip_mode=clip_mode,
                    )
                rows.append(
                    {
                        "split": record.split,
                        "scenario_id": record.scenario_id,
                        "scenario_type": record.scenario_type,
                        "view": view,
                        "camera_id": camera_id,
                        "phase": phase["label"],
                        "start_time": start,
                        "end_time": end,
                        "duration": duration,
                        "source_video": as_path_string(video_path_obj, absolute=absolute_paths),
                        "clip_path": as_path_string(clip_path, absolute=absolute_paths),
                    }
                )
    return rows


def _clip_path(
    output_root: Path,
    *,
    record: ScenarioRecord,
    view: str,
    camera_id: str,
    phase: str,
) -> Path:
    scenario_dir = (
        output_root
        / "clips"
        / record.split
        / record.scenario_type
        / record.scenario_id
        / view
    )
    return scenario_dir / f"{camera_id}_phase_{phase}.mp4"


def cut_video_clip(
    *,
    ffmpeg_bin: str,
    input_path: str | Path,
    output_path: str | Path,
    start: float,
    duration: float,
    overwrite: bool,
    clip_mode: str,
) -> None:
    """Public wrapper used by test-time exporters to create a single clip."""
    _run_ffmpeg_clip(
        ffmpeg_bin=ffmpeg_bin,
        input_path=Path(input_path),
        output_path=Path(output_path),
        start=start,
        duration=duration,
        overwrite=overwrite,
        clip_mode=clip_mode,
    )


def _run_ffmpeg_clip(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    overwrite: bool,
    clip_mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return
    if clip_mode == "copy":
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    elif clip_mode == "h264":
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-reset_timestamps",
            "1",
            str(output_path),
        ]
    elif clip_mode == "mpeg4":
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-reset_timestamps",
            "1",
            str(output_path),
        ]
    else:
        raise ValueError(f"Unsupported clip_mode: {clip_mode}")
    subprocess.run(command, check=True)
