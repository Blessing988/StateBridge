"""Convert LLaMA-Factory video datasets into image-frame datasets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .io import read_json, write_json


def export_frame_dataset(
    *,
    dataset: str | Path,
    output: str | Path,
    frame_root: str | Path,
    dataset_info_output: str | Path | None = None,
    dataset_name: str | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    frame_time: str = "middle",
    max_frames_per_row: int = 0,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    rows = read_json(dataset)
    if not isinstance(rows, list):
        raise ValueError("Input dataset must be a JSON list.")
    frame_root = Path(frame_root)
    output_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        videos = [str(path) for path in row.get("videos", [])]
        if max_frames_per_row > 0:
            videos = videos[:max_frames_per_row]
        images = [
            str(
                extract_video_frame(
                    video,
                    frame_root=frame_root,
                    ffmpeg_bin=ffmpeg_bin,
                    ffprobe_bin=ffprobe_bin,
                    frame_time=frame_time,
                    overwrite=overwrite,
                )
            )
            for video in videos
        ]
        item = dict(row)
        instruction = str(item.get("instruction", ""))
        item["instruction"] = _replace_video_placeholders(instruction, len(images))
        item.pop("videos", None)
        item["images"] = images
        metadata = dict(item.get("metadata", {}))
        metadata["frame_export_source_videos"] = len(row.get("videos", []))
        metadata["frame_export_images"] = len(images)
        metadata["frame_time"] = frame_time
        metadata["source_row_index"] = row_idx
        item["metadata"] = metadata
        output_rows.append(item)
    write_json(output, output_rows)
    if dataset_info_output:
        if not dataset_name:
            raise ValueError("dataset_name is required when dataset_info_output is set.")
        write_frame_dataset_info(
            dataset_info_output,
            dataset_name=dataset_name,
            file_name=Path(output).name,
        )
    return output_rows


def write_frame_dataset_info(output: str | Path, *, dataset_name: str, file_name: str) -> None:
    write_json(
        output,
        {
            dataset_name: {
                "file_name": file_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "images": "images",
                },
            }
        },
    )


def extract_video_frame(
    video: str | Path,
    *,
    frame_root: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    frame_time: str,
    overwrite: bool,
) -> Path:
    video_path = Path(video)
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:16]
    stem = f"{video_path.stem}_{frame_time}_{digest}.jpg"
    out = frame_root / stem[:2] / stem
    if out.exists() and out.stat().st_size > 0 and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp_for(video_path, ffprobe_bin=ffprobe_bin, frame_time=frame_time)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    subprocess.run(command, check=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract frame from {video_path}")
    return out


def _timestamp_for(video_path: Path, *, ffprobe_bin: str, frame_time: str) -> float:
    if frame_time == "start":
        return 0.05
    if frame_time == "one_sec":
        return 1.0
    if frame_time != "middle":
        try:
            return max(float(frame_time), 0.0)
        except ValueError as exc:
            raise ValueError(f"Unsupported frame_time: {frame_time}") from exc
    duration = _duration(video_path, ffprobe_bin=ffprobe_bin)
    if duration <= 0:
        return 0.05
    return max(0.05, min(duration * 0.5, max(duration - 0.05, 0.05)))


def _duration(video_path: Path, *, ffprobe_bin: str) -> float:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _replace_video_placeholders(instruction: str, image_count: int) -> str:
    count = instruction.count("<video>")
    if count == 0:
        return instruction
    if count != image_count:
        # Keep text/media alignment strict for MiniCPM image plugin.
        parts = instruction.split("<video>")
        kept = min(count, image_count)
        out = []
        for idx, part in enumerate(parts):
            out.append(part)
            if idx < kept:
                out.append("<image>")
        return "".join(out)
    return instruction.replace("<video>", "<image>")
