"""Create a paper teaser figure from one SynWTS sample scenario.

The script extracts one frame per behavioral phase, overlays pedestrian and
vehicle boxes, and adds compact caption/VQA text from the same scenario.
It requires ffmpeg for frame extraction and Pillow for layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PHASE_NAMES = {
    "0": "Pre-recognition",
    "1": "Recognition",
    "2": "Judgment",
    "3": "Action",
    "4": "Avoidance",
    "pre-recognition": "Pre-recognition",
    "recognition": "Recognition",
    "judgment": "Judgment",
    "action": "Action",
    "avoidance": "Avoidance",
}

PED_COLOR = (255, 212, 59)
VEH_COLOR = (34, 211, 238)
RED = (224, 49, 49)
BLUE = (25, 113, 194)
TEXT = (24, 24, 27)
MUTED = (82, 82, 91)
PANEL = (250, 250, 250)
BORDER = (214, 214, 220)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {root / pattern}")
    return matches[0]


def load_bbox_by_frame(path: Path) -> dict[int, list[float]]:
    data = read_json(path)
    rows = data.get("annotations", data if isinstance(data, list) else [])
    out: dict[int, list[float]] = {}
    for row in rows:
        if "image_id" in row and "bbox" in row:
            out[int(row["image_id"])] = [float(v) for v in row["bbox"]]
    return out


def extract_frame(ffmpeg_bin: str, video_path: Path, time_sec: float, output_path: Path) -> None:
    if ffmpeg_bin.lower() == "wpf":
        helper = Path(__file__).with_name("grab_wpf_frame.ps1")
        cmd = [
            "powershell",
            "-STA",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(helper),
            "-Video",
            str(video_path),
            "-Output",
            str(output_path),
            "-TimeSec",
            f"{time_sec:.3f}",
        ]
        subprocess.run(cmd, check=True)
        return

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def nearest_bbox(boxes: dict[int, list[float]], frame_idx: int) -> list[float] | None:
    if not boxes:
        return None
    if frame_idx in boxes:
        return boxes[frame_idx]
    key = min(boxes.keys(), key=lambda k: abs(k - frame_idx))
    return boxes[key]


def draw_box(draw: ImageDraw.ImageDraw, bbox_xywh: list[float] | None, scale_x: float, scale_y: float, color, label: str) -> None:
    if bbox_xywh is None:
        return
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y
    width = max(3, int(5 * min(scale_x, scale_y)))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    font = load_font(18, bold=True)
    pad = 5
    tw = draw.textlength(label, font=font)
    th = 22
    draw.rectangle([x1, max(0, y1 - th - 2), x1 + tw + 2 * pad, max(th, y1 - 2)], fill=color)
    draw.text((x1 + pad, max(0, y1 - th)), label, fill=(0, 0, 0), font=font)


def wrap_text(text: str, width: int = 82, max_chars: int = 620) -> str:
    text = " ".join(str(text).split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return "\n".join(textwrap.wrap(text, width=width))


def draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, line_gap: int = 5) -> int:
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, fill=fill, font=font)
        y += font.size + line_gap
    return y


def phase_label(phase: dict) -> str:
    labels = phase.get("labels") or []
    raw = str(labels[0]) if labels else ""
    return PHASE_NAMES.get(raw.lower(), raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="sample_data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenario", default="20231013_101813_normal_192.168.0.11_1_event_2")
    parser.add_argument("--view", default="overhead_view")
    parser.add_argument("--output", default="eccv_paper/ECCV_AICity26_Track2/figures/dataset_teaser.png")
    parser.add_argument("--pdf-output", default="")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    args = parser.parse_args()

    if args.ffmpeg_bin.lower() != "wpf" and shutil.which(args.ffmpeg_bin) is None and not Path(args.ffmpeg_bin).exists():
        raise SystemExit(
            f"ffmpeg not found: {args.ffmpeg_bin}. Install ffmpeg or pass --ffmpeg-bin /path/to/ffmpeg."
        )

    root = Path(args.dataset_root)
    split = args.split
    scenario = args.scenario
    view = args.view

    video_path = find_one(root / "videos" / split / "normal_trimmed" / scenario / view, "*.mp4")
    caption_path = find_one(root / "annotations" / "caption" / split / "normal_trimmed" / scenario / view, "*_caption.json")
    vqa_path = find_one(root / "annotations" / "vqa" / split / "normal_trimmed" / scenario / view, "*.json")
    ped_bbox_path = find_one(root / "annotations" / "bbox_annotated" / "pedestrian" / split / "normal_trimmed" / scenario / view, "*_bbox.json")
    veh_bbox_path = find_one(root / "annotations" / "bbox_annotated" / "vehicle" / split / "normal_trimmed" / scenario / view, "*_bbox.json")

    captions = read_json(caption_path)["event_phase"]
    captions = sorted(captions, key=lambda p: float(p.get("start_time", 0.0)))
    ped_boxes = load_bbox_by_frame(ped_bbox_path)
    veh_boxes = load_bbox_by_frame(veh_bbox_path)
    max_end = max(float(p.get("end_time", 0.0)) for p in captions)
    approx_fps = max(max(ped_boxes.keys(), default=0), max(veh_boxes.keys(), default=0)) / max(max_end, 1e-6)

    vqa_data = read_json(vqa_path)
    vqa_item = vqa_data[0] if isinstance(vqa_data, list) else vqa_data
    vqa_phase = None
    for ph in vqa_item.get("event_phase", []):
        labels = [str(x).lower() for x in ph.get("labels", [])]
        if "avoidance" in labels or "4" in labels:
            vqa_phase = ph
            break
    if vqa_phase is None:
        vqa_phase = (vqa_item.get("event_phase") or [{}])[0]

    selected_caption = captions[-1]
    vqa_questions = vqa_phase.get("conversations", [])[:3]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_w, frame_h = 420, 236
    margin, gap = 34, 18
    title_h = 72
    text_top = title_h + frame_h + 58
    canvas_w = margin * 2 + 5 * frame_w + 4 * gap
    canvas_h = text_top + 570
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(36, bold=True)
    label_font = load_font(22, bold=True)
    body_font = load_font(21, bold=False)
    small_font = load_font(18, bold=False)

    draw.text((margin, 22), "SynWTS sample: phase-level traffic safety captioning and VQA", fill=TEXT, font=title_font)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for idx, phase in enumerate(captions[:5]):
            start = float(phase["start_time"])
            end = float(phase["end_time"])
            mid = (start + end) / 2.0
            frame_idx = int(round(mid * approx_fps))
            raw_path = tmp / f"phase_{idx}.png"
            extract_frame(args.ffmpeg_bin, video_path, mid, raw_path)
            frame = Image.open(raw_path).convert("RGB")
            orig_w, orig_h = frame.size
            frame = frame.resize((frame_w, frame_h), Image.Resampling.LANCZOS)
            fd = ImageDraw.Draw(frame)
            sx, sy = frame_w / orig_w, frame_h / orig_h
            draw_box(fd, nearest_bbox(ped_boxes, frame_idx), sx, sy, PED_COLOR, "ped")
            draw_box(fd, nearest_bbox(veh_boxes, frame_idx), sx, sy, VEH_COLOR, "veh")
            x = margin + idx * (frame_w + gap)
            y = title_h
            canvas.paste(frame, (x, y))
            draw.rectangle([x, y, x + frame_w, y + frame_h], outline=BORDER, width=2)
            draw.text((x, y + frame_h + 8), phase_label(phase), fill=TEXT, font=label_font)
            draw.text((x, y + frame_h + 34), f"{start:.1f}s-{end:.1f}s", fill=MUTED, font=small_font)

    left_x = margin
    right_x = canvas_w // 2 + 20
    panel_y = text_top
    panel_w = canvas_w // 2 - margin - 28
    panel_h = canvas_h - panel_y - margin

    draw.rounded_rectangle([left_x, panel_y, left_x + panel_w, panel_y + panel_h], radius=14, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle([right_x, panel_y, right_x + panel_w, panel_y + panel_h], radius=14, fill=PANEL, outline=BORDER, width=2)

    y = panel_y + 22
    draw.text((left_x + 24, y), "Phase caption fields", fill=TEXT, font=label_font)
    y += 42
    draw.text((left_x + 24, y), "Caption pedestrian", fill=RED, font=label_font)
    y += 32
    y = draw_wrapped(draw, (left_x + 24, y), wrap_text(selected_caption["caption_pedestrian"], 74, 520), body_font, TEXT)
    y += 22
    draw.text((left_x + 24, y), "Caption vehicle", fill=BLUE, font=label_font)
    y += 32
    draw_wrapped(draw, (left_x + 24, y), wrap_text(selected_caption["caption_vehicle"], 74, 520), body_font, TEXT)

    y = panel_y + 22
    draw.text((right_x + 24, y), "VQA examples from the same scenario", fill=TEXT, font=label_font)
    y += 44
    for i, q in enumerate(vqa_questions, start=1):
        correct = str(q.get("correct", "")).strip()
        ans = q.get(correct, "") if correct else ""
        q_text = f"Q{i}: {q.get('question', '')}"
        a_text = f"A: {correct}. {ans}"
        y = draw_wrapped(draw, (right_x + 24, y), wrap_text(q_text, 68, 260), body_font, TEXT)
        y = draw_wrapped(draw, (right_x + 24, y + 2), wrap_text(a_text, 68, 160), body_font, BLUE)
        y += 22

    legend_y = canvas_h - margin - 38
    draw.rectangle([right_x + 24, legend_y, right_x + 58, legend_y + 24], outline=PED_COLOR, width=4)
    draw.text((right_x + 70, legend_y - 2), "pedestrian box", fill=TEXT, font=small_font)
    draw.rectangle([right_x + 260, legend_y, right_x + 294, legend_y + 24], outline=VEH_COLOR, width=4)
    draw.text((right_x + 306, legend_y - 2), "vehicle box", fill=TEXT, font=small_font)

    canvas.save(output)
    if args.pdf_output:
        pdf_output = Path(args.pdf_output)
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(pdf_output, "PDF", resolution=300.0)
    print(output)


if __name__ == "__main__":
    main()
