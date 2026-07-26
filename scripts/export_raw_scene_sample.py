"""Export unannotated SynWTS phase frames plus caption/VQA text."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image


PHASE_NAMES = {
    "0": "pre_recognition",
    "1": "recognition",
    "2": "judgment",
    "3": "action",
    "4": "avoidance",
    "pre-recognition": "pre_recognition",
    "recognition": "recognition",
    "judgment": "judgment",
    "action": "action",
    "avoidance": "avoidance",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_base(root: Path, kind: str, split: str, scenario: str, view: str) -> Path:
    normal = root / kind / split / "normal_trimmed" / scenario / view
    if normal.exists():
        return normal
    direct = root / kind / split / scenario / view
    if direct.exists():
        return direct
    raise FileNotFoundError(f"No {kind} path for {split}/{scenario}/{view}")


def scenario_env_base(root: Path, split: str, scenario: str) -> Path | None:
    normal = root / "annotations" / "vqa" / split / "normal_trimmed" / scenario / "environment"
    direct = root / "annotations" / "vqa" / split / scenario / "environment"
    if normal.exists():
        return normal
    if direct.exists():
        return direct
    return None


def phase_name(phase: dict) -> str:
    labels = phase.get("labels") or []
    raw = str(labels[0]).lower() if labels else "phase"
    return PHASE_NAMES.get(raw, raw.replace(" ", "_"))


def extract_wpf(video: Path, output: Path, time_sec: float) -> None:
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
        str(video),
        "-Output",
        str(output),
        "-TimeSec",
        f"{time_sec:.3f}",
    ]
    subprocess.run(cmd, check=True)


def make_grid(frame_paths: list[Path], output: Path, thumb_w: int = 480) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    thumbs = []
    for img in images:
        w, h = img.size
        thumb_h = int(round(h * (thumb_w / w)))
        thumbs.append(img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS))
    gap = 12
    canvas_w = len(thumbs) * thumb_w + (len(thumbs) - 1) * gap
    canvas_h = max(img.height for img in thumbs)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    x = 0
    for img in thumbs:
        canvas.paste(img, (x, 0))
        x += thumb_w + gap
    canvas.save(output)


def concise_vqa(rows: list[dict], limit: int) -> list[dict]:
    out = []
    for row in rows[:limit]:
        correct = str(row.get("correct", "")).strip()
        out.append(
            {
                "question": row.get("question", ""),
                "options": {k: row[k] for k in ["a", "b", "c", "d"] if k in row},
                "correct": correct,
                "answer": row.get(correct, "") if correct else "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="sample_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scenario", default="20230707_15_SY4_T1")
    parser.add_argument("--view", default="overhead_view")
    parser.add_argument("--video-name", default="")
    parser.add_argument("--output-dir", default="eccv_paper/ECCV_AICity26_Track2/figures/raw_synthetic_scene")
    parser.add_argument("--vqa-limit", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = scenario_base(root, "videos", args.split, args.scenario, args.view)
    caption_dir = scenario_base(root, "annotations/caption", args.split, args.scenario, args.view)
    vqa_dir = scenario_base(root, "annotations/vqa", args.split, args.scenario, args.view)

    video_path = video_dir / args.video_name if args.video_name else sorted(video_dir.glob("*.mp4"))[0]
    caption_path = sorted(caption_dir.glob("*_caption.json"))[0]
    vqa_path = sorted(vqa_dir.glob("*.json"))[0]

    caption_data = read_json(caption_path)
    phases = sorted(caption_data["event_phase"], key=lambda p: float(p.get("start_time", 0.0)))
    frame_paths: list[Path] = []
    exported = []

    for idx, phase in enumerate(phases):
        start = float(phase["start_time"])
        end = float(phase["end_time"])
        mid = (start + end) / 2.0
        name = phase_name(phase)
        frame_path = output_dir / f"{idx}_{name}_{mid:.2f}s.png"
        extract_wpf(video_path, frame_path, mid)
        frame_paths.append(frame_path)
        exported.append(
            {
                "phase": name,
                "start_time": start,
                "end_time": end,
                "mid_time": mid,
                "frame": str(frame_path),
                "caption_pedestrian": phase.get("caption_pedestrian", ""),
                "caption_vehicle": phase.get("caption_vehicle", ""),
            }
        )

    grid_path = output_dir / "raw_phase_grid.png"
    make_grid(frame_paths, grid_path)

    vqa_data = read_json(vqa_path)
    vqa_item = vqa_data[0] if isinstance(vqa_data, list) else vqa_data
    vqa_phases = []
    for phase in vqa_item.get("event_phase", []):
        vqa_phases.append(
            {
                "phase": phase_name(phase),
                "start_time": float(phase.get("start_time", 0.0)),
                "end_time": float(phase.get("end_time", 0.0)),
                "vqa": concise_vqa(phase.get("conversations", []), args.vqa_limit),
            }
        )

    env_vqa = []
    env_dir = scenario_env_base(root, args.split, args.scenario)
    if env_dir:
        env_data = read_json(sorted(env_dir.glob("*.json"))[0])
        env_item = env_data[0] if isinstance(env_data, list) else env_data
        env_vqa = concise_vqa(env_item.get("environment", []), args.vqa_limit)

    summary = {
        "scenario": args.scenario,
        "split": args.split,
        "view": args.view,
        "video": str(video_path),
        "grid": str(grid_path),
        "phases": exported,
        "vqa_phases": vqa_phases,
        "environment_vqa": env_vqa,
    }
    json_path = output_dir / "captions_vqa.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"Scenario: {args.scenario}",
        f"Split: {args.split}",
        f"View: {args.view}",
        f"Video: {video_path}",
        "",
        "Caption text",
        "============",
    ]
    for item in exported:
        lines.extend(
            [
                "",
                f"Phase: {item['phase']} ({item['start_time']:.3f}s-{item['end_time']:.3f}s)",
                "Caption pedestrian:",
                item["caption_pedestrian"],
                "Caption vehicle:",
                item["caption_vehicle"],
            ]
        )
    lines.extend(["", "VQA text", "========"])
    for item in vqa_phases:
        lines.append("")
        lines.append(f"Phase: {item['phase']} ({item['start_time']:.3f}s-{item['end_time']:.3f}s)")
        for idx, row in enumerate(item["vqa"], start=1):
            opts = " | ".join(f"{k}. {v}" for k, v in row["options"].items())
            lines.append(f"Q{idx}: {row['question']}")
            lines.append(f"Options: {opts}")
            lines.append(f"Answer: {row['correct']}. {row['answer']}")
    if env_vqa:
        lines.extend(["", "Environment VQA", "---------------"])
        for idx, row in enumerate(env_vqa, start=1):
            opts = " | ".join(f"{k}. {v}" for k, v in row["options"].items())
            lines.append(f"Q{idx}: {row['question']}")
            lines.append(f"Options: {opts}")
            lines.append(f"Answer: {row['correct']}. {row['answer']}")

    txt_path = output_dir / "captions_vqa.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(grid_path)
    print(json_path)
    print(txt_path)


if __name__ == "__main__":
    main()
