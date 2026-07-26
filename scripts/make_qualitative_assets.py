"""Build qualitative-result assets for the StateBridge paper.

The script creates PowerPoint-friendly PNG panels and raw text files:
- a synthetic phase-grid evidence panel copied from existing exported frames,
- a caption fusion comparison panel from actual submission JSONs,
- a VQA consensus panel from actual public-test prediction JSONs.
"""

from __future__ import annotations

import json
import shutil
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "eccv_paper" / "ECCV_AICity26_Track2" / "figures"
OUT = FIG_DIR / "qualitative_results"

TEXT = (15, 23, 42)
MUTED = (71, 85, 105)
BORDER = (203, 213, 225)
PANEL = (248, 250, 252)
BLUE = (37, 99, 235)
ORANGE = (234, 88, 12)
GREEN = (22, 163, 74)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json_from_zip(path: Path, member: str):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(member).decode("utf-8"))


def wrap(text: str, width: int, max_chars: int | None = None) -> str:
    text = " ".join(str(text).split())
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return "\n".join(textwrap.wrap(text, width=width))


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=TEXT, gap: int = 5) -> int:
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, color=BLUE) -> int:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline=BORDER, width=2)
    draw.rectangle([x1, y1, x2, y1 + 48], fill=(255, 255, 255), outline=None)
    draw.line([x1, y1 + 48, x2, y1 + 48], fill=BORDER, width=2)
    draw.text((x1 + 18, y1 + 13), title, font=font(23, True), fill=color)
    return y1 + 68


def make_caption_fusion_panel() -> None:
    baseline = read_json(ROOT / "results" / "caption_submission_joint.json")
    final_zip = ROOT / "results" / "submission_agentic_vqa_router_no_synreal.zip"
    if final_zip.exists():
        locked = read_json_from_zip(final_zip, "caption_submission.json")
    else:
        locked = read_json(ROOT / "results" / "caption_submission_fact_locked_gate_v2_loose.json")
    report = read_json(ROOT / "results" / "caption_fact_locked_gate_v2_loose_report.json")

    # Pick a readable accepted edit that illustrates conservative field control.
    preferred = ("20230922_7_SN1_T1", 4, "caption_vehicle")
    chosen = next(
        (
            item
            for item in report["accepted"]
            if (item["scenario_id"], int(item["row_index"]), item["key"]) == preferred
        ),
        None,
    )
    for item in report["accepted"]:
        if chosen is not None:
            break
        sid, idx, key = item["scenario_id"], int(item["row_index"]), item["key"]
        b = baseline[sid][idx][key]
        l = locked[sid][idx][key]
        if key == "caption_vehicle" and "vehicle" in b.lower() and "vehicle" in l.lower():
            chosen = item
            break
    if chosen is None:
        chosen = report["accepted"][0]

    sid, idx, key = chosen["scenario_id"], int(chosen["row_index"]), chosen["key"]
    phase = locked[sid][idx].get("labels", [""])[0]
    base_text = baseline[sid][idx][key]
    lock_text = locked[sid][idx][key]
    reasons = ", ".join(chosen.get("reasons", []))

    W, H = 2100, 980
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 36), "Qualitative caption fusion example", font=font(40, True), fill=TEXT)
    draw.text(
        (52, 88),
        f"Public-test scenario {sid}, phase {phase}, field {key}. Text comes from actual submission files.",
        font=font(22),
        fill=MUTED,
    )

    y = 142
    col_w = 630
    gap = 42
    x0 = 52
    y_body = rounded_panel(draw, (x0, y, x0 + col_w, H - 70), "Direct VLM field", BLUE)
    draw_wrapped(draw, (x0 + 24, y_body), wrap(base_text, 46, 780), font(24), TEXT, 7)

    x1 = x0 + col_w + gap
    y_body = rounded_panel(draw, (x1, y, x1 + col_w, H - 70), "StateBridge selected field", ORANGE)
    draw_wrapped(draw, (x1 + 24, y_body), wrap(lock_text, 46, 780), font(24), TEXT, 7)

    x2 = x1 + col_w + gap
    y_body = rounded_panel(draw, (x2, y, x2 + col_w, H - 70), "Why the field is accepted", GREEN)
    rationale = (
        "The controller compares role-phase fields rather than rewriting the full caption freely. "
        "A field is accepted only when it preserves supported traffic facts such as role, phase, "
        "relative position, visibility, speed, action, and context. "
        f"Recorded gate reason: {reasons}."
    )
    draw_wrapped(draw, (x2 + 24, y_body), wrap(rationale, 48), font(25), TEXT, 8)
    draw.text((x2 + 24, H - 134), "Use this panel to illustrate conservative output control.", font=font(25, True), fill=GREEN)

    out = OUT / "qual_caption_fusion_panel.png"
    img.save(out)

    txt = OUT / "qual_caption_fusion_example.txt"
    txt.write_text(
        "\n".join(
            [
                f"scenario_id: {sid}",
                f"row_index: {idx}",
                f"phase: {phase}",
                f"field: {key}",
                f"gate_reasons: {reasons}",
                "",
                "Direct VLM caption:",
                base_text,
                "",
                "StateBridge fact-locked caption:",
                lock_text,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def vqa_lookup(rows: list[dict]) -> dict[str, str]:
    return {str(r["id"]): str(r["correct"]) for r in rows if "id" in r and "correct" in r}


def iter_public_questions():
    data = read_json(ROOT / "WTS_VQA_PUBLIC_TEST.json")
    for scenario_idx, item in enumerate(data):
        for phase in item.get("event_phase", []):
            label = (phase.get("labels") or [""])[0]
            for row in phase.get("conversations", []):
                yield scenario_idx, item.get("videos", []), label, row


def make_vqa_consensus_panel() -> None:
    candidates: list[tuple[str, dict[str, str]]] = []
    paths = [
        ("Qwen3-VL VQA", ROOT / "results" / "vqa_submission_fast.json"),
        ("VQA ensemble", ROOT / "results" / "vqa_submission_ensemble.json"),
        ("StateBridge router", ROOT / "results" / "submission_agentic_vqa_router_no_synreal.zip"),
    ]
    for name, path in paths:
        if not path.exists():
            continue
        if path.suffix == ".zip":
            rows = read_json_from_zip(path, "vqa_submission.json")
        else:
            rows = read_json(path)
        candidates.append((name, vqa_lookup(rows)))

    examples = []
    for scenario_idx, videos, phase, row in iter_public_questions():
        qid = str(row.get("id", ""))
        answers = [(name, pred.get(qid, "")) for name, pred in candidates]
        if len({a for _, a in answers if a}) > 1:
            examples.append((scenario_idx, videos, phase, row, answers))
        if len(examples) >= 3:
            break
    if not examples:
        for scenario_idx, videos, phase, row in iter_public_questions():
            qid = str(row.get("id", ""))
            answers = [(name, pred.get(qid, "")) for name, pred in candidates]
            examples.append((scenario_idx, videos, phase, row, answers))
            if len(examples) >= 3:
                break

    W, H = 2100, 1120
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 36), "Qualitative VQA consensus examples", font=font(40, True), fill=TEXT)
    draw.text(
        (52, 88),
        "Rows show public-test questions where candidate answer streams may disagree. The final router answer is selected from synthetic-trained predictions.",
        font=font(22),
        fill=MUTED,
    )

    y = 150
    row_h = 296
    for ex_idx, (scenario_idx, videos, phase, row, answers) in enumerate(examples, start=1):
        draw.rounded_rectangle([52, y, W - 52, y + row_h], radius=18, fill=PANEL, outline=BORDER, width=2)
        qid = row.get("id", "")
        video = videos[0] if videos else ""
        title = f"Example {ex_idx}: phase {phase}, question id {qid}"
        draw.text((78, y + 22), title, font=font(24, True), fill=TEXT)
        draw.text((78, y + 55), f"Video: {video}", font=font(18), fill=MUTED)
        yy = y + 88
        yy = draw_wrapped(draw, (78, yy), wrap("Question: " + row.get("question", ""), 92), font(23), TEXT, 6)
        opts = "   ".join(f"{k}. {row.get(k, '')}" for k in ["a", "b", "c", "d"] if k in row)
        yy = draw_wrapped(draw, (78, yy + 6), wrap("Options: " + opts, 105), font(20), MUTED, 5)

        x = 78
        yy += 18
        for name, ans in answers:
            color = GREEN if name == "StateBridge router" else BLUE
            draw.rounded_rectangle([x, yy, x + 315, yy + 50], radius=12, fill="white", outline=color, width=2)
            draw.text((x + 16, yy + 14), f"{name}: {ans}", font=font(20, True), fill=color)
            x += 338
        y += row_h + 28

    out = OUT / "qual_vqa_consensus_panel.png"
    img.save(out)

    txt_lines = []
    for ex_idx, (scenario_idx, videos, phase, row, answers) in enumerate(examples, start=1):
        txt_lines.extend(
            [
                f"Example {ex_idx}",
                f"scenario_index: {scenario_idx}",
                f"video: {videos[0] if videos else ''}",
                f"phase: {phase}",
                f"id: {row.get('id', '')}",
                f"question: {row.get('question', '')}",
                "options: " + " | ".join(f"{k}. {row.get(k, '')}" for k in ["a", "b", "c", "d"] if k in row),
                "predictions: " + " | ".join(f"{name}: {ans}" for name, ans in answers),
                "",
            ]
        )
    (OUT / "qual_vqa_consensus_examples.txt").write_text("\n".join(txt_lines), encoding="utf-8")


def make_caption_repair_panel() -> None:
    before_path = ROOT / "results" / "caption_submission_fused_qwen35_internvl_m10p0.json"
    after_path = ROOT / "results" / "caption_submission_best_internvl_m10p0_complete_repair.json"
    if not before_path.exists() or not after_path.exists():
        return

    before = read_json(before_path)
    after = read_json(after_path)
    sid, idx, key = "20230707_11_SY3_T1", 2, "caption_vehicle"
    phase = after[sid][idx].get("labels", [""])[0]
    before_text = before[sid][idx][key]
    after_text = after[sid][idx][key]

    W, H = 1850, 760
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 36), "Clean caption repair example", font=font(38, True), fill=TEXT)
    draw.text(
        (52, 86),
        f"Public-test scenario {sid}, phase {phase}, field {key}. The repair removes an incomplete final clause while preserving the preceding facts.",
        font=font(21),
        fill=MUTED,
    )

    y = 142
    col_w = 830
    gap = 70
    x0 = 52
    y_body = rounded_panel(draw, (x0, y, x0 + col_w, H - 58), "Before repair: dangling ending", BLUE)
    draw_wrapped(draw, (x0 + 24, y_body), wrap(before_text, 68, 980), font(24), TEXT, 7)
    draw.text((x0 + 24, H - 112), "Problem: the last sentence stops after an unsupported fragment.", font=font(22, True), fill=BLUE)

    x1 = x0 + col_w + gap
    y_body = rounded_panel(draw, (x1, y, x1 + col_w, H - 58), "After repair: complete field", GREEN)
    draw_wrapped(draw, (x1 + 24, y_body), wrap(after_text, 68, 980), font(24), TEXT, 7)
    draw.text((x1 + 24, H - 112), "Fix: drop the incomplete tail and keep the verified description.", font=font(22, True), fill=GREEN)

    img.save(OUT / "qual_caption_repair_panel.png")
    (OUT / "qual_caption_repair_example.txt").write_text(
        "\n".join(
            [
                f"scenario_id: {sid}",
                f"row_index: {idx}",
                f"phase: {phase}",
                f"field: {key}",
                "",
                "Before repair:",
                before_text,
                "",
                "After repair:",
                after_text,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_evidence_assets() -> None:
    src_dir = FIG_DIR / "raw_synthetic_scene_20230707_15_SY4_T1"
    grid_src = src_dir / "raw_phase_grid.png"
    txt_src = src_dir / "captions_vqa.txt"
    json_src = src_dir / "captions_vqa.json"
    if grid_src.exists():
        shutil.copy2(grid_src, OUT / "qual_synthetic_phase_grid.png")
    if txt_src.exists():
        shutil.copy2(txt_src, OUT / "qual_synthetic_caption_vqa_text.txt")
    if json_src.exists():
        shutil.copy2(json_src, OUT / "qual_synthetic_caption_vqa_text.json")
    frame_paths = sorted(src_dir.glob("[0-4]_*.png"))
    for path in frame_paths:
        shutil.copy2(path, OUT / f"qual_frame_{path.name}")
    if frame_paths:
        phases = ["Pre-recognition", "Recognition", "Judgment", "Action", "Avoidance"]
        imgs = [Image.open(path).convert("RGB") for path in frame_paths[:5]]
        thumb_w = 390
        thumbs = []
        for img in imgs:
            w, h = img.size
            thumbs.append(img.resize((thumb_w, int(round(h * thumb_w / w))), Image.Resampling.LANCZOS))
        gap = 12
        label_h = 56
        canvas_w = len(thumbs) * thumb_w + (len(thumbs) - 1) * gap
        canvas_h = label_h + max(img.height for img in thumbs)
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        colors = [(220, 38, 38), (220, 38, 38), (234, 179, 8), (22, 101, 52), (37, 99, 235)]
        x = 0
        for idx, img in enumerate(thumbs):
            label = phases[idx]
            tw = draw.textlength(label, font=font(28, True))
            draw.text((x + (thumb_w - tw) / 2, 12), label, font=font(28, True), fill=colors[idx])
            canvas.paste(img, (x, label_h))
            draw.rectangle([x, label_h, x + thumb_w, label_h + img.height], outline=BORDER, width=2)
            x += thumb_w + gap
        canvas.save(OUT / "qual_synthetic_phase_grid_labeled.png")


def make_index() -> None:
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    lines = [
        "# Qualitative Result Assets",
        "",
        "Use these files to assemble the qualitative-results figures in PowerPoint or LaTeX.",
        "",
        "- `qual_synthetic_phase_grid.png`: local synthetic sample frames across behavioral phases.",
        "- `qual_synthetic_phase_grid_labeled.png`: same frames with phase labels.",
        "- `qual_frame_*.png`: individual phase frames for manual PowerPoint layouts.",
        "- `qual_synthetic_caption_vqa_text.txt`: caption and VQA text for the synthetic sample.",
        "- `qual_caption_fusion_panel.png`: actual public-test caption comparison from submission JSON files.",
        "- `qual_caption_fusion_example.txt`: raw text behind the caption fusion panel.",
        "- `qual_caption_repair_panel.png`: clean example of incomplete-caption repair.",
        "- `qual_caption_repair_example.txt`: raw text behind the repair panel.",
        "- `qual_vqa_consensus_panel.png`: actual public-test VQA question/prediction examples.",
        "- `qual_vqa_consensus_examples.txt`: raw text behind the VQA panel.",
        "",
        "Generated files:",
    ]
    lines.extend(f"- `{name}`" for name in files)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_evidence_assets()
    make_caption_fusion_panel()
    make_caption_repair_panel()
    make_vqa_consensus_panel()
    make_index()
    print(OUT)
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
