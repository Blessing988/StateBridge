#!/usr/bin/env python
"""Build VQA-scene-graph caption candidates and gated submission zips."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.caption_facts import _predicted_wts_fact_map
from synwts.caption_slots import (
    rerank_caption_slot_variants,
    rewrite_caption_submission_slots,
)
from synwts.validators import validate_caption_submission, validate_vqa_submission


def _margin_name(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "n").replace(".", "p")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _zip_submission(caption_path: Path, vqa_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def _parse_margins(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--margins must contain at least one value.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-caption", type=Path, required=True)
    parser.add_argument("--wts-vqa-json", type=Path, required=True)
    parser.add_argument("--vqa-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="scenegraph")
    parser.add_argument("--margins", default="0.8,1.0,1.2")
    parser.add_argument("--min-source-overlap", type=float, default=0.58)
    parser.add_argument("--max-changed-rows", type=int, default=60)
    parser.add_argument("--reward-slot-order", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenegraph_path = args.output_dir / f"{args.name}_predicted_scenegraph.json"
    template_caption_path = args.output_dir / f"caption_submission_{args.name}_template.json"
    template_report_path = args.output_dir / f"caption_{args.name}_template_report.json"

    scenegraph = _predicted_wts_fact_map(args.wts_vqa_json, args.vqa_submission)
    _write_json(scenegraph_path, scenegraph)

    _template, template_report = rewrite_caption_submission_slots(
        caption=args.base_caption,
        output=template_caption_path,
        wts_vqa_json=args.wts_vqa_json,
        vqa_submission=args.vqa_submission,
        report_output=template_report_path,
        mode="template",
        max_added_sentences=0,
    )

    vqa_validation = validate_vqa_submission(args.vqa_submission)
    if not vqa_validation["ok"]:
        raise ValueError(f"VQA validation failed: {vqa_validation['errors'][:3]}")

    outputs = {
        "scenegraph": str(scenegraph_path),
        "template_caption": str(template_caption_path),
        "template_report": str(template_report_path),
        "template_changed_rows": template_report.get("changed_rows"),
        "variants": [],
    }

    for margin in _parse_margins(args.margins):
        suffix = f"{args.name}_m{_margin_name(margin)}"
        caption_path = args.output_dir / f"caption_submission_{suffix}.json"
        report_path = args.output_dir / f"caption_{suffix}_report.json"
        zip_path = args.output_dir / f"submission_{suffix}.zip"
        _submission, report = rerank_caption_slot_variants(
            base_caption=args.base_caption,
            candidates={"scenegraph_template": template_caption_path},
            output=caption_path,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            report_output=report_path,
            min_switch_margin=margin,
            min_source_overlap=args.min_source_overlap,
            max_changed_rows=args.max_changed_rows,
            preserve_base_order=not args.reward_slot_order,
        )
        caption_validation = validate_caption_submission(caption_path)
        if not caption_validation["ok"]:
            raise ValueError(f"Caption validation failed for {caption_path}: {caption_validation['errors'][:3]}")
        _zip_submission(caption_path, args.vqa_submission, zip_path)
        outputs["variants"].append(
            {
                "margin": margin,
                "caption": str(caption_path),
                "report": str(report_path),
                "zip": str(zip_path),
                "changed_rows": report.get("changed_rows"),
                "source_counts": report.get("source_counts"),
            }
        )

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
