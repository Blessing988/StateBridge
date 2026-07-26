#!/usr/bin/env python
"""Create top-k variants from gated fact-locked caption candidates."""

from __future__ import annotations

import argparse
import json
import zipfile
from copy import deepcopy
from pathlib import Path

from gate_fact_locked_caption import CAPTION_KEYS, dump_json, local_quality, norm, should_accept, tokens


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_zip(caption_path: Path, vqa_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def candidate_score(candidate: str, baseline: str) -> float:
    cand = norm(candidate)
    base = norm(baseline)
    base_len = max(1, len(tokens(base)))
    ratio = len(tokens(cand)) / base_len
    trunc_bonus = 3.0 if base.endswith((" with", " on", " no", " traffic", " road", " is", " and")) else 0.0
    sentence_bonus = 1.0 if cand.endswith(".") and not base.endswith(".") else 0.0
    length_penalty = abs(1.0 - ratio) * 0.6
    return local_quality(cand) - local_quality(base) + trunc_bonus + sentence_bonus - length_penalty


def collect_candidates(best: dict, candidate: dict, source_policy: str) -> list[dict]:
    rows = []
    for scenario_id, base_rows in best.items():
        cand_rows = candidate.get(scenario_id, [])
        for row_index, base_row in enumerate(base_rows):
            if row_index >= len(cand_rows):
                continue
            cand_row = cand_rows[row_index]
            for key in CAPTION_KEYS:
                base_text = base_row.get(key, "")
                cand_text = cand_row.get(key, "")
                ok, reasons = should_accept(cand_text, base_text, source_policy)
                if not ok or norm(cand_text) == norm(base_text):
                    continue
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "row_index": row_index,
                        "key": key,
                        "score": candidate_score(cand_text, base_text),
                        "reasons": reasons,
                    }
                )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def build_subset(best: dict, candidate: dict, selected: list[dict]) -> dict:
    fused = deepcopy(best)
    for item in selected:
        sid = item["scenario_id"]
        idx = item["row_index"]
        key = item["key"]
        fused[sid][idx][key] = norm(candidate[sid][idx][key])
    return fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-caption", type=Path, required=True)
    parser.add_argument("--candidate-caption", type=Path, required=True)
    parser.add_argument("--vqa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-policy", default="loose")
    parser.add_argument("--limits", default="25,35,45")
    parser.add_argument("--prefix", default="fact_locked_gate_v3")
    args = parser.parse_args()

    best = load_json(args.best_caption)
    candidate = load_json(args.candidate_caption)
    candidates = collect_candidates(best, candidate, args.source_policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        "source_policy": args.source_policy,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:100],
    }
    dump_json(full_report, args.output_dir / f"caption_{args.prefix}_candidate_pool.json")

    for raw_limit in [x.strip() for x in args.limits.split(",") if x.strip()]:
        limit = int(raw_limit)
        selected = candidates[:limit]
        fused = build_subset(best, candidate, selected)
        caption_path = args.output_dir / f"caption_submission_{args.prefix}_{limit}.json"
        report_path = args.output_dir / f"caption_{args.prefix}_{limit}_report.json"
        zip_path = args.output_dir / f"submission_{args.prefix}_{limit}.zip"
        dump_json(fused, caption_path)
        dump_json(
            {
                "source_policy": args.source_policy,
                "requested_limit": limit,
                "fields_changed": len(selected),
                "selected": selected,
            },
            report_path,
        )
        make_zip(caption_path, args.vqa, zip_path)
        print(json.dumps({"limit": limit, "fields_changed": len(selected), "zip": str(zip_path)}))


if __name__ == "__main__":
    main()
