#!/usr/bin/env python
"""Role/phase controlled fact-locked caption fusion variants."""

from __future__ import annotations

import argparse
import json
import zipfile
from copy import deepcopy
from pathlib import Path

from gate_fact_locked_caption import CAPTION_KEYS, dump_json, norm, should_accept
from subset_fact_locked_gate import candidate_score


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_zip(caption_path: Path, vqa_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def collect(best: dict, candidate: dict, source_policy: str) -> list[dict]:
    rows = []
    for sid, base_rows in best.items():
        cand_rows = candidate.get(sid, [])
        for idx, base_row in enumerate(base_rows):
            if idx >= len(cand_rows):
                continue
            phase = str((base_row.get("labels") or [""])[0])
            for key in CAPTION_KEYS:
                base_text = base_row.get(key, "")
                cand_text = cand_rows[idx].get(key, "")
                ok, reasons = should_accept(cand_text, base_text, source_policy)
                if not ok or norm(base_text) == norm(cand_text):
                    continue
                rows.append(
                    {
                        "scenario_id": sid,
                        "row_index": idx,
                        "key": key,
                        "role": key.replace("caption_", ""),
                        "phase": phase,
                        "score": candidate_score(cand_text, base_text),
                        "reasons": reasons,
                    }
                )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def parse_caps(spec: str) -> dict[tuple[str, str], int]:
    caps = {}
    if not spec:
        return caps
    for part in spec.split(","):
        role_phase, raw_limit = part.split("=", 1)
        role, phase = role_phase.split(":", 1)
        caps[(role, phase)] = int(raw_limit)
    return caps


def select(rows: list[dict], total: int, role_caps: dict[str, int], phase_caps: dict[tuple[str, str], int]) -> list[dict]:
    selected = []
    role_counts: dict[str, int] = {}
    phase_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if len(selected) >= total:
            break
        role = row["role"]
        phase_key = (role, row["phase"])
        if role in role_caps and role_counts.get(role, 0) >= role_caps[role]:
            continue
        if phase_key in phase_caps and phase_counts.get(phase_key, 0) >= phase_caps[phase_key]:
            continue
        selected.append(row)
        role_counts[role] = role_counts.get(role, 0) + 1
        phase_counts[phase_key] = phase_counts.get(phase_key, 0) + 1
    return selected


def build(best: dict, candidate: dict, selected: list[dict]) -> dict:
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
    parser.add_argument("--name", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--vehicle-cap", type=int)
    parser.add_argument("--pedestrian-cap", type=int)
    parser.add_argument("--phase-caps", default="")
    args = parser.parse_args()

    best = load_json(args.best_caption)
    candidate = load_json(args.candidate_caption)
    rows = collect(best, candidate, args.source_policy)
    role_caps = {}
    if args.vehicle_cap is not None:
        role_caps["vehicle"] = args.vehicle_cap
    if args.pedestrian_cap is not None:
        role_caps["pedestrian"] = args.pedestrian_cap
    phase_caps = parse_caps(args.phase_caps)
    selected = select(rows, args.total, role_caps, phase_caps)
    fused = build(best, candidate, selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    caption_path = args.output_dir / f"caption_submission_{args.name}.json"
    report_path = args.output_dir / f"caption_{args.name}_report.json"
    zip_path = args.output_dir / f"submission_{args.name}.zip"
    dump_json(fused, caption_path)
    dump_json(
        {
            "name": args.name,
            "total": args.total,
            "fields_changed": len(selected),
            "role_caps": role_caps,
            "phase_caps": {f"{k[0]}:{k[1]}": v for k, v in phase_caps.items()},
            "selected": selected,
        },
        report_path,
    )
    make_zip(caption_path, args.vqa, zip_path)
    print(json.dumps({"name": args.name, "fields_changed": len(selected), "zip": str(zip_path)}))


if __name__ == "__main__":
    main()
