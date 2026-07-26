#!/usr/bin/env python
"""Route WTS VQA answers across specialist submissions by question type."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.io import read_json, write_json
from synwts.submission import _parse_vqa_letter
from synwts.vqa_fusion import classify_vqa_question


ROUTES = {
    "weather_lighting": ("env", "joint", "qwen35", "best"),
    "road_surface": ("env", "joint", "qwen35", "best"),
    "obstacle": ("env", "geometry", "joint", "best"),
    "pedestrian_attribute": ("env", "joint", "qwen35", "best"),
    "pedestrian_clothing": ("env", "joint", "qwen35", "best"),
    "road_context": ("env", "joint", "qwen35", "best"),
    "environment_other": ("env", "joint", "qwen35", "best"),
    "vehicle_fov": ("vehicle", "visual", "joint", "best"),
    "vehicle_action": ("vehicle", "visual", "joint", "best"),
    "vehicle_position": ("geometry", "vehicle", "visual", "joint", "best"),
    "vehicle_distance": ("geometry", "vehicle", "visual", "joint", "best"),
    "pedestrian_orientation": ("geometry", "visual", "joint", "best"),
    "pedestrian_position": ("geometry", "visual", "joint", "best"),
    "pedestrian_distance": ("geometry", "visual", "joint", "best"),
    "pedestrian_line_of_sight": ("ped_dynamic", "visual", "joint", "best"),
    "pedestrian_attention": ("ped_dynamic", "visual", "joint", "best"),
    "pedestrian_direction": ("ped_dynamic", "visual", "joint", "best"),
    "pedestrian_speed": ("ped_dynamic", "visual", "joint", "best"),
    "pedestrian_action": ("ped_dynamic", "visual", "joint", "best"),
    "phase_other": ("joint", "visual", "best"),
}

DEFAULT_WEIGHTS = {
    "best": 1.0,
    "joint": 1.15,
    "qwen35": 1.10,
    "visual": 1.05,
    "env": 1.30,
    "vehicle": 1.30,
    "ped_dynamic": 1.30,
    "geometry": 1.30,
}


def route_vqa(
    *,
    vqa_json: Path,
    submissions: dict[str, Path],
    output: Path,
    fallback_name: str,
    weights: dict[str, float],
    report_output: Path | None,
    caption_submission: Path | None,
    zip_output: Path | None,
) -> list[dict[str, str]]:
    rows = _load_rows(vqa_json)
    named = {name: _load_submission(path) for name, path in submissions.items()}
    if fallback_name not in named:
        raise ValueError(f"Missing fallback submission: {fallback_name}")
    fallback = named[fallback_name]

    out: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    changed_by_type: Counter[str] = Counter()
    used_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    missing_by_name: Counter[str] = Counter()

    for row in rows:
        qid = row["id"]
        qtype = row["question_type"]
        fallback_answer = fallback.get(qid, "")
        route = [name for name in ROUTES.get(qtype, (fallback_name,)) if name in named]
        if fallback_name not in route:
            route.append(fallback_name)
        answer, used, scores = _weighted_vote(qid, route=route, named=named, weights=weights, fallback=fallback_answer)
        out.append({"id": qid, "correct": answer})
        if answer != fallback_answer:
            changed_by_type[qtype] += 1
        for name in route:
            if qid not in named[name]:
                missing_by_name[name] += 1
        used_by_type[qtype].update(used)
        report_rows.append(
            {
                "id": qid,
                "scenario_id": row["scenario_id"],
                "question_type": qtype,
                "scope": row["scope"],
                "phase": row["phase"],
                "question": row["question"],
                "route": route,
                "selected": answer,
                "fallback": fallback_answer,
                "changed": answer != fallback_answer,
                "used": used,
                "scores": scores,
            }
        )

    write_json(output, out)
    if report_output:
        write_json(
            report_output,
            {
                "total": len(out),
                "fallback": fallback_name,
                "submissions": sorted(named),
                "changed_from_fallback": sum(changed_by_type.values()),
                "changed_by_type": dict(changed_by_type.most_common()),
                "used_by_type": {key: dict(value) for key, value in sorted(used_by_type.items())},
                "missing_by_name": dict(missing_by_name.most_common()),
                "rows": report_rows,
            },
        )
    if zip_output:
        if caption_submission is None:
            raise ValueError("--caption-submission is required with --zip-output")
        with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(caption_submission, "caption_submission.json")
            zf.write(output, "vqa_submission.json")
    return out


def _weighted_vote(
    qid: str,
    *,
    route: list[str],
    named: dict[str, dict[str, str]],
    weights: dict[str, float],
    fallback: str,
) -> tuple[str, list[str], dict[str, float]]:
    scores: Counter[str] = Counter()
    used: list[str] = []
    for name in route:
        answer = named[name].get(qid)
        if answer not in {"a", "b", "c", "d", "e"}:
            continue
        scores[answer] += weights.get(name, 1.0)
        used.append(name)
    if not scores:
        return fallback, [], {}
    best_score = max(scores.values())
    winners = {answer for answer, score in scores.items() if score == best_score}
    for name in route:
        answer = named[name].get(qid)
        if answer in winners:
            return answer, used, {key: float(value) for key, value in scores.items()}
    if fallback in winners:
        return fallback, used, {key: float(value) for key, value in scores.items()}
    return sorted(winners)[0], used, {key: float(value) for key, value in scores.items()}


def _load_submission(path: Path) -> dict[str, str]:
    rows = read_json(path)
    return {
        str(row.get("id", "")).strip(): _parse_vqa_letter(str(row.get("correct", "")))
        for row in rows
        if str(row.get("id", "")).strip()
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(read_json(path)):
        scenario_id = _scenario_from_item(item, fallback=f"item_{item_idx:05d}")
        if "event_phase" not in item:
            for q_idx, question in enumerate(item.get("conversations", [])):
                rows.append(_row(question, item_idx, scenario_id, "environment", None, None, q_idx))
            continue
        scope = _infer_scope(item)
        for phase_idx, phase in enumerate(item.get("event_phase", [])):
            labels = phase.get("labels") or []
            phase_label = str(labels[0]) if labels else ""
            for q_idx, question in enumerate(phase.get("conversations", [])):
                rows.append(_row(question, item_idx, scenario_id, scope, phase_label, phase_idx, q_idx))
    return rows


def _row(
    question: dict[str, Any],
    item_idx: int,
    scenario_id: str,
    scope: str,
    phase: str | None,
    phase_idx: int | None,
    q_idx: int,
) -> dict[str, Any]:
    text = str(question.get("question", "")).strip()
    qid = str(question.get("id", "")).strip() or f"item_{item_idx:05d}_phase_{phase_idx if phase_idx is not None else 'env'}_q_{q_idx:03d}"
    return {
        "id": qid,
        "scenario_id": scenario_id,
        "question": text,
        "question_type": classify_vqa_question(text, scope=scope),
        "scope": scope,
        "phase": phase,
    }


def _scenario_from_item(item: dict[str, Any], *, fallback: str) -> str:
    names = [str(name) for name in item.get("videos", [])]
    if not names:
        return fallback
    stem = Path(names[0]).stem
    if "_normal_" in stem:
        return stem
    parts = stem.split("_")
    return "_".join(parts[:4]) if len(parts) >= 4 else stem


def _infer_scope(item: dict[str, Any]) -> str:
    for phase in item.get("event_phase", []):
        for question in phase.get("conversations", []):
            q = str(question.get("question", "")).lower()
            if "vehicle" in q and any(term in q for term in ("field of view", "action taken by vehicle", "position of the vehicle relative", "relative distance of vehicle")):
                return "vehicle_view"
    return "overhead_view"


def _parse_named_paths(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_\\-]+", name):
            raise ValueError(f"Bad submission name: {name}")
        out[name] = Path(path)
    return out


def _parse_weights(value: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not value:
        return weights
    for item in value.split(","):
        if not item.strip():
            continue
        name, raw = item.split("=", 1)
        weights[name.strip()] = float(raw)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-json", required=True, type=Path)
    parser.add_argument("--submission", action="append", required=True, help="Named submission: name=path")
    parser.add_argument("--fallback-name", default="best")
    parser.add_argument("--weights", help="Optional comma list name=weight")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--caption-submission", type=Path)
    parser.add_argument("--zip-output", type=Path)
    args = parser.parse_args()

    rows = route_vqa(
        vqa_json=args.vqa_json,
        submissions=_parse_named_paths(args.submission),
        output=args.output,
        fallback_name=args.fallback_name,
        weights=_parse_weights(args.weights),
        report_output=args.report_output,
        caption_submission=args.caption_submission,
        zip_output=args.zip_output,
    )
    print(f"Wrote {len(rows)} routed VQA rows to {args.output}")


if __name__ == "__main__":
    main()
