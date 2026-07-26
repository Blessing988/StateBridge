#!/usr/bin/env python
"""Evaluate gold scene-graph facts converted back into caption templates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.caption_slots import _clean_caption_text, _template_caption
from synwts.evaluation import evaluate_caption_predictions
from synwts.facts import build_fact_rows
from synwts.index import build_index
from synwts.io import write_json
from synwts.parsers import load_caption_phases
from synwts.schema import ScenarioRecord


def _preferred_caption_file(record: ScenarioRecord) -> str | None:
    for view in ("overhead_view", "vehicle_view"):
        if view in record.caption_files:
            return record.caption_files[view]
    return next(iter(record.caption_files.values()), None)


def _fact_bundle(facts: dict[str, Any], phase: str) -> dict[str, Any]:
    return {
        "global": facts.get("global", {}),
        "phase": facts.get("phases", {}).get(phase, {}),
    }


def _word_count(text: str) -> int:
    return len(text.split())


def build_gold_scenegraph_caption_oracle(
    dataset_root: str | Path,
    *,
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records = build_index(dataset_root, splits=(split,))
    facts_by_scenario = {row["scenario_id"]: row for row in build_fact_rows(records)}

    predictions: dict[str, list[dict[str, Any]]] = {}
    coverage = {
        "records": len(records),
        "rows": 0,
        "missing_pedestrian_template": 0,
        "missing_vehicle_template": 0,
        "pedestrian_words": [],
        "vehicle_words": [],
        "samples": [],
    }

    for record in records:
        caption_path = _preferred_caption_file(record)
        if not caption_path:
            continue
        facts = facts_by_scenario.get(record.scenario_id, {})
        rows: list[dict[str, Any]] = []
        for phase_row in load_caption_phases(caption_path):
            phase = str(phase_row["label"])
            bundle = _fact_bundle(facts, phase)
            pedestrian = _clean_caption_text(_template_caption("pedestrian", bundle))
            vehicle = _clean_caption_text(_template_caption("vehicle", bundle))
            if not pedestrian:
                coverage["missing_pedestrian_template"] += 1
            if not vehicle:
                coverage["missing_vehicle_template"] += 1
            coverage["pedestrian_words"].append(_word_count(pedestrian))
            coverage["vehicle_words"].append(_word_count(vehicle))
            out_row = {
                "labels": [phase],
                "caption_pedestrian": pedestrian,
                "caption_vehicle": vehicle,
            }
            rows.append(out_row)
            coverage["rows"] += 1
            if len(coverage["samples"]) < 5:
                coverage["samples"].append({"scenario_id": record.scenario_id, **out_row})
        predictions[record.scenario_id] = rows

    coverage["avg_pedestrian_words"] = round(mean(coverage["pedestrian_words"] or [0]), 2)
    coverage["avg_vehicle_words"] = round(mean(coverage["vehicle_words"] or [0]), 2)
    del coverage["pedestrian_words"]
    del coverage["vehicle_words"]
    return predictions, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    predictions, coverage = build_gold_scenegraph_caption_oracle(args.dataset_root, split=args.split)
    write_json(args.output, predictions)
    eval_report = evaluate_caption_predictions(args.dataset_root, args.output, split=args.split)
    report = {
        "split": args.split,
        "prediction_output": str(args.output),
        "coverage": coverage,
        "evaluation": eval_report,
        "interpretation": (
            "Gold VQA facts converted to deterministic templates. "
            "If this is below current caption scores, wording/template realization is the bottleneck."
        ),
    }
    write_json(args.report_output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
