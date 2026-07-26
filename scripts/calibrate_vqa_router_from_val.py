#!/usr/bin/env python
"""Build VQA submissions from prior-real validation-calibrated specialist routing."""

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
from synwts.vqa_fusion import classify_vqa_question, load_vqa_question_rows


DEFAULT_MODELS = ("real", "real_aug", "env", "vehicle", "ped_dynamic", "geometry")

MODE_PRESETS = {
    "high_precision": {"min_n": 25, "min_acc": 0.72, "margin": 0.035, "support": 2, "change_cap": 0.035},
    "balanced": {"min_n": 20, "min_acc": 0.68, "margin": 0.015, "support": 1, "change_cap": 0.065},
    "aggressive": {"min_n": 12, "min_acc": 0.62, "margin": -0.005, "support": 1, "change_cap": 0.100},
}


def load_generated_predictions(path: Path) -> list[str]:
    preds: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            preds.append(_parse_vqa_letter(str(row.get("predict", ""))))
    return preds


def load_submission(path: Path) -> dict[str, str]:
    return {
        str(row.get("id", "")).strip(): _parse_vqa_letter(str(row.get("correct", "")))
        for row in read_json(path)
        if str(row.get("id", "")).strip()
    }


def load_val_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(read_json(path)):
        meta = item.get("metadata", {}) or {}
        question = str(meta.get("question") or _question_from_instruction(str(item.get("instruction", ""))))
        scope = str(meta.get("scope") or "")
        qtype = str(meta.get("question_type") or classify_vqa_question(question, scope=scope))
        rows.append(
            {
                "idx": idx,
                "gold": _parse_vqa_letter(str(item.get("output", ""))),
                "question": question,
                "question_type": qtype,
                "scope": scope,
                "phase": str(meta.get("phase") or ""),
                "scenario_type": str(meta.get("scenario_type") or ""),
            }
        )
    return rows


def _question_from_instruction(text: str) -> str:
    match = re.search(r"Question:\s*(.+?)\nOptions:", text, flags=re.S)
    return " ".join(match.group(1).split()) if match else ""


def calibrate(
    val_rows: list[dict[str, Any]],
    val_preds: dict[str, list[str]],
) -> dict[str, Any]:
    by_group: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    overall: dict[str, Counter[str]] = defaultdict(Counter)
    for i, row in enumerate(val_rows):
        qtype = row["question_type"]
        gold = row["gold"]
        for name, preds in val_preds.items():
            pred = preds[i] if i < len(preds) else ""
            ok = pred == gold
            by_group[qtype][name]["total"] += 1
            by_group[qtype][name]["correct"] += int(ok)
            overall[name]["total"] += 1
            overall[name]["correct"] += int(ok)

    groups: dict[str, dict[str, Any]] = {}
    for qtype, model_stats in sorted(by_group.items()):
        models = {}
        for name, counts in sorted(model_stats.items()):
            total = counts["total"]
            correct = counts["correct"]
            models[name] = {
                "n": total,
                "correct": correct,
                "acc": round(correct / total, 6) if total else 0.0,
            }
        groups[qtype] = {"models": models}
    return {
        "overall": {
            name: {
                "n": counts["total"],
                "correct": counts["correct"],
                "acc": round(counts["correct"] / counts["total"], 6) if counts["total"] else 0.0,
            }
            for name, counts in sorted(overall.items())
        },
        "groups": groups,
    }


def select_model_for_group(
    *,
    qtype: str,
    stats: dict[str, Any],
    mode: dict[str, float],
    model_order: tuple[str, ...],
    baseline_names: tuple[str, ...],
) -> tuple[str | None, dict[str, Any]]:
    models = stats["groups"].get(qtype, {}).get("models", {})
    candidates = [
        (name, models[name]["acc"], models[name]["n"])
        for name in model_order
        if name in models and models[name]["n"] >= mode["min_n"]
    ]
    if not candidates:
        return None, {"reason": "no_candidate"}

    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_name, best_acc, best_n = candidates[0]
    baseline_accs = [
        models[name]["acc"]
        for name in baseline_names
        if name in models and models[name]["n"] >= mode["min_n"]
    ]
    baseline_acc = max(baseline_accs) if baseline_accs else 0.0
    accepted = best_acc >= mode["min_acc"] and (best_acc - baseline_acc) >= mode["margin"]
    return (
        best_name if accepted else None,
        {
            "best": best_name,
            "best_acc": best_acc,
            "best_n": best_n,
            "baseline_acc": baseline_acc,
            "margin": best_acc - baseline_acc,
            "accepted": accepted,
        },
    )


def answer_support(qid: str, answer: str, public_preds: dict[str, dict[str, str]], names: tuple[str, ...]) -> int:
    return sum(1 for name in names if public_preds.get(name, {}).get(qid) == answer)


def build_mode_submission(
    *,
    mode_name: str,
    mode: dict[str, float],
    public_rows: list[dict[str, Any]],
    public_preds: dict[str, dict[str, str]],
    fallback_name: str,
    calibration: dict[str, Any],
    model_order: tuple[str, ...],
    baseline_names: tuple[str, ...],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    fallback = public_preds[fallback_name]
    rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    used_by_type: Counter[str] = Counter()
    selected_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    decisions: dict[str, Any] = {}

    for public_row in public_rows:
        qid = public_row["id"]
        qtype = public_row["question_type"]
        fallback_answer = fallback.get(qid, "")
        model_name, decision = decisions.get(qtype, (None, None))
        if decision is None:
            model_name, decision = select_model_for_group(
                qtype=qtype,
                stats=calibration,
                mode=mode,
                model_order=model_order,
                baseline_names=baseline_names,
            )
            decisions[qtype] = (model_name, decision)

        selected = fallback_answer
        support = 0
        if model_name and model_name in public_preds:
            candidate = public_preds[model_name].get(qid, "")
            support = answer_support(qid, candidate, public_preds, model_order)
            if candidate in {"a", "b", "c", "d", "e"} and support >= mode["support"]:
                selected = candidate
                used_by_type[model_name] += 1
                selected_by_type[qtype][model_name] += 1

        row = {"id": qid, "correct": selected}
        rows.append(row)
        if selected != fallback_answer:
            changed.append(
                {
                    "id": qid,
                    "question_type": qtype,
                    "question": public_row.get("question", ""),
                    "fallback": fallback_answer,
                    "selected": selected,
                    "model": model_name,
                    "support": support,
                    "decision": decision,
                }
            )
        report_rows.append(
            {
                "id": qid,
                "question_type": qtype,
                "scope": public_row.get("scope", ""),
                "phase": public_row.get("phase", ""),
                "selected": selected,
                "fallback": fallback_answer,
                "changed": selected != fallback_answer,
                "model": model_name,
                "support": support,
            }
        )

    cap = int(round(len(rows) * mode["change_cap"]))
    if len(changed) > cap:
        keep = {item["id"] for item in changed[:cap]}
        capped_rows = []
        for public_row, row in zip(public_rows, rows):
            if public_row["id"] not in keep:
                capped_rows.append({"id": public_row["id"], "correct": fallback.get(public_row["id"], "")})
            else:
                capped_rows.append(row)
        rows = capped_rows
        report_rows = [
            {**row, "selected": next(out["correct"] for out in rows if out["id"] == row["id"]), "changed": row["id"] in keep}
            for row in report_rows
        ]
        changed = [item for item in changed if item["id"] in keep]

    report = {
        "mode": mode_name,
        "preset": mode,
        "total": len(rows),
        "fallback": fallback_name,
        "changed_from_fallback": len(changed),
        "changed_share": round(len(changed) / max(len(rows), 1), 6),
        "used_by_model": dict(used_by_type),
        "selected_by_type": {qtype: dict(counter) for qtype, counter in sorted(selected_by_type.items())},
        "decisions": {qtype: decision for qtype, (_, decision) in sorted(decisions.items())},
        "changed_rows": changed,
        "rows": report_rows,
    }
    return rows, report


def write_zip(path: Path, caption_path: Path, vqa_path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value}")
        name, raw = value.split("=", 1)
        parsed[name.strip()] = Path(raw.strip())
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dataset", type=Path, required=True)
    parser.add_argument("--val-pred", action="append", default=[], help="name=generated_predictions.jsonl")
    parser.add_argument("--public-vqa-json", type=Path, required=True)
    parser.add_argument("--public-submission", action="append", required=True, help="name=vqa_submission.json")
    parser.add_argument("--fallback-name", default="partial")
    parser.add_argument("--caption-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="submission_calibrated_vqa_router")
    parser.add_argument("--model-order", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--baseline-names", default="real_aug,real")
    args = parser.parse_args()

    val_rows = load_val_rows(args.val_dataset)
    val_pred_paths = parse_named_paths(args.val_pred)
    public_submission_paths = parse_named_paths(args.public_submission)
    val_preds = {name: load_generated_predictions(path) for name, path in val_pred_paths.items()}
    for name, preds in val_preds.items():
        if len(preds) != len(val_rows):
            raise ValueError(f"{name} val predictions length {len(preds)} != val rows {len(val_rows)}")

    calibration = calibrate(val_rows, val_preds)
    public_rows = load_vqa_question_rows(args.public_vqa_json)
    public_preds = {name: load_submission(path) for name, path in public_submission_paths.items()}
    if args.fallback_name not in public_preds:
        raise ValueError(f"Missing fallback public submission {args.fallback_name}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "calibration_report.json", calibration)

    model_order = tuple(item.strip() for item in args.model_order.split(",") if item.strip())
    baseline_names = tuple(item.strip() for item in args.baseline_names.split(",") if item.strip())
    summary: dict[str, Any] = {"calibration": calibration["overall"], "modes": {}}

    for mode_name, mode in MODE_PRESETS.items():
        rows, report = build_mode_submission(
            mode_name=mode_name,
            mode=mode,
            public_rows=public_rows,
            public_preds=public_preds,
            fallback_name=args.fallback_name,
            calibration=calibration,
            model_order=model_order,
            baseline_names=baseline_names,
        )
        mode_dir = output_dir / f"{args.prefix}_{mode_name}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        vqa_path = mode_dir / "vqa_submission.json"
        write_json(vqa_path, rows)
        write_json(mode_dir / "report.json", report)
        caption_out = mode_dir / "caption_submission.json"
        caption_out.write_bytes(args.caption_submission.read_bytes())
        zip_path = output_dir.parent / f"{args.prefix}_{mode_name}.zip"
        write_zip(zip_path, caption_out, vqa_path)
        summary["modes"][mode_name] = {
            "zip": str(zip_path),
            "changed_from_fallback": report["changed_from_fallback"],
            "changed_share": report["changed_share"],
        }
        print(f"{mode_name}: wrote {zip_path} changed={report['changed_from_fallback']}")

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
