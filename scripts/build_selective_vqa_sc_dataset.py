from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_vqa(path: str | Path) -> dict[str, str]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    return {str(row["id"]): normalize(row.get("correct", "")) for row in rows}


def normalize(value: object) -> str:
    text = str(value).strip().lower()
    return text[:1] if text[:1] in {"a", "b", "c", "d", "e"} else "a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fallback-vqa", required=True)
    parser.add_argument("--candidate-vqa", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info-output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--min-disagree", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    rows = json.load(open(args.dataset, "r", encoding="utf-8"))
    fallback = load_vqa(args.fallback_vqa)
    candidates = [load_vqa(path) for path in args.candidate_vqa]

    selected = []
    report_rows = []
    for row in rows:
        meta = row.get("metadata", {})
        qid = str(meta.get("vqa_id", "")).strip()
        if not qid:
            continue
        fb = fallback.get(qid)
        answers = [cand.get(qid) for cand in candidates if cand.get(qid)]
        counts = Counter(answers)
        disagree = sum(1 for answer in answers if answer != fb)
        has_nonfallback_vote = any(answer != fb and count >= 2 for answer, count in counts.items())
        if disagree >= args.min_disagree and has_nonfallback_vote:
            selected.append(row)
            report_rows.append(
                {
                    "id": qid,
                    "fallback": fb,
                    "candidate_counts": dict(sorted(counts.items())),
                    "num_disagree": disagree,
                    "scope": meta.get("scope"),
                    "phase": meta.get("phase"),
                    "scenario_type": meta.get("scenario_type"),
                }
            )
            if args.max_rows and len(selected) >= args.max_rows:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(selected, output.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    info = {
        args.dataset_name: {
            "file_name": output.name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "videos": "videos",
            },
        }
    }
    info_path = Path(args.dataset_info_output)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(info, info_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    report = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "min_disagree": args.min_disagree,
        "candidate_files": args.candidate_vqa,
        "fallback_vqa": args.fallback_vqa,
        "selected_examples": report_rows[:50],
    }
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, report_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("input_rows", "selected_rows", "min_disagree")}, indent=2))


if __name__ == "__main__":
    main()
