from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OPTIONS = {"a", "b", "c", "d", "e"}


def parse_letter(text: object) -> str:
    clean = str(text).strip().lower()
    match = re.search(r"\b([abcde])\b", clean)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([abcde])[\).:\s-]?", clean)
    if match:
        return match.group(1)
    return "a"


def prediction_text(obj: object) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("predict", "prediction", "response", "output", "generated_text", "text"):
            if key in obj:
                return str(obj[key])
    return str(obj)


def read_predictions(path: str | Path) -> list[str]:
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(prediction_text(json.loads(line)))
        return rows
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, list):
        return [prediction_text(row) for row in data]
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return [prediction_text(row) for row in data["predictions"]]
    raise ValueError(f"Unsupported predictions format: {path}")


def load_vqa(path: str | Path) -> list[dict[str, str]]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    return [{"id": str(row["id"]), "correct": parse_letter(row.get("correct", ""))} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dataset", required=True)
    parser.add_argument("--prediction-files", nargs="+", required=True)
    parser.add_argument("--fallback-vqa", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--output-vqa", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--min-votes", type=int, default=8)
    parser.add_argument("--min-margin", type=int, default=3)
    args = parser.parse_args()

    rows = json.load(open(args.subset_dataset, "r", encoding="utf-8"))
    qids = [str(row.get("metadata", {}).get("vqa_id", "")).strip() for row in rows]
    if any(not qid for qid in qids):
        raise ValueError("Subset dataset contains rows without metadata.vqa_id.")

    prediction_sets = [read_predictions(path) for path in args.prediction_files]
    for path, preds in zip(args.prediction_files, prediction_sets):
        if len(preds) != len(qids):
            raise ValueError(f"Prediction count mismatch for {path}: got {len(preds)} expected {len(qids)}")

    fallback_rows = load_vqa(args.fallback_vqa)
    fallback = {row["id"]: row["correct"] for row in fallback_rows}
    output_map = dict(fallback)

    changed = []
    considered = 0
    rejected = Counter()
    for idx, qid in enumerate(qids):
        votes = [parse_letter(preds[idx]) for preds in prediction_sets]
        counts = Counter(votes)
        winner, top_count = max(counts.items(), key=lambda item: (item[1], item[0]))
        runner_up = max((count for ans, count in counts.items() if ans != winner), default=0)
        fb = fallback.get(qid, "a")
        considered += 1
        if winner == fb:
            rejected["same_as_fallback"] += 1
            continue
        if top_count < args.min_votes:
            rejected["low_votes"] += 1
            continue
        if top_count - runner_up < args.min_margin:
            rejected["low_margin"] += 1
            continue
        output_map[qid] = winner
        changed.append(
            {
                "id": qid,
                "fallback": fb,
                "winner": winner,
                "votes": dict(sorted(counts.items())),
                "top_count": top_count,
                "margin": top_count - runner_up,
            }
        )

    output_rows = [{"id": row["id"], "correct": output_map[row["id"]]} for row in fallback_rows]
    out_vqa = Path(args.output_vqa)
    out_vqa.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output_rows, out_vqa.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = Path(args.caption)
    caption_out = out_dir / "caption_submission.json"
    vqa_out = out_dir / "vqa_submission.json"
    caption_out.write_bytes(caption.read_bytes())
    vqa_out.write_bytes(out_vqa.read_bytes())

    zip_path = Path(args.zip_output)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        archive.write(caption_out, "caption_submission.json")
        archive.write(vqa_out, "vqa_submission.json")

    report = {
        "subset_rows": len(qids),
        "num_prediction_files": len(args.prediction_files),
        "considered": considered,
        "changed": len(changed),
        "min_votes": args.min_votes,
        "min_margin": args.min_margin,
        "rejected": dict(rejected),
        "changed_examples": changed[:100],
        "prediction_files": args.prediction_files,
    }
    json.dump(report, open(args.report_output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("subset_rows", "num_prediction_files", "changed")}, indent=2))


if __name__ == "__main__":
    main()
