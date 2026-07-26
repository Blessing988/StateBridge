#!/usr/bin/env python
"""Conservative gated fusion for fact-locked caption rewrite candidates."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from copy import deepcopy
from pathlib import Path


CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")

BAD_ENDING_RE = re.compile(
    r"(\b(and|or|with|of|in|on|to|from|as|which|that|while|because|where|clear and)\.?)$",
    re.IGNORECASE,
)
PROMPT_ARTIFACT_RE = re.compile(
    r"(caption_pedestrian|caption_vehicle|locked fact|segment notes|return json|target segment|complete description)",
    re.IGNORECASE,
)
ROLE_SELF_RE = re.compile(
    r"(pedestrian\s+(?:is|was)?\s*positioned[^.]{0,80}\bin front of the pedestrian|"
    r"vehicle\s+(?:is|was)?\s*positioned[^.]{0,80}\bin front of the vehicle)",
    re.IGNORECASE,
)
DRIVER_AWARE_RE = re.compile(r"\bdriver\b.{0,40}\b(aware|noticed|mindful|saw)\b", re.IGNORECASE)
ZERO_COLLISION_RE = re.compile(r"(0\s*km/h.{0,80}collid|collid.{0,80}0\s*km/h)", re.IGNORECASE)
FRAGMENT_RE = re.compile(
    r"(measuring\s+\d+\s+meters\s+in\s+(?:height|width)\s+and\.|"
    r"\bThere is\.|\bstood still on a clear and\.|"
    r"\bbut not both sides\.|"
    r"\bin front of the road\.|"
    r"\bhas a cloudy field of view\b)",
    re.IGNORECASE,
)

PROTECTED_PATTERNS = [
    re.compile(r"\b\d+\s*km/h\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*meters?\s+in\s+height\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*meters?\s+in\s+width\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*cm\b", re.IGNORECASE),
    re.compile(r"\b(?:one|two)-way traffic\b", re.IGNORECASE),
    re.compile(r"\b(?:light|usual|heavy)\s+traffic(?:\s+volume)?\b", re.IGNORECASE),
]
PROTECTED_TERMS = (
    "obstacle",
    "sidewalk",
    "roadside strip",
    "street light",
    "street lights",
    "residential road",
    "asphalt",
    "dry",
    "wet",
    "level",
    "slope",
    "cloudy",
    "sunny",
    "rainy",
    "dim",
    "bright",
    "black",
    "white",
    "blue",
    "navy",
    "beige",
    "gray",
    "grey",
    "red",
    "green",
    "yellow",
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text.lower())


def protected_items(text: str) -> set[str]:
    low = text.lower()
    items = {term for term in PROTECTED_TERMS if term in low}
    for pattern in PROTECTED_PATTERNS:
        for match in pattern.findall(text):
            items.add(re.sub(r"\s+", " ", match.lower()))
    return items


def bad_reasons(candidate: str, baseline: str, min_ratio: float, max_ratio: float) -> list[str]:
    cand = norm(candidate)
    base = norm(baseline)
    reasons: list[str] = []
    if not cand:
        reasons.append("empty")
        return reasons
    if PROMPT_ARTIFACT_RE.search(cand):
        reasons.append("prompt_artifact")
    if BAD_ENDING_RE.search(cand):
        reasons.append("bad_ending")
    if ROLE_SELF_RE.search(cand):
        reasons.append("role_self")
    if DRIVER_AWARE_RE.search(cand):
        reasons.append("driver_aware")
    if ZERO_COLLISION_RE.search(cand):
        reasons.append("zero_collision")
    if FRAGMENT_RE.search(cand):
        reasons.append("fragment")
    base_len = max(1, len(tokens(base)))
    cand_len = len(tokens(cand))
    ratio = cand_len / base_len
    if ratio < min_ratio:
        reasons.append(f"too_short:{ratio:.2f}")
    if ratio > max_ratio:
        reasons.append(f"too_long:{ratio:.2f}")

    missing = protected_items(base) - protected_items(cand)
    if missing:
        compact = ",".join(sorted(missing)[:5])
        reasons.append(f"missing_protected:{compact}")
    return reasons


def local_quality(text: str) -> float:
    t = norm(text)
    score = 0.0
    score += 0.015 * len(tokens(t))
    score += 0.15 * min(8, t.count("."))
    for term in ("pedestrian", "vehicle", "weather", "road", "traffic", "speed", "line of sight"):
        if term in t.lower():
            score += 0.3
    if BAD_ENDING_RE.search(t) or FRAGMENT_RE.search(t):
        score -= 5.0
    return score


def should_accept(candidate: str, baseline: str, policy: str) -> tuple[bool, list[str]]:
    if policy == "repair":
        min_ratio, max_ratio, margin = 0.90, 1.20, -0.20
    elif policy == "strict":
        min_ratio, max_ratio, margin = 0.96, 1.08, 0.20
    elif policy == "moderate":
        min_ratio, max_ratio, margin = 0.90, 1.15, 0.08
    elif policy == "loose":
        min_ratio, max_ratio, margin = 0.84, 1.22, -0.05
    else:
        raise ValueError(f"unknown policy: {policy}")

    reasons = bad_reasons(candidate, baseline, min_ratio, max_ratio)
    if reasons:
        return False, reasons

    if policy == "repair":
        baseline_bad = bool(BAD_ENDING_RE.search(norm(baseline)) or FRAGMENT_RE.search(norm(baseline)))
        if not baseline_bad:
            return False, ["baseline_not_truncated"]

    delta = local_quality(candidate) - local_quality(baseline)
    if delta < margin:
        return False, [f"quality_delta:{delta:.2f}"]
    return True, [f"quality_delta:{delta:.2f}"]


def build(policy: str, best: dict, candidate: dict) -> tuple[dict, dict]:
    fused = deepcopy(best)
    report = {
        "policy": policy,
        "scenarios": len(best),
        "rows": 0,
        "fields_changed": 0,
        "accepted": [],
        "reject_counts": {},
    }

    for scenario_id, rows in fused.items():
        cand_rows = candidate.get(scenario_id, [])
        for idx, row in enumerate(rows):
            report["rows"] += 1
            if idx >= len(cand_rows):
                continue
            cand_row = cand_rows[idx]
            for key in CAPTION_KEYS:
                base_text = row.get(key, "")
                cand_text = cand_row.get(key, "")
                ok, reasons = should_accept(cand_text, base_text, policy)
                if ok and norm(cand_text) != norm(base_text):
                    row[key] = norm(cand_text)
                    report["fields_changed"] += 1
                    report["accepted"].append(
                        {
                            "scenario_id": scenario_id,
                            "row_index": idx,
                            "key": key,
                            "reasons": reasons,
                        }
                    )
                else:
                    for reason in reasons:
                        reason_key = reason.split(":", 1)[0]
                        report["reject_counts"][reason_key] = report["reject_counts"].get(reason_key, 0) + 1
    return fused, report


def make_zip(caption_path: Path, vqa_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-caption", type=Path, required=True)
    parser.add_argument("--candidate-caption", type=Path, required=True)
    parser.add_argument("--vqa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="fact_locked_gate")
    parser.add_argument("--policies", default="strict,moderate,loose")
    args = parser.parse_args()

    best = load_json(args.best_caption)
    candidate = load_json(args.candidate_caption)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        fused, report = build(policy, best, candidate)
        caption_path = args.output_dir / f"caption_submission_{args.prefix}_{policy}.json"
        report_path = args.output_dir / f"caption_{args.prefix}_{policy}_report.json"
        zip_path = args.output_dir / f"submission_{args.prefix}_{policy}.zip"
        dump_json(fused, caption_path)
        dump_json(report, report_path)
        make_zip(caption_path, args.vqa, zip_path)
        print(
            json.dumps(
                {
                    "policy": policy,
                    "caption": str(caption_path),
                    "zip": str(zip_path),
                    "fields_changed": report["fields_changed"],
                    "reject_counts": report["reject_counts"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
