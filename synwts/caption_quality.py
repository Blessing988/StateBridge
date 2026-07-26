"""Caption completeness audit and conservative repair."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_json, write_json


CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")

BAD_TAILS = (
    "there is",
    "there are",
    "there was",
    "there were",
    "with",
    "and",
    "or",
    "of",
    "an",
    "a",
    "the",
    "to",
    "to the",
    "in",
    "in the",
    "on",
    "on the",
    "at",
    "at a",
    "at the",
    "from",
    "from the",
    "which",
    "that",
    "as",
    "for",
    "while",
    "during",
    "diagonally to the right",
    "diagonally to the left",
)


def audit_caption_completeness(
    *,
    caption: str | Path,
    output: str | Path | None = None,
    max_examples: int = 50,
) -> dict[str, Any]:
    data = _load_caption(caption)
    errors = _find_truncated_rows(data, max_examples=max_examples)
    report = {
        "ok": not errors,
        "num_errors": len(errors),
        "error_counts": dict(Counter(error["reason"] for error in errors)),
        "examples": errors[:max_examples],
    }
    if output:
        write_json(output, report)
    return report


def repair_caption_completeness(
    *,
    caption: str | Path,
    fallback_caption: str | Path,
    output: str | Path,
    report_output: str | Path | None = None,
    max_examples: int = 50,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data = _load_caption(caption)
    fallback = _load_caption(fallback_caption)
    repaired: dict[str, list[dict[str, Any]]] = {}
    changed = 0
    examples: list[dict[str, Any]] = []
    for scenario_id, rows in data.items():
        out_rows: list[dict[str, Any]] = []
        fallback_rows = fallback.get(scenario_id, [])
        for idx, row in enumerate(rows):
            new_row = dict(row)
            row_errors = _row_errors(row)
            if row_errors:
                fb = fallback_rows[idx] if idx < len(fallback_rows) else {}
                for key in CAPTION_KEYS:
                    if any(error["field"] == key for error in row_errors):
                        repaired_text = _repair_text(str(row.get(key, "")).strip())
                        if _truncation_reason(repaired_text):
                            repaired_text = _repair_text(str(fb.get(key, row.get(key, ""))).strip())
                        new_row[key] = repaired_text
                changed += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "scenario_id": scenario_id,
                            "row_index": idx,
                            "errors": row_errors,
                        }
                    )
            out_rows.append(new_row)
        repaired[scenario_id] = out_rows

    remaining = _find_truncated_rows(repaired, max_examples=max_examples)
    report = {
        "changed_rows": changed,
        "remaining_errors": len(remaining),
        "examples": examples,
        "remaining_examples": remaining[:max_examples],
    }
    write_json(output, repaired)
    if report_output:
        write_json(report_output, report)
    return repaired, report


def _load_caption(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Caption submission must be a JSON object: {path}")
    return data


def _find_truncated_rows(data: dict[str, list[dict[str, Any]]], *, max_examples: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for scenario_id, rows in data.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for error in _row_errors(row):
                if len(errors) < max_examples:
                    errors.append({"scenario_id": scenario_id, "row_index": idx, **error})
                else:
                    errors.append({"reason": error["reason"], "field": error["field"]})
    return errors


def _row_errors(row: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for key in CAPTION_KEYS:
        reason = _truncation_reason(str(row.get(key, "")).strip())
        if reason:
            errors.append({"field": key, "reason": reason, "tail": str(row.get(key, ""))[-120:]})
    return errors


def _truncation_reason(text: str) -> str:
    if not text:
        return "empty"
    clean = text.strip()
    lower = clean.lower().rstrip(" ,;:")
    if clean[-1] not in ".!?":
        return "missing_terminal_punctuation"
    for tail in BAD_TAILS:
        if lower.endswith(tail):
            return f"dangling_tail:{tail}"
    return ""


def _repair_text(text: str) -> str:
    clean = text.strip()
    if not clean:
        return clean
    reason = _truncation_reason(clean)
    if not reason:
        return clean
    lower = clean.lower().rstrip(" ,;:")
    if reason.startswith("dangling_tail:") or any(lower.endswith(tail) for tail in BAD_TAILS):
        for punct in (".", "!", "?"):
            idx = clean.rfind(punct)
            if idx >= 0:
                return clean[: idx + 1].strip()
    if clean[-1] not in ".!?":
        return clean.rstrip(" ,;:") + "."
    return clean
