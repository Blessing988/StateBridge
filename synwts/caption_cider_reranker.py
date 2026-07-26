"""CIDEr-style caption reranking with synthetic-train phrase banks."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from pathlib import Path
from typing import Any

from .exporters import load_records
from .io import read_json, write_json
from .parsers import load_caption_phases

CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")
CONTEXT_TERMS = (
    "weather",
    "brightness",
    "road surface",
    "asphalt",
    "residential",
    "two-way",
    "traffic volume",
    "sidewalk",
    "street light",
    "dry",
    "cloudy",
    "dim",
    "level",
)
ATTENTION_TERMS = (
    "line of sight",
    "noticed",
    "aware",
    "unaware",
    "attention",
    "looking",
    "field of view",
    "visual",
    "closely watching",
)
PHASE_TERMS = {
    "0": ("unaware", "presence", "looking around", "approached", "prior"),
    "1": ("noticed", "aware", "line of sight", "recognized", "presence"),
    "2": ("planned", "continue", "straight", "judgment", "intended"),
    "3": ("action", "speed", "slowed", "braked", "stopped", "moving"),
    "4": ("avoidance", "avoid", "close", "near", "collision", "passed"),
}


def build_caption_phrase_bank(
    *,
    index: str | Path,
    output: str | Path,
    splits: set[str] | None = None,
    max_ngram: int = 4,
) -> dict[str, Any]:
    rows = []
    splits = splits or {"train"}
    for record in load_records(index):
        if record.split not in splits:
            continue
        for view, caption_path in sorted(record.caption_files.items()):
            for phase in load_caption_phases(caption_path):
                phase_label = str(phase["label"])
                row = {
                    "scenario_type": record.scenario_type,
                    "view": view,
                    "phase": phase_label,
                    "caption_pedestrian": phase["caption_pedestrian"],
                    "caption_vehicle": phase["caption_vehicle"],
                }
                rows.append(row)

    bank = _make_bank(rows, max_ngram=max_ngram)
    write_json(output, bank)
    return bank


def rerank_caption_cider_bank(
    *,
    phrase_bank: str | Path,
    base_caption: str | Path,
    candidates: dict[str, str | Path],
    output: str | Path,
    report_output: str | Path | None = None,
    fallback_name: str = "base",
    max_changed_rows: int = 10,
    min_margin: float = 0.06,
    min_context_delta: int = 0,
    min_attention_delta: int = 0,
    min_source_overlap: float = 0.72,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bank = read_json(phrase_bank)
    base = _load_caption_submission(base_caption)
    loaded = {name: _load_caption_submission(path) for name, path in candidates.items()}
    if fallback_name != "base" and fallback_name in loaded:
        base = loaded[fallback_name]

    proposals: list[dict[str, Any]] = []
    for scenario_id, base_rows in base.items():
        for idx, base_row in enumerate(base_rows):
            phase = _phase_label(base_row, idx)
            base_score, base_parts = _score_row(base_row, bank=bank, phase=phase, base_row=base_row)
            best = {
                "scenario_id": scenario_id,
                "row_index": idx,
                "phase": phase,
                "source": "base",
                "row": base_row,
                "score": base_score,
                "parts": base_parts,
                "margin": 0.0,
            }
            for source_name, source_rows in loaded.items():
                row = _candidate_row(source_rows, scenario_id, phase, idx)
                if row is None:
                    continue
                score, parts = _score_row(row, bank=bank, phase=phase, base_row=base_row)
                margin = score - base_score
                if margin > best["margin"]:
                    best = {
                        "scenario_id": scenario_id,
                        "row_index": idx,
                        "phase": phase,
                        "source": source_name,
                        "row": row,
                        "score": score,
                        "parts": parts,
                        "base_score": base_score,
                        "base_parts": base_parts,
                        "margin": margin,
                    }
            if best["source"] != "base" and _passes_gates(
                best,
                min_margin=min_margin,
                min_context_delta=min_context_delta,
                min_attention_delta=min_attention_delta,
                min_source_overlap=min_source_overlap,
            ):
                proposals.append(best)

    proposals.sort(key=lambda item: (-item["margin"], item["source"], item["scenario_id"], item["row_index"]))
    accepted = proposals[: max(0, max_changed_rows)]
    accepted_keys = {(p["scenario_id"], p["row_index"]): p for p in accepted}

    output_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: Counter[str] = Counter()
    for scenario_id, base_rows in base.items():
        rows = []
        for idx, base_row in enumerate(base_rows):
            proposal = accepted_keys.get((scenario_id, idx))
            if proposal is None:
                rows.append(_normalize_row(base_row, phase=_phase_label(base_row, idx)))
                source_counts["base"] += 1
            else:
                rows.append(_normalize_row(proposal["row"], phase=proposal["phase"]))
                source_counts[proposal["source"]] += 1
        output_rows[scenario_id] = rows

    report = {
        "total_rows": sum(len(rows) for rows in base.values()),
        "proposed_rows": len(proposals),
        "changed_rows": len(accepted),
        "source_counts": dict(source_counts),
        "max_changed_rows": max_changed_rows,
        "min_margin": min_margin,
        "min_context_delta": min_context_delta,
        "min_attention_delta": min_attention_delta,
        "min_source_overlap": min_source_overlap,
        "accepted": [_report_proposal(p) for p in accepted],
        "rejected_sample": [_report_proposal(p) for p in proposals[max_changed_rows : max_changed_rows + 30]],
    }
    write_json(output, output_rows)
    if report_output:
        write_json(report_output, report)
    return output_rows, report


def _make_bank(rows: list[dict[str, Any]], *, max_ngram: int) -> dict[str, Any]:
    docs_by_key: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        phase = str(row["phase"])
        for role, caption_key in (("pedestrian", "caption_pedestrian"), ("vehicle", "caption_vehicle")):
            text = str(row.get(caption_key, ""))
            docs_by_key[f"phase:{phase}:role:{role}"].append(text)
            docs_by_key[f"phase:{phase}:role:any"].append(text)
            docs_by_key[f"global:role:{role}"].append(text)
            docs_by_key["global:role:any"].append(text)

    out: dict[str, Any] = {"max_ngram": max_ngram, "banks": {}}
    for key, docs in docs_by_key.items():
        out["banks"][key] = _bank_for_docs(docs, max_ngram=max_ngram)
    return out


def _bank_for_docs(docs: list[str], *, max_ngram: int) -> dict[str, Any]:
    document_frequency: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    for text in docs:
        grams = _text_ngrams(text, max_ngram=max_ngram)
        document_frequency.update(grams.keys())
        phrase_counts.update(grams)
    num_docs = max(len(docs), 1)
    weights = {}
    for gram, count in phrase_counts.items():
        df = document_frequency[gram]
        idf = math.log((num_docs + 1.0) / (df + 0.5))
        weights[gram] = round(float(count) * max(idf, 0.05), 6)
    top = dict(sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:6000])
    return {"num_docs": len(docs), "weights": top}


def _score_row(
    row: dict[str, Any],
    *,
    bank: dict[str, Any],
    phase: str,
    base_row: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    row_text = _row_text(row)
    base_text = _row_text(base_row)
    max_ngram = int(bank.get("max_ngram", 4))
    ped_score = _role_score(str(row.get("caption_pedestrian", "")), bank=bank, phase=phase, role="pedestrian", max_ngram=max_ngram)
    veh_score = _role_score(str(row.get("caption_vehicle", "")), bank=bank, phase=phase, role="vehicle", max_ngram=max_ngram)
    source_overlap = _token_f1(row_text, base_text)
    ngram_overlap = _ngram_f1(row_text, base_text, n=4)
    context_delta = _term_count(row_text, CONTEXT_TERMS) - _term_count(base_text, CONTEXT_TERMS)
    attention_delta = _term_count(row_text, ATTENTION_TERMS) - _term_count(base_text, ATTENTION_TERMS)
    phase_hits = _term_count(row_text, PHASE_TERMS.get(phase, ()))
    length_score = _length_score(row_text, base_text)
    trunc_penalty = -2.0 if _is_truncated(row_text) else 0.0
    repetition_penalty = -1.5 if _looks_repetitive(row_text) else 0.0
    score = (
        0.55 * ped_score
        + 0.55 * veh_score
        + 0.35 * source_overlap
        + 0.25 * ngram_overlap
        + 0.035 * context_delta
        + 0.04 * attention_delta
        + 0.025 * phase_hits
        + length_score
        + trunc_penalty
        + repetition_penalty
    )
    parts = {
        "ped_bank": round(ped_score, 4),
        "veh_bank": round(veh_score, 4),
        "source_overlap": round(source_overlap, 4),
        "ngram_overlap": round(ngram_overlap, 4),
        "context_delta": float(context_delta),
        "attention_delta": float(attention_delta),
        "phase_hits": float(phase_hits),
        "length": round(length_score, 4),
        "trunc_penalty": trunc_penalty,
        "repetition_penalty": repetition_penalty,
    }
    return score, parts


def _role_score(text: str, *, bank: dict[str, Any], phase: str, role: str, max_ngram: int) -> float:
    grams = _text_ngrams(text, max_ngram=max_ngram)
    if not grams:
        return 0.0
    phase_bank = bank.get("banks", {}).get(f"phase:{phase}:role:{role}", {}).get("weights", {})
    any_bank = bank.get("banks", {}).get(f"global:role:{role}", {}).get("weights", {})
    return 0.75 * _weighted_overlap(grams, phase_bank) + 0.25 * _weighted_overlap(grams, any_bank)


def _weighted_overlap(grams: Counter[str], weights: dict[str, float]) -> float:
    if not grams or not weights:
        return 0.0
    hit = sum(min(count, 3) * float(weights.get(gram, 0.0)) for gram, count in grams.items())
    denom = sum(min(count, 3) for count in grams.values())
    return hit / max(denom, 1)


def _passes_gates(
    proposal: dict[str, Any],
    *,
    min_margin: float,
    min_context_delta: int,
    min_attention_delta: int,
    min_source_overlap: float,
) -> bool:
    parts = proposal["parts"]
    return (
        proposal["margin"] >= min_margin
        and parts["context_delta"] >= min_context_delta
        and parts["attention_delta"] >= min_attention_delta
        and parts["source_overlap"] >= min_source_overlap
        and parts["trunc_penalty"] == 0.0
        and parts["repetition_penalty"] == 0.0
    )


def _load_caption_submission(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Caption submission must be an object: {path}")
    return data


def _candidate_row(data: dict[str, list[dict[str, Any]]], scenario_id: str, phase: str, idx: int) -> dict[str, Any] | None:
    rows = data.get(scenario_id)
    if not isinstance(rows, list):
        return None
    for row_idx, row in enumerate(rows):
        if _phase_label(row, row_idx) == phase:
            return row
    if idx < len(rows):
        return rows[idx]
    return None


def _phase_label(row: dict[str, Any], idx: int) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return str(idx)


def _normalize_row(row: dict[str, Any], *, phase: str) -> dict[str, Any]:
    return {
        "labels": [phase],
        "caption_pedestrian": str(row.get("caption_pedestrian", "")).strip(),
        "caption_vehicle": str(row.get("caption_vehicle", "")).strip(),
    }


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "")) for key in CAPTION_KEYS)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _text_ngrams(text: str, *, max_ngram: int) -> Counter[str]:
    toks = _tokens(text)
    out: Counter[str] = Counter()
    for n in range(1, max_ngram + 1):
        for idx in range(max(0, len(toks) - n + 1)):
            out[" ".join(toks[idx : idx + n])] += 1
    return out


def _token_f1(text: str, reference: str) -> float:
    return _counter_f1(Counter(_tokens(text)), Counter(_tokens(reference)))


def _ngram_f1(text: str, reference: str, *, n: int) -> float:
    toks = _tokens(text)
    refs = _tokens(reference)
    return _counter_f1(
        Counter(tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))),
        Counter(tuple(refs[i : i + n]) for i in range(max(0, len(refs) - n + 1))),
    )


def _counter_f1(left: Counter[Any], right: Counter[Any]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / max(sum(left.values()), 1)
    recall = overlap / max(sum(right.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _length_score(text: str, base_text: str) -> float:
    length = max(len(_tokens(text)), 1)
    base_length = max(len(_tokens(base_text)), 1)
    ratio = length / base_length
    return max(-0.25, 0.12 - abs(math.log(ratio)) * 0.35)


def _is_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped.endswith((".", "!", "?")):
        return True
    tail = stripped[-80:].lower()
    return bool(re.search(r"\b(there is an|there is a|with an|with a|and|or|in|on|to|of|at)$", tail))


def _looks_repetitive(text: str) -> bool:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 4:
        return False
    return max(Counter(sentences).values()) >= 3


def _report_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": proposal["scenario_id"],
        "row_index": proposal["row_index"],
        "phase": proposal["phase"],
        "source": proposal["source"],
        "margin": round(float(proposal["margin"]), 5),
        "score": round(float(proposal["score"]), 5),
        "parts": proposal["parts"],
    }
