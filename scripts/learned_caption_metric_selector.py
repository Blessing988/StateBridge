#!/usr/bin/env python
"""Train/apply a lightweight learned selector for caption candidates.

Training uses synthetic validation references only. It learns a linear pairwise
ranker over source/style/fact-preservation features, then applies the same
weights to public-test candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


CAPTION_KEYS = ("caption_pedestrian", "caption_vehicle")
SOURCE_PREFIX = "source:"


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_named(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path: {value}")
        name, path = value.split("=", 1)
        out[name] = Path(path)
    return out


def row_text(row: dict) -> str:
    return " ".join(str(row.get(k, "")) for k in CAPTION_KEYS).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def f1_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    p = overlap / max(1, sum(a.values()))
    r = overlap / max(1, sum(b.values()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def token_f1(a: str, b: str) -> float:
    return f1_counts(Counter(toks(a)), Counter(toks(b)))


def ngram_f1(a: str, b: str, n: int) -> float:
    return f1_counts(ngrams(toks(a), n), ngrams(toks(b), n))


def target_score(candidate: str, reference: str) -> float:
    return (
        0.15 * token_f1(candidate, reference)
        + 0.20 * ngram_f1(candidate, reference, 1)
        + 0.25 * ngram_f1(candidate, reference, 2)
        + 0.20 * ngram_f1(candidate, reference, 3)
        + 0.20 * ngram_f1(candidate, reference, 4)
    )


def protected_terms(text: str) -> set[str]:
    low = text.lower()
    terms = {
        "pedestrian",
        "vehicle",
        "obstacle",
        "sidewalk",
        "roadside strip",
        "street light",
        "residential road",
        "intersection",
        "signal",
        "asphalt",
        "two-way traffic",
        "one-way traffic",
        "0 km/h",
        "5 km/h",
        "10 km/h",
        "20s",
        "30s",
        "40s",
        "male",
        "female",
        "cloudy",
        "clear",
        "bright",
        "dim",
        "dry",
        "wet",
    }
    found = {term for term in terms if term in low}
    for match in re.findall(r"\b\d+\s*(?:km/h|cm|meters?)\b", low):
        found.add(re.sub(r"\s+", " ", match))
    return found


def bad_tail(text: str) -> float:
    clean = norm(text)
    if not clean:
        return 1.0
    if clean[-1] not in ".!?":
        return 1.0
    if re.search(r"\b(and|or|with|of|in|on|to|which|that|there is)\.?$", clean, re.I):
        return 1.0
    return 0.0


def phase_of(row: dict, idx: int) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return str(idx)


def features(*, source: str, candidate: dict, fallback: dict, idx: int, source_names: list[str]) -> dict[str, float]:
    text = row_text(candidate)
    base = row_text(fallback)
    words = toks(text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", norm(text)) if s]
    base_terms = protected_terms(base)
    cand_terms = protected_terms(text)
    missing = len(base_terms - cand_terms) / max(1, len(base_terms))
    added = len(cand_terms - base_terms) / max(1, len(cand_terms))
    feats = {
        "bias": 1.0,
        "len_log": math.log1p(len(words)),
        "sent_log": math.log1p(len(sentences)),
        "token_overlap_base": token_f1(text, base),
        "ngram2_overlap_base": ngram_f1(text, base, 2),
        "ngram4_overlap_base": ngram_f1(text, base, 4),
        "protected_missing_base": -missing,
        "protected_added": -added,
        "bad_tail": -bad_tail(text),
        "has_vehicle": 1.0 if "vehicle" in text.lower() else 0.0,
        "has_pedestrian": 1.0 if "pedestrian" in text.lower() else 0.0,
        f"phase:{phase_of(candidate, idx)}": 1.0,
    }
    for name in source_names:
        feats[SOURCE_PREFIX + name] = 1.0 if source == name else 0.0
    return feats


def dot(weights: dict[str, float], feats: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in feats.items())


def add_scaled(weights: dict[str, float], feats: dict[str, float], scale: float) -> None:
    for k, v in feats.items():
        weights[k] = weights.get(k, 0.0) + scale * v


def rows_by_phase(rows: list[dict]) -> dict[str, tuple[int, dict]]:
    return {phase_of(row, idx): (idx, row) for idx, row in enumerate(rows)}


def iter_groups(reference: dict, candidates: dict[str, dict], fallback_name: str):
    fallback = candidates[fallback_name]
    for sid, ref_rows in reference.items():
        source_phase = {name: rows_by_phase(data.get(sid, [])) for name, data in candidates.items()}
        for idx, ref_row in enumerate(ref_rows):
            phase = phase_of(ref_row, idx)
            options = []
            for name, by_phase in source_phase.items():
                if phase in by_phase:
                    _cand_idx, cand = by_phase[phase]
                    fb = source_phase[fallback_name].get(phase, (idx, fallback.get(sid, ref_rows)[idx]))[1]
                    options.append((name, cand, fb, ref_row, idx))
            if options:
                yield sid, idx, phase, options


def train(args: argparse.Namespace) -> None:
    reference = read_json(args.reference)
    cand_paths = parse_named(args.candidate)
    candidates = {name: read_json(path) for name, path in cand_paths.items()}
    source_names = sorted(candidates)
    if args.fallback_name not in candidates:
        raise ValueError(f"fallback not in candidates: {args.fallback_name}")

    groups = list(iter_groups(reference, candidates, args.fallback_name))
    weights: dict[str, float] = {}
    updates = 0
    for _epoch in range(args.epochs):
        for _sid, _idx, _phase, options in groups:
            scored = []
            for name, cand, fb, ref, idx in options:
                feat = features(source=name, candidate=cand, fallback=fb, idx=idx, source_names=source_names)
                target = target_score(row_text(cand), row_text(ref))
                scored.append((target, name, feat))
            oracle = max(scored, key=lambda x: x[0])
            pred = max(scored, key=lambda x: dot(weights, x[2]))
            if oracle[1] != pred[1] and oracle[0] - pred[0] >= args.min_target_gap:
                add_scaled(weights, oracle[2], args.lr)
                add_scaled(weights, pred[2], -args.lr)
                updates += 1

    correct = total = 0
    source_oracle = Counter()
    source_pred = Counter()
    for _sid, _idx, _phase, options in groups:
        scored = []
        for name, cand, fb, ref, idx in options:
            feat = features(source=name, candidate=cand, fallback=fb, idx=idx, source_names=source_names)
            target = target_score(row_text(cand), row_text(ref))
            scored.append((target, dot(weights, feat), name))
        oracle = max(scored, key=lambda x: x[0])
        pred = max(scored, key=lambda x: x[1])
        correct += int(oracle[2] == pred[2])
        total += 1
        source_oracle[oracle[2]] += 1
        source_pred[pred[2]] += 1

    model = {
        "weights": weights,
        "source_names": source_names,
        "fallback_name": args.fallback_name,
        "train_groups": len(groups),
        "updates": updates,
        "train_source_match": correct / max(1, total),
        "source_oracle": dict(source_oracle),
        "source_pred": dict(source_pred),
    }
    write_json(args.output_model, model)
    print(json.dumps({k: model[k] for k in model if k != "weights"}, indent=2))


def apply(args: argparse.Namespace) -> None:
    model = read_json(args.model)
    weights = model["weights"]
    cand_paths = parse_named(args.candidate)
    candidates = {name: read_json(path) for name, path in cand_paths.items()}
    source_names = sorted(candidates)
    fallback_name = args.fallback_name or model["fallback_name"]
    fallback = candidates[fallback_name]
    selected = deepcopy(fallback)
    changes = []
    source_counts = Counter()

    for sid, fallback_rows in fallback.items():
        source_phase = {name: rows_by_phase(data.get(sid, [])) for name, data in candidates.items()}
        for idx, fb_row in enumerate(fallback_rows):
            phase = phase_of(fb_row, idx)
            options = []
            for name, by_phase in source_phase.items():
                if phase not in by_phase:
                    continue
                _cand_idx, cand = by_phase[phase]
                feat = features(source=name, candidate=cand, fallback=fb_row, idx=idx, source_names=source_names)
                score = dot(weights, feat)
                if name != fallback_name and token_f1(row_text(cand), row_text(fb_row)) < args.min_source_overlap:
                    score -= 100.0
                options.append((score, name, cand))
            if not options:
                continue
            best_score, best_name, best_row = max(options, key=lambda x: x[0])
            base_score = max((s for s, n, _r in options if n == fallback_name), default=-1e9)
            if best_name != fallback_name and best_score - base_score >= args.min_margin:
                selected[sid][idx] = best_row
                changes.append(
                    {
                        "scenario_id": sid,
                        "row_index": idx,
                        "phase": phase,
                        "source": best_name,
                        "margin": round(best_score - base_score, 4),
                    }
                )
                source_counts[best_name] += 1
            else:
                source_counts[fallback_name] += 1

    if args.max_changed_rows and len(changes) > args.max_changed_rows:
        keep = {(c["scenario_id"], c["row_index"]) for c in sorted(changes, key=lambda x: x["margin"], reverse=True)[: args.max_changed_rows]}
        selected = deepcopy(fallback)
        source_counts = Counter()
        new_changes = []
        for change in sorted(changes, key=lambda x: x["margin"], reverse=True):
            sid, idx = change["scenario_id"], change["row_index"]
            if (sid, idx) in keep:
                phase = str(change["phase"])
                selected[sid][idx] = rows_by_phase(candidates[change["source"]][sid])[phase][1]
                source_counts[change["source"]] += 1
                new_changes.append(change)
        total = sum(len(rows) for rows in fallback.values())
        source_counts[fallback_name] += total - len(new_changes)
        changes = new_changes

    write_json(args.output, selected)
    report = {
        "changed_rows": len(changes),
        "source_counts": dict(source_counts),
        "changes": changes[:200],
        "min_margin": args.min_margin,
        "min_source_overlap": args.min_source_overlap,
        "max_changed_rows": args.max_changed_rows,
    }
    if args.report_output:
        write_json(args.report_output, report)
    if args.vqa and args.zip_output:
        with zipfile.ZipFile(args.zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.output, "caption_submission.json")
            zf.write(args.vqa, "vqa_submission.json")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--reference", required=True)
    p_train.add_argument("--candidate", action="append", required=True)
    p_train.add_argument("--fallback-name", required=True)
    p_train.add_argument("--output-model", required=True)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=0.05)
    p_train.add_argument("--min-target-gap", type=float, default=0.002)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--model", required=True)
    p_apply.add_argument("--candidate", action="append", required=True)
    p_apply.add_argument("--fallback-name")
    p_apply.add_argument("--output", required=True)
    p_apply.add_argument("--report-output")
    p_apply.add_argument("--vqa")
    p_apply.add_argument("--zip-output")
    p_apply.add_argument("--min-margin", type=float, default=0.0)
    p_apply.add_argument("--min-source-overlap", type=float, default=0.70)
    p_apply.add_argument("--max-changed-rows", type=int, default=45)
    p_apply.set_defaults(func=apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
