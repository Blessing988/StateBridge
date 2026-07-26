"""Dependency-light caption and VQA metrics.

These metrics are intended for local iteration and regression testing. The
official leaderboard remains authoritative. If pycocoevalcap is installed on
the HPC, it can be added later as an official-compatible backend.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(candidates: list[str], references: list[str], max_n: int = 4) -> float:
    clipped = [0] * max_n
    total = [0] * max_n
    cand_len = 0
    ref_len = 0
    for cand, ref in zip(candidates, references):
        cand_tokens = tokenize(cand)
        ref_tokens = tokenize(ref)
        cand_len += len(cand_tokens)
        ref_len += len(ref_tokens)
        for n in range(1, max_n + 1):
            cand_ngrams = ngrams(cand_tokens, n)
            ref_ngrams = ngrams(ref_tokens, n)
            total[n - 1] += sum(cand_ngrams.values())
            clipped[n - 1] += sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
    precisions = [
        (clipped[i] + 1.0) / (total[i] + 1.0)
        for i in range(max_n)
    ]
    if cand_len == 0:
        return 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / max(cand_len, 1)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def rouge_l(candidate: str, reference: str) -> float:
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return _f_score(precision, recall, beta=1.2)


def meteor_like(candidate: str, reference: str) -> float:
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)
    matches = sum(min(cand_counts[token], ref_counts[token]) for token in cand_counts)
    if matches == 0:
        return 0.0
    precision = matches / len(cand_tokens)
    recall = matches / len(ref_tokens)
    return _f_score(precision, recall, beta=3.0)


def cider_like(candidates: list[str], references: list[str], max_n: int = 4) -> float:
    if not candidates:
        return 0.0
    ref_doc_freq: list[defaultdict[tuple[str, ...], int]] = [defaultdict(int) for _ in range(max_n)]
    for ref in references:
        ref_tokens = tokenize(ref)
        for n in range(1, max_n + 1):
            for gram in set(ngrams(ref_tokens, n)):
                ref_doc_freq[n - 1][gram] += 1
    scores = []
    doc_count = max(len(references), 1)
    for cand, ref in zip(candidates, references):
        cand_tokens = tokenize(cand)
        ref_tokens = tokenize(ref)
        per_n = []
        for n in range(1, max_n + 1):
            cand_vec = _tfidf(ngrams(cand_tokens, n), ref_doc_freq[n - 1], doc_count)
            ref_vec = _tfidf(ngrams(ref_tokens, n), ref_doc_freq[n - 1], doc_count)
            per_n.append(_cosine(cand_vec, ref_vec))
        scores.append(sum(per_n) / max_n)
    return sum(scores) / len(scores)


def caption_metric_bundle(candidates: list[str], references: list[str]) -> dict[str, float]:
    if len(candidates) != len(references):
        raise ValueError("Candidate and reference counts must match.")
    if not candidates:
        return {"BLEU-4": 0.0, "METEOR": 0.0, "ROUGE-L": 0.0, "CIDEr": 0.0, "mean": 0.0}
    bleu = corpus_bleu(candidates, references)
    meteor = sum(meteor_like(c, r) for c, r in zip(candidates, references)) / len(candidates)
    rouge = sum(rouge_l(c, r) for c, r in zip(candidates, references)) / len(candidates)
    cider = cider_like(candidates, references)
    return {
        "BLEU-4": round(bleu, 6),
        "METEOR": round(meteor, 6),
        "ROUGE-L": round(rouge, 6),
        "CIDEr": round(cider, 6),
        "mean": round((bleu + meteor + rouge + cider) / 4.0, 6),
    }


def vqa_accuracy(predictions: dict[str, str], references: dict[str, str]) -> dict[str, float | int]:
    total = len(references)
    correct = sum(
        1
        for qid, answer in references.items()
        if predictions.get(qid, "").strip().lower() == answer.strip().lower()
    )
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
    }


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for idx, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def _f_score(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def _tfidf(
    counts: Counter[tuple[str, ...]],
    doc_freq: defaultdict[tuple[str, ...], int],
    doc_count: int,
) -> dict[tuple[str, ...], float]:
    total = max(sum(counts.values()), 1)
    vec: dict[tuple[str, ...], float] = {}
    for gram, count in counts.items():
        tf = count / total
        idf = math.log((doc_count + 1.0) / (doc_freq[gram] + 1.0))
        vec[gram] = tf * idf
    return vec


def _cosine(a: dict[tuple[str, ...], float], b: dict[tuple[str, ...], float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

