#!/usr/bin/env python
"""Discriminative VQA scorer using frozen V-JEPA2 video embeddings.

This is not a generative VLM. It follows the VL-JEPA idea for Track 2 VQA:
predict/select answer semantics in embedding space, then only emit option
letters. The visual encoder is frozen `facebook/vjepa2-*`; a small MLP learns
cross-modal interactions between video embeddings and compressed
question-option text features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synwts.io import read_json, write_json
from synwts.submission import _parse_vqa_letter
from synwts.validators import validate_caption_submission, validate_vqa_submission
from synwts.vqa_fusion import classify_vqa_question


LETTERS = ("a", "b", "c", "d", "e")


@dataclass
class VqaRow:
    row_id: str
    question: str
    options: dict[str, str]
    videos: list[str]
    correct: str | None
    question_type: str
    scope: str
    phase: str


def parse_rows(path: Path, *, require_labels: bool, max_rows: int = 0) -> list[VqaRow]:
    rows: list[VqaRow] = []
    for idx, item in enumerate(read_json(path)):
        if max_rows and len(rows) >= max_rows:
            break
        instruction = str(item.get("instruction", ""))
        metadata = item.get("metadata", {}) or {}
        options = _parse_options(instruction)
        if len(options) < 2:
            continue
        question = str(metadata.get("question") or _parse_question(instruction)).strip()
        scope = str(metadata.get("scope") or "")
        phase = str(metadata.get("phase") or "")
        qtype = str(metadata.get("question_type") or classify_vqa_question(question, scope=scope))
        row_id = str(metadata.get("vqa_id") or metadata.get("id") or item.get("id") or f"row_{idx:06d}").strip()
        correct_raw = str(item.get("output", "")).strip()
        correct = _parse_vqa_letter(correct_raw) if correct_raw else None
        if require_labels and correct not in options:
            continue
        rows.append(
            VqaRow(
                row_id=row_id,
                question=question,
                options=options,
                videos=[str(v) for v in item.get("videos", [])],
                correct=correct,
                question_type=qtype,
                scope=scope,
                phase=phase,
            )
        )
    return rows


def _parse_options(instruction: str) -> dict[str, str]:
    options: dict[str, str] = {}
    match = re.search(r"(?ms)^Options:\s*\n(?P<body>.*?)(?:\n\s*\nReturn|\nReturn)", instruction)
    if not match:
        return options
    for item in re.finditer(r"(?m)^\s*([a-eA-E])\.\s*(.+?)\s*$", match.group("body")):
        options[item.group(1).lower()] = " ".join(item.group(2).split())
    return options


def _parse_question(instruction: str) -> str:
    match = re.search(r"(?ms)Question:\s*(.+?)\nOptions:", instruction)
    return " ".join(match.group(1).split()) if match else ""


class Vjepa2Extractor:
    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: Path,
        num_frames: int,
        max_videos_per_row: int,
        device: str,
        dtype: str,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoVideoProcessor

        self.torch = torch
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.max_videos_per_row = max_videos_per_row
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        print(f"Loading {model_name} on {self.device} dtype={dtype}", flush=True)
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()
        self.feature_dim: int | None = None

    def row_features(self, rows: list[VqaRow]) -> np.ndarray:
        feats: list[np.ndarray] = []
        started = time.time()
        for i, row in enumerate(rows):
            video_feats = []
            for path in row.videos[: self.max_videos_per_row or None]:
                video_feats.append(self.video_feature(path))
            if video_feats:
                feat = np.mean(np.stack(video_feats, axis=0), axis=0).astype(np.float32)
            else:
                feat = np.zeros(self.feature_dim or 1024, dtype=np.float32)
            if self.feature_dim is None:
                self.feature_dim = int(feat.shape[0])
            feats.append(feat)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - started
                print(f"encoded rows {i + 1}/{len(rows)} elapsed={elapsed/60:.1f}m", flush=True)
        return np.stack(feats, axis=0).astype(np.float32)

    def video_feature(self, path: str) -> np.ndarray:
        cache_path = self.cache_dir / f"{_stable_hash(path)}_f{self.num_frames}.npy"
        if cache_path.exists():
            feat = np.load(cache_path)
            if self.feature_dim is None:
                self.feature_dim = int(feat.shape[0])
            return feat.astype(np.float32)
        try:
            frames = _read_video_frames(Path(path), self.num_frames)
            feat = self._encode_frames(frames)
        except Exception as exc:
            print(f"WARN video_failed path={path} error={type(exc).__name__}: {exc}", flush=True)
            feat = np.zeros(self.feature_dim or 1024, dtype=np.float32)
        np.save(cache_path, feat.astype(np.float32))
        if self.feature_dim is None:
            self.feature_dim = int(feat.shape[0])
        return feat.astype(np.float32)

    def _encode_frames(self, frames_hwc: np.ndarray) -> np.ndarray:
        torch = self.torch
        frames_chw = np.transpose(frames_hwc, (0, 3, 1, 2))
        try:
            inputs = self.processor(frames_chw, return_tensors="pt")
        except Exception:
            inputs = self.processor(frames_hwc, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_vision_features"):
                out = self.model.get_vision_features(**inputs)
            else:
                out = self.model(**inputs)
                out = getattr(out, "last_hidden_state", out[0] if isinstance(out, (tuple, list)) else out)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feat = out.float()
        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.ndim > 2:
            feat = feat.reshape(feat.shape[0], -1, feat.shape[-1]).mean(dim=1)
        feat = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / max(norm, 1e-6)


def _read_video_frames(path: Path, num_frames: int) -> np.ndarray:
    from decord import VideoReader, cpu

    if not path.exists():
        raise FileNotFoundError(path)
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    n = len(vr)
    if n <= 0:
        raise ValueError("empty_video")
    idx = np.linspace(0, n - 1, num_frames).round().astype("int64")
    return vr.get_batch(idx).asnumpy()


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:24]


def build_option_table(rows: list[VqaRow]) -> tuple[list[str], list[int], list[str], list[list[str]], np.ndarray | None]:
    texts: list[str] = []
    row_indices: list[int] = []
    letters: list[str] = []
    meta_values: list[list[str]] = []
    labels: list[int] = []
    has_labels = all(row.correct for row in rows)
    for row_idx, row in enumerate(rows):
        for letter, option_text in sorted(row.options.items()):
            texts.append(f"question: {row.question} option: {option_text}")
            row_indices.append(row_idx)
            letters.append(letter)
            meta_values.append([row.question_type, row.scope, row.phase])
            if has_labels:
                labels.append(int(letter == row.correct))
    return texts, row_indices, letters, meta_values, np.asarray(labels, dtype=np.float32) if has_labels else None


def make_features(
    *,
    rows: list[VqaRow],
    row_video_features: np.ndarray,
    vectorizer: Any,
    svd: Any,
    onehot: Any,
    scaler: Any,
    fit: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[str]]:
    from sklearn.preprocessing import OneHotEncoder

    texts, row_indices, letters, meta_values, labels = build_option_table(rows)
    if fit:
        text_sparse = vectorizer.fit_transform(texts)
        text_dense = svd.fit_transform(text_sparse).astype(np.float32)
        try:
            meta_dense = onehot.fit_transform(meta_values).astype(np.float32)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
            meta_dense = onehot.fit_transform(meta_values).astype(np.float32)
    else:
        text_dense = svd.transform(vectorizer.transform(texts)).astype(np.float32)
        meta_dense = onehot.transform(meta_values).astype(np.float32)
        if not isinstance(meta_dense, np.ndarray):
            meta_dense = meta_dense.toarray().astype(np.float32)
    video_dense = row_video_features[np.asarray(row_indices, dtype=np.int64)].astype(np.float32)
    x = np.concatenate([video_dense, text_dense, meta_dense], axis=1).astype(np.float32)
    if fit:
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = scaler.transform(x).astype(np.float32)
    return x, labels, row_indices, letters


def train_mlp(x: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int, lr: float, seed: int) -> Any:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 512),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(512, 128),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(128, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.reshape(-1, 1)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        print(f"epoch {epoch + 1}/{epochs} loss={total / max(seen, 1):.5f}", flush=True)
    model.eval()
    return model


def predict_scores(model: Any, x: np.ndarray, *, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False, num_workers=0)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device)).squeeze(1).float().detach().cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(scores, axis=0)


def rows_from_scores(
    rows: list[VqaRow],
    row_indices: list[int],
    letters: list[str],
    scores: np.ndarray,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    grouped: list[list[tuple[str, float]]] = [[] for _ in rows]
    for row_idx, letter, score in zip(row_indices, letters, scores):
        grouped[row_idx].append((letter, float(score)))
    out: list[dict[str, str]] = []
    detail: list[dict[str, Any]] = []
    for row, items in zip(rows, grouped):
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        answer = ranked[0][0] if ranked else sorted(row.options)[0]
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
        out.append({"id": row.row_id, "correct": answer})
        detail.append(
            {
                "id": row.row_id,
                "answer": answer,
                "margin": margin,
                "scores": {letter: round(score, 6) for letter, score in ranked},
                "question_type": row.question_type,
                "scope": row.scope,
                "phase": row.phase,
                "question": row.question,
            }
        )
    return out, detail


def accuracy(rows: list[VqaRow], preds: list[dict[str, str]]) -> float:
    correct = 0
    total = 0
    for row, pred in zip(rows, preds):
        if row.correct:
            total += 1
            correct += int(pred["correct"] == row.correct)
    return correct / max(total, 1)


def load_vqa_submission(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            data = json.loads(zf.read("vqa_submission.json"))
    else:
        data = read_json(path)
    return {str(row["id"]): _parse_vqa_letter(str(row.get("correct", ""))) for row in data}


def write_zip(path: Path, caption_path: Path, vqa_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(caption_path, "caption_submission.json")
        zf.write(vqa_path, "vqa_submission.json")


def write_gated(
    *,
    name: str,
    output_dir: Path,
    caption_path: Path,
    fallback: dict[str, str],
    raw_rows: list[dict[str, str]],
    details: list[dict[str, Any]],
    threshold: float,
    cap: int,
) -> dict[str, Any]:
    candidates = []
    for pred, detail in zip(raw_rows, details):
        qid = pred["id"]
        fallback_answer = fallback.get(qid, pred["correct"])
        if pred["correct"] != fallback_answer and detail["margin"] >= threshold:
            candidates.append({**detail, "fallback": fallback_answer, "selected": pred["correct"]})
    candidates.sort(key=lambda item: item["margin"], reverse=True)
    if cap > 0:
        candidates = candidates[:cap]
    keep = {item["id"]: item["selected"] for item in candidates}
    gated_rows = [{"id": row["id"], "correct": keep.get(row["id"], fallback.get(row["id"], row["correct"]))} for row in raw_rows]
    mode_dir = output_dir / name
    mode_dir.mkdir(parents=True, exist_ok=True)
    vqa_path = mode_dir / "vqa_submission.json"
    cap_path = mode_dir / "caption_submission.json"
    write_json(vqa_path, gated_rows)
    cap_path.write_bytes(caption_path.read_bytes())
    validation = validate_vqa_submission(vqa_path)
    write_json(mode_dir / "vqa_validation.json", validation)
    report = {
        "mode": name,
        "threshold": threshold,
        "cap": cap,
        "changed_from_fallback": len(candidates),
        "changed_by_type": _count_by(candidates, "question_type"),
        "validation": validation,
        "changed_rows": candidates,
    }
    write_json(mode_dir / "report.json", report)
    zip_path = output_dir.parent / f"{output_dir.name}_{name}.zip"
    write_zip(zip_path, cap_path, vqa_path)
    report["zip"] = str(zip_path)
    return report


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", action="append", required=True, type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--fallback-vqa", required=True, type=Path)
    parser.add_argument("--caption-submission", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--model-name", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--max-videos-per-row", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--tfidf-max-features", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-calibration-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--gate", default="hp:0.40:75,balanced:0.30:250,aggressive:0.20:700")
    args = parser.parse_args()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    train_rows: list[VqaRow] = []
    for path in args.train_json:
        part = parse_rows(path, require_labels=True, max_rows=args.max_train_rows)
        print(f"train rows from {path}: {len(part)}", flush=True)
        train_rows.extend(part)
    cal_rows = parse_rows(args.calibration_json, require_labels=True, max_rows=args.max_calibration_rows) if args.calibration_json else []
    test_rows = parse_rows(args.test_json, require_labels=False, max_rows=args.max_test_rows)
    print(f"total train={len(train_rows)} calibration={len(cal_rows)} test={len(test_rows)}", flush=True)
    if not train_rows or not test_rows:
        raise ValueError("Need non-empty train and test rows.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = Vjepa2Extractor(
        model_name=args.model_name,
        cache_dir=args.feature_cache,
        num_frames=args.num_frames,
        max_videos_per_row=args.max_videos_per_row,
        device=args.device,
        dtype=args.dtype,
    )
    train_video = extractor.row_features(train_rows)
    cal_video = extractor.row_features(cal_rows) if cal_rows else np.zeros((0, train_video.shape[1]), dtype=np.float32)
    test_video = extractor.row_features(test_rows)

    vectorizer = TfidfVectorizer(max_features=args.tfidf_max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    svd = TruncatedSVD(n_components=args.text_dim, random_state=args.seed)
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    scaler = StandardScaler()

    x_train, y_train, _, _ = make_features(
        rows=train_rows,
        row_video_features=train_video,
        vectorizer=vectorizer,
        svd=svd,
        onehot=onehot,
        scaler=scaler,
        fit=True,
    )
    model = train_mlp(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)

    summary: dict[str, Any] = {
        "model_name": args.model_name,
        "train_rows": len(train_rows),
        "calibration_rows": len(cal_rows),
        "test_rows": len(test_rows),
        "num_frames": args.num_frames,
        "max_videos_per_row": args.max_videos_per_row,
    }

    if cal_rows:
        x_cal, _, cal_row_indices, cal_letters = make_features(
            rows=cal_rows,
            row_video_features=cal_video,
            vectorizer=vectorizer,
            svd=svd,
            onehot=onehot,
            scaler=scaler,
            fit=False,
        )
        cal_scores = predict_scores(model, x_cal, batch_size=args.batch_size)
        cal_preds, cal_details = rows_from_scores(cal_rows, cal_row_indices, cal_letters, cal_scores)
        summary["calibration_raw_acc"] = round(accuracy(cal_rows, cal_preds), 6)
        write_json(args.output_dir / "calibration_predictions.json", cal_preds)
        write_json(args.output_dir / "calibration_details.json", cal_details)
        print(f"calibration raw acc={summary['calibration_raw_acc']}", flush=True)

    x_test, _, test_row_indices, test_letters = make_features(
        rows=test_rows,
        row_video_features=test_video,
        vectorizer=vectorizer,
        svd=svd,
        onehot=onehot,
        scaler=scaler,
        fit=False,
    )
    test_scores = predict_scores(model, x_test, batch_size=args.batch_size)
    raw_rows, details = rows_from_scores(test_rows, test_row_indices, test_letters, test_scores)

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_vqa = raw_dir / "vqa_submission.json"
    raw_cap = raw_dir / "caption_submission.json"
    write_json(raw_vqa, raw_rows)
    raw_cap.write_bytes(args.caption_submission.read_bytes())
    write_json(raw_dir / "details.json", details)
    write_json(raw_dir / "vqa_validation.json", validate_vqa_submission(raw_vqa))
    write_json(raw_dir / "caption_validation.json", validate_caption_submission(raw_cap))
    raw_zip = args.output_dir.parent / f"{args.output_dir.name}_raw.zip"
    write_zip(raw_zip, raw_cap, raw_vqa)
    summary["raw_zip"] = str(raw_zip)

    fallback = load_vqa_submission(args.fallback_vqa)
    summary["gates"] = {}
    for item in args.gate.split(","):
        if not item.strip():
            continue
        name, raw_threshold, raw_cap = item.split(":")
        report = write_gated(
            name=name,
            output_dir=args.output_dir,
            caption_path=args.caption_submission,
            fallback=fallback,
            raw_rows=raw_rows,
            details=details,
            threshold=float(raw_threshold),
            cap=int(raw_cap),
        )
        summary["gates"][name] = {
            "zip": report["zip"],
            "threshold": report["threshold"],
            "cap": report["cap"],
            "changed_from_fallback": report["changed_from_fallback"],
            "changed_by_type": report["changed_by_type"],
        }
        print(f"{name}: {report['zip']} changed={report['changed_from_fallback']}", flush=True)

    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
