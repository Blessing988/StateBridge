"""Conditional option scoring for Qwen-family multimodal VQA candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .io import read_json


def score_qwen3vl_options(
    *,
    candidates_path: str | Path,
    output: str | Path,
    model_name_or_path: str,
    adapter_name_or_path: str | None = None,
    dtype: str = "bf16",
    attn_implementation: str | None = "sdpa",
    device_map: str = "auto",
    video_max_pixels: int | None = 65536,
    fps: float | None = 2.0,
    response_mode: str = "letter",
    include_eos: bool = False,
    resume: bool = True,
    max_rows: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Score each answer with the SFT model and write resumable JSONL output."""

    if response_mode not in {"letter", "letter_text"}:
        raise ValueError(f"Unsupported response_mode: {response_mode}")
    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("Candidate file must be a JSON list.")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")
    shard_candidates = candidates[shard_index::num_shards]

    torch, processor, model = _load_qwen3vl(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output) if resume else set()
    mode = "a" if resume and output.exists() else "w"

    attempted = 0
    written = 0
    errors = 0
    with output.open(mode, encoding="utf-8") as handle:
        for candidate in shard_candidates:
            qid = str(candidate.get("metadata", {}).get("vqa_id", "")).strip()
            if not qid:
                raise ValueError("Candidate is missing metadata.vqa_id.")
            if qid in completed:
                continue
            if max_rows and attempted >= max_rows:
                break
            attempted += 1
            try:
                scores, token_counts = _score_candidate(
                    torch=torch,
                    processor=processor,
                    model=model,
                    candidate=candidate,
                    response_mode=response_mode,
                    include_eos=include_eos,
                    video_max_pixels=video_max_pixels,
                    fps=fps,
                )
                predicted = min(scores, key=lambda letter: (-scores[letter], letter))
                correct = str(candidate.get("correct", "")).strip().lower()
                row = {
                    "vqa_id": qid,
                    "scores": scores,
                    "token_counts": token_counts,
                    "predicted": predicted,
                    "model_name_or_path": model_name_or_path,
                    "adapter_name_or_path": adapter_name_or_path,
                    "response_mode": response_mode,
                    "include_eos": include_eos,
                }
                if correct:
                    row["correct"] = correct
                    row["is_correct"] = predicted == correct
                written += 1
            except Exception as exc:
                row = {
                    "vqa_id": qid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_name_or_path": model_name_or_path,
                    "adapter_name_or_path": adapter_name_or_path,
                }
                errors += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if attempted == 1 or attempted % 25 == 0:
                print(
                    f"scored={written} errors={errors} skipped={len(completed)} "
                    f"attempted_this_run={attempted}",
                    flush=True,
                )

    return {
        "total_candidates": len(candidates),
        "shard_candidates": len(shard_candidates),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "already_completed": len(completed),
        "attempted": attempted,
        "written": written,
        "errors": errors,
        "output": str(output),
    }


def _load_qwen3vl(
    *,
    model_name_or_path: str,
    adapter_name_or_path: str | None,
    dtype: str,
    attn_implementation: str | None,
    device_map: str,
    trust_remote_code: bool,
):
    try:
        import torch
        from transformers import AutoConfig
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Option scoring requires torch and a recent Transformers multimodal build."
        ) from exc

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype],
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForMultimodalLM as ModelClass
    else:
        from transformers import AutoModelForMultimodalLM as ModelClass
    model = ModelClass.from_pretrained(model_name_or_path, **load_kwargs)
    if adapter_name_or_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading an SFT adapter requires peft.") from exc
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    return torch, processor, model


def _score_candidate(
    *,
    torch,
    processor,
    model,
    candidate: dict[str, Any],
    response_mode: str,
    include_eos: bool,
    video_max_pixels: int | None,
    fps: float | None,
) -> tuple[dict[str, float], dict[str, int]]:
    content = _interleave_prompt_and_videos(
        instruction=str(candidate["instruction"]),
        videos=[str(path) for path in candidate.get("videos", [])],
        video_max_pixels=video_max_pixels,
        fps=fps,
    )
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("video_metadata", None)
    inputs = inputs.to(model.device)

    answer_ids: dict[str, list[int]] = {}
    for letter, text in sorted(candidate["options"].items()):
        answer = letter if response_mode == "letter" else f"{letter}. {text}".strip()
        ids = processor.tokenizer(answer, add_special_tokens=False).input_ids
        if include_eos and processor.tokenizer.eos_token_id is not None:
            ids = [*ids, processor.tokenizer.eos_token_id]
        if not ids:
            raise ValueError(f"Option {letter} produced no answer tokens.")
        answer_ids[str(letter)] = ids

    scores: dict[str, float] = {}
    token_counts = {letter: len(ids) for letter, ids in answer_ids.items()}
    if all(len(ids) == 1 for ids in answer_ids.values()):
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)[0]
        for letter, ids in answer_ids.items():
            scores[letter] = float(log_probs[ids[0]].item())
        return scores, token_counts

    for letter, ids in answer_ids.items():
        scores[letter] = _conditional_log_probability(
            torch=torch,
            model=model,
            prompt_inputs=inputs,
            answer_ids=ids,
        )
    return scores, token_counts


def _conditional_log_probability(*, torch, model, prompt_inputs, answer_ids: list[int]) -> float:
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]
    suffix = torch.tensor([answer_ids], dtype=input_ids.dtype, device=input_ids.device)
    batch = dict(prompt_inputs)
    batch["input_ids"] = torch.cat((input_ids, suffix), dim=1)
    batch["attention_mask"] = torch.cat(
        (
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], len(answer_ids)),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ),
        dim=1,
    )
    with torch.inference_mode():
        logits = model(**batch).logits
    prompt_length = input_ids.shape[1]
    answer_logits = logits[:, prompt_length - 1 : prompt_length + len(answer_ids) - 1, :]
    targets = suffix
    token_log_probs = torch.log_softmax(answer_logits.float(), dim=-1).gather(
        -1,
        targets.unsqueeze(-1),
    )
    value = float(token_log_probs.squeeze(-1).mean().item())
    if not math.isfinite(value):
        raise ValueError("Model produced a non-finite option score.")
    return value


def _interleave_prompt_and_videos(
    *,
    instruction: str,
    videos: list[str],
    video_max_pixels: int | None,
    fps: float | None,
) -> list[dict[str, Any]]:
    parts = instruction.split("<video>")
    placeholder_count = len(parts) - 1
    if placeholder_count != len(videos):
        raise ValueError(
            f"Prompt has {placeholder_count} video placeholders but row has {len(videos)} videos."
        )
    content: list[dict[str, Any]] = []
    for index, text in enumerate(parts):
        if text:
            content.append({"type": "text", "text": text})
        if index < len(videos):
            video_item: dict[str, Any] = {
                "type": "video",
                "video": _as_media_uri(videos[index]),
            }
            if video_max_pixels:
                video_item["max_pixels"] = video_max_pixels
            if fps:
                video_item["fps"] = fps
            content.append(video_item)
    return content


def _as_media_uri(path: str) -> str:
    if "://" in path:
        return path
    return str(Path(path).expanduser().resolve())


def _completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("vqa_id", "")).strip()
            if qid and "scores" in row:
                completed.add(qid)
    return completed
