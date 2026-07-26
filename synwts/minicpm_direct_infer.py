"""Direct MiniCPM-V inference for exported LLaMA-Factory style datasets.

This bypasses LLaMA-Factory prediction because MiniCPM-V remote-code generate()
expects pixel_values that the generic Trainer path can drop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_idx = _count_existing_rows(output) if args.resume else 0

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    if args.adapter_name_or_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.eval()

    mode = "a" if args.resume and start_idx else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows[start_idx:], initial=start_idx, total=len(rows), desc="MiniCPM direct infer"):
            prediction = infer_one(
                model=model,
                processor=processor,
                row=row,
                max_new_tokens=args.max_new_tokens,
                max_images=args.max_images,
            )
            handle.write(
                json.dumps(
                    {
                        "predict": prediction,
                        "label": row.get("output", ""),
                        "metadata": row.get("metadata", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()


def infer_one(
    *,
    model: Any,
    processor: Any,
    row: dict[str, Any],
    max_new_tokens: int,
    max_images: int | None,
) -> str:
    prompt = _build_prompt(row)
    images = _load_images(row, max_images=max_images)
    content: list[Any] = [*images, prompt] if images else [prompt]
    msgs = [{"role": "user", "content": content}]
    with torch.inference_mode():
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=processor.tokenizer,
            processor=processor,
            sampling=False,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, list):
        response = response[0] if response else ""
    return str(response).strip()


def _build_prompt(row: dict[str, Any]) -> str:
    parts = [str(row.get("instruction", "")), str(row.get("input", ""))]
    text = "\n".join(part for part in parts if part.strip())
    text = text.replace("<image>", "").replace("<video>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_images(row: dict[str, Any], *, max_images: int | None) -> list[Image.Image]:
    paths: list[str] = []
    for key in ("images", "image", "videos", "video"):
        value = row.get(key)
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, list):
            paths.extend(str(item) for item in value)
    if max_images is not None:
        paths = paths[:max_images]

    images: list[Image.Image] = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception:
            continue
    return images


def _count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", default="openbmb/MiniCPM-V-4_5")
    parser.add_argument("--adapter-name-or-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
