"""Check the software stack required for Qwen3.6 multimodal QLoRA."""

from importlib.util import find_spec

import torch
import transformers
from transformers import AutoConfig, AutoProcessor


MODEL_ID = "Qwen/Qwen3.6-27B"


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("model_type:", config.model_type)
    print("architectures:", config.architectures)
    print("processor:", type(processor).__name__)
    print("bitsandbytes:", find_spec("bitsandbytes"))
    print("deepspeed:", find_spec("deepspeed"))

    if find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes is required for 4-bit QLoRA")


if __name__ == "__main__":
    main()
