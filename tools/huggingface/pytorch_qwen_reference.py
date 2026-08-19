#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Create a Transformers FP32 Qwen logit oracle")
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    token_ids = [int(value) for value in args.tokens.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_directory, torch_dtype=torch.float32, local_files_only=True
    ).eval()
    with torch.inference_mode():
        logits = model(torch.tensor([token_ids], dtype=torch.long)).logits[0, -1].float()
    logits.numpy().tofile(args.output / "pytorch_logits.f32")
    values, indices = torch.topk(logits, 10)
    report = {
        "schema_version": 1,
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "tokens": token_ids,
        "top_logits": [
            {"token": int(index), "logit": float(value)}
            for value, index in zip(values, indices)
        ],
    }
    (args.output / "pytorch_result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
