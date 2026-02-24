from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.utils import ensure_dir, load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Train YAML config")
    parser.add_argument("--ollama-config", default="configs/ollama.yaml", help="Ollama YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.train_config)
    ollama_cfg = load_yaml(args.ollama_config)

    base_model_id = train_cfg["model"]["base_model_id"]
    adapter_dir = resolve_path(train_cfg["training"]["adapter_output_dir"])
    merged_dir = ensure_dir(ollama_cfg["paths"]["merged_model_dir"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(merged_dir))

    print(f"Merged model exported to: {merged_dir}")
    print("Next step: run packages/serving/convert_to_gguf.sh")


if __name__ == "__main__":
    main()
