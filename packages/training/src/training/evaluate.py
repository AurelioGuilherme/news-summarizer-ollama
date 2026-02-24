from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.utils import build_prompt, ensure_dir, load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate summarization LoRA model")
    parser.add_argument("--config", default="configs/train.yaml", help="Train YAML config")
    return parser.parse_args()


def trim_generated(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    return generated_text.strip()


def load_model(base_model_id: str, adapter_dir: Path) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def generate_summaries(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    instruction_template: str,
    generation_cfg: dict[str, Any],
) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []

    max_new_tokens = int(generation_cfg["max_new_tokens"])
    num_beams = int(generation_cfg["num_beams"])
    length_penalty = float(generation_cfg["length_penalty"])
    no_repeat_ngram_size = int(generation_cfg["no_repeat_ngram_size"])
    do_sample = bool(generation_cfg["do_sample"])

    for row in dataset:
        prompt = build_prompt(instruction_template, row["article"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        summary = trim_generated(prompt, generated)

        predictions.append(summary)
        references.append(row["summary"])

    return predictions, references


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    base_model_id = cfg["model"]["base_model_id"]
    adapter_dir = resolve_path(cfg["training"]["adapter_output_dir"])
    dataset_dir = resolve_path(cfg["data"]["dataset_dir"])
    report_dir = ensure_dir(cfg["training"]["report_dir"])

    dataset = load_from_disk(str(dataset_dir))
    test_ds = dataset["test"]

    max_samples = int(cfg["evaluation"].get("max_eval_generation_samples", len(test_ds)))
    if max_samples < len(test_ds):
        test_ds = test_ds.select(range(max_samples))

    model, tokenizer = load_model(base_model_id, adapter_dir)

    predictions, references = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        dataset=test_ds,
        instruction_template=cfg["data"]["instruction_template"],
        generation_cfg=cfg["evaluation"],
    )

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=False)

    samples = []
    for idx in range(min(5, len(predictions))):
        samples.append(
            {
                "article": test_ds[idx]["article"][:500],
                "reference": references[idx],
                "prediction": predictions[idx],
            }
        )

    report = {
        "num_samples": len(predictions),
        "rouge": rouge_scores,
        "samples": samples,
    }

    output_path = Path(report_dir) / "eval_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Evaluation report written to: {output_path}")
    print(json.dumps(rouge_scores, indent=2))


if __name__ == "__main__":
    main()
