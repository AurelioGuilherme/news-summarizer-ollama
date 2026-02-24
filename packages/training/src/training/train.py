from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from training.utils import build_prompt, ensure_dir, load_yaml, resolve_path


@dataclass
class TrainBundle:
    config: dict[str, Any]
    model_cfg: dict[str, Any]
    lora_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    data_cfg: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA summarization model")
    parser.add_argument("--config", default="configs/train.yaml", help="Train YAML config")
    return parser.parse_args()


def load_bundle(config_path: str) -> TrainBundle:
    cfg = load_yaml(config_path)
    return TrainBundle(
        config=cfg,
        model_cfg=cfg["model"],
        lora_cfg=cfg["lora"],
        train_cfg=cfg["training"],
        data_cfg=cfg["data"],
    )


def _compute_dtype(name: str) -> torch.dtype:
    if name.lower() == "float16":
        return torch.float16
    if name.lower() == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def tokenize_example(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
    template: str,
    max_len: int,
    max_target_tokens: int,
) -> dict[str, list[int]]:
    prompt = build_prompt(template=template, article=example["article"])
    target = example["summary"].strip()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer does not define eos_token_id")

    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        target_ids = [eos_token_id]

    capped_target_tokens = min(max_target_tokens, max_len - 8)
    target_ids = target_ids[:capped_target_tokens]

    reserved_for_target = len(target_ids) + 1
    max_prompt_tokens = max(1, max_len - reserved_for_target)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_tokens)["input_ids"]

    input_ids = prompt_ids + target_ids + [eos_token_id]
    labels = ([-100] * len(prompt_ids)) + target_ids + [eos_token_id]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.config)

    data_dir = resolve_path(bundle.data_cfg["dataset_dir"])
    dataset = load_from_disk(str(data_dir))

    base_model_id = bundle.model_cfg["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = bool(bundle.model_cfg.get("load_in_4bit", False))
    quantization_config = None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bundle.model_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=_compute_dtype(bundle.model_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if bool(bundle.train_cfg.get("gradient_checkpointing", False)) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=bundle.lora_cfg["r"],
        lora_alpha=bundle.lora_cfg["alpha"],
        lora_dropout=bundle.lora_cfg["dropout"],
        target_modules=bundle.lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    max_len = int(bundle.model_cfg["max_seq_length"])
    template = bundle.data_cfg["instruction_template"]
    max_target_tokens = int(bundle.data_cfg.get("max_target_tokens", 140))

    def mapper(example: dict[str, str]) -> dict[str, list[int]]:
        return tokenize_example(
            example=example,
            tokenizer=tokenizer,
            template=template,
            max_len=max_len,
            max_target_tokens=max_target_tokens,
        )

    max_train_samples = bundle.train_cfg.get("max_train_samples")
    max_eval_samples = bundle.train_cfg.get("max_eval_samples")

    train_raw = dataset["train"]
    eval_raw = dataset["validation"]

    if max_train_samples:
        train_raw = train_raw.select(range(min(int(max_train_samples), len(train_raw))))
    if max_eval_samples:
        eval_raw = eval_raw.select(range(min(int(max_eval_samples), len(eval_raw))))

    remove_columns = train_raw.column_names
    train_ds = train_raw.map(mapper, remove_columns=remove_columns)
    eval_ds = eval_raw.map(mapper, remove_columns=remove_columns)

    output_dir = ensure_dir(bundle.train_cfg["output_dir"])
    report_dir = ensure_dir(bundle.train_cfg["report_dir"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=float(bundle.train_cfg["num_train_epochs"]),
        learning_rate=float(bundle.train_cfg["learning_rate"]),
        warmup_ratio=float(bundle.train_cfg["warmup_ratio"]),
        per_device_train_batch_size=int(bundle.train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(bundle.train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(bundle.train_cfg["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(bundle.train_cfg["gradient_checkpointing"]),
        weight_decay=float(bundle.train_cfg["weight_decay"]),
        max_grad_norm=float(bundle.train_cfg["max_grad_norm"]),
        logging_steps=int(bundle.train_cfg["logging_steps"]),
        save_steps=int(bundle.train_cfg["save_steps"]),
        eval_steps=int(bundle.train_cfg["eval_steps"]),
        save_total_limit=int(bundle.train_cfg["save_total_limit"]),
        seed=int(bundle.train_cfg["seed"]),
        fp16=bool(bundle.train_cfg["fp16"]),
        bf16=bool(bundle.train_cfg["bf16"]),
        optim=bundle.train_cfg["optim"],
        lr_scheduler_type=bundle.train_cfg["lr_scheduler_type"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    adapter_output_dir = ensure_dir(bundle.train_cfg["adapter_output_dir"])
    trainer.model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))

    metrics_path = Path(report_dir) / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Adapter saved to: {adapter_output_dir}")
    print(f"Training metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
