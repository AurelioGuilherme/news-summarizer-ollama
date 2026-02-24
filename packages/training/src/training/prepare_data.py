from __future__ import annotations

import argparse
from dataclasses import dataclass

from datasets import DatasetDict, load_dataset

from training.utils import ensure_dir, load_yaml, normalize_whitespace, resolve_path


@dataclass
class DataConfig:
    dataset_name: str
    language: str
    train_size: float
    validation_size: float
    test_size: float
    min_article_chars: int
    min_summary_chars: int
    max_article_chars: int
    seed: int
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PT-BR summarization dataset")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data YAML config",
    )
    return parser.parse_args()


def load_config(config_path: str) -> DataConfig:
    cfg = load_yaml(config_path)
    ds_cfg = cfg["dataset"]
    return DataConfig(
        dataset_name=ds_cfg["name"],
        language=ds_cfg["language"],
        train_size=float(ds_cfg["train_size"]),
        validation_size=float(ds_cfg["validation_size"]),
        test_size=float(ds_cfg["test_size"]),
        min_article_chars=int(ds_cfg["min_article_chars"]),
        min_summary_chars=int(ds_cfg["min_summary_chars"]),
        max_article_chars=int(ds_cfg["max_article_chars"]),
        seed=int(ds_cfg["seed"]),
        output_dir=cfg["paths"]["processed_dataset_dir"],
    )


def normalize_record(example: dict[str, str]) -> dict[str, str]:
    article = normalize_whitespace(example.get("text", ""))
    summary = normalize_whitespace(example.get("summary", ""))
    return {
        "article": article,
        "summary": summary,
    }


def filter_record(example: dict[str, str], cfg: DataConfig) -> bool:
    article = example["article"]
    summary = example["summary"]
    return (
        len(article) >= cfg.min_article_chars
        and len(summary) >= cfg.min_summary_chars
        and len(article) <= cfg.max_article_chars
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    full_dataset = load_dataset(cfg.dataset_name, cfg.language, split="train")
    full_dataset = full_dataset.map(normalize_record, remove_columns=full_dataset.column_names)
    full_dataset = full_dataset.filter(lambda x: filter_record(x, cfg))

    first_split = full_dataset.train_test_split(
        test_size=(cfg.validation_size + cfg.test_size),
        seed=cfg.seed,
        shuffle=True,
    )
    remaining = first_split["test"].train_test_split(
        test_size=cfg.test_size / (cfg.validation_size + cfg.test_size),
        seed=cfg.seed,
        shuffle=True,
    )

    dataset = DatasetDict(
        {
            "train": first_split["train"],
            "validation": remaining["train"],
            "test": remaining["test"],
        }
    )

    output_dir = ensure_dir(cfg.output_dir)
    dataset.save_to_disk(str(output_dir))

    print(f"Saved dataset to: {output_dir}")
    print(dataset)


if __name__ == "__main__":
    main()
