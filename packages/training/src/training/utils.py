from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def load_yaml(path_value: str | Path) -> dict[str, Any]:
    with resolve_path(path_value).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(path_value: str | Path) -> Path:
    path = resolve_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def build_prompt(template: str, article: str) -> str:
    return template.format(article=article)
