from __future__ import annotations

import argparse
import json

import requests


def clean_summary(text: str) -> str:
    summary = text.strip()
    for marker in [
        "\n\nResumo:",
        "\n\nO texto",
        "\n\nExtrair",
        "\n\nNoticia:",
        "\n\nNotícia:",
        "\n\nAqui est",
    ]:
        if marker in summary:
            summary = summary.split(marker)[0].strip()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for Ollama summarization model"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:11434", help="Ollama base URL"
    )
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Model name in Ollama")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt = (
        "Voce e um assistente especialista em resumir noticias em portugues do Brasil.\n"
        "Responda somente em portugues, com texto objetivo e fiel aos fatos.\n"
        "Nao traduza, nao invente e nao adicione comentarios extras.\n"
        "Resuma a noticia em ate 3 frases, sem listas.\n\n"
        "Noticia:\n"
        "O governo anunciou hoje um novo programa de incentivo fiscal para pequenas "
        "empresas de tecnologia. O objetivo e ampliar a contratacao de profissionais "
        "e acelerar a inovacao regional. Especialistas apontam que a medida pode "
        "gerar impacto no curto prazo, mas depende de regulamentacao clara para "
        "evitar assimetrias de acesso entre empresas de diferentes portes.\n\n"
        "Resumo:"
    )

    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "num_predict": 96,
            "repeat_penalty": 1.2,
        },
    }

    response = requests.post(f"{args.base_url}/api/generate", json=payload, timeout=180)
    response.raise_for_status()

    body = response.json()
    result = {
        "model": args.model,
        "response": clean_summary(body.get("response", "")),
        "total_duration": body.get("total_duration"),
        "eval_count": body.get("eval_count"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
