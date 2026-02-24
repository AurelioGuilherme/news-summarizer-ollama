from __future__ import annotations

import argparse
import json
import time

import requests

DEFAULT_CASES = [
    (
        "economia",
        "O Banco Central indicou que podera reduzir os juros nos proximos meses, "
        "caso a inflacao siga em queda. Analistas esperam impacto positivo no credito, "
        "mas alertam para incertezas no cenario internacional.",
    ),
    (
        "cidades",
        "A prefeitura anunciou um pacote de obras para ampliar corredores de onibus e "
        "modernizar estacoes de transferencia. A gestao afirma que a medida pode reduzir "
        "o tempo medio de deslocamento nos horarios de pico.",
    ),
    (
        "saude",
        "O ministerio divulgou uma campanha nacional de vacinacao com foco em grupos de "
        "risco. Especialistas destacam que a adesao nas primeiras semanas sera decisiva "
        "para conter a circulacao de novos surtos.",
    ),
]


def clean_summary(text: str) -> str:
    summary = text.strip()
    for marker in [
        "\n\nResumo:",
        "\n\nO texto",
        "\n\nExtrair",
        "\n\n1)",
        "\n\nNoticia:",
        "\n\nNotícia:",
        "\n\nAqui est",
    ]:
        if marker in summary:
            summary = summary.split(marker)[0].strip()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Ollama models for summaries"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:11434", help="Ollama base URL"
    )
    parser.add_argument("--model-a", default="qwen2.5:0.5b", help="Primary model")
    parser.add_argument(
        "--model-b", default="resumo-noticias-pt-gguf", help="Secondary model"
    )
    parser.add_argument("--max-tokens", type=int, default=96, help="Max output tokens")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Generation temperature"
    )
    return parser.parse_args()


def call_model(
    *,
    base_url: str,
    model: str,
    text: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    prompt = (
        "Voce e um assistente especialista em resumir noticias em portugues do Brasil.\n"
        "Responda somente em portugues, com objetividade e fidelidade aos fatos.\n"
        "Resuma em ate 3 frases, sem listas.\n\n"
        "Noticia:\n"
        f"{text}\n\n"
        "Resumo:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.8,
            "num_predict": max_tokens,
            "repeat_penalty": 1.2,
        },
    }
    start = time.time()
    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    body = response.json()
    elapsed = time.time() - start
    output = clean_summary(body.get("response", ""))
    return output, elapsed


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for label, text in DEFAULT_CASES:
        model_a_output, model_a_time = call_model(
            base_url=args.base_url,
            model=args.model_a,
            text=text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        model_b_output, model_b_time = call_model(
            base_url=args.base_url,
            model=args.model_b,
            text=text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        rows.append(
            {
                "case": label,
                "model_a": {
                    "name": args.model_a,
                    "seconds": round(model_a_time, 3),
                    "summary": model_a_output,
                },
                "model_b": {
                    "name": args.model_b,
                    "seconds": round(model_b_time, 3),
                    "summary": model_b_output,
                },
            }
        )

    print(json.dumps({"comparisons": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
