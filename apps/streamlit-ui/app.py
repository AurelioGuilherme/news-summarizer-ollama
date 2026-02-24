from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import urlparse

import requests
import streamlit as st
import trafilatura

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
STABLE_MODEL = "qwen2.5:0.5b"
EXPERIMENTAL_MODEL = "resumo-noticias-pt-gguf"

STYLE_RULES = {
    "Curto (2-3 frases)": {
        "min_sentences": 2,
        "max_sentences": 3,
        "paragraphs": 1,
        "token_cap": 96,
    },
    "Medio (1 paragrafo)": {
        "min_sentences": 4,
        "max_sentences": 6,
        "paragraphs": 1,
        "token_cap": 160,
    },
    "Detalhado (2 paragrafos)": {
        "min_sentences": 4,
        "max_sentences": 6,
        "paragraphs": 2,
        "token_cap": 220,
    },
}


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def scrape_article(url: str) -> dict[str, str]:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Nao foi possivel baixar o conteudo da URL.")

    extracted_json = trafilatura.extract(
        downloaded,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_formatting=False,
        favor_recall=True,
        deduplicate=True,
    )
    if not extracted_json:
        raise ValueError("Nao foi possivel extrair texto limpo dessa URL.")

    payload = json.loads(extracted_json)
    article_text = (payload.get("text") or "").strip()
    if len(article_text) < 200:
        raise ValueError("Texto extraido muito curto para resumir com qualidade.")

    parsed = urlparse(url)
    return {
        "url": url,
        "source": parsed.netloc,
        "title": (payload.get("title") or "").strip(),
        "date": (payload.get("date") or "").strip(),
        "text": article_text,
    }


def clean_summary(text: str) -> str:
    summary = text.strip()
    for marker in [
        "\n\nAqui est",
        "\n\nHere are",
        "\n\n1.",
        "\n\nResumo:",
        "\n\nO texto",
        "\n\nExtrair",
        "\n\nNoticia:",
        "\n\nNotícia:",
    ]:
        if marker in summary:
            summary = summary.split(marker)[0].strip()
    return summary


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = []
    for chunk in chunks:
        sentence = chunk.strip(" -\n\t")
        if not sentence:
            continue
        sentence = re.sub(r"^\d+[\).:-]\s*", "", sentence)
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        sentences.append(sentence)
    return sentences


def has_repetition(text: str) -> bool:
    sentences = split_sentences(text)
    if len(sentences) >= 3:
        lowered = [s.lower() for s in sentences]
        unique_ratio = len(set(lowered)) / len(lowered)
        if unique_ratio < 0.75:
            return True

    words = re.findall(r"[\wÀ-ÿ]+", text.lower())
    if len(words) < 24:
        return False

    trigrams: dict[tuple[str, str, str], int] = {}
    for i in range(len(words) - 2):
        tri = (words[i], words[i + 1], words[i + 2])
        trigrams[tri] = trigrams.get(tri, 0) + 1
        if trigrams[tri] >= 3:
            return True
    return False


def enforce_structure(text: str, style_name: str) -> str:
    rules = STYLE_RULES[style_name]
    sentences = split_sentences(clean_summary(text))
    if not sentences:
        return ""

    capped = sentences[: int(rules["max_sentences"])]
    if int(rules["paragraphs"]) == 1:
        return " ".join(capped).strip()

    if len(capped) <= 2:
        return " ".join(capped).strip()

    first_size = min(3, max(2, len(capped) // 2))
    if len(capped) - first_size < 1:
        first_size = len(capped) - 1
    first = " ".join(capped[:first_size]).strip()
    second = " ".join(capped[first_size:]).strip()
    if not second:
        return first
    return f"{first}\n\n{second}"


def build_generation_options(
    *, model_name: str, temperature: float, max_tokens: int, style_name: str
) -> dict[str, float | int]:
    top_p = 0.8
    repeat_penalty = 1.2
    effective_temperature = temperature
    if model_name == EXPERIMENTAL_MODEL:
        top_p = 0.7
        repeat_penalty = 1.25
        effective_temperature = min(temperature, 0.1)

    style_cap = int(STYLE_RULES[style_name]["token_cap"])
    num_predict = min(max_tokens, style_cap)
    return {
        "temperature": effective_temperature,
        "top_p": top_p,
        "num_predict": num_predict,
        "repeat_penalty": repeat_penalty,
    }


def should_retry(raw_text: str, processed_text: str, style_name: str) -> bool:
    if not processed_text:
        return True
    leaked_markers = ["Resumo:", "Noticia:", "Notícia:", "Aqui est", "Here are"]
    if any(marker in raw_text for marker in leaked_markers):
        return True
    if has_repetition(processed_text):
        return True

    sentence_count = len(split_sentences(processed_text))
    max_sentences = int(STYLE_RULES[style_name]["max_sentences"])
    return sentence_count > max_sentences + 1


def generate_summary(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    style_name: str,
) -> tuple[str, float]:
    def run_once(options: dict[str, float | int]) -> tuple[str, float]:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        start = time.time()
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        body = response.json()
        elapsed = time.time() - start
        return body.get("response", ""), elapsed

    options = build_generation_options(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        style_name=style_name,
    )
    raw, elapsed = run_once(options)
    summary = enforce_structure(raw, style_name)

    if should_retry(raw, summary, style_name):
        retry_options = dict(options)
        retry_options["temperature"] = min(float(retry_options["temperature"]), 0.05)
        retry_options["top_p"] = min(float(retry_options["top_p"]), 0.6)
        retry_options["repeat_penalty"] = float(retry_options["repeat_penalty"]) + 0.1
        retry_options["num_predict"] = max(
            64, int(float(retry_options["num_predict"]) * 0.75)
        )
        raw_retry, elapsed_retry = run_once(retry_options)
        retry_summary = enforce_structure(raw_retry, style_name)
        if retry_summary:
            summary = retry_summary
        elapsed += elapsed_retry

    return summary, elapsed


st.set_page_config(
    page_title="Resumo de Noticias PT-BR", page_icon="📰", layout="centered"
)

st.title("📰 Resumo de Noticias com Ollama")
st.caption("Cole uma noticia longa e gere um resumo objetivo em portugues.")

if "scraped_url" not in st.session_state:
    st.session_state.scraped_url = ""
if "scraped_text" not in st.session_state:
    st.session_state.scraped_text = ""
if "scraped_title" not in st.session_state:
    st.session_state.scraped_title = ""
if "scraped_source" not in st.session_state:
    st.session_state.scraped_source = ""
if "scraped_date" not in st.session_state:
    st.session_state.scraped_date = ""

with st.sidebar:
    st.subheader("Configuracoes")
    model_options = [
        STABLE_MODEL,
        EXPERIMENTAL_MODEL,
        "Personalizado...",
    ]
    default_index = (
        model_options.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_options else 2
    )
    selected_model = st.selectbox(
        "Modelo Ollama", options=model_options, index=default_index
    )
    if selected_model == "Personalizado...":
        model_name = st.text_input("Nome do modelo personalizado", value=DEFAULT_MODEL)
    else:
        model_name = selected_model
    base_url = st.text_input("Ollama Base URL", value=DEFAULT_BASE_URL)
    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )
    max_tokens = st.slider(
        "Tokens maximos", min_value=64, max_value=512, value=96, step=16
    )
    st.caption("O limite real de tokens e ajustado automaticamente pelo estilo.")
    input_mode = st.radio("Entrada", ["Texto manual", "Link (scraping)"], index=0)
    estilo = st.selectbox(
        "Estilo",
        ["Curto (2-3 frases)", "Medio (1 paragrafo)", "Detalhado (2 paragrafos)"],
    )
    compare_mode = st.checkbox("Modo comparativo", value=False)
    compare_model = ""
    if compare_mode:
        if model_name == STABLE_MODEL:
            compare_default = EXPERIMENTAL_MODEL
        else:
            compare_default = STABLE_MODEL
        compare_model = st.selectbox(
            "Comparar com",
            options=[STABLE_MODEL, EXPERIMENTAL_MODEL],
            index=0 if compare_default == STABLE_MODEL else 1,
        )

if model_name.strip() == STABLE_MODEL:
    st.info("Modo estavel ativo: modelo base `qwen2.5:0.5b`.")
elif model_name.strip() == EXPERIMENTAL_MODEL:
    st.warning(
        "Modo experimental ativo: `resumo-noticias-pt-gguf` ainda esta em validacao."
    )

style_instruction = {
    "Curto (2-3 frases)": "Resuma em 2 a 3 frases.",
    "Medio (1 paragrafo)": "Resuma em 1 paragrafo de 4 a 6 frases.",
    "Detalhado (2 paragrafos)": "Resuma em 2 paragrafos concisos.",
}[estilo]

news_text = ""
if input_mode == "Texto manual":
    news_text = st.text_area(
        "Texto da noticia",
        placeholder="Cole aqui o texto da noticia...",
        height=280,
    )
else:
    url_input = st.text_input(
        "URL da noticia",
        value=st.session_state.scraped_url,
        placeholder="https://exemplo.com/noticia",
    ).strip()
    extract_clicked = st.button("Extrair texto do link", use_container_width=True)
    if extract_clicked:
        if not is_valid_url(url_input):
            st.error("Informe uma URL valida com http(s).")
        else:
            try:
                with st.spinner("Extraindo conteudo da URL..."):
                    extracted = scrape_article(url_input)
                st.session_state.scraped_url = extracted["url"]
                st.session_state.scraped_text = extracted["text"]
                st.session_state.scraped_title = extracted["title"]
                st.session_state.scraped_source = extracted["source"]
                st.session_state.scraped_date = extracted["date"]
                st.success(
                    "Conteudo extraido com sucesso. Revise o texto antes de resumir."
                )
            except json.JSONDecodeError:
                st.error("Falha ao interpretar o conteudo extraido da pagina.")
            except ValueError as exc:
                st.error(str(exc))

    if st.session_state.scraped_text:
        meta_line = f"Fonte: {st.session_state.scraped_source}"
        if st.session_state.scraped_date:
            meta_line += f" | Data: {st.session_state.scraped_date}"
        st.caption(meta_line)
        if st.session_state.scraped_title:
            st.markdown(f"**Titulo extraido:** {st.session_state.scraped_title}")
    else:
        st.info("Cole uma URL e clique em 'Extrair texto do link'.")

    news_text = st.text_area(
        "Texto extraido (editavel)",
        value=st.session_state.scraped_text,
        placeholder="O texto extraido aparecera aqui para revisao.",
        height=280,
    )

if st.button("Gerar resumo", type="primary", use_container_width=True):
    if not news_text.strip():
        st.error("Insira um texto de noticia antes de gerar o resumo.")
    else:
        prompt = (
            "Voce e um assistente especialista em resumir noticias em portugues do Brasil.\n"
            "Mantenha fidelidade aos fatos e objetividade.\n"
            "Responda somente em portugues e sem listas.\n"
            f"{style_instruction}\n\n"
            "Noticia:\n"
            f"{news_text.strip()}\n\n"
            "Resumo:"
        )

        try:
            with st.spinner("Gerando resumo..."):
                summary, elapsed = generate_summary(
                    base_url=base_url,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    style_name=estilo,
                )
                compare_summary = ""
                compare_elapsed = 0.0
                if compare_mode and compare_model != model_name:
                    compare_summary, compare_elapsed = generate_summary(
                        base_url=base_url,
                        model_name=compare_model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        style_name=estilo,
                    )

            if compare_mode and compare_model != model_name:
                st.subheader("Comparacao")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**{model_name}**")
                    st.write(summary)
                    st.caption(f"Tempo: {elapsed:.2f}s")
                with col_b:
                    st.markdown(f"**{compare_model}**")
                    st.write(compare_summary)
                    st.caption(f"Tempo: {compare_elapsed:.2f}s")
            else:
                st.subheader("Resumo")
                st.write(summary)
                st.caption(f"Tempo de resposta: {elapsed:.2f}s")
        except requests.RequestException as exc:
            st.error(
                "Nao foi possivel conectar ao Ollama. Verifique se o servico esta ativo."
            )
            st.exception(exc)
