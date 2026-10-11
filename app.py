"""Competitor Review & Feature Insight Generator.

Pipeline: parse reviews -> sentiment classification (local Transformers model)
-> pain-point extraction and feature recommendations (Hugging Face Inference API LLM)
-> evidence-grounded report. Every pain point and recommendation is traced back to
the customer reviews that support it; ids cited by the LLM are validated in code.
"""

from __future__ import annotations

import json
import importlib.util
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import gradio as gr
import httpx
import pandas as pd
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

import ingest


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a local .env file without overriding variables that are already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv(Path(__file__).parent / ".env")

SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini
LLM_TIMEOUT_SECONDS = 180
# "transformers" (PyTorch), "onnx" (same model, quantized, no PyTorch: fits 512 MB hosts) or "auto"
SENTIMENT_BACKEND = os.environ.get("SENTIMENT_BACKEND", "auto").strip().lower()
ONNX_SENTIMENT_REPO = os.environ.get("ONNX_SENTIMENT_MODEL", "Xenova/twitter-roberta-base-sentiment-latest")
ONNX_SENTIMENT_FILE = "onnx/model_quantized.onnx"
MAX_ANALYSES_PER_HOUR = int(os.environ.get("MAX_ANALYSES_PER_HOUR", "0"))  # 0 = unlimited; set on public deployments

MAX_REVIEWS = 200
MAX_REVIEW_CHARS = 600
LOW_CONFIDENCE = 0.60
MIN_REVIEWS_PER_PERIOD = 3
NUM_RECOMMENDATIONS = 3
LLM_ATTEMPTS = 2

SENTIMENTS = ["positive", "neutral", "negative"]
PAIN_POINT_COLUMNS = [
    "#",
    "Pain point",
    "Severity (1-5)",
    "Reviews citing",
    "Share of reviews",
    "What customers say",
    "Example evidence",
]
REVIEW_TABLE_COLUMNS = ["review", "date", "rating", "sentiment", "confidence", "low_confidence"]
NONE_CHOICE = "(none)"
SAMPLE_PATH = Path(__file__).parent / "sample_reviews.csv"
SAMPLE_PRODUCT = "Smartwatch (sample)"


class AnalysisError(Exception):
    """A user-facing failure; the message is shown in the UI."""


@dataclass
class PainPoint:
    theme: str
    description: str
    severity: int
    review_ids: list[int]  # 1-based ids of supporting reviews


@dataclass
class Recommendation:
    title: str
    rationale: str
    impact: int
    effort: int
    pain_point_numbers: list[int]  # 1-based positions in the pain-point list
    review_ids: list[int] = field(default_factory=list)
    score: float = 0.0


# ---------------------------------------------------------------- input parsing


def parse_reviews(
    file_path: str | None,
    schema: ingest.Schema | None = None,
    group_value: str | None = None,
    limit: int = MAX_REVIEWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Return DataFrame[review, date, rating, group] (sampled to `limit`) and non-fatal warnings."""
    try:
        if not file_path:
            raise AnalysisError("Upload a review dataset first (or click 'Load sample dataset').")
        raw = ingest.load_table(file_path)
        df, warnings = ingest.normalize(raw, schema or ingest.detect_schema(raw), group_value)
        df, english_warnings = ingest.filter_english(df)
        df, sample_warnings = ingest.sample_reviews(df, limit)
    except ingest.IngestError as exc:
        raise AnalysisError(str(exc)) from exc
    return df, warnings + english_warnings + sample_warnings


# -------------------------------------------------------------------- sentiment


@lru_cache(maxsize=1)
def _sentiment_pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model=SENTIMENT_MODEL)


def classify_sentiment(reviews: list[str]) -> pd.DataFrame:
    """Return DataFrame[sentiment, confidence, low_confidence] aligned with `reviews`."""
    results = _sentiment_pipeline()(reviews, truncation=True, max_length=512, batch_size=16)
    out = pd.DataFrame(
        {
            "sentiment": [r["label"].lower() for r in results],
            "confidence": [round(float(r["score"]), 3) for r in results],
        }
    )
    unknown = set(out["sentiment"]) - set(SENTIMENTS)
    if unknown:
        raise AnalysisError(f"Sentiment model returned unexpected labels: {sorted(unknown)}")
    out["low_confidence"] = out["confidence"] < LOW_CONFIDENCE
    return out


@lru_cache(maxsize=1)
def _onnx_sentiment():
    """Quantized ONNX export of the same RoBERTa sentiment model, run with onnxruntime (no PyTorch)."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    def fetch(name: str) -> str:
        return hf_hub_download(ONNX_SENTIMENT_REPO, name)

    tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    labels = json.loads(Path(fetch("config.json")).read_text())["id2label"]
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False  # lower peak memory on small hosts
    session = ort.InferenceSession(fetch(ONNX_SENTIMENT_FILE), options, providers=["CPUExecutionProvider"])
    return tokenizer, session, [labels[str(i)].lower() for i in range(len(labels))]


def classify_sentiment_onnx(reviews: list[str], batch_size: int = 16) -> pd.DataFrame:
    import numpy as np

    tokenizer, session, labels = _onnx_sentiment()
    sentiments: list[str] = []
    confidences: list[float] = []
    for start in range(0, len(reviews), batch_size):
        encoded = tokenizer.encode_batch(reviews[start : start + batch_size])
        feeds = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        logits = session.run(["logits"], feeds)[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        sentiments += [labels[i] for i in probs.argmax(axis=1)]
        confidences += [round(float(p), 3) for p in probs.max(axis=1)]
    out = pd.DataFrame({"sentiment": sentiments, "confidence": confidences})
    out["low_confidence"] = out["confidence"] < LOW_CONFIDENCE
    return out


def sentiment_backend() -> str:
    if SENTIMENT_BACKEND == "auto":
        return "transformers" if importlib.util.find_spec("torch") else "onnx"
    return SENTIMENT_BACKEND


def run_sentiment(reviews: list[str]) -> pd.DataFrame:
    backend = sentiment_backend()
    if backend == "transformers":
        return classify_sentiment(reviews)
    if backend == "onnx":
        return classify_sentiment_onnx(reviews)
    raise AnalysisError(f"Unknown SENTIMENT_BACKEND '{SENTIMENT_BACKEND}'; use 'auto', 'transformers' or 'onnx'.")


def sentiment_distribution(sentiments: pd.Series) -> pd.DataFrame:
    counts = sentiments.value_counts().reindex(SENTIMENTS, fill_value=0)
    return pd.DataFrame(
        {
            "sentiment": SENTIMENTS,
            "reviews": counts.values,
            "share_pct": (counts.values / counts.sum() * 100).round(1),
        }
    )


def sentiment_trend(df: pd.DataFrame) -> pd.DataFrame | None:
    """Share of each sentiment per period; None when fewer than two periods have enough reviews."""
    dated = df.dropna(subset=["date"])
    if dated.empty:
        return None
    span_days = (dated["date"].max() - dated["date"].min()).days
    freq = "Y" if span_days > 3 * 365 else "M" if span_days > 60 else "W"
    period = dated["date"].dt.to_period(freq).astype(str)
    shares = (
        pd.crosstab(period, dated["sentiment"], normalize="index")
        .reindex(columns=SENTIMENTS, fill_value=0)
        .mul(100)
        .round(1)
    )
    shares = shares[period.value_counts().reindex(shares.index) >= MIN_REVIEWS_PER_PERIOD]
    if len(shares) < 2:
        return None
    return shares.reset_index(names="period").melt(id_vars="period", var_name="sentiment", value_name="share_pct")


# -------------------------------------------------------------------------- LLM

SYSTEM_PROMPT = (
    "You are a product strategy analyst. You turn raw customer reviews into evidence-based product insights. "
    "Use ONLY the reviews provided. Never invent reviews, quotes, or numbers. Reply with a single JSON object and nothing else."
)


def _stars(rating: float) -> str:
    return "" if pd.isna(rating) else f", {rating:g} stars"


def build_prompt(df: pd.DataFrame, target: str, product: str) -> str:
    lines = [
        f"[R{i}] ({row.sentiment}{_stars(row.rating)}) {row.review[:MAX_REVIEW_CHARS]}"
        for i, row in enumerate(df.itertuples(), start=1)
    ]
    name = product.strip() or "the product"
    if target == "Competitor product":
        goal = (
            f"These are reviews of a COMPETITOR product ({name}). Treat its customers' complaints as market gaps and "
            "recommend features that let OUR product win those customers."
        )
    else:
        goal = f"These are reviews of OUR product ({name}). Recommend features that fix what customers complain about."
    return f"""{goal}

Sentiment labels come from an automatic classifier and can be wrong (e.g. sarcasm); star ratings, when given, are the customer's own score. Judge the text yourself.

REVIEWS:
{chr(10).join(lines)}

Return JSON with exactly this shape:
{{
  "pain_points": [
    {{"theme": "short name", "description": "one sentence", "severity": 1-5, "review_ids": [integers of supporting reviews, e.g. 3]}}
  ],
  "recommendations": [
    {{"title": "concrete feature", "rationale": "why, grounded in the reviews", "impact": 1-5, "effort": 1-5, "pain_points": [1-based positions in the pain_points list this addresses]}}
  ]
}}
Rules: give 3 to 6 pain points sorted by importance, each citing at least one real review id. Give exactly {NUM_RECOMMENDATIONS} recommendations, each addressing at least one pain point. Severity and impact: 5 = highest. Effort: 5 = hardest. Recommendations must be concrete features, not vague advice."""


def _extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("response contains no JSON object")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON root is not an object")
    return payload


def _rating(value: object, name: str) -> int:
    number = int(value)  # type: ignore[arg-type]
    if not 1 <= number <= 5:
        raise ValueError(f"{name} must be between 1 and 5, got {number}")
    return number


def _int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [int(v) for v in value]


def validate_payload(payload: dict, review_count: int) -> tuple[list[PainPoint], list[Recommendation]]:
    """Validate LLM output; drop claims not grounded in real reviews. Raises ValueError if unusable."""
    raw_points = payload.get("pain_points")
    raw_recs = payload.get("recommendations")
    if not isinstance(raw_points, list) or not isinstance(raw_recs, list):
        raise ValueError("'pain_points' and 'recommendations' must be lists")

    # Pain-point positions are kept (None for dropped ones) so recommendation references stay correct.
    positional: list[PainPoint | None] = []
    for item in raw_points:
        ids = sorted({i for i in _int_list(item.get("review_ids"), "review_ids") if 1 <= i <= review_count})
        if not ids or not str(item.get("theme", "")).strip():
            positional.append(None)
            continue
        positional.append(
            PainPoint(
                theme=str(item["theme"]).strip(),
                description=str(item.get("description", "")).strip(),
                severity=_rating(item.get("severity"), "severity"),
                review_ids=ids,
            )
        )

    recs: list[Recommendation] = []
    for item in raw_recs:
        refs = [n for n in _int_list(item.get("pain_points"), "pain_points") if 1 <= n <= len(positional)]
        refs = [n for n in refs if positional[n - 1] is not None]
        title = str(item.get("title", "")).strip()
        if not refs or not title:
            continue
        evidence = sorted({i for n in refs for i in positional[n - 1].review_ids})  # type: ignore[union-attr]
        impact, effort = _rating(item.get("impact"), "impact"), _rating(item.get("effort"), "effort")
        recs.append(
            Recommendation(
                title=title,
                rationale=str(item.get("rationale", "")).strip(),
                impact=impact,
                effort=effort,
                pain_point_numbers=refs,
                review_ids=evidence,
                score=round(impact * len(evidence) / effort, 2),
            )
        )

    if len(recs) < NUM_RECOMMENDATIONS:
        raise ValueError(f"only {len(recs)} grounded recommendations returned, need {NUM_RECOMMENDATIONS}")
    recs = sorted(recs, key=lambda r: r.score, reverse=True)[:NUM_RECOMMENDATIONS]

    # Re-number pain points to those that survived, ordered by evidence volume x severity.
    kept = [(n, p) for n, p in enumerate(positional, start=1) if p is not None]
    if not kept:
        raise ValueError("no grounded pain points returned")
    kept.sort(key=lambda np_: (len(np_[1].review_ids) * np_[1].severity), reverse=True)
    renumber = {old: new for new, (old, _) in enumerate(kept, start=1)}
    for rec in recs:
        rec.pain_point_numbers = sorted(renumber[n] for n in rec.pain_point_numbers)
    return [p for _, p in kept], recs


TRANSIENT_STATUS = {429, 500, 502, 503, 504}
TRANSIENT_RETRY_DELAYS = (3, 8, 15)  # seconds to wait before each retry


def _call_openai_compatible(prompt: str, base_url: str, api_key: str, system: str) -> str:
    """Any OpenAI-compatible chat endpoint (Gemini, Groq, OpenRouter, GitHub Models, ...). Retries transient errors."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 16000,  # generous: reasoning models count thinking tokens against this limit
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    for attempt in range(len(TRANSIENT_RETRY_DELAYS) + 1):
        try:
            response = httpx.post(
                url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=LLM_TIMEOUT_SECONDS
            )
            if response.status_code in TRANSIENT_STATUS and attempt < len(TRANSIENT_RETRY_DELAYS):
                time.sleep(TRANSIENT_RETRY_DELAYS[attempt])
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300].replace(api_key, "***")
            raise AnalysisError(
                f"The language model request failed ({LLM_MODEL}, HTTP {exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            if attempt < len(TRANSIENT_RETRY_DELAYS):
                time.sleep(TRANSIENT_RETRY_DELAYS[attempt])
                continue
            raise AnalysisError(f"The language model request failed ({LLM_MODEL}): {type(exc).__name__} {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnalysisError(f"The language model returned an unexpected response format ({LLM_MODEL}).") from exc
    raise AnalysisError(f"The language model request failed ({LLM_MODEL}).")  # unreachable; satisfies type checkers


def call_llm(prompt: str) -> str:
    """Use an OpenAI-compatible provider if LLM_API_KEY is set, otherwise the Hugging Face Inference API."""
    system = SYSTEM_PROMPT
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        return _call_openai_compatible(prompt, os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL), api_key, system)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise AnalysisError(
            "No LLM credentials found. Set LLM_API_KEY (any OpenAI-compatible provider, see README) or HF_TOKEN "
            "(Hugging Face Inference Providers; requires credits) as an environment variable, Space secret or in .env."
        )
    client = InferenceClient(model=LLM_MODEL, token=token, timeout=LLM_TIMEOUT_SECONDS)
    try:
        response = client.chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.2,
        )
    except (HfHubHTTPError, InferenceTimeoutError, httpx.HTTPError) as exc:
        raise AnalysisError(f"The language model request failed ({LLM_MODEL}): {exc}") from exc
    return response.choices[0].message.content or ""


def generate_insights(df: pd.DataFrame, target: str, product: str) -> tuple[list[PainPoint], list[Recommendation]]:
    prompt = build_prompt(df, target, product)
    last_error = ""
    for _ in range(LLM_ATTEMPTS):
        raw = call_llm(prompt)
        try:
            return validate_payload(_extract_json(raw), len(df))
        except (ValueError, TypeError, AttributeError) as exc:  # malformed or ungrounded output: retry
            last_error = str(exc)
    raise AnalysisError(f"The model did not return usable, evidence-backed insights ({last_error}). Please try again.")


# ---------------------------------------------------------------------- reports


def _excerpt(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def pain_point_table(points: list[PainPoint], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, p in enumerate(points, start=1):
        examples = " | ".join(f'"{_excerpt(df.loc[i - 1, "review"], 120)}"' for i in p.review_ids[:2])
        rows.append(
            {
                "#": n,
                "Pain point": p.theme,
                "Severity (1-5)": p.severity,
                "Reviews citing": len(p.review_ids),
                "Share of reviews": f"{len(p.review_ids) / len(df):.0%}",
                "What customers say": p.description,
                "Example evidence": examples,
            }
        )
    return pd.DataFrame(rows)


def recommendations_markdown(recs: list[Recommendation], points: list[PainPoint], df: pd.DataFrame) -> str:
    parts = []
    for rank, r in enumerate(recs, start=1):
        addressed = ", ".join(f"#{n} {points[n - 1].theme}" for n in r.pain_point_numbers)
        quotes = "\n".join(f'  - "{_excerpt(df.loc[i - 1, "review"])}"' for i in r.review_ids[:3])
        parts.append(
            f"### {rank}. {r.title}\n"
            f"{r.rationale}\n\n"
            f"- **Priority score:** {r.score} (impact {r.impact}/5 x {len(r.review_ids)} supporting reviews / effort {r.effort}/5)\n"
            f"- **Addresses:** {addressed}\n"
            f"- **Evidence ({len(r.review_ids)} reviews):**\n{quotes}"
        )
    return "\n\n".join(parts)


def summary_markdown(df: pd.DataFrame, warnings: list[str]) -> str:
    dist = sentiment_distribution(df["sentiment"])
    share = dict(zip(dist["sentiment"], dist["share_pct"]))
    low = int(df["low_confidence"].sum())
    text = (
        f"**{len(df)} reviews analyzed** - "
        f"{share['positive']}% positive, {share['neutral']}% neutral, {share['negative']}% negative."
    )
    if low:
        text += (
            f"\n\n{low} review(s) were classified with low confidence (< {LOW_CONFIDENCE:.0%}); "
            "these often involve sarcasm or mixed feelings. See the review-level table."
        )
    if df["rating"].notna().any():
        mismatch = int(
            (((df["rating"] >= 4) & (df["sentiment"] == "negative")) | ((df["rating"] <= 2) & (df["sentiment"] == "positive"))).sum()
        )
        text += f"\n\nAverage star rating: {df['rating'].mean():.2f}."
        if mismatch:
            text += f" In {mismatch} review(s) the star rating and the text sentiment disagree (possible sarcasm or misclassification)."
    if df["date"].notna().any():
        text += f"\n\nDate range: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}."
    for w in warnings:
        text += f"\n\n> Note: {w}"
    return text


LIMITATIONS = """
- **Sarcasm and mixed sentiment:** classifiers misread statements like "great, another update that kills the battery". Check the low-confidence flags and review-level table.
- **English only:** only English reviews are supported; non-English reviews are detected heuristically and excluded.
- **Domain specificity:** the RoBERTa sentiment model is trained on general social media text (lightweight deployments run a quantized copy of it); industry jargon can reduce accuracy.
- **Sample bias:** reviews over-represent very happy and very unhappy customers and may not reflect your whole customer base.
- **LLM output is decision support:** themes and recommendations are generated by a language model. Cited reviews are verified to exist, but whether they truly support a claim should be confirmed by a person before roadmap decisions.
- **Privacy:** do not upload confidential or personal customer data to a public demo.
"""


_usage_lock = threading.Lock()
_recent_analyses: deque[float] = deque()


def check_usage_cap(now: float | None = None) -> None:
    """Global hourly cap so a public demo can't exhaust the LLM quota."""
    if MAX_ANALYSES_PER_HOUR <= 0:
        return
    now = time.time() if now is None else now
    with _usage_lock:
        while _recent_analyses and now - _recent_analyses[0] > 3600:
            _recent_analyses.popleft()
        if len(_recent_analyses) >= MAX_ANALYSES_PER_HOUR:
            wait_min = int((3600 - (now - _recent_analyses[0])) // 60) + 1
            raise AnalysisError(
                f"This demo is limited to {MAX_ANALYSES_PER_HOUR} analyses per hour to protect its free API quota. "
                f"Please try again in about {wait_min} minute(s)."
            )
        _recent_analyses.append(now)


def _schema_from_ui(review, title, date, rating, group) -> ingest.Schema:
    pick = lambda v: None if v in (None, "", NONE_CHOICE) else v  # noqa: E731
    return ingest.Schema(review=pick(review), title=pick(title), date=pick(date), rating=pick(rating), group=pick(group))


def analyze(
    file_path: str | None,
    target: str,
    product: str,
    review_col: str | None,
    title_col: str | None,
    date_col: str | None,
    rating_col: str | None,
    group_col: str | None,
    group_value: str | None,
    limit: float,
):
    try:
        schema = (
            _schema_from_ui(review_col, title_col, date_col, rating_col, group_col) if review_col else None
        )  # no mapping chosen: auto-detect
        df, warnings = parse_reviews(file_path, schema, group_value or None, int(limit))
        check_usage_cap()
        df = pd.concat([df, run_sentiment(df["review"].tolist())], axis=1)
        points, recs = generate_insights(df, target, product.strip() or (group_value or ""))
    except AnalysisError as exc:
        raise gr.Error(str(exc)) from exc

    trend = sentiment_trend(df)
    review_table = df.assign(date=df["date"].dt.strftime("%Y-%m-%d").fillna(""))[REVIEW_TABLE_COLUMNS]
    return (
        summary_markdown(df, warnings),
        sentiment_distribution(df["sentiment"]),
        gr.update(value=trend, visible=trend is not None),
        pain_point_table(points, df),
        recommendations_markdown(recs, points, df),
        review_table,
    )


def _product_after_file_change(file_path: str | None, product: str) -> dict:
    """Keep the product-name box consistent with the file: set it for the sample, clear a stale sample name."""
    if file_path and Path(file_path).name == SAMPLE_PATH.name:
        return gr.update(value=SAMPLE_PRODUCT)
    return gr.update(value="") if product == SAMPLE_PRODUCT else gr.update()


def inspect_upload(file_path: str | None, product: str = ""):
    """Load an uploaded file, auto-detect its columns, and populate the mapping controls."""
    empty = gr.update(choices=[], value=None)
    name = _product_after_file_change(file_path, product)
    if not file_path:
        return "", empty, empty, empty, empty, empty, empty, name
    try:
        raw = ingest.load_table(file_path)
    except ingest.IngestError as exc:
        gr.Warning(str(exc))
        return f"Could not read the file: {exc}", empty, empty, empty, empty, empty, empty, name
    schema = ingest.detect_schema(raw)
    columns = [str(c) for c in raw.columns]
    optional = [NONE_CHOICE, *columns]
    groups = ingest.group_choices(raw, schema.group)
    info = f"**{len(raw):,} rows** loaded; columns detected automatically (adjust under *Data mapping* if needed)."
    if schema.review is None:
        info += " **No review text column was detected - please select one.**"
    return (
        info,
        gr.update(choices=columns, value=schema.review),
        gr.update(choices=optional, value=schema.title or NONE_CHOICE),
        gr.update(choices=optional, value=schema.date or NONE_CHOICE),
        gr.update(choices=optional, value=schema.rating or NONE_CHOICE),
        gr.update(choices=optional, value=schema.group or NONE_CHOICE),
        gr.update(choices=groups, value=groups[0][1] if groups else None, visible=bool(groups)),
        name,
    )


def refresh_groups(file_path: str | None, group_col: str | None):
    if not file_path or group_col in (None, "", NONE_CHOICE):
        return gr.update(choices=[], value=None, visible=False)
    groups = ingest.group_choices(ingest.load_table(file_path), group_col)
    return gr.update(choices=groups, value=groups[0][1] if groups else None, visible=bool(groups))


def load_sample():
    return str(SAMPLE_PATH), SAMPLE_PRODUCT


# --------------------------------------------------------------------------- UI


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Competitor Review & Feature Insight Generator") as demo:
        gr.Markdown(
            "# Competitor Review & Feature Insight Generator\n"
            "Turn raw customer reviews into sentiment trends, ranked pain points, and **three prioritized, "
            "evidence-backed feature recommendations**."
        )
        with gr.Row():
            with gr.Column():
                file = gr.File(
                    label="Upload a review dataset (CSV, TSV, JSON, JSONL or zip) - any column names work",
                    file_types=ingest.UPLOAD_TYPES,
                    type="filepath",
                    height=140,
                )
                file_info = gr.Markdown()
                group_value = gr.Dropdown(label="Product / app to analyze", choices=[], visible=False)
            with gr.Column():
                target = gr.Radio(["My product", "Competitor product"], value="My product", label="Whose reviews are these?")
                product = gr.Textbox(label="Product name (optional)", placeholder="e.g. Acme Smartwatch")
                limit = gr.Slider(50, 300, value=MAX_REVIEWS, step=10, label="Max reviews to analyze (random sample if larger)")
                with gr.Row():
                    sample_btn = gr.Button("Load sample dataset")
                    run_btn = gr.Button("Analyze", variant="primary")

        with gr.Accordion("Data mapping (auto-detected from your file)", open=False):
            with gr.Row():
                review_col = gr.Dropdown(label="Review text column", choices=[])
                title_col = gr.Dropdown(label="Title column", choices=[])
                date_col = gr.Dropdown(label="Date column", choices=[])
                rating_col = gr.Dropdown(label="Rating column", choices=[])
                group_col = gr.Dropdown(label="Product / app column", choices=[])

        summary = gr.Markdown()
        with gr.Row():
            dist_plot = gr.BarPlot(
                x="sentiment", y="reviews", title="Sentiment distribution", sort=SENTIMENTS, y_lim=[0, None]
            )
            trend_plot = gr.LinePlot(
                x="period",
                y="share_pct",
                color="sentiment",
                title=f"Sentiment trend (% per period, periods with {MIN_REVIEWS_PER_PERIOD}+ reviews)",
                y_lim=[0, 100],
                x_label_angle=45,
                visible=False,
            )
        gr.Markdown("## Key customer pain points")
        pain_table = gr.Dataframe(
            headers=PAIN_POINT_COLUMNS,
            wrap=True,
            interactive=False,
            column_widths=["5%", "16%", "9%", "9%", "9%", "26%", "26%"],
        )
        gr.Markdown("## Top 3 feature recommendations (prioritized)")
        recs_md = gr.Markdown()
        with gr.Accordion("Review-level sentiment", open=False):
            review_table = gr.Dataframe(headers=REVIEW_TABLE_COLUMNS, wrap=True, interactive=False)
        with gr.Accordion("Limitations and responsible use", open=False):
            gr.Markdown(LIMITATIONS)

        mapping = [review_col, title_col, date_col, rating_col, group_col]
        sample_btn.click(load_sample, outputs=[file, product])
        file.change(inspect_upload, inputs=[file, product], outputs=[file_info, *mapping, group_value, product])
        group_col.input(refresh_groups, inputs=[file, group_col], outputs=group_value)
        run_btn.click(
            analyze,
            inputs=[file, target, product, *mapping, group_value, limit],
            outputs=[summary, dist_plot, trend_plot, pain_table, recs_md, review_table],
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    port = os.environ.get("PORT")  # set by hosts such as Render
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0" if port else None, server_port=int(port) if port else None
    )
