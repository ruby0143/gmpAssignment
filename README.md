---
title: Competitor Review & Feature Insight Generator
emoji: 📊
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Competitor Review & Feature Insight Generator

Paste or upload customer or competitor reviews and get:
- a sentiment breakdown (plus a trend chart when reviews are dated),
- ranked customer pain points with supporting quotes,
- **three prioritized feature recommendations**, each traced back to the reviews that support it.

Sentiment runs locally (`cardiffnlp/twitter-roberta-base-sentiment-latest`). Pain points and recommendations come from an
LLM via the Hugging Face Inference API (default `meta-llama/Llama-3.1-8B-Instruct`). Review ids cited by the LLM are
validated in code, and claims that cite no real review are dropped.

For a full explanation of the design and data flow, see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Setup (local)

Requires Python 3.10+ (3.12 recommended) and **one LLM credential**:
- **Free option:** a Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey), set as `LLM_API_KEY`
  together with `LLM_MODEL` (see `.env.example`). Any OpenAI-compatible provider also works via `LLM_BASE_URL`.
- **Hugging Face option:** an `HF_TOKEN` with *Make calls to Inference Providers*. Hugging Face no longer gives free
  users inference credits, so this needs pre-paid credits or PRO.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env and paste your LLM key (see below)
```

Alternatively export the variables instead of using `.env` (for example `export LLM_API_KEY=...`).
The `.env` file is read automatically at startup and is git-ignored; never share it.

The first run downloads the sentiment model (about 0.5 GB) into `~/.cache/huggingface`.

## Start the app

```bash
python app.py
```

Open http://127.0.0.1:7860. To use another port: `GRADIO_SERVER_PORT=7861 python app.py`.
Stop it with `Ctrl+C`.

Without an LLM credential the app still starts and runs sentiment analysis, but **Analyze** then fails with a clear
"No LLM credentials found" message because the recommendations step needs one.

## Using it

1. Click **Load sample dataset** then **Analyze** for a quick demo, or paste reviews (one per line), or upload a file.
2. Choose whether the reviews are for *your product* or a *competitor* (this changes how recommendations are framed).
3. If the file contains several products, pick one in **Product / app to analyze**.

**Supported files:** CSV, TSV, JSON, JSONL, or a zip containing one of these. Review text, title, date, rating and
product columns are auto-detected; override them under *Data mapping* if needed. Large files are randomly sampled
(reproducibly; default 200 reviews, adjustable with the slider).

**English only:** non-English reviews are detected with an offline heuristic and excluded. Short, ambiguous text
(for example "Super") may slip through.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | none | Key for an OpenAI-compatible provider (takes priority over `HF_TOKEN`) |
| `LLM_BASE_URL` | Gemini's OpenAI-compatible endpoint | Provider endpoint used with `LLM_API_KEY` |
| `HF_TOKEN` | none | Hugging Face token (needs inference credits) |
| `LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model name for the chosen provider, e.g. a Gemini model |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | English sentiment classifier |

## Deploy to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `ingest.py`, `requirements.txt`, `README.md` and `sample_reviews.csv`.
3. In *Settings > Variables and secrets*, add `LLM_API_KEY` (secret) and `LLM_MODEL` (variable). Note that Gradio Spaces now require a paid Hugging Face plan.
4. Wait for the build, then run the sample dataset once to confirm it works.

## Tests

```bash
pip install pytest
pytest tests
```

The tests use a stubbed LLM, so no token is needed. The end-to-end test loads the sentiment model locally.

## Responsible use

Output is decision support, not an automated roadmap: classifiers misread sarcasm, reviews over-represent extreme
opinions, and LLM themes should be confirmed by a person. Do not upload confidential or personal customer data to a
public Space.
