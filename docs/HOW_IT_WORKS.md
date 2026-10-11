# How the Competitor Review & Feature Insight Generator Works

This document explains the assignment, the design decisions, how the app processes data end to end, and how to
test, deploy and extend it. For quick setup and run instructions, see the [README](../README.md).

## 1. The assignment and what we built

**Assignment (Option 2, Marketing & Product Strategy):** build a Gradio app that takes raw customer reviews for a
product or competitor, classifies sentiment trends, extracts key customer pain points, and generates three concrete,
prioritized product feature recommendations. Deliver it as a Hugging Face Space plus a 2-page business memo
(roadmap integration, unit economics, competitive-advantage risks).

**What the app does:**

| Requirement | How the app meets it |
|---|---|
| Takes raw reviews | Upload CSV / TSV / JSON / JSONL / zip with any column names |
| Classifies sentiment trends | Per-review sentiment (positive / neutral / negative), distribution chart, and a trend chart when reviews have dates |
| Extracts pain points | LLM groups complaints into themes with severity, review counts and example quotes |
| 3 prioritized recommendations | Exactly three, each ranked by a transparent score and linked to supporting reviews |
| Recognises model limitations | Low-confidence flags, star-vs-text disagreement count, and a limitations panel |

The main design idea is **traceability**: every pain point and recommendation points back to real customer reviews,
and the code checks that those references exist. The output is decision support for a product manager, not an
automated roadmap.

## 2. Architecture at a glance

```
 Upload
      |
      v
 ingest.py:  load -> detect columns -> filter product -> normalize -> English filter -> random sample
      |
      v
 app.py:     sentiment classification (local RoBERTa model)
      |
      v
 app.py:     LLM (HF Inference API) -> pain points + recommendations (JSON)
      |
      v
 app.py:     validate / ground / rank  ->  summary, charts, tables, recommendations in the Gradio UI
```

| File | Role |
|---|---|
| `app.py` | Gradio UI, sentiment, LLM call, validation and ranking, report rendering |
| `ingest.py` | Everything about getting arbitrary data into one standard shape |
| `sample_reviews.csv` | 24 dated smartwatch reviews for a quick demo |
| `tests/test_app.py` | 17 automated tests (LLM stubbed, so no token needed) |
| `requirements.txt` | Dependencies for local use and Hugging Face Spaces |
| `README.md` | Setup, run, deploy |
| `modal_app.py` | Deployment definition for Modal (image, secret, scaling) |
| `requirements-lite.txt` | Dependencies for the PyTorch-free ONNX mode |

Two models are involved, with different jobs:

| Model | Where it runs | Job |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Locally inside the app (downloaded once, about 0.5 GB) | Classify each review as positive, neutral or negative with a confidence score |
| An LLM chosen by `LLM_MODEL` (tested with `gemini-2.5-flash` via Google's free OpenAI-compatible API; `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face Inference Providers also supported) | Remotely, with `LLM_API_KEY` or `HF_TOKEN` | Read the reviews and produce pain points and recommendations |

Why split it this way: a small classifier is fast, cheap and consistent for per-review labels, while turning
hundreds of reviews into themes and feature ideas needs a language model's reasoning. Splitting the work also keeps
the per-analysis cost low (one LLM call per analysis, not one per review).

## 3. Step-by-step data flow

### Step 1: Input

Users upload a review file (or click **Load sample dataset**).
The "Whose reviews are these?" switch (my product vs competitor) is sent to the LLM so recommendations are framed
correctly: fix complaints for your own product, or exploit gaps for a competitor's.

### Step 2: Ingestion (`ingest.py`)

The goal is to turn any reasonable review dataset into one standard table:

`review | date | rating | group`

1. **Load.** `load_table` reads CSV, TSV, JSON (a list of records, a list of strings, or an object wrapping a list),
   JSONL, or a zip containing one of these (it picks the largest supported file). Files over 50 MB are rejected.
2. **Detect columns.** `detect_schema` matches column names against known aliases (for example `review`, `text`,
   `comment` for the text; `date`, `created_at` for dates; `rating`, `stars`; `app_name`, `product` for the product).
   If no text column matches, it falls back to the column whose values are mostly strings with the longest average
   length. The UI shows the result and lets the user override each column under **Data mapping**.
3. **Filter by product.** If the product column has several values, a dropdown lists them with counts. Analysing
   mixed products together would give meaningless themes, so the app asks you to pick one (and warns if you don't).
4. **Normalize.** `normalize` combines title and body into one review text, parses dates and ratings (bad values
   become blank rather than errors), drops empty and duplicate reviews, and collects warnings to show the user.
5. **English filter.** `filter_english` removes non-English reviews with an offline heuristic: text that is mostly
   non-Latin script, or contains more common French/German/Spanish/Italian/Portuguese/Dutch function words than
   English ones, is excluded. No extra model is needed. Short ambiguous text (for example "Super") may slip through.
6. **Sample.** If more than the chosen limit (default 200, slider 50 to 300) remain, `sample_reviews` takes a
   **random, reproducible** sample (fixed seed). Random rather than "first N" so a long history isn't biased toward
   the oldest reviews, and reproducible so reruns give the same result.

### Step 3: Sentiment (`classify_sentiment`)

The RoBERTa model labels each review positive, neutral or negative with a confidence score. Reviews with confidence
below 60% are flagged as **low confidence**, which commonly indicates sarcasm or mixed feelings. Long reviews are
truncated to the model's input limit.

The summary then shows:
- the percentage split by sentiment and a bar chart;
- if ratings exist, the average star rating and how many reviews have a **star/text disagreement** (4 to 5 stars but
  negative text, or 1 to 2 stars but positive text). These are likely sarcasm or misclassification;
- if dates exist, a **trend chart** of sentiment share per period. The period is weekly, monthly or yearly
  depending on the date span, and periods with fewer than 3 reviews are hidden because their percentages would be
  noise. If fewer than two periods remain, no trend chart is shown, which avoids implying a trend that isn't there.

### Step 4: Insight generation (`generate_insights`)

One prompt is sent to the LLM containing:
- the framing (my product or competitor) and the product name;
- every sampled review as `[R<id>] (<sentiment>, <n> stars) <text, up to 600 characters>`;
- instructions to use only the supplied reviews, to treat the sentiment labels as fallible, and to return a single
  JSON object with `pain_points` (theme, description, severity 1 to 5, supporting review ids) and exactly three
  `recommendations` (title, rationale, impact 1 to 5, effort 1 to 5, which pain points they address).

The temperature is low (0.2) for consistent output.

### Step 5: Validation and grounding (`validate_payload`)

LLM output is never trusted blindly. The code:

1. Extracts the JSON object even if the model wraps it in prose or code fences.
2. Checks types and that all ratings are between 1 and 5.
3. **Drops any review id that doesn't exist** in the sampled data, and drops pain points left with no valid review.
4. **Drops recommendations** that address no surviving pain point.
5. Requires at least three grounded recommendations. If the result is malformed or too thin, the app **retries once**
   (`LLM_ATTEMPTS = 2`) and then shows a clear error instead of a partial result.
6. Computes the evidence for each recommendation from the *reviews actually linked to its pain points*, not from
   the model's say-so.

### Step 6: Ranking

Recommendations are ranked by a transparent score computed in code:

```
priority score = impact x severity x number of supporting reviews / effort
```

Impact, severity (of the most severe pain point the recommendation addresses) and effort come from the LLM, each on a
1 to 5 scale; the supporting-review count comes from the validated evidence. Including severity means a recommendation
that fixes the most severe, widely cited pain point is not outranked by a cheaper fix to a minor one. The top three
are kept. Pain points are ordered by supporting reviews x severity. The formula is printed in the UI and in the
downloaded report so a product manager can challenge the numbers rather than treat them as magic.

### Step 7: Report

The UI shows, top to bottom:
- a row of **key-number tiles** (reviews analyzed, % negative, average stars, low-confidence labels, star/text
  disagreements; the last two only when ratings exist) and a short summary with data notes;
- the **sentiment distribution** chart and, for dated data, the **trend** chart, both using one colour scheme
  (green positive, grey neutral, red negative);
- **pain-point cards** (severity, number and share of reviews, description, two example quotes). Cards instead of a
  table so they stay readable on phones;
- the **three recommendations** with their score breakdown, the pain points they address and quotes as evidence;
- a **Download report (Markdown)** button that exports the summary, pain points, recommendations, data notes and
  limitations for use in a roadmap meeting;
- an expandable review-level table (sentiment, confidence, low-confidence flag) and a limitations panel.

Quotes are taken from your data, not generated by the model. Review text and model output are escaped so stray
Markdown or HTML in a review cannot change the layout, and internal review ids the model sometimes writes into prose
(for example "R12") are removed. A progress bar shows the current stage, and a note under the Analyze button says an
analysis takes about 30 to 60 seconds.

## 4. Error handling

All failures, expected or not, are shown **inside the page** (in the summary area, where they stay until the next run)
rather than as a pop-up that disappears or a stack trace. Every other section is cleared so stale results are never
shown next to an error:

| Situation | Behaviour |
|---|---|
| Empty input or no reviews after cleaning | Clear message |
| Unreadable file, bad zip, unsupported type, no review column | Message naming the problem and the columns found |
| Unparseable dates | Warning; trend is disabled or the affected reviews are excluded from it |
| No English reviews | Message that only English is supported |
| No LLM credentials | Message explaining how to set `LLM_API_KEY` or `HF_TOKEN` |
| LLM request fails or times out | Message naming the model and HTTP status. Temporary errors (429, 5xx) are retried up to 3 times with waiting |
| LLM returns unusable output twice | Message asking the user to retry |

## 5. Configuration

| Setting | Default | Notes |
|---|---|---|
| `LLM_API_KEY` (environment variable, Space secret or `.env`) | none | Key for any OpenAI-compatible provider (for example Gemini). Takes priority over `HF_TOKEN`. Never commit it |
| `LLM_BASE_URL` | Gemini's OpenAI-compatible endpoint | Endpoint used with `LLM_API_KEY` |
| `HF_TOKEN` | none | Hugging Face Inference Providers. Free accounts have no credits, so this needs credits or PRO |
| `LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model name for the chosen provider, e.g. `gemini-2.5-flash` |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Must output positive / neutral / negative labels |
| `SENTIMENT_BACKEND` | `auto` | `transformers`, or `onnx` = the same model quantized to int8, without PyTorch (`requirements-lite.txt`) (97.4% label agreement with the full model on 3,880 real reviews; identical 95.1% agreement with star polarity) |
| `MAX_ANALYSES_PER_HOUR` | `0` | Global hourly cap for public deployments |
| `MAX_REVIEWS`, `LOW_CONFIDENCE`, `MIN_REVIEWS_PER_PERIOD`, `LLM_ATTEMPTS` | 200 / 0.60 / 3 / 2 | Constants at the top of `app.py` |

## 6. Testing

Run `pytest tests` (see the README). The 17 tests cover:

- validation and grounding (invalid ids dropped, ranking, too few recommendations, out-of-range ratings, JSON
  extraction from fenced output);
- ingestion (text, CSV, JSON, JSONL, wrapped JSON, list of strings, zip, bad files, column detection including the
  real app-store schema, product filtering, reproducible sampling);
- the English filter and the trend logic;
- error handling (missing token, retry then fail);
- one end-to-end run with the real sentiment model and a stubbed LLM.

**Tested live:** the sample dataset and the Talking Tom app-store subset were run end to end against `gemini-2.5-flash`
(about 15 to 30 seconds each) and produced grounded pain points and recommendations. The Hugging Face Llama path was
not usable because the account had no inference credits.

**Suggested manual checks:** sample dataset; the Talking Tom subset of the app-store file (English); empty input;
a CSV with the wrong columns; an undated CSV (no trend chart expected); a CSV containing a sarcastic line such as
"Oh great, another update that kills the battery" (expect a low-confidence flag or a star/text mismatch).

## 7. Limitations and risks

These are built into the app's limitations panel and are good material for the business memo.

- **Sarcasm and mixed sentiment** are misread by the classifier. Mitigated by flags, not eliminated.
- **English only.** The sentiment model is English; the language filter is a heuristic and not perfect (on the
  provided app-store file, German reviews of WhatsApp partly leak through).
- **Domain specificity.** The sentiment model is trained on general social text, so specialist jargon can reduce accuracy.
- **Sample bias.** Reviews over-represent very happy and very unhappy customers, and random sampling of large files
  may miss rare issues.
- **LLM judgement.** Cited reviews are verified to exist, but whether they really support a claim, and the impact and
  effort ratings, are the model's judgement. A person should confirm before roadmap decisions.
- **Privacy.** A public Space sends review text to the Inference API. Don't use confidential or personal data.
- **Single product per analysis.** Mixed products are discouraged because themes would blur.

## 8. Unit economics (input for the memo)

Per analysis, cost has two parts:

- **Sentiment:** runs locally on the Space's CPU; no per-call fee, but it adds latency (a few seconds for 200 reviews)
  and requires hosting.
- **LLM:** one call. With Gemini's free tier there is no charge, subject to rate limits and Google's free-tier data terms. The prompt is roughly the sampled reviews at up to 600 characters each (about 200 reviews is
  tens of thousands of tokens at most) plus a short JSON answer. Cost = tokens x your provider's per-token price, so
  check the current pricing for your chosen model and provider. Retries (at most one) can double it in the worst case.

Because only one LLM call is made per analysis rather than one per review, cost scales with the sample size cap, not
with dataset size. Cite real prices and measured token counts in the memo rather than the estimates here.

## 9. Deployment

The app is deployed on **Modal** (`modal_app.py`, deploy with `python -m modal deploy modal_app.py`). This was chosen
after checking the alternatives:

| Option | Outcome |
|---|---|
| Hugging Face Spaces | Gradio Spaces now need a paid plan; free ZeroGPU Spaces need an account older than 30 days |
| Render free tier (512 MB) | Too small: the app peaks at about 1 GB with PyTorch, and about 520 to 590 MB even with the quantized ONNX model |
| **Modal free Starter plan** | Works: full app, 2 GB container, scales to zero when idle, within the monthly free credit |

Measured on the live deployment: cold start about 31 seconds, then about 30 seconds for the 24-review sample and about
55 seconds for 200 app-store reviews (sentiment plus one LLM call). The image includes the sentiment model, the LLM key
is stored as a Modal secret, and a cap of 20 analyses per hour protects the free LLM quota.

## 10. Extending the app

| Goal | Where to change |
|---|---|
| Different or larger LLM | Set `LLM_MODEL` |
| Support another language | Replace `SENTIMENT_MODEL` with a multilingual classifier and remove or relax `filter_english` in `parse_reviews` |
| More input formats (Excel, database) | Add a branch to `load_table`; the rest of the pipeline is unchanged |
| More or different recommendation count | `NUM_RECOMMENDATIONS` and the prompt in `build_prompt` |
| Different ranking logic | `validate_payload` (score) and `recommendations_markdown` (display) |
| Other export formats (CSV, PDF) | Extend `write_report` / `build_report_markdown` in `app.py` |

## 11. How this maps to the memo

| Memo section | Material in this document |
|---|---|
| Strategic integration with product roadmaps | Sections 1 and 3: evidence-linked, ranked recommendations feed prioritisation; keep a human in the loop |
| Unit economic cost structure | Sections 2 and 8: one LLM call per analysis, local sentiment, sample cap |
| Competitive-advantage risks | Section 7: bias, sarcasm, privacy, easily copied tooling, and over-trusting model output |
