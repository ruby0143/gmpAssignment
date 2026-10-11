"""Deploy the Gradio app to Modal (https://modal.com) with a public, persistent URL.

    modal deploy modal_app.py

The full app runs as-is: Gradio UI, the PyTorch RoBERTa sentiment model (baked into the image, so cold starts don't
download it) and the LLM via LLM_API_KEY. The container scales to zero when idle, so usage stays within Modal's
free monthly credit for a demo.
"""

import os
from pathlib import Path

import modal

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HERE = Path(__file__).parent


def _local_llm_key() -> str:
    """Read LLM_API_KEY from the environment or the local .env (deploy time only); never printed."""
    key = os.environ.get("LLM_API_KEY", "")
    env_file = HERE / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "LLM_API_KEY":
                key = value.strip().strip("\"'")
    return key


def _download_sentiment_model() -> None:
    from transformers import pipeline

    pipeline("text-classification", model=SENTIMENT_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gradio==6.30.0", "transformers==5.19.0", "huggingface_hub==2.2.0", "pandas==3.0.6", "httpx")
    .run_function(_download_sentiment_model)
    .env(
        {
            "HF_HUB_OFFLINE": "1",  # model is already in the image
            "SENTIMENT_BACKEND": "transformers",
            "LLM_MODEL": "gemini-2.5-flash",
            "MAX_ANALYSES_PER_HOUR": "20",
        }
    )
    .add_local_python_source("app", "ingest")
    .add_local_file(HERE / "sample_reviews.csv", "/root/sample_reviews.csv")
)

if modal.is_local() and not _local_llm_key():
    raise SystemExit("LLM_API_KEY is not set (export it or add it to .env) - the app needs it for recommendations.")

app = modal.App("review-insight-generator")


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({"LLM_API_KEY": _local_llm_key()})],
    max_containers=1,  # Gradio sessions are stateful, so keep a single container
    scaledown_window=600,  # stay warm 10 minutes after the last request, then scale to zero
    cpu=1.0,
    memory=2048,
    timeout=600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI

    import app as review_app

    review_app._sentiment_pipeline()  # load the model at container start so the first analysis is fast
    review_app.demo.queue(default_concurrency_limit=2)
    return gr.mount_gradio_app(FastAPI(), review_app.demo, path="/")
