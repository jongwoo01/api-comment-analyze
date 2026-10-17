"""FastAPI service that scores text with the local emotion model.

A pinned Korean emotion classifier is loaded once at import time. Endpoints are lightweight and keep responses aligned with request order
to make the frontend integration predictable.
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "Jinuuuu/KoELECTRA_fine_tunning_emotion"
MODEL_REVISION = "f3b1c1b572dd555048ce2150e3b77033222ff990"
LABEL_MAP = {
    "angry": "anger", "anxious": "fear", "embarrassed": "embarrassment",
    "happy": "joy", "heartache": "hurt", "sad": "sadness",
}
MODEL_DIR = os.getenv("EMOTION_MODEL_PATH", MODEL_ID)
MAX_BATCH = 128
MAX_TOKENS = 256
INFERENCE_BATCH = 8
# Bound CPU work on the shared two-core Oracle server.
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

logger = logging.getLogger("comment_analyze")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="Comment Analyze Backend", version="1.0.0")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Single comment text to score")


class AnalyzeBatchRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=MAX_BATCH, description="List of comment texts"
    )


def _load_model():
    """Load tokenizer/model once; place on CUDA if available for speed."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = {"revision": MODEL_REVISION} if MODEL_DIR == MODEL_ID else {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, **options)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, use_safetensors=True, **options
    )
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if len(labels) != len(LABEL_MAP) or set(labels) != set(LABEL_MAP):
        raise RuntimeError(f"Unexpected model labels: {labels}")
    model.to(device).eval()
    logger.info("Model loaded for device=%s", device)
    return tokenizer, model, device


tokenizer, model, device = _load_model()


def _validate_texts(texts: Sequence[str]) -> List[str]:
    """Ensure texts are non-empty strings and strip whitespace."""

    cleaned: List[str] = []
    bad_indexes: List[int] = []

    for idx, raw in enumerate(texts):
        if not isinstance(raw, str) or not raw.strip():
            bad_indexes.append(idx)
            continue
        cleaned.append(raw.strip())

    if bad_indexes:
        logger.warning("Rejected empty/invalid texts at indexes=%s", bad_indexes)
        raise HTTPException(
            status_code=400,
            detail=f"Empty or invalid text at index(es): {bad_indexes}",
        )

    return cleaned


def _predict_probabilities(texts: Sequence[str]):
    """Run the model and return probability dicts aligned with EMOTION_LABELS."""

    results = []
    for start in range(0, len(texts), INFERENCE_BATCH):
        encoded = tokenizer(
            list(texts[start:start + INFERENCE_BATCH]), padding=True,
            truncation=True, max_length=MAX_TOKENS, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            probabilities = F.softmax(model(**encoded).logits, dim=-1)
        for row in probabilities:
            scores = {
                LABEL_MAP[model.config.id2label[idx]]: float(value)
                for idx, value in enumerate(row)
            }
            results.append({"emotions": scores, "dominant": max(scores, key=scores.get)})
    return results


@app.get("/health")
def health_check():
    """Liveness probe for load balancers and uptime checks."""

    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REVISION}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Score a single comment and return per-emotion probabilities."""

    texts = _validate_texts([request.text])
    logger.info("/analyze request size=1")
    result = _predict_probabilities(texts)[0]
    logger.info("/analyze response dominant=%s", result["dominant"])
    return result


@app.post("/analyze-batch")
def analyze_batch(request: AnalyzeBatchRequest):
    """Score multiple comments while preserving request order."""

    texts = _validate_texts(request.texts)
    logger.info(
        "/analyze-batch request size=%d",
        len(texts),
    )
    results = _predict_probabilities(texts)
    logger.info(
        "/analyze-batch response size=%d dominants=%s",
        len(results),
        [item["dominant"] for item in results[:5]],
    )
    return results


if __name__ == "__main__":
    # Enable `python app/main.py` for quick local smoke tests.
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
