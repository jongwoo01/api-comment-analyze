"""FastAPI service that scores text with the local emotion model.

The model artifacts live in ../my_emotion_model and are loaded once at import
time. Endpoints are lightweight and keep responses aligned with request order
to make the frontend integration predictable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Known emotion labels in the intended order. Update here if the model's label
# mapping changes; the index must match the model's logits order.
# dair-ai/emotion dataset label order
EMOTION_LABELS: Sequence[str] = (
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "my_emotion_model"
MAX_BATCH = 128
MAX_TOKENS = 256

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
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

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits  # shape: [batch, num_labels]
        probabilities = F.softmax(logits, dim=-1)

    results = []
    for row in probabilities:
        scores = {label: float(row[idx]) for idx, label in enumerate(EMOTION_LABELS)}
        dominant = max(scores, key=scores.get)
        results.append({"emotions": scores, "dominant": dominant})

    return results


def _preview_texts(texts: Sequence[str], limit: int = 60) -> list[str]:
    """Shorten texts for safe logging to avoid dumping full content."""

    return [text[:limit] + ("..." if len(text) > limit else "") for text in texts]


@app.get("/health")
def health_check():
    """Liveness probe for load balancers and uptime checks."""

    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Score a single comment and return per-emotion probabilities."""

    texts = _validate_texts([request.text])
    logger.info("/analyze request text_preview=%s", _preview_texts(texts)[0])
    result = _predict_probabilities(texts)[0]
    logger.info("/analyze response dominant=%s", result["dominant"])
    return result


@app.post("/analyze-batch")
def analyze_batch(request: AnalyzeBatchRequest):
    """Score multiple comments while preserving request order."""

    texts = _validate_texts(request.texts)
    logger.info(
        "/analyze-batch request size=%d text_previews=%s",
        len(texts),
        _preview_texts(texts[:5]),
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
