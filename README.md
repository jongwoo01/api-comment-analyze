# Comment Analyze Backend

FastAPI service that scores Korean YouTube comments with the bundled BERT
classifier in `my_emotion_model/`.

## Setup

```bash
cd comment-analyze-back
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `curl http://localhost:8000/health`

Single analyze:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "이 영상 너무 좋네요!"}'
```

Batch analyze:

```bash
curl -X POST http://localhost:8000/analyze-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["이 영상 너무 좋네요!", "좀 지루했어요"]}'
```

## Notes

- Labels are fixed to `[joy, sadness, anger, surprise, disgust, fear]` and must
  match the model's training order.
- The model is loaded once at startup and runs on CUDA if available; otherwise
  it falls back to CPU.
- Keep `my_emotion_model/` files untouched; they are referenced relative to the
  service code.
