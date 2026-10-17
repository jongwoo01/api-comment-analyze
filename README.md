# Korean Comment Emotion API

FastAPI + a pinned Korean KoELECTRA classifier. Existing custom BERT artifacts
remain in `my_emotion_model/` for rollback; the application no longer loads them.

## Run

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
.venv/bin/python -m unittest discover -s tests
```

First startup downloads the model; subsequent runs use the Hugging Face cache.
For an offline deployment set `EMOTION_MODEL_PATH` to the downloaded snapshot.
Use the model and tokenizer from the same revision. The loader checks label names.

## Model

- Source: https://huggingface.co/Jinuuuu/KoELECTRA_fine_tunning_emotion
- Revision: `f3b1c1b572dd555048ce2150e3b77033222ff990`
- Publisher-declared license: MIT; source model card retained in the HF snapshot.
- Outputs: `anger`, `fear`, `embarrassment`, `joy`, `hurt`, `sadness`.
- Korean text only; no neutral class. Sarcasm, mixed emotions and factual questions
  can be misclassified. Scores are model probabilities, not calibrated certainty.
- The eight hand-written Korean regression examples passed before deployment;
  they are a smoke check, not a representative accuracy benchmark.

`GET /health` reports model identity. `POST /analyze` accepts `{"text":"..."}`;
`POST /analyze-batch` accepts `{"texts":["..."]}` and preserves order.
Requests allow 1–128 nonblank texts, truncated to 256 tokens, inferred in chunks
of eight. `TORCH_NUM_THREADS` defaults to 1 for the shared Oracle CPU.
Logs contain request counts and predicted labels, not comment contents.

## Oracle deployment

Checkout: `/srv/comment-analyze/api-comment-analyze`.
Service: `comment-analyze.service` (Gunicorn, two Uvicorn workers).
Caddy forwards the public API to the existing private port 8010.
Download the pinned model before restarting; test a candidate on localhost first.
Update code with a fast-forward Git pull, then restart only this service.
Check `/health`, a Korean positive/negative batch, and the frontend comments route.

Rollback: restore the previous deployed Git commit, restart the service, and
restore the matching frontend deployment. Keep the original model directory.
