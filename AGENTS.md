# Repository Guidelines

## Project Structure & Module Organization
- `app/main.py`: FastAPI app with `/health`, `/analyze`, `/analyze-batch`; loads the local BERT model once at import time and keeps batch order stable.
- `my_emotion_model/`: tokenizer/config/checkpoint artifacts referenced by relative path; do not rename or edit unless retraining the model.
- `requirements.txt`: runtime dependencies. Add dev-only tools (formatters, tests) in a separate section or file if needed.
- Suggested test location: create `tests/` with FastAPI `TestClient` suites; keep fixtures small to avoid loading the full model when not required.

## Build, Test, and Development Commands
- Create venv and install deps: `python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt`.
- Run locally with reload: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.
- Quick smoke run: `python app/main.py` (same app, reload enabled).
- Health probe: `curl http://localhost:8000/health`.
- Force CPU if GPU causes issues: `CUDA_VISIBLE_DEVICES="" uvicorn app.main:app --reload`.

## Coding Style & Naming Conventions
- Python 3.12+, PEP8, 4-space indentation; prefer explicit type hints and small helper functions (`_validate_texts`, `_predict_probabilities` pattern).
- Keep constants upper snake case near the top (e.g., `EMOTION_LABELS`, `MAX_BATCH`, `MAX_TOKENS`).
- Use `pathlib.Path` for filesystem references; avoid hardcoding absolute paths.
- Maintain docstrings for FastAPI endpoints and helpers; keep response schemas predictable and ordered.

## Testing Guidelines
- Testing framework of choice: `pytest` with `httpx`/FastAPI `TestClient`; add to dev dependencies before use.
- Name tests by feature (`test_analyze_endpoint.py`, `test_batch_validation.py`); assert label ordering and batch size limits.
- For deterministic outputs, stub model calls or freeze logits; avoid loading the heavy model when unit-testing validation.

## Commit & Pull Request Guidelines
- Commit messages: imperative, concise, optional scope (e.g., `fix: guard empty texts`, `chore: document health check`).
- PRs should link issues when applicable and describe: what changed, why, how to validate (commands/curl examples), and any model-impacting notes.
- Include before/after behavior for API changes; add screenshots only when relevant to responses or docs.

## Security & Configuration Tips
- Do not commit new model artifacts or large data files unless intentionally updating `my_emotion_model/`.
- Keep secrets out of the repo; prefer environment variables and document any required values in the PR.
- Monitor resource usage: large batches and long texts are clamped by `MAX_BATCH` and `MAX_TOKENS`; adjust cautiously and document rationale.
