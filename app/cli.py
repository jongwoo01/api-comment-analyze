"""Simple CLI to score texts with the loaded emotion model.

Usage:
    python -m app.cli "This video was amazing" "I was disappointed"

Outputs JSON-friendly dicts so you can quickly sanity-check the model
outside the web service.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from app.main import _predict_probabilities, _validate_texts


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score texts with emotion model")
    parser.add_argument(
        "texts",
        nargs="+",
        help="One or more sentences to score",
    )
    args = parser.parse_args(argv)

    texts = _validate_texts(args.texts)
    results = _predict_probabilities(texts)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
