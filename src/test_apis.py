"""Quick connectivity test for every configured free-tier LLM model."""

from __future__ import annotations

import argparse
import logging
import time

from config_loader import load_config
from llm_clients import get_all_model_configs, query_model

TEST_PROMPT = (
    "Rate the crime risk of a neighborhood on a scale of 1 to 10. "
    "You must reply with ONLY a single integer between 1 and 10. "
    "Do not write any words, explanation, punctuation, or formatting. "
    "Correct example response: 6"
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for API smoke tests."""
    parser = argparse.ArgumentParser(
        description="Send a single connectivity test request to every configured LLM."
    )
    return parser.parse_args()


def main() -> None:
    """Run one test prompt against every configured model and print pass/fail results."""
    configure_logging()
    parse_args()
    config = load_config()

    all_configs = get_all_model_configs(config)
    if not all_configs:
        print("No models found in config.yaml — check your models: section.")
        return

    passed = 0
    failed = 0

    print(f"\nTesting {len(all_configs)} model(s)...\n")

    for model_config in all_configs:
        start = time.perf_counter()
        result = query_model(TEST_PROMPT, model_config, config)
        elapsed = time.perf_counter() - start
        label = f"{model_config['provider'].title()} / {model_config['display_name']}"

        if result["success"]:
            score = result["score"]
            score_display = int(score) if score.is_integer() else score
            print(f"  ✓ {label} — Score: {score_display} ({elapsed:.1f}s)")
            passed += 1
        else:
            print(f"  ✗ {label} — Error: {result['error']}")
            LOGGER.error("API test failed for %s in %.1fs", label, elapsed)
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(all_configs)} models.\n")

    if failed > 0:
        print("Tip: Check your API keys in config.yaml for any failing models.")
        print("     Run: python src/test_apis.py to re-test after fixing keys.\n")


if __name__ == "__main__":
    main()