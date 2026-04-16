"""Master runner for LinguisticRedline extended EMNLP workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    ("fetch_census", [sys.executable, "src/fetch_census.py"], Path("data/census_tracts.csv")),
    ("fetch_osm", [sys.executable, "src/fetch_osm.py"], Path("data/tracts_with_amenities.csv")),
    ("generate_descriptions", [sys.executable, "src/generate_descriptions.py"], Path("data/neighborhood_descriptions.csv")),
    ("query_llm", [sys.executable, "src/query_llm.py"], Path("data/llm_responses.csv")),
    ("counterfactual", [sys.executable, "src/counterfactual.py"], Path("outputs/counterfactual_results.csv")),
    ("ground_truth", [sys.executable, "src/ground_truth.py"], Path("outputs/ground_truth_comparison.csv")),
    ("analysis", [sys.executable, "src/analysis.py"], Path("outputs/anova_results.csv")),
    ("debiasing", [sys.executable, "src/debiasing.py"], Path("outputs/debiasing_results.csv")),
    ("fairness", [sys.executable, "src/fairness.py"], Path("outputs/fairness_report.csv")),
    ("experiment_tracker", [sys.executable, "src/experiment_tracker.py"], Path("outputs/experiment_log.csv")),
]


def parse_args() -> argparse.Namespace:
    """Parse skip flags for each pipeline stage."""
    parser = argparse.ArgumentParser(description="Run extended LinguisticRedline pipeline.")
    for name, _, _ in STEPS:
        parser.add_argument(f"--skip-{name.replace('_', '-')}", action="store_true")
    return parser.parse_args()


def run_step(name: str, cmd: list[str], expected_output: Path, skip: bool) -> None:
    """Run one pipeline step unless skipped by CLI flag."""
    if skip:
        print(f"[skip] {name}")
        return
    print(f"[run] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if expected_output.exists():
        print(f"[ok] {name} -> {expected_output}")


def main() -> None:
    """Run all configured stages with explicit skip controls."""
    args = parse_args()
    for name, cmd, output in STEPS:
        attr = f"skip_{name}".replace("-", "_")
        run_step(name, cmd, output, skip=getattr(args, attr, False))


if __name__ == "__main__":
    main()
