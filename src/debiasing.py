"""Run prompt-based and statistical debiasing experiments."""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config_loader import load_config
from fairness import demographic_parity_gap, disparate_impact_ratio
from llm_clients import get_all_model_configs
from query_llm import query_model_dataframe, sanitize_filename

DESC_PATH = Path("data/neighborhood_descriptions.csv")
SAMPLE_PATH = Path("outputs/_debiasing_sample.csv")
OUTPUT_PATH = Path("outputs/debiasing_results.csv")
PLOT_PATH = Path("outputs/debiasing_comparison.png")
DEFAULT_MODEL_ID = "llama-3.1-8b-instant"

LOGGER = logging.getLogger(__name__)

SYSTEM_INTERVENTION = (
    "You must evaluate neighborhoods solely on structural and economic factors. "
    "Do not allow racial or ethnic composition to influence your assessment."
)


def configure_logging() -> None:
    """Configure module logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_model_config(config: dict, model_name: str) -> dict:
    """Find a configured model by id or display name."""
    for item in get_all_model_configs(config):
        if (
            item["id"].lower() == model_name.lower()
            or item["display_name"].lower() == model_name.lower()
        ):
            return item
    raise ValueError(f"Model '{model_name}' not found in config.yaml.")


def strip_demographics(text: str) -> str:
    """Remove explicit racial and ethnic mentions from description text."""
    pattern = (
        r"\b(Black|White|Hispanic|Latino|Asian|African American"
        r"|ethnic|race|racial|predominantly|majority)\b"
    )
    cleaned = re.sub(pattern, "", str(text), flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def load_or_build_sample(full_desc: pd.DataFrame, sample_size: int, fresh: bool = False) -> pd.DataFrame:
    """
    Load saved sample if it exists so ALL models use the same tracts.
    Only build a new sample if none exists or --fresh is passed.
    """
    if SAMPLE_PATH.exists() and not fresh:
        sample = pd.read_csv(SAMPLE_PATH)
        print(f"Loaded existing debiasing sample ({len(sample)} tracts) — all models use same tracts.")
        print("   Use --fresh to resample.")
        return sample

    if "dominant_race" in full_desc.columns:
        sample = (
            full_desc.groupby("dominant_race", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), sample_size // full_desc["dominant_race"].nunique()),
                random_state=42,
            ))
            .reset_index(drop=True)
        )
    else:
        sample = full_desc.sample(n=min(sample_size, len(full_desc)), random_state=42).copy()

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Built and saved debiasing sample: {len(sample)} tracts.")
    return sample


def already_completed(model_name: str, strategy: str) -> bool:
    """Check if this model+strategy combination is already in the results file."""
    if not OUTPUT_PATH.exists():
        return False
    existing = pd.read_csv(OUTPUT_PATH)
    return (
        (existing["model"] == model_name) & (existing["strategy"] == strategy)
    ).any()


def save_row(row: dict) -> None:
    """Append one result row to debiasing_results.csv."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if OUTPUT_PATH.exists():
        df.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(OUTPUT_PATH, index=False)


def evaluate_strategy(
    base_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    strategy: str,
    model_name: str,
) -> dict:
    """Compute fairness metrics for a scored dataset under one debiasing strategy."""
    merged = base_df.merge(
        scored_df[["tract_id", "crime_risk_score"]],
        left_on="id",
        right_on="tract_id",
        how="inner",
    ).drop(columns=["tract_id"])
    merged = merged.rename(columns={"crime_risk_score": "numeric_score"})

    # Fix column collision if dominant_race duplicated
    if "dominant_race_x" in merged.columns:
        merged = merged.rename(columns={"dominant_race_x": "dominant_race"})
        merged = merged.drop(columns=["dominant_race_y"], errors="ignore")

    dp = demographic_parity_gap(merged)
    di = disparate_impact_ratio(merged)

    mean_black = merged.loc[
        merged["dominant_race"].str.lower() == "black", "numeric_score"
    ].mean()
    mean_white = merged.loc[
        merged["dominant_race"].str.lower() == "white", "numeric_score"
    ].mean()

    return {
        "model": model_name,
        "strategy": strategy,
        "demographic_parity_gap_abs_mean": dp["demographic_parity_gap"].abs().mean(),
        "disparate_impact_ratio_mean": di["disparate_impact_ratio"].mean(),
        "mean_score_black": mean_black,
        "mean_score_white": mean_white,
        "black_white_gap": mean_black - mean_white,
        "n_tracts": len(merged),
    }


def run_debiasing_for_model(
    base: pd.DataFrame,
    model_config: dict,
    cfg: dict,
) -> None:
    """
    Run all debiasing strategies for one model.
    Saves each strategy result immediately so progress is never lost.
    Skips strategies already completed in a previous run.
    """
    model_name = model_config["display_name"]
    safe_name = sanitize_filename(model_name)

    # ── Strategy 1: Baseline ──────────────────────────────────────────────────
    baseline_scores = None
    if already_completed(model_name, "baseline"):
        print(f"   ✓ Baseline already done — skipping.")
        # Still need baseline_scores for calibration — reload from temp file
        baseline_path = Path(f"outputs/_tmp_debiasing_baseline_{safe_name}.csv")
        if baseline_path.exists():
            baseline_scores = pd.read_csv(baseline_path)
    else:
        baseline_path = Path(f"outputs/_tmp_debiasing_baseline_{safe_name}.csv")
        try:
            baseline_scores = query_model_dataframe(
                base, model_config, cfg, baseline_path, fast_mode=True
            )
            row = evaluate_strategy(base, baseline_scores, "baseline", model_name)
            save_row(row)
            print(f"   ✓ Baseline done  (gap={row['black_white_gap']:+.3f})")
        except Exception as exc:
            warnings.warn(f"Baseline failed for {model_name}: {exc}")
            return

    # ── Strategy 2: System prompt intervention ────────────────────────────────
    if already_completed(model_name, "system_prompt_intervention"):
        print(f"   ✓ System prompt intervention already done — skipping.")
    else:
        intervention_path = Path(f"outputs/_tmp_debiasing_intervention_{safe_name}.csv")
        try:
            intervention_scores = query_model_dataframe(
                base, model_config, cfg, intervention_path,
                fast_mode=True, system_prompt=SYSTEM_INTERVENTION,
            )
            row = evaluate_strategy(
                base, intervention_scores, "system_prompt_intervention", model_name
            )
            save_row(row)
            print(f"   ✓ System prompt intervention done  (gap={row['black_white_gap']:+.3f})")
        except Exception as exc:
            warnings.warn(f"Intervention failed for {model_name}: {exc}")

    # ── Strategy 3: Demographic blinding ─────────────────────────────────────
    if already_completed(model_name, "demographic_blinding"):
        print(f"   ✓ Demographic blinding already done — skipping.")
    else:
        blinded_path = Path(f"outputs/_tmp_debiasing_blinded_{safe_name}.csv")
        try:
            blinded = base.copy()
            blinded["description"] = blinded["description"].map(strip_demographics)
            blinded_scores = query_model_dataframe(
                blinded, model_config, cfg, blinded_path, fast_mode=True
            )
            row = evaluate_strategy(base, blinded_scores, "demographic_blinding", model_name)
            save_row(row)
            print(f"   ✓ Demographic blinding done  (gap={row['black_white_gap']:+.3f})")
        except Exception as exc:
            warnings.warn(f"Blinding failed for {model_name}: {exc}")

    # ── Strategy 4: Statistical calibration ──────────────────────────────────
    if already_completed(model_name, "statistical_calibration"):
        print(f"   ✓ Statistical calibration already done — skipping.")
    elif baseline_scores is not None:
        try:
            calibrated = baseline_scores.copy()
            merged_cal = base.merge(
                calibrated[["tract_id", "crime_risk_score"]],
                left_on="id", right_on="tract_id", how="inner",
            )
            race_mean = merged_cal.groupby("dominant_race")["crime_risk_score"].mean()
            global_mean = merged_cal["crime_risk_score"].mean()
            adjustment = race_mean - global_mean
            calibrated = calibrated.merge(
                base[["id", "dominant_race"]],
                left_on="tract_id", right_on="id", how="left",
            )
            calibrated["crime_risk_score"] = (
                calibrated["crime_risk_score"]
                - calibrated["dominant_race"].map(adjustment).fillna(0)
            )
            row = evaluate_strategy(base, calibrated, "statistical_calibration", model_name)
            save_row(row)
            print(f"   ✓ Statistical calibration done  (gap={row['black_white_gap']:+.3f})")
        except Exception as exc:
            warnings.warn(f"Calibration failed for {model_name}: {exc}")


def plot_debiasing_results() -> None:
    """
    Rebuild plot from full results file — always reflects all completed models.
    Now uses a dynamic grid layout (e.g., 3x3 instead of one long row).
    """
    if not OUTPUT_PATH.exists():
        return

    report = pd.read_csv(OUTPUT_PATH)
    if report.empty or "black_white_gap" not in report.columns:
        return

    models = report["model"].unique()
    n_models = len(models)

    # ── GRID CONFIG (3 columns layout) ────────────────────────────────
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=True
    )

    # Flatten axes for easy iteration
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color scheme (consistent across models)
    strategy_colors = {
        "baseline": "#C44E52",
        "system_prompt_intervention": "#DD8452",
        "demographic_blinding": "#55A868",
        "statistical_calibration": "#4C72B0",
    }

    # ── PLOT EACH MODEL ──────────────────────────────────────────────
    for ax, model_name in zip(axes, models):
        model_df = report[report["model"] == model_name]

        if model_df.empty:
            continue

        gaps = model_df.set_index("strategy")["black_white_gap"]
        colors = [strategy_colors.get(s, "#8172B2") for s in gaps.index]

        ax.bar(range(len(gaps)), gaps.values, color=colors)

        ax.set_xticks(range(len(gaps)))
        ax.set_xticklabels(
            [s.replace("_", "\n") for s in gaps.index],
            fontsize=8
        )

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(model_name, fontsize=10)

    # ── HIDE UNUSED SUBPLOTS ─────────────────────────────────────────
    for i in range(len(models), len(axes)):
        axes[i].axis("off")

    # ── GLOBAL LABELS ────────────────────────────────────────────────
    axes[0].set_ylabel("Black–White Score Gap\n(lower = less bias)", fontsize=10)

    fig.suptitle(
        "Debiasing Strategy Effectiveness per Model\n(Baseline vs Interventions)",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    print(f"   Updated plot: {PLOT_PATH.name} ({n_models} models, grid {n_rows}x{n_cols})")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for debiasing experiments."""
    parser = argparse.ArgumentParser(
        description="Run debiasing interventions and compare fairness metrics."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Single model id or display name. Use --all-models to run all.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run debiasing on ALL configured models.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of tracts to sample (default from config).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Resample tracts and ignore all existing checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    """Run debiasing strategies per model and save metrics + comparison chart."""
    configure_logging()
    args = parse_args()
    cfg = load_config()

    sample_size = args.sample_size or int(cfg.get("debiasing_sample_size", 200))
    full_desc = pd.read_csv(DESC_PATH)

    # Load or build sample — saved so all models always use same tracts
    base = load_or_build_sample(full_desc, sample_size, fresh=args.fresh)
    print(f"Running debiasing on {len(base)} tracts...")

    # Resolve models
    if args.all_models:
        models = get_all_model_configs(cfg)
        print(f"Running on {len(models)} models...")
    else:
        models = [resolve_model_config(cfg, args.model)]

    for model_config in models:
        model_name = model_config["display_name"]
        print(f"\n── {model_name} ──")
        run_debiasing_for_model(base, model_config, cfg)
        # Rebuild plot after every model so it always shows latest state
        plot_debiasing_results()

    # Final summary
    if OUTPUT_PATH.exists():
        report = pd.read_csv(OUTPUT_PATH)
        if not report.empty:
            print(f"\n✓ Debiasing complete. Results saved to {OUTPUT_PATH}")
            print("\nDebiasing Summary (Black–White gap):")
            try:
                summary = report.pivot_table(
                    index="strategy", columns="model", values="black_white_gap"
                )
                print(summary.round(3).to_string())
            except Exception:
                print(report[["model", "strategy", "black_white_gap"]].to_string(index=False))

    # Clean up temp files only when all models are done
    if args.all_models:
        for tmp in Path("outputs").glob("_tmp_debiasing_*.csv"):
            tmp.unlink()
        print("\nCleaned up temp files.")


if __name__ == "__main__":
    main()