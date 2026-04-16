"""Log experiment metadata and fairness summary for each pipeline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config_loader import load_config
from fairness import demographic_parity_gap, disparate_impact_ratio

DESC_PATH = Path("data/neighborhood_descriptions.csv")
RESP_PATH = Path("data/llm_responses_all.csv")
LOG_PATH = Path("outputs/experiment_log.csv")


# ─────────────────────────────────────────────────────────────
# FIX: robust column normalization
# ─────────────────────────────────────────────────────────────

def fix_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist after merge."""

    # --- dominant_race fix ---
    if "dominant_race" not in df.columns:
        if "dominant_race_x" in df.columns:
            df["dominant_race"] = df["dominant_race_x"]
        elif "dominant_race_y" in df.columns:
            df["dominant_race"] = df["dominant_race_y"]
        else:
            raise KeyError(
                "dominant_race not found. Check merge inputs (desc/resp mismatch)."
            )

    # --- city fix ---
    if "city" not in df.columns:
        if "city_x" in df.columns:
            df["city"] = df["city_x"]
        elif "city_y" in df.columns:
            df["city"] = df["city_y"]

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track experiment runs.")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def load_responses() -> pd.DataFrame:
    if RESP_PATH.exists():
        resp = pd.read_csv(RESP_PATH)
    else:
        fallback = Path("data/llm_responses.csv")
        if fallback.exists():
            warnings.warn(f"Using fallback {fallback}")
            resp = pd.read_csv(fallback)
        else:
            raise FileNotFoundError("No response file found.")

    resp = resp.rename(columns={"crime_risk_score": "numeric_score"})

    if "success" in resp.columns:
        resp = resp[resp["success"] == True]

    return resp.dropna(subset=["numeric_score"])


# ─────────────────────────────────────────────────────────────
# CORE LOGGING FUNCTION
# ─────────────────────────────────────────────────────────────

def log_single_model(merged, model_name, run_id, cfg, cfg_hash):
    """Compute metrics for one model."""

    # safety check
    if "dominant_race" not in merged.columns:
        raise KeyError(
            f"dominant_race missing in model {model_name}. "
            f"Available columns: {merged.columns.tolist()}"
        )

    dp = demographic_parity_gap(merged)
    di = disparate_impact_ratio(merged)

    black = merged.loc[
        merged["dominant_race"].str.lower() == "black", "numeric_score"
    ].mean()

    white = merged.loc[
        merged["dominant_race"].str.lower() == "white", "numeric_score"
    ].mean()

    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "num_tracts": int(merged["id"].nunique()),
        "cities": ",".join(sorted(map(str, merged["city"].dropna().unique()))),
        "mean_score_black": black,
        "mean_score_white": white,
        "black_white_gap": (black - white) if pd.notna(black) and pd.notna(white) else None,
        "demographic_parity_gap": dp["demographic_parity_gap"].abs().mean(),
        "disparate_impact_ratio": di["disparate_impact_ratio"].mean(),
        "success_rate": merged["success"].mean() if "success" in merged.columns else None,
        "config_hash": cfg_hash,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config()

    desc = pd.read_csv(DESC_PATH)
    resp = load_responses()

    merged_all = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")

    # FIX HERE (CRITICAL)
    merged_all = fix_merge_columns(merged_all)

    if merged_all.empty:
        print("No data found after merge.")
        return

    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    run_id = args.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    new_rows = []

    if "model_display_name" in merged_all.columns:
        models = merged_all["model_display_name"].unique()
        print(f"Logging {len(models)} models...")

        for model in sorted(models):
            model_df = merged_all[merged_all["model_display_name"] == model].copy()

            # extra safety
            model_df = fix_merge_columns(model_df)

            row = log_single_model(model_df, model, run_id, cfg, cfg_hash)
            new_rows.append(row)

            print(
                f"{model:35s} | "
                f"gap={row['black_white_gap']:+.3f} | "
                f"DPG={row['demographic_parity_gap']:.3f}"
            )
    else:
        merged_all = fix_merge_columns(merged_all)
        row = log_single_model(merged_all, "unknown", run_id, cfg, cfg_hash)
        new_rows.append(row)

    # ── Save log ─────────────────────────────────────────────
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(LOG_PATH, index=False)

    print(f"\n✓ Logged {len(new_rows)} rows → {LOG_PATH}")


if __name__ == "__main__":
    main()