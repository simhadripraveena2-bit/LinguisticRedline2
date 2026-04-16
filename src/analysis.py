"""Statistical and NLP analysis for LLM neighborhood crime-risk responses."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge

DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")
RESPONSES_PATH = Path("data/llm_responses_all.csv")
OUTPUT_DIR = Path("outputs")

THREAT_KEYWORDS = [
    "danger", "unsafe", "crime", "risk", "violence",
    "fear", "theft", "robbery", "assault",
]

MODEL_PARAM_BILLIONS = {
    "Llama 3.1 8B": 8,
    "Llama 3.1 8B (Cerebras)": 8,
    "Llama 3.3 70B": 70,
    "Llama 4 Scout": 17,
    "Qwen 3 32B": 32,
    "Qwen 3 235B (Cerebras)": 235,
    "Mistral 7B": 7,
    "Mistral Small 3.1": 24,
}

MODEL_ORIGIN = {
    "Llama 3.1 8B": "US",
    "Llama 3.1 8B (Cerebras)": "US",
    "Llama 3.3 70B": "US",
    "Llama 4 Scout": "US",
    "Qwen 3 32B": "Chinese",
    "Qwen 3 235B (Cerebras)": "Chinese",
    "Mistral 7B": "EU",
    "Mistral Small 3.1": "EU",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def load_merged_data(responses_path: Path = RESPONSES_PATH) -> pd.DataFrame:
    desc = pd.read_csv(DESCRIPTIONS_PATH)

    if responses_path.exists():
        resp = pd.read_csv(responses_path)
    else:
        fallback = Path("data/llm_responses.csv")
        if fallback.exists():
            warnings.warn(f"{responses_path} not found, using fallback")
            resp = pd.read_csv(fallback)
        else:
            raise FileNotFoundError("Run query_llm.py first.")

    resp = resp.rename(columns={
        "crime_risk_score": "numeric_score",
        "raw_response": "qualitative_response",
    })

    if "success" in resp.columns:
        resp = resp[resp["success"] == True]

    merged = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")

    # FIX: handle duplicate columns
    if "dominant_race_x" in merged.columns:
        merged = merged.rename(columns={"dominant_race_x": "dominant_race"})
        merged = merged.drop(columns=["dominant_race_y"], errors="ignore")

    for col in ["city", "income_bucket"]:
        if f"{col}_x" in merged.columns:
            merged = merged.rename(columns={f"{col}_x": col})
            merged = merged.drop(columns=[f"{col}_y"], errors="ignore")

    # Normalize race labels
    if "dominant_race" not in merged.columns:
        raise ValueError("dominant_race column missing after merge!")

    merged["dominant_race"] = (
        merged["dominant_race"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    print("Unique race labels:", merged["dominant_race"].unique())

    return merged.dropna(subset=["numeric_score"]).copy()


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_anova(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ["dominant_race", "income_bucket", "amenity_bucket", "city"]:
        if factor not in merged.columns:
            continue
        groups = [
            grp["numeric_score"].dropna().values
            for _, grp in merged.groupby(factor)
        ]
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            rows.append({"factor": factor, "f_stat": f_stat, "p_value": p_val})
    return pd.DataFrame(rows)


def run_regression(merged: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c for c in ["pct_black", "pct_white", "pct_hispanic", "pct_asian", "vacancy_rate"]
        if c in merged.columns
    ]

    if not feature_cols:
        warnings.warn("No regression features found.")
        return pd.DataFrame()

    model_df = merged[["numeric_score"] + feature_cols].fillna(0)
    X = model_df[feature_cols]
    y = model_df["numeric_score"]

    coef = Ridge(alpha=1.0).fit(X, y).coef_

    return pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef
    }).sort_values("coefficient", key=np.abs, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_per_model_race_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    if "model_display_name" not in merged.columns:
        warnings.warn("Missing model_display_name")
        return pd.DataFrame()

    result = (
        merged.groupby(["model_display_name", "dominant_race"])["numeric_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "model_display_name": "model",
            "mean": "mean_score",
            "std": "std_score",
            "count": "n_tracts",
        })
    )

    result["param_billions"] = result["model"].map(MODEL_PARAM_BILLIONS)
    result["origin"] = result["model"].map(MODEL_ORIGIN).fillna("Unknown")

    return result


def run_origin_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    data = merged.copy()
    data["origin"] = data["model_display_name"].map(MODEL_ORIGIN).fillna("Unknown")

    rows = []
    for origin, grp in data.groupby("origin"):
        black_mean = grp[grp["dominant_race"] == "Black"]["numeric_score"].mean()
        white_mean = grp[grp["dominant_race"] == "White"]["numeric_score"].mean()

        rows.append({
            "origin": origin,
            "raw_bias_gap": black_mean - white_mean,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_race_heatmap(merged: pd.DataFrame) -> None:
    pivot = merged.pivot_table(
        index="model_display_name",
        columns="dominant_race",
        values="numeric_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r")
    plt.title("Model vs Race Crime Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_model_race_scores.png")
    plt.close()


def plot_model_bias_by_origin(merged: pd.DataFrame) -> None:
    bw = merged[merged["dominant_race"].isin(["Black", "White"])]

    pivot = bw.pivot_table(
        index="model_display_name",
        columns="dominant_race",
        values="numeric_score",
        aggfunc="mean",
    )

    pivot["gap"] = pivot["Black"] - pivot["White"]
    pivot = pivot.sort_values("gap", ascending=False)

    pivot["gap"].plot(kind="barh")
    plt.title("Bias Gap (Black - White)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_bias_black_vs_white.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# NLP (UNCHANGED CORE)
# ─────────────────────────────────────────────────────────────────────────────

def run_nlp_analysis(merged: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text_df = merged.dropna(subset=["qualitative_response"]).copy()
    if text_df.empty:
        return

    vec = CountVectorizer(max_features=1000, stop_words="english")
    dtm = vec.fit_transform(text_df["qualitative_response"].astype(str))

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    print("NLP done")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    merged = load_merged_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRunning analysis...")

    run_anova(merged).to_csv(OUTPUT_DIR / "anova_results.csv", index=False)
    run_regression(merged).to_csv(OUTPUT_DIR / "regression.csv", index=False)

    print("\nRunning per-model analysis...")
    per_model = run_per_model_race_analysis(merged)
    per_model.to_csv(OUTPUT_DIR / "per_model_race_scores.csv", index=False)

    run_origin_comparison(merged).to_csv(
        OUTPUT_DIR / "origin_bias.csv", index=False
    )

    print("\nGenerating plots...")
    plot_model_race_heatmap(merged)
    plot_model_bias_by_origin(merged)

    print("\nRunning NLP...")
    run_nlp_analysis(merged)

    print("\n✓ All done. Outputs saved in /outputs")


if __name__ == "__main__":
    main()