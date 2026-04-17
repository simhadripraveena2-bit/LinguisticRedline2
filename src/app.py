"""Streamlit dashboard for LinguisticRedline free multi-model analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

BASE_DESC = Path("data/neighborhood_descriptions.csv")
BASE_RESP = Path("data/llm_responses_all.csv")

COUNTERFACTUAL = Path("outputs/counterfactual_results.csv")
GROUND_TRUTH = Path("outputs/ground_truth_comparison.csv")
DEBIASING = Path("outputs/debiasing_results.csv")

PROVIDER_COLORS = {
    "groq": "#16a34a",
    "gemini": "#2563eb",
    "github": "#6b7280",
    "openrouter": "#7c3aed",
    "huggingface": "#ca8a04",
}


@st.cache_data
def load_base() -> pd.DataFrame:
    """Load merged dataset safely (multi-model)."""

    if not BASE_DESC.exists():
        st.error("Missing data/neighborhood_descriptions.csv")
        return pd.DataFrame()

    if not BASE_RESP.exists():
        st.error("Missing data/llm_responses_all.csv — run query_llm.py --all-models")
        return pd.DataFrame()

    desc = pd.read_csv(BASE_DESC)
    resp = pd.read_csv(BASE_RESP)

    df = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")

    # ── SAFE FALLBACKS (IMPORTANT FIX) ─────────────────────────────
    if "model_display_name" not in df.columns:
        df["model_display_name"] = "unknown"

    if "provider" not in df.columns:
        df["provider"] = "unknown"

    return df


@st.cache_data
def load_multi_model_data() -> pd.DataFrame:
    """Robust loader for multi-model data (prevents empty crashes)."""

    desc = pd.read_csv(BASE_DESC)

    file = BASE_RESP  # force single source (ALL FILE)

    if not file.exists():
        return pd.DataFrame()

    model_df = pd.read_csv(file)

    df = desc.merge(model_df, left_on="id", right_on="tract_id", how="inner")

    # ---- SAFE GUARANTEES ----
    if df.empty:
        return df

    if "model_display_name" not in df.columns:
        df["model_display_name"] = "unknown"

    if "provider" not in df.columns:
        df["provider"] = "unknown"

    # IMPORTANT: avoid seaborn crash
    required = ["model_display_name", "crime_risk_score", "dominant_race"]

    for col in required:
        if col not in df.columns:
            df[col] = None

    df = df.dropna(subset=["crime_risk_score"])

    return df


def render_model_badges(model_df: pd.DataFrame) -> None:
    """Render provider-colored model badges."""
    models = (
        model_df[["model_display_name", "provider"]]
        .drop_duplicates()
        .sort_values(["provider", "model_display_name"])
    )

    chips = []
    for row in models.itertuples(index=False):
        color = PROVIDER_COLORS.get(str(row.provider).lower(), "#334155")
        chips.append(
            f'<span style="background:{color};color:white;padding:0.35rem 0.6rem;'
            f'border-radius:999px;margin:0.2rem;display:inline-block;">'
            f'{row.model_display_name} · {str(row.provider).title()} · FREE</span>'
        )

    st.markdown("".join(chips), unsafe_allow_html=True)


def tab_multi_model() -> None:
    all_models = load_multi_model_data()

    if all_models.empty:
        st.info("Run `python src/query_llm.py --all-models` first.")
        return

    st.markdown("#### Free multi-model roster")
    render_model_badges(all_models)

    summary = (
        all_models.groupby(["model_display_name", "provider"], as_index=False)
        .agg(
            mean_score=("crime_risk_score", "mean"),
            rows=("tract_id", "count"),
            failures=("success", lambda x: int((~x).sum())),
        )
        .sort_values(["provider", "model_display_name"])
    )

    st.dataframe(summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    if all_models.empty or "crime_risk_score" not in all_models.columns:
        st.warning("No valid multi-model data found.")
        return

    sns.boxplot(
        data=all_models.dropna(subset=["crime_risk_score"]),
        x="model_display_name",
        y="crime_risk_score",
        hue="dominant_race",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=20)
    ax.set_xlabel("Model")
    ax.set_ylabel("Crime risk score")
    st.pyplot(fig)


def tab_counterfactual() -> None:
    if not COUNTERFACTUAL.exists():
        st.info("Run `python src/counterfactual.py` first.")
        return

    df = pd.read_csv(COUNTERFACTUAL)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="original_score",
        y="counterfactual_score",
        hue="original_race",
        alpha=0.6,
        ax=ax,
    )
    ax.plot([1, 10], [1, 10], linestyle="--", color="black")
    st.pyplot(fig)


def tab_ground_truth() -> None:
    if not GROUND_TRUTH.exists():
        st.info("Run `python src/ground_truth.py` first.")
        return

    df = pd.read_csv(GROUND_TRUTH)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="actual_crime_rate",
        y="crime_risk_score",
        hue="dominant_race",
        alpha=0.5,
        ax=ax,
    )
    st.pyplot(fig)


def tab_debiasing() -> None:
    if not DEBIASING.exists():
        st.info("Run `python src/debiasing.py` first.")
        return

    df = pd.read_csv(DEBIASING)

    plot_df = df.melt(
        id_vars=["strategy"],
        value_vars=[
            "demographic_parity_gap_abs_mean",
            "disparate_impact_ratio_mean",
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="strategy", y="value", hue="variable", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="LinguisticRedline Dashboard", layout="wide")
    st.title("LinguisticRedline: Zero-Cost Multi-Provider Dashboard")

    base = load_base()

    if base.empty:
        st.stop()

    st.subheader("Overview")

    # SAFE COLUMN DISPLAY
    cols = [
        "id",
        "city",
        "dominant_race",
        "income_bucket",
        "crime_risk_score",
        "model_display_name",
        "provider",
    ]

    available_cols = [c for c in cols if c in base.columns]

    st.write(base[available_cols].head(20))

    t1, t2, t3, t4 = st.tabs(
        [
            "Multi-Model Comparison",
            "Counterfactual Analysis",
            "Ground Truth Calibration",
            "Debiasing Results",
        ]
    )

    with t1:
        tab_multi_model()
    with t2:
        tab_counterfactual()
    with t3:
        tab_ground_truth()
    with t4:
        tab_debiasing()


if __name__ == "__main__":
    main()
