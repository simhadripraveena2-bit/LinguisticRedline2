"""Streamlit dashboard for LinguisticRedline free multi-model analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

BASE_DESC = Path("data/neighborhood_descriptions.csv")
BASE_RESP = Path("data/llm_responses.csv")
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
    """Load the default single-model dataset used across dashboard tabs."""
    return pd.read_csv(BASE_DESC).merge(pd.read_csv(BASE_RESP), left_on="id", right_on="tract_id", how="inner")


@st.cache_data
def load_multi_model_data() -> pd.DataFrame:
    """Load and combine all per-model response files in the data directory."""
    desc = pd.read_csv(BASE_DESC)
    parts: list[pd.DataFrame] = []
    for file in sorted(Path("data").glob("llm_responses_*.csv")):
        if file.name == "llm_responses_all.csv":
            continue
        model_df = pd.read_csv(file)
        joined = desc[["id", "dominant_race", "city", "income_bucket"]].merge(
            model_df, left_on="id", right_on="tract_id", how="inner"
        )
        parts.append(joined)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def render_model_badges(model_df: pd.DataFrame) -> None:
    """Render provider-colored model badges with FREE indicators."""
    models = model_df[["model_display_name", "provider"]].drop_duplicates().sort_values(["provider", "model_display_name"])
    chips = []
    for row in models.itertuples(index=False):
        color = PROVIDER_COLORS.get(str(row.provider).lower(), "#334155")
        chips.append(
            f'<span style="background:{color};color:white;padding:0.35rem 0.6rem;border-radius:999px;margin:0.2rem;display:inline-block;">'
            f'{row.model_display_name} · {str(row.provider).title()} · FREE</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def tab_multi_model() -> None:
    """Render side-by-side model comparison plots and free-tier model badges."""
    all_models = load_multi_model_data()
    if all_models.empty:
        st.info("Run `python src/query_llm.py --all-models` to populate model comparison data.")
        return

    st.markdown("#### Free multi-model roster")
    render_model_badges(all_models)

    summary = (
        all_models.groupby(["model_display_name", "provider"], as_index=False)
        .agg(mean_score=("crime_risk_score", "mean"), rows=("tract_id", "count"), failures=("success", lambda x: int((~x).sum())))
        .sort_values(["provider", "model_display_name"])
    )
    st.dataframe(summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=all_models, x="model_display_name", y="crime_risk_score", hue="dominant_race", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    ax.set_xlabel("Model")
    ax.set_ylabel("Crime risk score")
    st.pyplot(fig)


def tab_counterfactual() -> None:
    """Render original vs counterfactual scatter."""
    if not COUNTERFACTUAL.exists():
        st.info("Run `python src/counterfactual.py` first.")
        return
    df = pd.read_csv(COUNTERFACTUAL)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x="original_score", y="counterfactual_score", hue="original_race", alpha=0.6, ax=ax)
    ax.plot([1, 10], [1, 10], linestyle="--", color="black")
    st.pyplot(fig)


def tab_ground_truth() -> None:
    """Render LLM score vs actual crime rate by race."""
    if not GROUND_TRUTH.exists():
        st.info("Run `python src/ground_truth.py` first.")
        return
    df = pd.read_csv(GROUND_TRUTH)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="actual_crime_rate", y="crime_risk_score", hue="dominant_race", alpha=0.5, ax=ax)
    st.pyplot(fig)


def tab_debiasing() -> None:
    """Render debiasing before/after metrics comparison."""
    if not DEBIASING.exists():
        st.info("Run `python src/debiasing.py` first.")
        return
    df = pd.read_csv(DEBIASING)
    plot_df = df.melt(id_vars=["strategy"], value_vars=["demographic_parity_gap_abs_mean", "disparate_impact_ratio_mean"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="strategy", y="value", hue="variable", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)


def main() -> None:
    """Run Streamlit dashboard layout with free multi-model analysis tabs."""
    st.set_page_config(page_title="LinguisticRedline Dashboard", layout="wide")
    st.title("LinguisticRedline: Zero-Cost Multi-Provider Dashboard")

    if not BASE_DESC.exists() or not BASE_RESP.exists():
        st.warning("Base data missing. Run pipeline stages before opening dashboard.")
        return

    base = load_base()
    st.subheader("Overview")
    st.write(base[["id", "city", "dominant_race", "income_bucket", "crime_risk_score", "model_display_name", "provider"]].head(20))

    t1, t2, t3, t4 = st.tabs([
        "Multi-Model Comparison",
        "Counterfactual Analysis",
        "Ground Truth Calibration",
        "Debiasing Results",
    ])
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
