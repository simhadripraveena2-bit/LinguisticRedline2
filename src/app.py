"""Streamlit dashboard for LinguisticRedline."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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


# ---------------- SAFE LOADING ----------------

@st.cache_data
def load_base() -> pd.DataFrame:
    if not BASE_DESC.exists() or not BASE_RESP.exists():
        return pd.DataFrame()

    desc = pd.read_csv(BASE_DESC)
    resp = pd.read_csv(BASE_RESP)

    # merge safely
    df = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")

    # normalize columns (VERY IMPORTANT)
    if "crime_risk_score" not in df.columns and "numeric_score" in df.columns:
        df["crime_risk_score"] = df["numeric_score"]

    return df


@st.cache_data
def load_multi_model_data() -> pd.DataFrame:
    desc = pd.read_csv(BASE_DESC)
    parts = []

    for file in Path("data").glob("llm_responses_*.csv"):
        if "all" in file.name:
            continue

        try:
            model_df = pd.read_csv(file)

            if "tract_id" not in model_df.columns:
                continue

            joined = desc.merge(model_df, left_on="id", right_on="tract_id", how="inner")

            # standardize
            if "crime_risk_score" not in joined.columns and "numeric_score" in joined.columns:
                joined["crime_risk_score"] = joined["numeric_score"]

            parts.append(joined)

        except Exception:
            continue

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


# ---------------- SAFE UI ----------------

def render_model_badges(df: pd.DataFrame) -> None:
    if df.empty or "model_display_name" not in df.columns:
        st.warning("No model data available.")
        return

    models = df[["model_display_name", "provider"]].drop_duplicates()

    chips = []
    for _, row in models.iterrows():
        color = PROVIDER_COLORS.get(str(row.get("provider", "unknown")).lower(), "#334155")

        chips.append(
            f"<span style='background:{color};color:white;padding:6px 10px;"
            f"border-radius:999px;margin:3px;display:inline-block;'>"
            f"{row['model_display_name']} · {row.get('provider','?')}</span>"
        )

    st.markdown(" ".join(chips), unsafe_allow_html=True)


# ---------------- TABS ----------------

def tab_multi_model():
    df = load_multi_model_data()

    if df.empty:
        st.error("No multi-model data found. Run query_llm.py first.")
        return

    st.subheader("Multi-model comparison")
    render_model_badges(df)

    required = {"model_display_name", "crime_risk_score", "dominant_race"}

    if not required.issubset(df.columns):
        st.error(f"Missing columns: {required - set(df.columns)}")
        return

    summary = (
        df.groupby(["model_display_name", "provider"], as_index=False)
        .agg(
            mean_score=("crime_risk_score", "mean"),
            rows=("tract_id", "count"),
        )
    )

    st.dataframe(summary, use_container_width=True)

    plot_df = df.dropna(subset=["crime_risk_score", "dominant_race"])

    if plot_df.empty:
        st.warning("No valid data for plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.boxplot(
        data=plot_df,
        x="model_display_name",
        y="crime_risk_score",
        hue="dominant_race",
        ax=ax
    )

    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)


def tab_counterfactual():
    if not COUNTERFACTUAL.exists():
        st.info("Run counterfactual pipeline first.")
        return

    df = pd.read_csv(COUNTERFACTUAL)

    if df.empty:
        st.warning("No counterfactual data.")
        return

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="original_score",
        y="counterfactual_score",
        hue="original_race",
        ax=ax
    )
    st.pyplot(fig)


def tab_ground_truth():
    if not GROUND_TRUTH.exists():
        st.info("Run ground truth pipeline first.")
        return

    df = pd.read_csv(GROUND_TRUTH)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="actual_crime_rate",
        y="crime_risk_score",
        hue="dominant_race",
        ax=ax
    )
    st.pyplot(fig)


def tab_debiasing():
    if not DEBIASING.exists():
        st.info("Run debiasing pipeline first.")
        return

    df = pd.read_csv(DEBIASING)

    if df.empty:
        st.warning("No debiasing data.")
        return

    plot_df = df.melt(
        id_vars=["strategy"],
        value_vars=[
            "demographic_parity_gap_abs_mean",
            "disparate_impact_ratio_mean"
        ],
        var_name="metric",
        value_name="value"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="strategy", y="value", hue="metric", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)


# ---------------- MAIN ----------------

def main():
    st.set_page_config(page_title="LinguisticRedline", layout="wide")
    st.title("LinguisticRedline Dashboard")

    base = load_base()

    if base.empty:
        st.warning("Base data missing or broken merge.")
    else:
        cols = [c for c in [
            "id", "city", "dominant_race", "income_bucket",
            "crime_risk_score", "model_display_name", "provider"
        ] if c in base.columns]

        st.subheader("Overview")
        st.dataframe(base[cols].head(20))

    t1, t2, t3, t4 = st.tabs([
        "Multi-Model",
        "Counterfactual",
        "Ground Truth",
        "Debiasing"
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
