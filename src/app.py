"""Streamlit dashboard for LinguisticRedline (robust version)."""

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


# ---------------- SAFE CSV LOADER ----------------

def safe_read_csv(path: Path) -> pd.DataFrame:
    """Robust CSV reader for broken / cloud CSVs."""
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


# ---------------- SCHEMA NORMALIZATION ----------------

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent column names across datasets."""
    rename_map = {
        "race": "dominant_race",
        "Race": "dominant_race",
        "ethnicity": "dominant_race",
        "score": "crime_risk_score",
        "risk_score": "crime_risk_score",
        "tract": "id",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ensure required columns exist (avoid KeyError)
    if "dominant_race" not in df.columns:
        df["dominant_race"] = "unknown"

    if "crime_risk_score" not in df.columns and "numeric_score" in df.columns:
        df["crime_risk_score"] = df["numeric_score"]

    return df


# ---------------- LOAD BASE DATA ----------------

@st.cache_data
def load_base() -> pd.DataFrame:
    if not BASE_DESC.exists() or not BASE_RESP.exists():
        return pd.DataFrame()

    desc = safe_read_csv(BASE_DESC)
    resp = safe_read_csv(BASE_RESP)

    desc = normalize_schema(desc)
    resp = normalize_schema(resp)

    if "tract_id" not in resp.columns:
        return pd.DataFrame()

    df = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")
    df = normalize_schema(df)

    return df


# ---------------- MULTI MODEL LOADER ----------------

@st.cache_data
def load_multi_model_data() -> pd.DataFrame:
    if not BASE_DESC.exists():
        return pd.DataFrame()

    desc = safe_read_csv(BASE_DESC)
    desc = normalize_schema(desc)

    parts = []

    for file in Path("data").glob("llm_responses_*.csv"):
        if "all" in file.name:
            continue

        try:
            resp = safe_read_csv(file)
            resp = normalize_schema(resp)

            if "tract_id" not in resp.columns:
                continue

            joined = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")
            joined = normalize_schema(joined)

            parts.append(joined)

        except Exception:
            continue

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


# ---------------- SAFE BADGES ----------------

def render_model_badges(df: pd.DataFrame):
    if df.empty or "model_display_name" not in df.columns:
        st.warning("No model metadata available.")
        return

    models = df[["model_display_name", "provider"]].drop_duplicates()

    html = ""
    for _, r in models.iterrows():
        html += f"""
        <span style="
            background:#334155;
            color:white;
            padding:6px 10px;
            border-radius:999px;
            margin:3px;
            display:inline-block;">
            {r.get('model_display_name','?')} · {r.get('provider','?')}
        </span>
        """

    st.markdown(html, unsafe_allow_html=True)


# ---------------- TAB 1 ----------------

def tab_multi_model():
    df = load_multi_model_data()

    if df.empty:
        st.error("No multi-model data found. Run pipeline first.")
        return

    df = normalize_schema(df)

    st.subheader("Multi-model comparison")
    render_model_badges(df)

    if "crime_risk_score" not in df.columns:
        st.error("Missing crime_risk_score column.")
        return

    summary = df.groupby("model_display_name").agg(
        mean_score=("crime_risk_score", "mean"),
        rows=("id", "count")
    ).reset_index()

    st.dataframe(summary)

    plot_df = df.dropna(subset=["crime_risk_score", "dominant_race"])

    if plot_df.empty:
        st.warning("No valid data for plot.")
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


# ---------------- TAB 2 ----------------

def tab_counterfactual():
    if not COUNTERFACTUAL.exists():
        st.info("Run counterfactual pipeline first.")
        return

    df = safe_read_csv(COUNTERFACTUAL)
    df = normalize_schema(df)

    if df.empty:
        st.warning("Empty counterfactual file.")
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


# ---------------- TAB 3 ----------------

def tab_ground_truth():
    if not GROUND_TRUTH.exists():
        st.info("Run ground truth pipeline first.")
        return

    df = safe_read_csv(GROUND_TRUTH)
    df = normalize_schema(df)

    if df.empty:
        return

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="actual_crime_rate",
        y="crime_risk_score",
        hue="dominant_race",
        ax=ax
    )
    st.pyplot(fig)


# ---------------- TAB 4 ----------------

def tab_debiasing():
    if not DEBIASING.exists():
        st.info("Run debiasing pipeline first.")
        return

    df = safe_read_csv(DEBIASING)

    if df.empty:
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

    st.subheader("Overview")

    if base.empty:
        st.warning("Base dataset missing or broken merge.")
    else:
        cols = [c for c in [
            "id", "city", "dominant_race", "income_bucket",
            "crime_risk_score", "model_display_name", "provider"
        ] if c in base.columns]

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
