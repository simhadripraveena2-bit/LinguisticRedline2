"""Streamlit dashboard for LinguisticRedline multi-model bias analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DESC        = Path("data/neighborhood_descriptions.csv")
BASE_RESP        = Path("data/llm_responses_all.csv")
COUNTERFACTUAL   = Path("outputs/counterfactual_results.csv")
CF_STATS         = Path("outputs/counterfactual_stats_per_model.csv")
GROUND_TRUTH     = Path("outputs/ground_truth_comparison.csv")
GT_PER_MODEL     = Path("outputs/ground_truth_per_model.csv")
DEBIASING        = Path("outputs/debiasing_results.csv")
COHENS_D         = Path("outputs/cohens_d_per_model.csv")
SUPERADDITIVITY  = Path("outputs/superadditivity_results.csv")
INTERSECTIONAL   = Path("outputs/intersectional_heatmap.png")
FAIRNESS_REPORT  = Path("outputs/fairness_report.csv")

# ── Provider colors (match actual providers in use) ───────────────────────────
PROVIDER_COLORS = {
    "groq":      "#16a34a",
    "cerebras":  "#dc2626",
    "mistral":   "#7c3aed",
    "sambanova": "#ea580c",
    "gemini":    "#2563eb",
    "github":    "#6b7280",
}

ORIGIN_COLORS = {
    "US":      "#4C72B0",
    "EU":      "#55A868",
    "Chinese": "#C44E52",
    "Unknown": "#8172B2",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_base() -> pd.DataFrame:
    if not BASE_DESC.exists():
        st.error("Missing data/neighborhood_descriptions.csv")
        return pd.DataFrame()

    if not BASE_RESP.exists():
        st.error("Missing data/llm_responses_all.csv — run query_llm.py first.")
        return pd.DataFrame()

    desc = pd.read_csv(BASE_DESC)
    resp = pd.read_csv(BASE_RESP)

    if "success" in resp.columns:
        resp = resp[resp["success"] == True]

    df = desc.merge(resp, left_on="id", right_on="tract_id", how="inner")

    # Fix column collisions
    if "dominant_race_x" in df.columns:
        df = df.rename(columns={"dominant_race_x": "dominant_race"})
        df = df.drop(columns=["dominant_race_y"], errors="ignore")

    for col in ["city", "income_bucket"]:
        if f"{col}_x" in df.columns:
            df = df.rename(columns={f"{col}_x": col})
            df = df.drop(columns=[f"{col}_y"], errors="ignore")

    # Safe defaults
    df["model_display_name"] = df.get("model_display_name", "unknown")
    df["provider"] = df.get("provider", "unknown")

    # Required column check
    required_cols = ["dominant_race", "crime_risk_score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return pd.DataFrame()

    return df


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists, return empty DataFrame otherwise."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_model_badges(model_df: pd.DataFrame) -> None:
    """Render provider-colored model badges."""
    if model_df.empty:
        return
    models = (
        model_df[["model_display_name", "provider"]]
        .drop_duplicates()
        .sort_values(["provider", "model_display_name"])
    )
    chips = []
    for row in models.itertuples(index=False):
        color = PROVIDER_COLORS.get(str(row.provider).lower(), "#334155")
        chips.append(
            f'<span style="background:{color};color:white;padding:0.3rem 0.6rem;'
            f'border-radius:999px;margin:0.2rem;display:inline-block;font-size:0.85rem;">'
            f'{row.model_display_name} · {str(row.provider).title()} · FREE</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def metric_row(label: str, value: str, delta: str = "") -> None:
    st.metric(label=label, value=value, delta=delta or None)


# ── Tab: Overview ─────────────────────────────────────────────────────────────

def tab_overview(base: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_row("Total Tracts", f"{base['id'].nunique():,}")
    with col2:
        metric_row("Models", str(base["model_display_name"].nunique()))
    with col3:
        metric_row("Cities", str(base["city"].nunique()) if "city" in base.columns else "—")
    with col4:
        metric_row("Mean Risk Score", f"{base['crime_risk_score'].mean():.2f}")

    render_model_badges(base)

    st.markdown("#### Sample rows")
    cols = ["id", "city", "dominant_race", "income_bucket",
            "crime_risk_score", "model_display_name", "provider"]
    st.dataframe(base[[c for c in cols if c in base.columns]].head(20),
                 use_container_width=True)


# ── Tab: Multi-Model Comparison ───────────────────────────────────────────────

def tab_multi_model(base: pd.DataFrame) -> None:
    st.subheader("Multi-Model Crime Risk Comparison")

    if base.empty:
        st.info("Run `python src/query_llm.py --all-models` first.")
        return

    # Summary table
    summary = (
        base.groupby(["model_display_name", "provider"], as_index=False)
        .agg(
            mean_score=("crime_risk_score", "mean"),
            std_score=("crime_risk_score", "std"),
            n_tracts=("tract_id", "count"),
        )
        .sort_values("mean_score", ascending=False)
        .round(3)
    )
    st.dataframe(summary, use_container_width=True)

    # Boxplot: score distribution by model × race
    st.markdown("#### Score Distribution by Model and Race")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=base,
        x="model_display_name",
        y="crime_risk_score",
        hue="dominant_race",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("Model")
    ax.set_ylabel("Crime Risk Score (1–10)")
    ax.set_title("Crime Risk Score Distribution by Model and Neighborhood Race")
    plt.tight_layout()
    st.pyplot(fig)

    # Cohen's d per model
    cohens_df = load_csv(COHENS_D)
    if not cohens_df.empty:
        st.markdown("#### Standardized Racial Bias Gap (Cohen's d) per Model")
        st.caption("Cohen's d > 0 means Black neighborhoods scored higher than White. "
                   "Comparable across models regardless of score scale.")

        plot_df = cohens_df.sort_values("cohens_d", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            ORIGIN_COLORS.get(str(o), "#8172B2")
            for o in plot_df.get("origin", ["Unknown"] * len(plot_df))
        ]
        ax.barh(plot_df["model"], plot_df["cohens_d"], color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Cohen's d (Black vs White neighborhoods)")
        ax.set_title("Standardized Racial Bias per Model\n(colored by training data origin)")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(
            cohens_df[["model", "cohens_d", "p_value", "significant", "origin",
                        "param_billions"]].round(4),
            use_container_width=True,
        )

    # Superadditivity
    super_df = load_csv(SUPERADDITIVITY)
    if not super_df.empty:
        st.markdown("#### Superadditivity Test (Race × Income Interaction)")
        st.caption("A positive, significant interaction coefficient means racial bias "
                   "compounds when neighborhoods are simultaneously Black and low-income.")
        st.dataframe(
            super_df[["model", "interaction_coefficient", "interaction_p_value",
                       "superadditive", "r2_improvement"]].round(4),
            use_container_width=True,
        )


# ── Tab: Counterfactual Analysis ──────────────────────────────────────────────

def tab_counterfactual() -> None:
    st.subheader("Counterfactual Racial Label Swap Analysis")
    st.caption("Same 500-tract sample used across all models. "
               "Score gap = original score − counterfactual score.")

    cf_df = load_csv(COUNTERFACTUAL)
    stats_df = load_csv(CF_STATS)

    if cf_df.empty:
        st.info("Run `python src/counterfactual.py --all-models` first.")
        return

    # Per-model stats table
    if not stats_df.empty:
        st.markdown("#### Per-Model Gap Statistics")
        display_cols = ["model", "mean_gap", "p_value", "significant",
                        "pct_black_scored_higher", "n_pairs"]
        st.dataframe(
            stats_df[[c for c in display_cols if c in stats_df.columns]].round(4),
            use_container_width=True,
        )

        # Bar chart of gaps
        plot_df = stats_df.dropna(subset=["mean_gap"]).sort_values("mean_gap", ascending=False)
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            bar_colors = [
                "#C44E52" if (row["mean_gap"] > 0 and row.get("significant", False))
                else "#F0A0A0" if row["mean_gap"] > 0
                else "#4C72B0"
                for _, row in plot_df.iterrows()
            ]
            ax.bar(plot_df["model"], plot_df["mean_gap"], color=bar_colors)
            ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticklabels(plot_df["model"], rotation=25, ha="right")
            ax.set_ylabel("Mean Score Gap")
            ax.set_title("Counterfactual Racial Bias Gap per Model\n"
                         "(dark red = significant p < 0.05)")
            plt.tight_layout()
            st.pyplot(fig)

    # Model selector for scatter
    models_available = cf_df["model"].unique().tolist() if "model" in cf_df.columns else []
    if models_available:
        selected = st.selectbox("Select model for scatter plot", models_available)
        model_cf = cf_df[cf_df["model"] == selected]

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(
            data=model_cf,
            x="original_score",
            y="counterfactual_score",
            hue="original_race",
            alpha=0.6,
            ax=ax,
        )
        ax.plot([1, 10], [1, 10], linestyle="--", color="black", linewidth=0.8)
        ax.set_title(f"Original vs Counterfactual Score — {selected}")
        ax.set_xlabel("Original Score")
        ax.set_ylabel("Counterfactual Score")
        plt.tight_layout()
        st.pyplot(fig)


# ── Tab: Ground Truth Calibration ─────────────────────────────────────────────

def tab_ground_truth() -> None:
    st.subheader("Ground Truth Calibration")
    st.caption("Bias residual = LLM score − expected score from actual crime rate. "
               "Positive gap = LLM overestimates crime risk in Black neighborhoods "
               "beyond what actual rates would predict.")

    gt_model = load_csv(GT_PER_MODEL)
    gt_full  = load_csv(GROUND_TRUTH)

    if gt_model.empty and gt_full.empty:
        st.info("Run `python src/ground_truth.py` first.")
        return

    if not gt_model.empty:
        st.markdown("#### Overestimation Gap per Model (Black vs White)")
        st.dataframe(
            gt_model[["model", "overall_pearson_corr", "black_bias_residual",
                       "white_bias_residual", "overestimation_gap_black_vs_white",
                       "n_tracts"]].round(4),
            use_container_width=True,
        )

        plot_df = gt_model.dropna(subset=["overestimation_gap_black_vs_white"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = [
                "#C44E52" if v > 0 else "#4C72B0"
                for v in plot_df["overestimation_gap_black_vs_white"]
            ]
            ax.barh(
                plot_df["model"],
                plot_df["overestimation_gap_black_vs_white"],
                color=colors,
            )
            ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Overestimation Gap (Black − White bias residual)")
            ax.set_title("LLM Overestimation of Crime Risk vs Actual Rates")
            plt.tight_layout()
            st.pyplot(fig)

    if not gt_full.empty and "actual_crime_rate" in gt_full.columns:
        st.markdown("#### LLM Score vs Actual Crime Rate (Representative Model)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=gt_full,
            x="actual_crime_rate",
            y="numeric_score",
            hue="dominant_race",
            alpha=0.4,
            ax=ax,
        )
        ax.set_xlabel("Actual Crime Rate (ACS proxy)")
        ax.set_ylabel("LLM Crime Risk Score")
        plt.tight_layout()
        st.pyplot(fig)


# ── Tab: Debiasing Results ────────────────────────────────────────────────────

def tab_debiasing() -> None:
    st.subheader("Debiasing Strategy Effectiveness")
    st.caption("Three strategies tested per model: system prompt intervention, "
               "demographic blinding, and statistical calibration.")

    df = load_csv(DEBIASING)
    if df.empty:
        st.info("Run `python src/debiasing.py --all-models` first.")
        return

    # Model selector
    models = df["model"].unique().tolist() if "model" in df.columns else ["all"]
    selected_model = st.selectbox("Select model", ["All models"] + models)

    if selected_model != "All models":
        df = df[df["model"] == selected_model]

    st.dataframe(
        df[["model", "strategy", "black_white_gap",
            "demographic_parity_gap_abs_mean", "disparate_impact_ratio_mean",
            "n_tracts"]].round(4),
        use_container_width=True,
    )

    # Bar chart: Black-White gap per strategy per model
    if "black_white_gap" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        strategy_colors = {
            "baseline": "#C44E52",
            "system_prompt_intervention": "#DD8452",
            "demographic_blinding": "#55A868",
            "statistical_calibration": "#4C72B0",
        }

        if "model" in df.columns and df["model"].nunique() > 1:
            pivot = df.pivot_table(
                index="strategy", columns="model", values="black_white_gap"
            )
            pivot.plot(kind="bar", ax=ax, colormap="tab10")
            ax.set_xlabel("Strategy")
        else:
            gaps = df.set_index("strategy")["black_white_gap"]
            colors = [strategy_colors.get(s, "#8172B2") for s in gaps.index]
            ax.bar(range(len(gaps)), gaps.values, color=colors)
            ax.set_xticks(range(len(gaps)))
            ax.set_xticklabels(
                [s.replace("_", "\n") for s in gaps.index], fontsize=9
            )

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Black–White Score Gap\n(lower = less bias)")
        ax.set_title("Debiasing Strategy Effectiveness")
        plt.tight_layout()
        st.pyplot(fig)


# ── Tab: Fairness & Intersectionality ─────────────────────────────────────────

def tab_fairness() -> None:
    st.subheader("Fairness Metrics & Intersectional Analysis")

    fairness_df = load_csv(FAIRNESS_REPORT)
    if not fairness_df.empty:
        st.markdown("#### Demographic Parity Gap by Race")
        st.dataframe(fairness_df.round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=fairness_df,
            x="dominant_race",
            y="demographic_parity_gap",
            ax=ax,
            palette="RdBu_r",
        )
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Demographic Parity Gap by Dominant Race\n"
                     "(positive = scored higher than global mean)")
        plt.tight_layout()
        st.pyplot(fig)

    if INTERSECTIONAL.exists():
        st.markdown("#### Race × Income Intersectional Heatmap")
        st.image(str(INTERSECTIONAL))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="LinguisticRedline Dashboard",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 LinguisticRedline: Racial Bias in LLM Urban Crime Risk Perception")
    st.caption("Multi-model empirical study — EMNLP 2026 | All models free-tier")

    base = load_base()
    if base.empty:
        st.stop()

    tabs = st.tabs([
        "📊 Overview",
        "🤖 Multi-Model Comparison",
        "🔄 Counterfactual Analysis",
        "📍 Ground Truth Calibration",
        "⚖️ Debiasing Results",
        "🧩 Fairness & Intersectionality",
    ])

    with tabs[0]:
        tab_overview(base)
    with tabs[1]:
        tab_multi_model(base)
    with tabs[2]:
        tab_counterfactual()
    with tabs[3]:
        tab_ground_truth()
    with tabs[4]:
        tab_debiasing()
    with tabs[5]:
        tab_fairness()


if __name__ == "__main__":
    main()
