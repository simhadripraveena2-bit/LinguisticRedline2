# LinguisticRedline: Uncovering Racial Bias in LLM Perceptions of Urban Crime Risk

*A Multi-Model Empirical Study of Racial and Socioeconomic Bias in LLM Perceptions of Urban Crime Risk*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status: Research](https://img.shields.io/badge/status-Research-orange)
![Models](https://img.shields.io/badge/models-Groq%20%7C%20Cerebras%20%7C%20Mistral-purple)
![Scale](https://img.shields.io/badge/scale-5%2C000%20Census%20Tracts-teal)
![Target Venue](https://img.shields.io/badge/target-EMNLP%202026-darkgreen)

## Live Demo

Experience the full multi-model bias analysis dashboard:

**[LinguisticRedline Dashboard](https://linguisticredline2-ve26kgbdv7ergrtiu8mwnm.streamlit.app/)**

- Multi-model comparison (Llama, Mistral, Qwen, etc.)
- Counterfactual bias analysis
- Debiasing strategy evaluation
- Ground truth calibration

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Research Questions](#research-questions)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Ethical Statement](#ethical-statement)
- [Citation](#citation)
- [Author](#author)

## Overview

LinguisticRedline investigates how large language models (LLMs) perceive urban crime risk when presented with neighborhood descriptions derived from real demographic and built-environment data. The EMNLP 2026 pipeline covers 5,000 census tracts across 20 U.S. cities, supports multi-model evaluation across 7 free-tier models spanning 3 model families (Meta Llama, Qwen/Alibaba, Mistral), includes counterfactual prompt-pair analysis, compares LLM scores against ground-truth crime signals (or ACS crime proxies), runs three debiasing strategies, and adds deep NLP analysis using topic modeling, SHAP-style attribution, and sentiment modeling. Total cost to replicate the full study: **$0**.

Key research contributions:
- First systematic cross-model comparison of racial bias in urban crime risk perception across US-origin, EU-origin, and Chinese-origin LLMs
- Standardized Cohen's d bias measurement enabling fair cross-model comparison across models with different score distributions
- Statistical test for superadditivity — does racial bias compound when neighborhoods are simultaneously Black and low-income?
- Counterfactual prompt-pair analysis with per-model t-test significance testing (same 500-tract pairs used across all models)
- Three debiasing strategies evaluated per model with full resume/checkpoint support

## Project Structure

```text
LinguisticRedline/
├── data/
│   ├── raw/
│   ├── census_tracts.csv
│   ├── tracts_with_amenities.csv
│   ├── neighborhood_descriptions.csv
│   ├── llm_responses.csv
│   ├── llm_responses_all.csv              # combined multi-model responses (rebuilt after every run)
│   ├── llm_responses_{model_name}.csv     # per-model response files
│   ├── counterfactual_pairs.csv           # stratified pairs shared across ALL models
│   └── osm_cache/
├── outputs/
│   ├── fairness_report.csv
│   ├── disparate_impact_by_vacancy.csv
│   ├── fairness_by_city.csv
│   ├── cohens_d_per_model.csv
│   ├── cohens_d_per_model.png
│   ├── superadditivity_results.csv
│   ├── cross_model_comparison.csv
│   ├── size_vs_bias.png
│   ├── per_model_race_scores.csv
│   ├── origin_bias_comparison.csv
│   ├── heatmap_model_race_scores.png
│   ├── model_bias_black_vs_white.png
│   ├── anova_results.csv
│   ├── regression_coefficients.csv
│   ├── city_breakdown.csv
│   ├── city_race_breakdown.csv
│   ├── threat_keyword_counts.csv
│   ├── merged_with_scores.csv
│   ├── counterfactual_results.csv
│   ├── counterfactual_stats_per_model.csv
│   ├── counterfactual_gap_per_model.png
│   ├── ground_truth_comparison.csv
│   ├── ground_truth_per_model.csv
│   ├── ground_truth_bias_residuals.png
│   ├── intersectional_heatmap.png
│   ├── intersectional_anova.csv
│   ├── intersectional_fairness.csv
│   ├── topic_model_results.csv
│   ├── sentiment_by_race.csv
│   ├── shap_top_features.csv
│   ├── debiasing_results.csv
│   ├── debiasing_comparison.png           # updated after every model, never overwritten
│   ├── _debiasing_sample.csv             # saved stratified sample shared across all debiasing runs
│   ├── experiment_log.csv
│   ├── boxplot_dominant_race_income_bucket.png
│   ├── heatmap_race_city_scores.png
│   └── city_mean_scores.png
├── src/
│   ├── fetch_census.py
│   ├── fetch_osm.py
│   ├── generate_descriptions.py
│   ├── query_llm.py
│   ├── llm_clients.py
│   ├── config_loader.py
│   ├── counterfactual.py
│   ├── ground_truth.py
│   ├── debiasing.py
│   ├── experiment_tracker.py
│   ├── analysis.py
│   ├── fairness.py
│   ├── pipeline.py
│   ├── test_apis.py
│   └── app.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 0 | test_apis.py | Smoke-tests all configured free-tier models. Supports `--provider` flag to test a single provider |
| 1 | fetch_census.py | Fetches ACS 2022 data for 20 U.S. cities, 5,000 tracts, stratified by race × income × region |
| 2 | fetch_osm.py | Fetches amenity counts per tract from OpenStreetMap via Overpass API with parallel processing and caching |
| 3 | generate_descriptions.py | Converts census and amenity features into standardized natural language neighborhood descriptions |
| 4 | query_llm.py | Queries free-tier LLMs across Groq, Cerebras, and Mistral via `--model`, `--provider`, or `--all-models`. Auto-resumes from checkpoints. All per-model CSVs are merged into `llm_responses_all.csv` via `rebuild_combined_csv()` after every run |
| 5 | counterfactual.py | Builds stratified counterfactual pairs **once** and reuses them across all models (ensuring valid cross-model comparison). Queries LLMs per model, computes score gaps and t-test significance. Results append to `counterfactual_results.csv` — safe to interrupt and resume |
| 6 | ground_truth.py | Compares LLM scores against actual crime data or ACS proxy variables; computes per-model bias residuals and overestimation gaps by race |
| 7 | analysis.py | Regression, ANOVA, per-model race scores, origin comparison, topic modeling, SHAP, sentiment, visualizations |
| 8 | fairness.py | Cohen's d per model, superadditivity (Race × Income interaction), cross-model comparison, disparate impact, demographic parity gap |
| 9 | debiasing.py | Tests 3 debiasing strategies per model: system prompt intervention, demographic blinding, statistical calibration. Resume-safe — skips already-completed model+strategy combinations. Same tract sample used across all models. Plot updated after every model |
| 10 | experiment_tracker.py | Logs per-model fairness metrics for every pipeline run to `experiment_log.csv` |
| 11 | pipeline.py | Orchestrates full pipeline with `--skip` flags for each step |
| 12 | app.py | Streamlit dashboard with tabs for multi-model comparison, counterfactual analysis, ground truth, and debiasing |

## Setup Instructions

```bash
git clone https://github.com/simhadripraveena2-bit/LinguisticRedline.git
cd LinguisticRedline
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### API Keys Required

This project runs entirely on free API tiers. Total cost to replicate the full study: **$0**.

| Service | Purpose | Get Key At | Cost |
|---------|---------|------------|------|
| U.S. Census Bureau | Tract-level demographic data | https://api.census.gov/data/key_signup.html | FREE |
| Groq | Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout, Qwen 3 32B (14,400 req/day) | https://console.groq.com | FREE |
| Cerebras | Llama 3.1 8B (1M tokens/day, 30 RPM) | https://cloud.cerebras.ai | FREE |
| Mistral | Mistral Small 3.1, Mistral 7B (1B tokens/month, 2 RPM) | https://console.mistral.ai | FREE — requires phone verification |

Add your keys to `config.yaml` (never commit this file to GitHub):

```yaml
census_api_key: YOUR_CENSUS_KEY
groq_api_key: YOUR_GROQ_KEY
cerebras_api_key: YOUR_CEREBRAS_KEY
mistral_api_key: YOUR_MISTRAL_KEY
```

## How to Run

### Step 0 — Test all providers first

```bash
python src/test_apis.py

# Test a single provider
python src/test_apis.py --provider groq
python src/test_apis.py --provider cerebras
python src/test_apis.py --provider mistral
```

### Full Pipeline

```bash
# Steps 1-3: Data collection (run once)
python src/fetch_census.py --sample-per-city 500
python src/fetch_osm.py
python src/generate_descriptions.py

# Step 4: Query all models
python src/query_llm.py --all-models

# Rebuild combined file after all models finish
python -c "import sys; sys.path.insert(0,'src'); from query_llm import rebuild_combined_csv; rebuild_combined_csv()"

# Steps 5-9: Analysis
python src/counterfactual.py --all-models --limit 500
python src/ground_truth.py
python src/analysis.py
python src/fairness.py
python src/debiasing.py --all-models --sample-size 200
python src/experiment_tracker.py

# Step 10: Dashboard
streamlit run src/app.py
```

### Run Individual Steps

```bash
# Query by provider (when some providers hit daily/rate limits)
python src/query_llm.py --provider groq
python src/query_llm.py --provider cerebras
python src/query_llm.py --provider mistral

# Query a single model
python src/query_llm.py --model "Llama 3.1 8B"
python src/query_llm.py --model "Qwen 3 32B"

# Resume interrupted run (automatic — just re-run the same command)
python src/query_llm.py --provider groq

# Force fresh start (ignore checkpoints)
python src/query_llm.py --provider groq --fresh

# Counterfactual — pairs built once on first run, reused by all subsequent models
python src/counterfactual.py --model "Llama 3.1 8B" --limit 500   # builds pairs
python src/counterfactual.py --model "Llama 3.3 70B"              # reuses existing pairs
python src/counterfactual.py --model "Llama 4 Scout"
python src/counterfactual.py --model "Qwen 3 32B"
python src/counterfactual.py --model "Llama 3.1 8B (Cerebras)"
python src/counterfactual.py --model "Mistral Small 3.1"
python src/counterfactual.py --model "Mistral 7B"
# Or run all at once:
python src/counterfactual.py --all-models --limit 500
# Regenerate pairs from scratch:
python src/counterfactual.py --all-models --fresh-pairs

# Ground truth (uses ACS proxy if no FBI UCR CSV available)
python src/ground_truth.py
python src/ground_truth.py --ground-truth-csv path/to/fbi_ucr.csv

# Debiasing — resume-safe, same sample reused across all models
python src/debiasing.py --model "Llama 3.1 8B" --sample-size 200
python src/debiasing.py --model "Llama 3.3 70B"   # resumes if interrupted
python src/debiasing.py --model "Qwen 3 32B"
python src/debiasing.py --model "Mistral Small 3.1"
python src/debiasing.py --model "Mistral 7B"
# Or run all at once:
python src/debiasing.py --all-models --sample-size 200
# Force fresh sample and ignore all checkpoints:
python src/debiasing.py --all-models --fresh
```

## Configuration

Full `config.yaml` structure:

```yaml
# API credentials (all free tier)
census_api_key: YOUR_CENSUS_KEY
groq_api_key: YOUR_GROQ_KEY
cerebras_api_key: YOUR_CEREBRAS_KEY
mistral_api_key: YOUR_MISTRAL_KEY

# Census and sampling
census_year: 2022
min_population: 500
sample_per_city: 200
counterfactual_sample_size: 500
debiasing_sample_size: 200
ground_truth_csv: null
request_delay: 0.5

# Models (all free tier)
models:
  groq:
    - id: llama-3.1-8b-instant
      display_name: Llama 3.1 8B
      provider: groq
    - id: llama-3.3-70b-versatile
      display_name: Llama 3.3 70B
      provider: groq
    - id: meta-llama/llama-4-scout-17b-16e-instruct
      display_name: Llama 4 Scout
      provider: groq
    - id: qwen/qwen3-32b
      display_name: Qwen 3 32B
      provider: groq
  cerebras:
    - id: llama3.1-8b
      display_name: Llama 3.1 8B (Cerebras)
      provider: cerebras
  mistral:
    - id: mistral-small-latest
      display_name: Mistral Small 3.1
      provider: mistral
    - id: open-mistral-7b
      display_name: Mistral 7B
      provider: mistral

# OSM controls
amenity_score_threshold:
  community_rich: 3
  financially_underserved: -1
osm_cache_dir: data/osm_cache
osm_max_workers: 5
osm_timeout_per_tract: 10
osm_request_delay: 0.5
```

| Field | Description |
|-------|-------------|
| `census_api_key` | API key for U.S. Census data access |
| `groq_api_key` | Groq inference — Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout, Qwen 3 32B |
| `cerebras_api_key` | Cerebras inference — Llama 3.1 8B (1M tokens/day free) |
| `mistral_api_key` | Mistral inference — Mistral Small 3.1, Mistral 7B (1B tokens/month free) |
| `request_delay` | Seconds between API requests (increase to 30+ for Mistral free tier) |
| `debiasing_sample_size` | Tracts per debiasing experiment (same sample reused across all models) |
| `counterfactual_sample_size` | Tracts for counterfactual analysis (same pairs reused across all models) |

## Dataset

- **Source**: U.S. Census ACS 2022 via Census Bureau API
- **Amenity data**: OpenStreetMap via Overpass API
- **Scale**: 5,000 census tracts across 20 U.S. cities
- **Sampling**: Stratified by race × income × region (Northeast/South/Midwest/West)
- **Models evaluated**:

| Model | Provider | Parameters | Origin |
|-------|----------|-----------|--------|
| Llama 3.1 8B | Groq | 8B | US (Meta) |
| Llama 3.3 70B | Groq | 70B | US (Meta) |
| Llama 4 Scout | Groq | 17B | US (Meta) |
| Qwen 3 32B | Groq | 32B | Chinese (Alibaba) |
| Llama 3.1 8B | Cerebras | 8B | US (Meta) |
| Mistral Small 3.1 | Mistral | 24B | EU (Mistral AI) |
| Mistral 7B | Mistral | 7B | EU (Mistral AI) |

- **Counterfactual pairs**: ~500 stratified matched description pairs with racial label swaps — same pairs used across all models for valid cross-model comparison
- **Debiasing sample**: stratified 200-tract sample saved to `outputs/_debiasing_sample.csv` and reused across all models
- **Data policy**: No raw data is committed to the repository

## Research Questions

- **RQ1**: Do LLMs systematically encode racial redlining patterns in urban crime risk perception, controlling for income, vacancy rate, and amenity density?
- **RQ2**: Does the language used in LLM qualitative responses differ systematically by neighborhood racial composition?
- **RQ3**: Which neighborhood features (race, income, vacancy rate, amenities) most strongly predict LLM-assigned crime risk?
- **RQ4**: How does LLM bias in crime risk perception vary across cities?
- **RQ5**: Does the LLM exhibit a uniform urban penalty — assigning consistently high crime risk to all urban neighborhoods regardless of demographic composition?
- **RQ6**: Does racial bias persist across model families with distinct pretraining data origins (US-centric Meta/Llama, EU-centric Mistral, Chinese-origin Qwen)?
- **RQ7**: Does model size (7B → 70B parameters) amplify or reduce racial bias as measured by standardized Cohen's d?
- **RQ8**: When racial labels are swapped in counterfactual pairs, how large is the score gap, and is it statistically significant per model?
- **RQ9**: Is racial bias superadditive — does it compound when neighborhoods are simultaneously Black and low-income beyond what either factor predicts alone?
- **RQ10**: Can system prompt interventions, demographic blinding, or statistical calibration meaningfully reduce measured bias?

## Outputs

| File | Description |
|------|-------------|
| `outputs/cohens_d_per_model.csv` | Standardized racial bias gap (Cohen's d) per model — core cross-model comparison metric |
| `outputs/cohens_d_per_model.png` | Bar chart of Cohen's d per model colored by training data origin |
| `outputs/superadditivity_results.csv` | Race × Income interaction regression per model — tests if bias compounds |
| `outputs/cross_model_comparison.csv` | Full cross-model comparison with parameter count and origin metadata |
| `outputs/size_vs_bias.png` | Scatter: model size (parameters) vs Cohen's d — does size predict bias? |
| `outputs/per_model_race_scores.csv` | Mean crime risk score by model × dominant race |
| `outputs/origin_bias_comparison.csv` | Mean bias gap by training data origin (US/EU/Chinese) |
| `outputs/heatmap_model_race_scores.png` | Heatmap: models × race groups mean scores |
| `outputs/model_bias_black_vs_white.png` | Grouped bar chart: Black vs White scores per model |
| `outputs/fairness_report.csv` | Demographic parity gap by dominant race |
| `outputs/disparate_impact_by_vacancy.csv` | Disparate impact ratios stratified by vacancy band |
| `outputs/fairness_by_city.csv` | City-level fairness breakdown by dominant race |
| `outputs/anova_results.csv` | One-way ANOVA results for each demographic factor |
| `outputs/regression_coefficients.csv` | Linear regression coefficients for all features |
| `outputs/city_breakdown.csv` | City-level mean score and tract count summary |
| `outputs/city_race_breakdown.csv` | Mean score by city and dominant race |
| `outputs/counterfactual_results.csv` | Score gaps for counterfactual pairs — all models combined, appended per run |
| `outputs/counterfactual_stats_per_model.csv` | Per-model t-test results and % Black scored higher |
| `outputs/counterfactual_gap_per_model.png` | Bar chart of counterfactual gaps per model |
| `outputs/ground_truth_comparison.csv` | Full merged data with bias residuals (representative model) |
| `outputs/ground_truth_per_model.csv` | Overestimation gap per model vs actual crime rates |
| `outputs/ground_truth_bias_residuals.png` | Per-model overestimation bar chart |
| `outputs/intersectional_heatmap.png` | Mean LLM score heatmap across race × income bucket |
| `outputs/intersectional_anova.csv` | Two-way ANOVA results for race × income interaction |
| `outputs/topic_model_results.csv` | LDA topic distribution per racial group |
| `outputs/sentiment_by_race.csv` | Sentiment scores for qualitative LLM responses by race |
| `outputs/shap_top_features.csv` | SHAP-style feature importances for LLM score prediction |
| `outputs/debiasing_results.csv` | Before/after fairness metrics for 3 strategies per model — appended per run |
| `outputs/debiasing_comparison.png` | Per-model debiasing chart — rebuilt from full results after every model |
| `outputs/_debiasing_sample.csv` | Saved stratified tract sample shared across all debiasing model runs |
| `outputs/experiment_log.csv` | Per-model fairness log across all pipeline runs |
| `data/llm_responses_{model}.csv` | Per-model raw LLM responses and scores |
| `data/llm_responses_all.csv` | Combined multi-model responses — rebuilt from all per-model files after every run |
| `data/counterfactual_pairs.csv` | Stratified counterfactual pairs shared across all models |

## Requirements

### Key Dependencies

| Package | Purpose |
|---------|---------|
| groq | Llama and Qwen inference via Groq |
| openai | Cerebras and Mistral inference (OpenAI-compatible SDK with custom base_url) |
| statsmodels | OLS regression for superadditivity testing and ANOVA |
| scipy | Cohen's d t-tests, Pearson correlation |
| scikit-learn | Ridge regression, TF-IDF, LDA topic modeling |
| vaderSentiment | Sentiment analysis of LLM qualitative responses |
| streamlit | Interactive results dashboard |
| matplotlib / seaborn | All visualizations |
| tqdm | Progress bars for long-running pipeline steps |

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Ethical Statement

This research is designed to study and expose potential bias in LLM systems, not to endorse or reinforce biased inferences. The analysis is conducted at the aggregated neighborhood (census tract) level, and no real individuals are identified or profiled. Findings are intended to support improved fairness, transparency, and accountability in AI systems deployed in socially sensitive domains such as urban planning, policing, and housing.

## Citation

```bibtex
Citation will be added upon acceptance.
```

## Author

### Simhadri Praveena
#### SDE / Dual Degree IIT KGP
Research Interests:
* Computer Vision
* Adversarial Machine Learning
* Explainable AI
