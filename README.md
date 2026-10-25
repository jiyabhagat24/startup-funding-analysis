# Global Startup Funding Analysis & Prediction

Exploratory analysis, a Random Forest model and an interactive Streamlit dashboard built on a dataset of global
startup funding rounds.

**Tech:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit

## The data

Four CSV files in `data/raw/`:

| File | Rows | Contents |
|---|---|---|
| `funding_rounds.csv` | 2,818 | One row per funding round: date, stage, amount, valuations, lead investor, sector, country |
| `startups.csv` | 1,747 | One row per startup: founding year, employees, female founder flag, revenue status |
| `investors.csv` | 73 | One row per lead investor |
| `quarterly_summary.csv` | 25 | Quarterly totals (verified against the rounds table) |

- 2,818 rounds from 1,747 startups, 35 countries, 20 sectors, 73 lead investors.
- Rounds are dated 2020-01-02 to 2026-03-29. **2026 contains Q1 only**, so it is shown as a partial year.
- Total funding in the data: US$33.5 billion.

### Data quality (all checks in `reports/data_quality.json`)

Clean: no missing values, no duplicate rows, unique round IDs, startup attributes match across tables,
`post_money = pre_money + amount` in 100% of rows, and the quarterly summary matches the rounds table.

Problems found (rows are kept, nothing is silently dropped):

- **704 rounds are dated before the startup's `founded_year`.** They are flagged (`funded_before_founding`) and the
  startup age used in the model is clipped at 0.
- **468 startups have funding stages out of chronological order** (for example a Series A dated before its Seed).
- `equity_dilution_pct` equals `amount / post_money` in only about 4% of rows, so it is treated as a descriptive field only.
- The investor type on a round matches the investor's type in `investors.csv` for only about 20% of rounds, so the two
  are treated as separate fields.
- `employee_count` is one value per startup (not the headcount at the time of each round).

## Project structure

```
app.py                  Streamlit dashboard
requirements.txt
src/
  config.py             paths, colours, chart style
  data_prep.py          load, clean, merge, quality checks
  eda.py                the 12 analyses / charts
  model.py              Random Forest training + evaluation
  compare_models.py     Random Forest vs 4 other models
  compare_models.py     Random Forest vs Logistic Regression, Decision Tree, Gradient Boosting
data/raw/               the four source CSVs
data/processed/         funding_master.csv (one row per round, merged)
figures/                01-12 EDA charts + 3 model charts + model_comparison.png
models/                 random_forest.joblib, metrics.json
reports/                data_quality.json, eda_summary.json
```

## How to run

```bash
pip install -r requirements.txt
python src/data_prep.py      # builds data/processed/funding_master.csv
python src/eda.py            # saves the 12 charts to figures/
python src/model.py          # trains the model, saves models/ and 3 model charts
python src/compare_models.py # optional: compares 5 models on the same split
python src/compare_models.py # optional: compares 4 other models
streamlit run app.py         # opens the dashboard
```

Run the commands from the project root. The dashboard builds any missing file on first launch, and retrains the model
automatically if the saved one was made with a different scikit-learn version.

## 12 exploratory analyses (`figures/01`-`12`)

| # | Analysis | What it shows |
|---|---|---|
| 01 | Round size distribution | Heavily right-skewed: median US$1.83M, mean US$11.90M, 90th percentile US$31.5M, max US$359.5M |
| 02 | Funding by year | 2020 is the largest full year (US$6,501M, 503 rounds); 2022 the smallest (US$4,603M); 2025 US$6,307M |
| 03 | Quarterly trend | Noisy, no clear long-run trend; peak 2020 Q3 (US$2,415M), low 2022 Q3 (US$819M) |
| 04 | Round size by stage | Median rises at every stage: US$0.49M (Pre-Seed) to US$142.46M (Series D) |
| 05 | Top 15 countries | United States 28.9% of funding, then India 11.5%, UK 9.7%, China 8.7%, Germany 4.7%; top 5 = 63.5% |
| 06 | Share by region | North America 31.8%, Asia 29.9%, Europe 23.2%, all other regions 15.1% combined |
| 07 | Funding by sector | AI is first (US$5,670M, 462 rounds), then Fintech (US$4,564M) and Healthcare & Biotech (US$4,287M) |
| 08 | Country x sector heatmap | Largest cell in the top-10 x top-10 grid: United States Fintech, US$1,620M |
| 09 | Valuation by stage | Median post-money rises from US$3.6M (Pre-Seed) to US$816.8M (Series D) |
| 10 | Investor type | Venture Capital led 1,177 of 2,818 rounds; median round size is similar across types (US$1.64M - US$2.04M) |
| 11 | Correlation heatmap | Round size correlates 0.98 with post-money valuation; startup traits (employees, age, investors, female founder) are all within +/-0.02 of zero |
| 12 | Startup traits | Median round: US$1.83M pre-revenue, US$1.87M revenue-generating, US$1.65M profitable; US$1.80M with a female founder vs US$1.83M without |

## Model: predicting funding potential

**Question.** Will a round be a *high-funding* round, meaning larger than the median round in the training data
(US$1.795M)?

**Features** (known before a round closes): funding stage, sector, country, region, revenue status, female founder,
employee count, startup age at the round, year. Valuations, dilution, number of investors and lead investor are
deliberately excluded because they are outcomes of the deal and would leak the answer.

**Method.** Random Forest classifier in a scikit-learn pipeline (one-hot encoding), tuned with a grid search over
`n_estimators`, `max_depth`, `min_samples_leaf` using 5-fold `GroupKFold`. Train/test split is 80/20 **by startup**,
so rounds of the same startup are never in both sets (2,234 train rounds / 584 test rounds; 1,397 / 350 startups).

**Results on the held-out test set**

| Metric | Value |
|---|---|
| ROC AUC | 0.920 (cross-validated on training data: 0.934) |
| Accuracy | 80.7% |
| Precision / Recall / F1 | 77.4% / 88.4% / 82.5% |
| Baseline: always guess the majority class | 51.7% accuracy |
| Baseline: funding stage only | 0.917 AUC, 79.8% accuracy |

**How to read this honestly.** The model beats the naive baseline by a wide margin, but almost all of its skill comes
from funding stage alone: permutation importance shows stage costs 0.42 AUC when shuffled, while every other feature
costs 0.005 or less. When the target is changed to "large *for its own stage*", the same model scores AUC 0.52, which
is close to chance. In this dataset, sector, country, revenue status and founder traits carry very little extra signal
about round size. Predictions are a statistical pattern in this dataset, not financial advice.

### Model comparison

Four other classifiers were tested on the same data, the same startup-grouped split and the same target, each with a
small grid search (`python src/compare_models.py`, results in `reports/model_comparison.json`).

| Model | CV ROC AUC | Test ROC AUC | Test accuracy | Test F1 |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.916 | 79.3% | 80.5% |
| Logistic Regression (funding stage only) | 0.923 | 0.917 | 79.8% | 83.5% |
| Decision Tree | 0.931 | 0.916 | 81.2% | 83.0% |
| **Random Forest (used)** | **0.934** | **0.920** | 80.6% | 82.5% |
| Gradient Boosting | 0.934 | 0.911 | 79.4% | 80.5% |

All models land within about 0.01 AUC of each other, and a logistic regression that sees only funding stage scores
the same as the full models. Random Forest has the highest test AUC, but the gaps are small enough (584 test rounds)
that this should not be read as Random Forest being clearly better. It was kept because it is robust and works
directly with permutation importance.

### Model comparison

`src/compare_models.py` trains other models on the same split, features and target, each with a small grid search
(5-fold `GroupKFold` on the training data only).

| Model | CV ROC AUC | Test ROC AUC | Test accuracy | Test F1 |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.916 | 79.3% | 0.805 |
| Logistic Regression (funding stage only) | 0.923 | 0.917 | 79.8% | 0.835 |
| Decision Tree | 0.931 | 0.916 | 81.2% | 0.830 |
| **Random Forest** | 0.934 | **0.920** | 80.6% | 0.825 |
| Gradient Boosting | 0.934 | 0.911 | 79.4% | 0.805 |

Random Forest has the highest test AUC, but all five models land within 0.01 of each other, which is too small a gap
to call a real winner on 584 test rounds. A logistic regression using only funding stage scores almost the same,
which confirms that stage carries the signal. Random Forest was kept for robustness and for its permutation
importance; the choice does not change the conclusions. Chart: `figures/model_comparison.png`.

Model charts: `figures/model_roc_curve.png`, `model_confusion_matrix.png`, `model_feature_importance.png`.

## Dashboard

`streamlit run app.py` opens an interactive dashboard with sidebar filters (year, region, country, industry, funding
stage, top-N) and seven tabs:

- **Trends**: quarterly and yearly funding, sector-by-year heatmap
- **By country**: rank countries by total funding, number of rounds or median round size; regional share; country table
- **By industry**: same views for sectors
- **Country x industry**: heatmap of countries against industries
- **Stages & investors**: round size by stage, investor types, top lead investors
- **Predict**: model scores plus a form that returns the probability of a high-funding round
- **Data**: filtered table with CSV download

## Limitations

- 2,818 rounds is a modest sample, and some fields do not reconcile (see Data quality above).
- 2026 is a partial year (Q1 only); compare it with other years per quarter, not in total.
- The model predicts round size relative to the dataset median, not the dollar amount, and not startup success.
