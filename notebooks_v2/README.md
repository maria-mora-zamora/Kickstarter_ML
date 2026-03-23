# Kickstarter ML Unified Notebook Pipeline (notebooks_v2)

## Overview
This folder contains 6 complete, production-quality Jupyter notebooks implementing a unified ML pipeline for predicting Kickstarter campaign success. All notebooks are valid Jupyter format (.ipynb files) with complete code, markdown documentation, and clear execution flow.

## Notebook Summary

### 01_data_cleaning.ipynb
**Purpose:** Load raw data, deduplicate, filter to binary outcomes, remove leakage and junk columns, apply post-drop temporal filter, and split data.

**Key steps:**
- Load 85 CSV shards from Webrobots export
- Deduplicate on campaign ID
- Filter to successful/failed campaigns only
- Parse JSON category and location columns
- Drop 10 leakage columns (spotlight, pledged, etc.)
- Drop 19 junk columns (URLs, metadata, etc.)
- Apply post-drop cut: filter to 2015-01-01 onwards (modern Kickstarter platform)
- Temporal 64/16/20 train/val/test split

**Outputs:** clean_df.parquet, train_df.parquet, val_df.parquet, test_df.parquet, leakage_columns.json

---

### 02_eda.ipynb
**Purpose:** Exploratory data analysis of the cleaned dataset. Understand distributions, key predictors, and inform feature engineering.

**Key sections:**
- Data overview and class balance
- Cleaning audit
- Target distribution by goal, category, staff pick, video
- Geographic analysis
- Text feature summaries
- Duration and timing patterns
- Missing data strategy
- Temporal trends (2015-2026)
- Correlation heatmap
- Split audit

**Outputs:** Visualizations (02_correlation_heatmap.png, etc.)

---

### 03b_tfidf_features.ipynb
**IMPORTANT: Run BEFORE 03a_feature_engineering.ipynb**

**Purpose:** Identify predictive words in campaign blurb and title using TF-IDF.

**Key steps:**
- TF-IDF vectorization on blurb (min_df=50, max_df=0.8, bigrams)
- TF-IDF vectorization on campaign title
- Compute success rates and statistical significance (z-test, p<0.01)
- Category-adjusted diff: term success rate minus category average
- Filter: |diff|>=10pp, n>=200, p<0.01, exclude years, exclude non-English tokens
- Export shortlist of ~150-160 predictive terms

**Outputs:** tfidf_word_list.json (used by 03a)

---

### 03a_feature_engineering.ipynb
**Purpose:** Transform raw data into ~193 ML-ready features.

**Key features engineered:**
- Numeric: log_goal, duration_days, prep_days, has_video, text lengths, etc.
- Goal comparison: log_goal_vs_cat_median
- Target encoding: cat_name_encoded, cat_parent_encoded (lambda=10, fit on train only)
- Geography: 20 country OHE features + Other
- Text: Binary features for each TF-IDF shortlist term (~150 features)

**Key principle:** All encoders fit on train_df only, applied identically to val_df and test_df (no leakage)

**Outputs:** X_train.parquet, X_val.parquet, X_test.parquet, y_train.parquet, y_val.parquet, y_test.parquet, feature_cols.json, feature_summary.csv

---

### 04_modelling.ipynb
**Purpose:** Train 11 candidate models, evaluate on test set, compare performance.

**Models evaluated:**
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. **XGBoost** (primary, best performer)
6. Linear SVC (calibrated)
7. Naive Bayes
8. K-Nearest Neighbors (subsampled)
9. Neural Network (MLP)
10. Voting Ensemble (LR + RF + XGB)
11. Stacking Ensemble (LR + DT + RF + GB)

**Configuration:**
- QUICK_TEST mode for fast prototyping (set to False for full run)
- TimeSeriesSplit CV (respects temporal ordering, prevents leakage)
- RandomizedSearchCV for hyperparameter tuning
- Evaluation metrics: ROC-AUC, F1, Precision, Recall, PR-AUC

**Outputs:** 
- all_model_results.csv (rankings)
- ROC curves, feature importance, SHAP summary, confusion matrix (in figures/)

---

### 05_reflection.ipynb
**Purpose:** Critical reflection on model limitations, ethical implications, and business recommendations.

**Key sections:**
1. **Problem Framing** — Business context and ML formulation
2. **Data Summary** — Post-drop rationale, temporal split
3. **Construct Gap** — Measurement problem (execution quality ≠ creative merit)
4. **Leakage Analysis** — Decisions made for data integrity
5. **Distributional Shift** — Temporal drift in success rates
6. **Goodhart's Law** — How deployment changes behavior
7. **Fairness Concerns** — Geographic bias in features
8. **Model Performance** — Rankings and interpretation
9. **Limitations** — Unobservable creative quality, missing creator reputation, external shocks
10. **Business Recommendations** — 5 evidence-backed actions for creators

---

## Execution Order

**Run in this sequence:**

```
01_data_cleaning.ipynb
    ↓
02_eda.ipynb
    ↓
03b_tfidf_features.ipynb ← MUST RUN BEFORE 03a
    ↓
03a_feature_engineering.ipynb ← Loads tfidf_word_list.json from 03b
    ↓
04_modelling.ipynb
    ↓
05_reflection.ipynb
```

---

## Key Design Decisions

### Post-Drop Temporal Filter
Early Kickstarter (2009-2014) had 71.7% success rate vs 60.8% post-2014. We filter to 2015-01-01 onwards to model the modern platform.

### Temporal Split (64/16/20)
Instead of random split: Train on oldest data, validate on middle data, test on newest data. This prevents temporal leakage and mimics real deployment.

### Data Leakage Prevention
- Removed 10 post-campaign columns (pledged, backers_count, spotlight, etc.)
- Spotlight (r=1.0 with success) is especially dangerous
- Removed 19 junk columns (URLs, IDs, metadata)
- Target encoders fit on train only

### Staff Pick Exclusion (in modeling)
Staff pick is included in EDA for context but excluded from final model because it encodes Kickstarter editorial judgment not available to creators pre-launch.

### Feature Engineering Scale
~193 total features:
- 12 numeric (goal, duration, prep time, text metrics, temporal)
- 1 target-encoded category (cat_name_encoded)
- 1 target-encoded parent category
- 21 country OHE features
- ~158 TF-IDF binary text features

---

## Output Files

**In ../outputs/:**
- clean_df.parquet (188,429 rows post-drop)
- train_df.parquet (120,594 rows)
- val_df.parquet (30,149 rows)
- test_df.parquet (37,686 rows)
- X_train.parquet, X_val.parquet, X_test.parquet (feature matrices)
- y_train.parquet, y_val.parquet, y_test.parquet (targets)
- feature_cols.json (list of 193 feature names)
- feature_summary.csv (feature metadata)
- tfidf_word_list.json (shortlisted predictive terms)
- leakage_columns.json (reference of dropped columns)

**In ../outputs/figures/:**
- 02_correlation_heatmap.png
- roc_curves.png
- feature_importance.png
- shap_summary.png
- confusion_matrix_xgb.png

**In ../outputs/results/:**
- all_model_results.csv (model rankings and metrics)

---

## Critical Notes

1. **Notebook 03b MUST run before 03a** — 03b produces tfidf_word_list.json required by 03a

2. **QUICK_TEST mode** — In 04_modelling.ipynb, set QUICK_TEST=False for full production run (it defaults to False). True uses only 5k rows for fast testing.

3. **Paths** — All notebooks use relative paths (../data, ../outputs). Ensure working directory is the notebooks_v2 folder or adjust paths as needed.

4. **Dependencies** — Install: pandas, sklearn, xgboost, shap, scipy, statsmodels, matplotlib, seaborn

5. **Data Not Included** — Raw CSV shards must be downloaded from Webrobots and placed in ../data/Kickstarter_2026-02-12T03_20_22_018Z/

6. **Temporal Drift** — Model should be retrained quarterly on rolling 12-month windows in production.

---

## Summary of Findings

**Best model:** XGBoost (AUC 0.8954 on test set)
**Baseline (LR):** AUC 0.8285
**Improvement:** +6.69pp

**Top 5 Feature Drivers:**
1. Goal amount (log scale)
2. Category success rate (target encoded)
3. Campaign duration (days)
4. Video presence (binary)
5. Goal relative to category median

**Key Insights:**
- Lower goals predict success (2-3x more successful than failed)
- 28-35 day campaigns outperform longer ones
- Videos add +15-25pp to success rate
- Subcategory matters enormously (30-80% success rate range)
- Geographic bias exists (US-centric platform)
- Model predicts execution quality, not creative merit

---

Created: March 2026
Version: 1.0
Status: Complete and production-ready
