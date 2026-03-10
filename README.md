# Kickstarter Campaign Success Prediction
### ESADE AI II — Final Project | March 2026

---

## Business Problem

Can we predict whether a Kickstarter crowdfunding campaign will successfully reach its funding goal — using **only information available before the campaign launches**?

**For creators**: know if your campaign is set up for success before you go live.  
**For Kickstarter**: allocate editorial attention (staff picks, homepage features) more efficiently.

This is a **binary classification problem**: `success = 1` (funded), `success = 0` (failed).

---

## Dataset

| Property | Value |
|---|---|
| Source | Kickstarter bulk export (Webrobots) |
| Files | 85 CSV shards, identical 42-column schema |
| Raw rows | ~276,000 (with duplicates) |
| Unique campaigns | ~209,000 after deduplication |
| Binary subset | ~188,000 (successful + failed only) |
| Time range | 2012–2026 |
| Class balance | ~37% success, ~63% failure |
| Split method | **Temporal** (chronological 80/20) — NOT random |

---

## Folder Structure

```
Kickstarter_ML_Project/
├── data/                      # Empty — raw data stays in Test Data/
├── notebooks/
│   ├── 01_data_loading_and_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelling.ipynb
│   └── 05_results_and_reflection.ipynb
├── src/
│   ├── data_loader.py         # Load + deduplicate all 85 CSV shards
│   ├── preprocessing.py       # Sklearn Pipeline + ColumnTransformer
│   ├── features.py            # All feature engineering functions
│   └── evaluation.py         # Unified model evaluation + plotting
├── outputs/
│   ├── figures/               # All PNG plots (150 dpi)
│   ├── models/                # Saved .pkl model files
│   └── results/               # CSVs with model scores
└── README.md
```

---

## How to Run

Run notebooks **in order** — each notebook saves outputs consumed by the next.

```bash
cd "Artificial Intelligence 2/Final Project/Kickstarter_ML_Project"

# 1. Load, clean, and split the data
jupyter nbconvert --to notebook --execute notebooks/01_data_loading_and_cleaning.ipynb

# 2. Exploratory data analysis
jupyter nbconvert --to notebook --execute notebooks/02_EDA.ipynb

# 3. Feature engineering
jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb

# 4. Train and evaluate all models
jupyter nbconvert --to notebook --execute notebooks/04_modelling.ipynb

# 5. Results synthesis and critical reflection
jupyter nbconvert --to notebook --execute notebooks/05_results_and_reflection.ipynb
```

Or open each notebook in Jupyter and run all cells manually.

---

## Models Trained

| # | Model | Type |
|---|---|---|
| 1 | Logistic Regression | Linear baseline |
| 2 | Decision Tree | Interpretable tree |
| 3 | Random Forest | Ensemble (bagging) — Core syllabus |
| 4 | Gradient Boosting | Ensemble (boosting) |
| 5 | XGBoost | Ensemble (boosting) |
| 6 | K-Nearest Neighbours | Instance-based |
| 7 | Naive Bayes | Probabilistic baseline |
| 8 | Linear SVC | Margin-based |
| 9 | Voting Ensemble | Soft voting (LR + RF + XGB) |
| 10 | Stacking Ensemble | Meta-learner (LR over base learners) |

---

## Key Results

*(Run notebooks to populate)*

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD |

---

## Critical Limitations

1. **Construct gap**: funding success ≠ quality of creative idea. A great idea can fail with a badly-set goal; a mediocre campaign with professional marketing can succeed.
2. **Selection bias**: only launched campaigns are in the dataset. We cannot model why creators chose not to launch at all.
3. **Data leakage risk**: post-campaign columns (`pledged`, `backers_count`, `spotlight`) are excluded unconditionally.
4. **Temporal drift**: consumer tastes and platform dynamics have shifted since 2012. The model should be retrained periodically.
5. **Fairness**: country and currency are predictive but structurally biased against non-US creators.
6. **Goodhart's Law**: if this model were made public, creators would optimise their campaigns to fool it, degrading its predictive power.

---

*Prof. Dr. Maria De-Arteaga | ESADE Ramon Llull University | AI II 2025–2026*
