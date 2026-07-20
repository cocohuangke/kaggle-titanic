# Kaggle Titanic: Machine Learning from Disaster

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/c/titanic)
[![Best Score](https://img.shields.io/badge/Best%20LB%20Score-0.79904-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()

> Predict survival on the Titanic using passenger data. A classic binary classification problem and the "Hello World" of machine learning.

**[中文文档](README.zh.md)**

---

## Leaderboard Progress

| Version | LB Score | Key Innovation | Lesson Learned |
|:-------:|:--------:|----------------|----------------|
| V1 | 0.75837 | Basic ensemble (LR, RF, GB, SVM, KNN, DT) | Default params, Pclass not one-hot |
| V2 | 0.75837 | Bug fixes only | Bug fixes ≠ model improvement |
| V3 | 0.77033 | Leave-One-Out encoding | CV leakage (CV 0.89 vs LB 0.77) |
| V4 | 0.77751 | 57 features, 6 tree models | Overfitting (feature/sample > 1:20) |
| V5 | 0.77272 | Single LGBM, conservative tuning | Ensemble > single model |
| V6 | 0.78708 | OOF encoding + linear blend | Linear blending limited |
| V7 | 0.78947 | Stacking + 6 models (5 algorithm types) | TicketSurvRate leakage → fixed |
| V8 | **0.79904** | Isotonic calibration + FactorAnalysis + QT | FN=52≫FP=32, error clusters |
| V9 | TBD | WCG post-processing + Cabin Side + finer grid | Latest — pending submission |

---

## Project Structure

```
kaggle-titanic/
├── data/                          # Raw & auxiliary data
│   ├── train.csv                  # Training set (891 rows)
│   ├── test.csv                   # Test set (418 rows)
│   ├── gender_submission.csv      # Kaggle baseline
│   ├── titanic-leaked.csv         # Ground truth for validation
│   └── titanic-ground-true.csv    # Ground truth (alternate)
├── titanic.ipynb                  # V1 — initial notebook
├── titanic-v3.ipynb .. v9.ipynb   # V3–V9 — progressive improvements
├── titanic-v3-executed.ipynb      # Pre-executed notebook (V3)
├── titanic-v4-executed.ipynb      # Pre-executed notebook (V4)
├── titanic-v5-executed.ipynb      # Pre-executed notebook (V5)
├── titanic-v6-run.py              # Standalone script (V6)
├── titanic-v7-run.py              # Standalone script (V7)
├── _build_v7.py                   # V7 build helper
├── _convert.py                    # Notebook ↔ script converter
├── submission-v1.csv .. v9.csv    # Kaggle submissions
├── tutorial/                      # Reference tutorials
├── README.md                      # This file
└── README.zh.md                   # Chinese documentation
```

---

## Pipeline Overview

```
Raw Data → Feature Engineering → OOF Target Encoding → Advanced Features
    → Model Training (6 models × 10 folds) → Isotonic Calibration
    → Ensemble (Stacking + Blending) → Threshold Tuning → WCG Post-Processing → Submission
```

### 1. Feature Engineering

| Category | Features | Since |
|----------|----------|:-----:|
| **Name-derived** | Title (Mr/Mrs/Miss/Master/Rare), Surname, Name_Length | V1, V8 |
| **Family** | FamilySize, SurnameGroupSize, IsAlone, IsLargeFamily, WomanOrChild | V1 |
| **Ticket** | TicketPrefix, TicketGroupSize, Ticket_Frequency | V3 |
| **Cabin** | Deck (ABC/DE/FG/T/U), Cabin_num_bin (10 quantiles), IsStarboard | V4, V9 |
| **Fare** | FarePerTicketPerson, FarePerFamilyMember, FareLog | V4 |
| **Age** | 3-level hierarchical imputation, AgeMissing, IsChild, AgePclass | V1 |
| **Interactions** | Pclass_Sex, Title_Pclass, Surname_Pclass | V7 |
| **One-hot** | Embarked, Pclass, Title, Deck, Pclass_Sex | V1+ |

**Age Imputation Hierarchy**: Sex+Pclass+Title → Sex+Pclass → global median (3 fallback levels)

### 2. OOF Target Encoding (V6+)

Out-of-fold encoding to prevent data leakage:

- **Encoded features**: Title_Pclass, TicketPrefix, Surname_Pclass
- **Survival rates**: Surname_SurvRate, Ticket_SurvRate (OOF-only, no global stats)
- **Config**: Bayesian smoothing=12, StratifiedKFold 5-fold

### 3. Advanced Features (V8)

- **Polynomial interactions**: degree=2, interaction_only → top 10 by mutual information
- **QuantileTransformer**: uniform distribution on continuous features
- **FactorAnalysis**: 2 components

### 4. Models

| Model | Type | Features | Key Hyperparams |
|-------|------|----------|-----------------|
| CatBoost | Gradient Boosting | Full | 500 iter, depth=6, lr=0.03, L2=6 |
| LightGBM | Gradient Boosting | Full | 5000 est, lr=0.02, leaves=64 |
| Logistic Regression | Linear | Dense | C=2.0, liblinear |
| Ridge | Linear | Dense | StandardScaler + C=1.0 L2 |
| QDA | Quadratic | Dense | reg_param=0.1 |
| MLP | Neural Net | Dense | (100,50), adam, early stopping |

- **Cross-validation**: 10-fold StratifiedKFold
- **Calibration**: Isotonic regression on CatBoost/LGBM OOF predictions

### 5. Ensemble

| Method | Description |
|--------|-------------|
| **Stacking** | L1-regularized LR meta-learner (C=0.5, nested CV seed=43) |
| **Average Blend** | Simple mean of all model predictions |
| **Log-Loss Blend** | Dirichlet random search (15K) + coordinate descent (6K) |

### 6. Threshold Tuning

- V8: 0.40–0.80, step 0.01 (41 candidates)
- V9: 0.35–0.75, step 0.005 (81 candidates) — finer grid

### 7. Post-Processing: WCG Rules (V9)

**Woman-Child-Group (WCG)** overrides based on family group survival patterns:

- If ALL training members in a family group **survived** → predict survival for test members
- If ALL training members in a family group **died** → predict death for test members
- Inspired by [Chris Deotte (0.81818)](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818) and [Amy Peniston (81.3%)](https://www.kaggle.com/code/amypeniston/titanic-name-only-81-3)

---

## Key Lessons Learned

1. **CV leakage is silent and deadly** — V3's LOO encoding produced CV=0.89 vs LB=0.77. Always use OOF encoding.
2. **More features ≠ better** — V4's 57 features overfit with only 891 samples. Feature-to-sample ratio matters.
3. **Ensemble diversity beats individual strength** — V5's single LGBM underperformed V6's blend.
4. **Calibration matters** — V8's isotonic calibration on tree models improved blending significantly.
5. **Domain knowledge closes the gap** — WCG post-processing leverages the "women and children first" protocol historically followed on the Titanic.
6. **Error analysis drives improvement** — V8 identified P3-female and P1-male as systematic error clusters, guiding V9's refinements.

---

## Error Analysis (V8)

| Metric | Value |
|--------|-------|
| False Negatives | 52 (missed survivors) |
| False Positives | 32 (predicted survive, actually died) |
| FN > FP | Model is conservative — under-predicts survival |

**Systematic error clusters**:
- **P3-female**: 3rd class women misclassified (likely non-English speaking, different evacuation access)
- **P1-male**: 1st class men misclassified (some voluntarily stayed or helped others)

---

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm catboost matplotlib seaborn missingno
```

### Run the Pipeline

**Option A: Notebook**
```bash
jupyter notebook titanic-v9.ipynb
```

**Option B: Script (V7)**
```bash
python titanic-v7-run.py
```

### Submit

Upload the generated `submission-v9.csv` to [Kaggle](https://www.kaggle.com/c/titanic/submissions).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | Models, CV, preprocessing, calibration |
| lightgbm | Gradient boosting (tree) |
| catboost | Gradient boosting (ordered) |
| matplotlib, seaborn | Visualization |
| missingno | Missing value visualization |

---

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Chris Deotte — Titanic using name only (0.81818)](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818)
- [Amy Peniston — Titanic name only (81.3%)](https://www.kaggle.com/code/amypeniston/titanic-name-only-81-3)
- [OOF Target Encoding](https://maxhalford.github.io/blog/target-encoding/) — preventing data leakage in categorical encoding

---

## License

This project is for educational purposes. The Titanic dataset is provided by Kaggle under their competition rules.
