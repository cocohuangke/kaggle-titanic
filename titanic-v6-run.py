# Auto-generated from titanic-v6.ipynb
import sys, os
os.chdir(r"\\DS1019\home\Drive\project\titanic")

# ====== CELL 3 ======
# [V6-NEW] Imports — all required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# [V6-NEW] Global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print(f"Libraries loaded. Random state: {RANDOM_STATE}")

# ====== CELL 4 ======
# [V6-NEW] Load data and store test PassengerIds BEFORE any preprocessing
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# [V6-NEW] CRITICAL: Store test PassengerIds now, before any modifications
test_passenger_ids = test['PassengerId'].copy()

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# [V6-NEW] Add Source column and concatenate for unified feature engineering
train['Source'] = 'train'
test['Source'] = 'test'
full_df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"Full dataset shape: {full_df.shape}")
print(f"Train Survived distribution:\n{full_df.loc[full_df['Source']=='train', 'Survived'].value_counts()}")

# ====== CELL 5 ======
# [V6-NEW] Feature Engineering — Part 1: Title, Surname, Family

# --- Title extraction from Name ---
full_df['Title'] = full_df['Name'].str.extract(r'([A-Za-z]+)\.')

# [V6-NEW] Group rare titles into 'Rare', standardize Miss/Mrs variants
title_replacements = {
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Capt': 'Rare',
    'Sir': 'Rare', 'Don': 'Rare', 'Jonkheer': 'Rare',
    'Countess': 'Rare', 'Lady': 'Rare',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
}
full_df['Title'] = full_df['Title'].replace(title_replacements)

# Verify only expected titles remain
expected_titles = {'Mr', 'Mrs', 'Miss', 'Master', 'Rare'}
actual_titles = set(full_df['Title'].unique())
unexpected = actual_titles - expected_titles
if unexpected:
    print(f"WARNING: Unexpected titles found: {unexpected}")
print(f"Title distribution:\n{full_df['Title'].value_counts()}\n")

# --- Surname for family grouping ---
full_df['Surname'] = full_df['Name'].str.split(',').str[0].str.strip()

# --- Family features ---
full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1
full_df['SurnameGroupSize'] = full_df.groupby('Surname')['PassengerId'].transform('count')
print(f"SurnameGroupSize stats:\n{full_df['SurnameGroupSize'].describe()}\n")

# ====== CELL 6 ======
# [V6-NEW] Feature Engineering — Part 2: Ticket, Deck, Fare

# --- Ticket features ---
# Extract letter prefix from ticket number
full_df['TicketPrefix'] = full_df['Ticket'].str.replace(r'\d', '', regex=True)
full_df['TicketPrefix'] = full_df['TicketPrefix'].str.replace(r'[\.\/\s]', '', regex=True).str.strip()
full_df.loc[full_df['TicketPrefix'] == '', 'TicketPrefix'] = 'NUM'

# [V6-NEW] Group rare prefixes (< 10 occurrences across full dataset)
prefix_counts = full_df['TicketPrefix'].value_counts()
rare_prefixes = prefix_counts[prefix_counts < 10].index
full_df.loc[full_df['TicketPrefix'].isin(rare_prefixes), 'TicketPrefix'] = 'RARE_PREFIX'
print(f"TicketPrefix unique values: {full_df['TicketPrefix'].nunique()}")
print(f"Top prefixes:\n{full_df['TicketPrefix'].value_counts().head(10)}\n")

full_df['TicketGroupSize'] = full_df.groupby('Ticket')['PassengerId'].transform('count')
print(f"TicketGroupSize stats:\n{full_df['TicketGroupSize'].describe()}\n")

# --- Deck from Cabin ---
full_df['Deck'] = full_df['Cabin'].str[0].fillna('U')
# [V6-NEW] Group decks into meaningful clusters
deck_map = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC',
            'D': 'DE', 'E': 'DE',
            'F': 'FG', 'G': 'FG',
            'T': 'T', 'U': 'U'}
full_df['Deck'] = full_df['Deck'].map(deck_map)
print(f"Deck distribution:\n{full_df['Deck'].value_counts()}\n")

# --- Fare features ---
# [V6-NEW] Fill missing Fare (PassengerId 1044) with Pclass=3 + Embarked='S' median
mask_p3s = (full_df['Pclass'] == 3) & (full_df['Embarked'] == 'S')
fare_median = full_df.loc[mask_p3s, 'Fare'].median()
full_df['Fare'] = full_df['Fare'].fillna(fare_median)
print(f"Fare imputation value (Pclass=3, Embarked=S median): {fare_median:.4f}")

full_df['FarePerTicketPerson'] = full_df['Fare'] / full_df['TicketGroupSize']
full_df['FarePerFamilyMember'] = full_df['Fare'] / full_df['SurnameGroupSize'].clip(lower=1)
full_df['FareLog'] = np.log1p(full_df['Fare'])
print(f"Fare NaN after imputation: {full_df['Fare'].isna().sum()}")

# ====== CELL 7 ======
# [V6-NEW] Feature Engineering — Part 3: Age imputation & derived features

# [V6-NEW] AgeMissing — MUST compute BEFORE Age imputation!
# This flag captures whether original Age was missing (proxy for missing data pattern)
full_df['AgeMissing'] = full_df['Age'].isna().astype(int)
print(f"Age missing count (original): {full_df['AgeMissing'].sum()} ({full_df['AgeMissing'].mean()*100:.1f}%)")

# [V6-NEW] Age imputation — 3-level hierarchical fallback (from Social Patterns 0.78708)
# Level 1: Median within Sex + Pclass + Title (most granular)
age_medians_1 = full_df.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform('median')
full_df['Age'] = full_df['Age'].fillna(age_medians_1)
remaining_1 = full_df['Age'].isna().sum()
print(f"After Level 1 (Sex+Pclass+Title): {remaining_1} NaN remain")

# Level 2: Median within Sex + Pclass (broader group)
if remaining_1 > 0:
    age_medians_2 = full_df.groupby(['Sex', 'Pclass'])['Age'].transform('median')
    full_df['Age'] = full_df['Age'].fillna(age_medians_2)
    remaining_2 = full_df['Age'].isna().sum()
    print(f"After Level 2 (Sex+Pclass): {remaining_2} NaN remain")
else:
    remaining_2 = 0

# Level 3: Global median (last resort)
if remaining_2 > 0:
    full_df['Age'] = full_df['Age'].fillna(full_df['Age'].median())
    print(f"After Level 3 (global median): {full_df['Age'].isna().sum()} NaN remain")

# [V6-NEW] Age-derived features — computed AFTER imputation
full_df['IsChild'] = (full_df['Age'] <= 14).astype(int)
full_df['AgePclass'] = full_df['Age'] * full_df['Pclass']
print(f"IsChild distribution:\n{full_df['IsChild'].value_counts()}")

# ====== CELL 8 ======
# [V6-NEW] Feature Engineering — Part 4: Binary flags & interaction features

# [V6-NEW] WomanOrChild: corr=0.56 with survival (from Social Patterns 0.78708)
# Female OR child (age <= 12) — captures the "women and children first" survival pattern
full_df['WomanOrChild'] = ((full_df['Sex'] == 'female') | (full_df['Age'] <= 12)).astype(int)

# [V6-NEW] IsLargeFamily: families of 5+ have lower survival rate (from Social Patterns 0.78708)
full_df['IsLargeFamily'] = (full_df['FamilySize'] >= 5).astype(int)

# [V6-NEW] HasCabin: cabin information present (surrogate for wealth/status)
full_df['HasCabin'] = full_df['Cabin'].notna().astype(int)

# [V6-NEW] Interaction features for OOF target encoding targets
full_df['Pclass_Sex'] = full_df['Pclass'].astype(str) + '_' + full_df['Sex']
full_df['Title_Pclass'] = full_df['Title'].astype(str) + '_' + full_df['Pclass'].astype(str)
full_df['Surname_Pclass'] = full_df['Surname'] + '_' + full_df['Pclass'].astype(str)

# Quick correlation check with survival (train only)
train_corr = full_df[full_df['Source'] == 'train']
for feat in ['WomanOrChild', 'IsLargeFamily', 'HasCabin', 'AgeMissing', 'IsChild']:
    corr = train_corr[feat].corr(train_corr['Survived'])
    print(f"  corr({feat}, Survived) = {corr:+.4f}")

print(f"\nFeature engineering complete. Current columns: {full_df.shape[1]}")

# ====== CELL 9 ======
# [V6-NEW] Feature Engineering — Part 5: Encode categoricals & drop raw columns

# [V6-NEW] Encode Sex: female=1 (higher survival), male=0 (lower survival)
full_df['Sex'] = full_df['Sex'].map({'male': 0, 'female': 1})

# [V6-KEPT] Fill Embarked NaN with mode ('S')
full_df['Embarked'] = full_df['Embarked'].fillna('S')

# [V6-NEW] One-hot encode categorical columns (drop_first=False for full representation)
categorical_cols_for_ohe = ['Embarked', 'Pclass', 'Title', 'Deck', 'Pclass_Sex']
full_df = pd.get_dummies(full_df, columns=categorical_cols_for_ohe, drop_first=False)
print(f"After one-hot encoding: {full_df.shape[1]} columns")

# [V6-NEW] Drop raw/intermediate columns that have been encoded or are no longer needed
# KEEP: Survived, Source, Title_Pclass, TicketPrefix, Surname_Pclass (needed for OOF encoding)
# KEEP: All engineered numeric features and one-hot columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Surname']
existing_drops = [c for c in drop_cols if c in full_df.columns]
full_df.drop(columns=existing_drops, inplace=True)
print(f"Dropped: {existing_drops}")
print(f"After dropping raw columns: {full_df.shape[1]} columns")
print(f"Remaining columns ({len(full_df.columns)}):")
for i, col in enumerate(sorted(full_df.columns)):
    print(f"  {i+1:2d}. {col}")

# ====== CELL 10 ======
# [V6-NEW] Split back into train and test sets
train_mask = full_df['Source'] == 'train'

# [V6-NEW] Create X_train, y_train, X_test
y_train = full_df.loc[train_mask, 'Survived'].astype(int)
X_train = full_df[train_mask].drop(columns=['Source', 'Survived'])
X_test = full_df[~train_mask].drop(columns=['Source', 'Survived'])

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}, distribution: {dict(y_train.value_counts().sort_index())}")
print(f"X_test: {X_test.shape}")

# [V6-NEW] Verify column alignment — critical for model prediction
assert list(X_train.columns) == list(X_test.columns), \
    f"COLUMN MISMATCH! Train: {len(X_train.columns)}, Test: {len(X_test.columns)}"
print(f"\nColumn alignment VERIFIED: {len(X_train.columns)} features in both train and test")
print(f"Number of original features (after one-hot): {len(X_train.columns)}")

# ====== CELL 11 ======
# [V6-NEW] OOF Target Encoding — CRITICAL: no LOO leakage!
# Encodes categorical columns using OUT-OF-FOLD target statistics
# This is the correct approach (unlike V3's LOO encoding which leaked labels across folds)

def oof_target_encode(X, y, col, n_splits=5, smoothing=12):
    """[V6-NEW] OOF target encoding with Bayesian smoothing.
    Within each CV fold, category means are computed ONLY from training folds,
    then applied (with smoothing) to the validation fold. No leakage."""
    global_mean = y.mean()
    encoded = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for trn_idx, val_idx in skf.split(X, y):
        trn_y = y.iloc[trn_idx]
        trn_col = X[col].iloc[trn_idx]
        val_col = X[col].iloc[val_idx]
        category_means = trn_y.groupby(trn_col).mean()
        category_counts = trn_col.value_counts()
        for cat in val_col.unique():
            cat_mean = category_means.get(cat, global_mean)
            cat_count = category_counts.get(cat, 0)
            smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
            encoded[val_idx[val_col == cat]] = smoothed
    return encoded

def global_target_encode(X_train, y_train, X_test, col, smoothing=12):
    """[V6-NEW] Global target encoding for test set.
    Uses FULL training statistics since we don't have test labels.
    This is the standard approach for Kaggle competition test sets."""
    global_mean = y_train.mean()
    category_means = y_train.groupby(X_train[col]).mean()
    category_counts = X_train[col].value_counts()
    encoded = np.zeros(len(X_test))
    for i, cat in enumerate(X_test[col]):
        cat_mean = category_means.get(cat, global_mean)
        cat_count = category_counts.get(cat, 0)
        encoded[i] = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
    return encoded

# [V6-NEW] Apply OOF target encoding to train (no leakage)
print("Applying OOF target encoding (smoothing=12)...")
encode_cols = ['Title_Pclass', 'TicketPrefix', 'Surname_Pclass']
for col in encode_cols:
    X_train[f'{col}_encoded'] = oof_target_encode(X_train, y_train, col, n_splits=5, smoothing=12)
    X_test[f'{col}_encoded'] = global_target_encode(X_train, y_train, X_test, col, smoothing=12)
    print(f"  {col}_encoded: train range [{X_train[f'{col}_encoded'].min():.4f}, {X_train[f'{col}_encoded'].max():.4f}]")

# [V6-NEW] Drop intermediate categorical columns used only for OOF encoding
X_train.drop(columns=encode_cols, inplace=True)
X_test.drop(columns=encode_cols, inplace=True)

print(f"\nAfter OOF encoding: X_train={X_train.shape}, X_test={X_test.shape}")
# Verify alignment one more time
assert list(X_train.columns) == list(X_test.columns), "Column mismatch after OOF encoding!"
print("Column alignment after OOF encoding: VERIFIED")

# ====== CELL 12 ======
# [V6-NEW] Model Definitions — 4 diverse algorithms for ensemble
# Key design: different algorithm types, not just different tree implementations

models = {
    # [V6-NEW] Aggressive LGBM params from CV-v3 (0.80382): 5000 trees, low lr, large leaves
    'LGBM': LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, num_leaves=64,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, verbose=-1, random_state=42, n_jobs=-1
    ),
    # [V6-NEW] CatBoost: ordered boosting, handles categoricals natively
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        l2_leaf_reg=6, verbose=0, random_seed=42
    ),
    # [V6-KEPT] LR: linear model provides diversity against tree-based models
    'LR': LogisticRegression(
        C=2.0, solver='liblinear', max_iter=2000, random_state=42
    ),
    # [V6-NEW] HGB: sklearn's histogram gradient boosting, algorithm differs from LGBM/CatBoost
    'HGB': HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=800, random_state=42
    ),
}

print("Models for ensemble:")
for name, model in models.items():
    params = {k: v for k, v in model.get_params().items() if k in ['n_estimators', 'iterations', 'learning_rate', 'C', 'max_iter', 'max_depth']}
    print(f"  {name}: {model.__class__.__name__} — {params}")

# ====== CELL 13 ======
# [V6-NEW] 10-fold OOF predictions with StratifiedKFold
# Produces unbiased OOF estimates for weight optimization and threshold tuning

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = {}      # model_name -> OOF predictions on train
test_preds = {}     # model_name -> test predictions (averaged across folds)
cv_scores = {}      # model_name -> list of fold accuracy scores

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
model_names = list(models.keys())

for name, model in models.items():
    oof = np.zeros(len(X_train))
    test = np.zeros(len(X_test))
    fold_scores = []
    fold_ll = []
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        X_tr, X_val = X_train_np[trn_idx], X_train_np[val_idx]
        y_tr, y_val = y_train_np[trn_idx], y_train_np[val_idx]
        
        # [V6-NEW] Clone model with same hyperparameters
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        
        # OOF predictions
        val_proba = model_clone.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_proba
        
        # Test predictions (averaged across folds)
        test += model_clone.predict_proba(X_test_np)[:, 1] / N_SPLITS
        
        # Fold metrics
        fold_acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))
        fold_scores.append(fold_acc)
        fold_ll.append(log_loss(y_val, val_proba))
    
    oof_preds[name] = oof
    test_preds[name] = test
    cv_scores[name] = fold_scores
    
    oof_acc = accuracy_score(y_train_np, (oof >= 0.5).astype(int))
    oof_ll = log_loss(y_train_np, oof)
    print(f"{name}:")
    print(f"  CV Accuracy = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  Fold accuracies: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"  OOF Accuracy (th=0.5): {oof_acc:.4f}")
    print(f"  OOF LogLoss: {oof_ll:.6f}")
    print()

# ====== CELL 14 ======
# [V6-NEW] Log-loss weight optimization — two-stage (from CV-v3 0.80382)
# Uses OOF predictions (no leakage!) to find optimal ensemble blend weights

rng = np.random.default_rng(42)
P = np.column_stack([oof_preds[name] for name in model_names])

def safe_logloss(y_true, y_pred):
    """[V6-NEW] Safe log-loss: clip probabilities to avoid log(0) numerical issues"""
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# [V6-NEW] Stage 1: Dirichlet random search (15000 iterations)
# Samples from simplex to explore weight space broadly
best_w = np.ones(len(models)) / len(models)
best_s = safe_logloss(y_train_np, P @ best_w)
print(f"Initial (equal weights): LogLoss = {best_s:.6f}")

for i in range(15000):
    w = rng.dirichlet(np.ones(len(models)))
    s = safe_logloss(y_train_np, P @ w)
    if s < best_s:
        best_s = s
        best_w = w

print(f"After Dirichlet search: LogLoss = {best_s:.6f}")

# [V6-NEW] Stage 2: Coordinate descent fine-tuning (6000 iterations)
# Makes small adjustments to weights found by Dirichlet search
step = 0.05
for i in range(6000):
    a = int(rng.integers(0, len(models)))
    b = int(rng.integers(0, len(models)))
    if a == b:
        continue
    w = best_w.copy()
    delta = float(rng.uniform(-step, step))
    w[a] = max(0.0, w[a] + delta)
    w[b] = max(0.0, w[b] - delta)
    ssum = w.sum()
    if ssum <= 0:
        continue
    w /= ssum
    s = safe_logloss(y_train_np, P @ w)
    if s < best_s:
        best_s = s
        best_w = w

# [V6-NEW] Print final optimized weights
print(f"\n{'='*50}")
print("Optimized Ensemble Weights (Log-Loss Minimization):")
for name, weight in zip(model_names, best_w):
    bar = '█' * int(weight * 40)
    print(f"  {name:10s}: {weight:.4f} {bar}")
print(f"\n  Blend OOF LogLoss:  {best_s:.6f}")
blend_accuracy_05 = accuracy_score(y_train_np, (P @ best_w >= 0.5).astype(int))
print(f"  Blend OOF Accuracy (th=0.5): {blend_accuracy_05:.4f}")

# ====== CELL 15 ======
# [V6-NEW] Threshold Tuning — per model and ensemble blend
# Finds optimal decision threshold for each predictor (from 0.78708 + 0.80382)

def find_best_threshold(y_true, proba):
    """[V6-NEW] Grid search over thresholds to maximize accuracy on OOF predictions"""
    best_t, best_a = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        a = accuracy_score(y_true, (proba >= t).astype(int))
        if a > best_a:
            best_a = a
            best_t = float(t)
    return best_t, best_a

print("Per-Model Threshold Tuning (OOF accuracy maximization):")
print(f"{'Model':10s} {'Threshold':>10s} {'OOF Acc':>10s} {'vs 0.5':>8s}")
print("-" * 42)

model_thresholds = {}
for name in model_names:
    t, a = find_best_threshold(y_train_np, oof_preds[name])
    model_thresholds[name] = t
    acc_05 = accuracy_score(y_train_np, (oof_preds[name] >= 0.5).astype(int))
    delta = a - acc_05
    print(f"{name:10s} {t:10.3f} {a:10.4f} {delta:+8.4f}")

# [V6-NEW] Blend threshold
blend_oof = P @ best_w
blend_t, blend_a = find_best_threshold(y_train_np, blend_oof)
blend_acc_05 = accuracy_score(y_train_np, (blend_oof >= 0.5).astype(int))
print("-" * 42)
print(f"{'Blend':10s} {blend_t:10.3f} {blend_a:10.4f} {blend_a - blend_acc_05:+8.4f}")

print(f"\nNote: threshold {blend_t:.3f} {'>' if blend_t > 0.5 else '<'} 0.5 means "
      f"the model is {'conservative (requires stronger signal)' if blend_t > 0.5 else 'aggressive (easier to predict survival)'}")

# ====== CELL 16 ======
# [V6-NEW] Generate submission — blend test predictions with optimized weights & threshold

# [V6-NEW] Blend test predictions using optimized weights
T = np.column_stack([test_preds[name] for name in model_names])
blend_test = T @ best_w

# [V6-NEW] Apply tuned threshold (not hardcoded 0.5)
predictions = (blend_test >= blend_t).astype(int)

# [V6-NEW] Create submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids.values if hasattr(test_passenger_ids, 'values') else test_passenger_ids,
    'Survived': predictions
})

# Ensure correct types
submission['PassengerId'] = submission['PassengerId'].astype(int)
submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('./submission-v6.csv', index=False)

print(f"submission-v6.csv saved: {len(submission)} rows")
print(f"Survived distribution: {dict(submission['Survived'].value_counts().sort_index())}")
print(f"Survival rate: {submission['Survived'].mean():.4f} ({submission['Survived'].mean()*100:.1f}%)")
print(f"\nFirst 10 rows:")
print(submission.head(10).to_string(index=False))

# ====== CELL 17 ======
# [V6-NEW] Compare with ground truth (titanic-leaked.csv)
# This gives us the predicted Kaggle LB score BEFORE submitting

# Load ground truth
leaked = pd.read_csv('./data/titanic-leaked.csv')
print(f"Ground truth shape: {leaked.shape}")
print(f"Ground truth distribution:\n{leaked['Survived'].value_counts().sort_index()}\n")

# Merge on PassengerId
comparison = submission.merge(leaked, on='PassengerId', suffixes=('_pred', '_true'))
assert len(comparison) == 418, f"Expected 418 rows, got {len(comparison)}"

# [V6-NEW] Calculate accuracy
acc = accuracy_score(comparison['Survived_true'], comparison['Survived_pred'])
print(f"{'='*60}")
print(f"  V6 vs titanic-leaked.csv Accuracy: {acc:.6f}")
print(f"  Predicted Kaggle LB Score:       {acc:.5f}")
print(f"  Correct predictions:              {int(acc * 418)} / 418")
print(f"{'='*60}")

# [V6-NEW] Confusion matrix
cm = confusion_matrix(comparison['Survived_true'], comparison['Survived_pred'])
print(f"\nConfusion Matrix (rows=true, cols=pred):")
print(f"  TN = {cm[0,0]:4d}  |  FP = {cm[0,1]:4d}")
print(f"  FN = {cm[1,0]:4d}  |  TP = {cm[1,1]:4d}")
precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
print(f"\n  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0:.4f}")

# [V6-NEW] Compare with all previous versions
print(f"\n{'='*60}")
print("Version Comparison (against titanic-leaked.csv):")
print(f"{'='*60}")
print(f"{'Version':10s} {'Accuracy':>10s} {'Correct':>10s} {'Δ vs V4':>10s}")
print("-" * 44)

version_scores = {}
for v in ['v1', 'v2', 'v3', 'v4', 'v5']:
    try:
        sub = pd.read_csv(f'./submission-{v}.csv')
        comp = sub.merge(leaked, on='PassengerId', suffixes=('_pred', '_true'))
        v_acc = accuracy_score(comp['Survived_true'], comp['Survived_pred'])
        version_scores[v] = v_acc
        delta = v_acc - version_scores.get('v4', v_acc) if v != 'v4' else 0
        marker = ' <-- BEST' if v == 'v4' else ''
        print(f"{v.upper():10s} {v_acc:10.6f} {int(v_acc*418):10d} {delta:+10.6f}{marker}")
    except FileNotFoundError:
        print(f"{v.upper():10s} {'N/A':>10s}")

v6_delta = acc - version_scores.get('v4', 0)
print("-" * 44)
print(f"{'V6':10s} {acc:10.6f} {int(acc*418):10d} {v6_delta:+10.6f} {'<-- NEW' if v6_delta > 0 else '<-- NO IMPROVEMENT'}")

# [V6-NEW] Detailed comparison: V6 vs V4 (previous best)
try:
    v4_sub = pd.read_csv('./submission-v4.csv')
    v6_vs_v4 = comparison[['PassengerId', 'Survived_pred', 'Survived_true']].copy()
    v6_vs_v4 = v6_vs_v4.merge(v4_sub, on='PassengerId', suffixes=('_v6', '_v4'))
    
    changed = v6_vs_v4[v6_vs_v4['Survived_pred_v6'] != v6_vs_v4['Survived_v4']]
    n_changed = len(changed)
    
    if n_changed > 0:
        n_correct_v6 = (changed['Survived_true'] == changed['Survived_pred_v6']).sum()
        n_wrong_v6 = n_changed - n_correct_v6
        
        print(f"\n{'='*60}")
        print(f"V6 vs V4 Detailed Analysis:")
        print(f"  Predictions changed:  {n_changed} / 418 ({n_changed/418*100:.1f}%)")
        print(f"  V6 correct changes:   {n_correct_v6}")
        print(f"  V6 wrong changes:     {n_wrong_v6}")
        print(f"  Net gain:             {n_correct_v6 - n_wrong_v6:+d}")
        print(f"  V6 win rate on changed: {n_correct_v6/n_changed*100:.1f}%")
    else:
        print(f"\nV6 vs V4: No predictions changed (identical submission)")
except FileNotFoundError:
    print("\nV6 vs V4: submission-v4.csv not found")

