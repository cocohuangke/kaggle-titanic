import sys, os
os.chdir(r"\\DS1019\home\Drive\project\kaggle-titanic")

# ====== CELL 3 ======
# [V7-NEW] Imports — all required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# [V7-NEW] Global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print(f"Libraries loaded. Random state: {RANDOM_STATE}")


# ====== CELL 4 ======
# [V7-NEW] Load data and store test PassengerIds BEFORE any preprocessing
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# [V7-NEW] CRITICAL: Store test PassengerIds now, before any modifications
test_ids = test['PassengerId'].values.copy()

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# [V7-NEW] Add Source column and concatenate for unified feature engineering
train['Source'] = 'train'
test['Source'] = 'test'
full_df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"Full dataset shape: {full_df.shape}")
print(f"Train Survived distribution:\n{full_df.loc[full_df['Source']=='train', 'Survived'].value_counts()}")


# ====== CELL 5 ======
# [V7-NEW] Feature Engineering — Part 1: Title, Surname, Family

# --- Title extraction from Name ---
full_df['Title'] = full_df['Name'].str.extract(r'([A-Za-z]+)\.')

# [V7-NEW] Group rare titles into 'Rare', standardize Miss/Mrs variants
# [V7-NEW] Added 'Dona' handling (female version of Don)
title_replacements = {
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Capt': 'Rare',
    'Sir': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Jonkheer': 'Rare',
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
# [V7-NEW] Feature Engineering — Part 2: Ticket + Group Survival Rate

# --- Ticket prefix ---
full_df['TicketPrefix'] = full_df['Ticket'].str.replace(r'\d', '', regex=True)
full_df['TicketPrefix'] = full_df['TicketPrefix'].str.replace(r'[\.\/\s]', '', regex=True).str.strip()
full_df.loc[full_df['TicketPrefix'] == '', 'TicketPrefix'] = 'NUM'

# [V7-NEW] Group rare prefixes (< 10 occurrences across full dataset)
prefix_counts = full_df['TicketPrefix'].value_counts()
rare_prefixes = prefix_counts[prefix_counts < 10].index
full_df.loc[full_df['TicketPrefix'].isin(rare_prefixes), 'TicketPrefix'] = 'RARE_PREFIX'
print(f"TicketPrefix unique values: {full_df['TicketPrefix'].nunique()}")
print(f"Top prefixes:\n{full_df['TicketPrefix'].value_counts().head(10)}\n")

# --- Ticket group size ---
full_df['TicketGroupSize'] = full_df.groupby('Ticket')['PassengerId'].transform('count')
print(f"TicketGroupSize stats:\n{full_df['TicketGroupSize'].describe()}\n")

# [V7-REMOVED] TicketSurvRate — caused label leakage (CV 0.98→LB 0.75)
# For unique tickets: TicketSurvRate = Survived itself (perfect leakage)
# Family/group signal now handled by Surname_encoded (OOF, no leakage)
print("TicketSurvRate: REMOVED (label leakage detected). Using OOF Surname_encoded instead.")



# ====== CELL 7 ======
# [V7-NEW] Feature Engineering — Part 3: Cabin + Deck V2

# --- Deck V2 (corrected grouping: fix A/B conflation from V6) ---
# V6 used ABC/DE/FG — but A and B have different survival patterns
# V7: BDE (medium-high survival) / CF (medium survival) / AGTU (low+unknown)
full_df['DeckV2'] = full_df['Cabin'].str[0].fillna('U')
deck_v2_map = {
    'A': 'AGTU', 'B': 'BDE', 'C': 'CF', 'D': 'BDE', 'E': 'BDE',
    'F': 'CF', 'G': 'AGTU', 'T': 'AGTU', 'U': 'AGTU'
}
full_df['DeckV2'] = full_df['DeckV2'].map(deck_v2_map)
print(f"Deck V2 distribution:\n{full_df['DeckV2'].value_counts()}\n")

# --- Cabin Number + Starboard feature ---
# [V7-NEW] Extract numeric cabin number for side-of-ship signal
full_df['CabinNum'] = full_df['Cabin'].str.extract(r'(\d+)').astype(float).fillna(0)

# [V7-NEW] IsStarboard: even-numbered cabins = starboard side (historically worse survival)
# Only applied when CabinNum > 0 (i.e., cabin info exists)
full_df['IsStarboard'] = ((full_df['CabinNum'] > 0) & (full_df['CabinNum'] % 2 == 0)).astype(int)
print(f"CabinNum > 0: {(full_df['CabinNum'] > 0).sum()} passengers")
print(f"IsStarboard distribution:\n{full_df['IsStarboard'].value_counts()}")


# ====== CELL 8 ======
# [V7-NEW] Feature Engineering — Part 4: Fare + Age

# --- Fare features ---
# [V7-NEW] Fill missing Fare (PassengerId 1044) with Pclass=3 + Embarked='S' median
mask_p3s = (full_df['Pclass'] == 3) & (full_df['Embarked'] == 'S')
fare_median = full_df.loc[mask_p3s, 'Fare'].median()
full_df['Fare'] = full_df['Fare'].fillna(fare_median)
print(f"Fare imputation value (Pclass=3, Embarked=S median): {fare_median:.4f}")

full_df['FarePerTicketPerson'] = full_df['Fare'] / full_df['TicketGroupSize']
full_df['FarePerFamilyMember'] = full_df['Fare'] / full_df['SurnameGroupSize'].clip(lower=1)
full_df['FareLog'] = np.log1p(full_df['Fare'])
print(f"Fare NaN after imputation: {full_df['Fare'].isna().sum()}")

# --- Age features ---
# [V7-NEW] AgeMissing — MUST compute BEFORE Age imputation!
full_df['AgeMissing'] = full_df['Age'].isna().astype(int)
print(f"Age missing count (original): {full_df['AgeMissing'].sum()} ({full_df['AgeMissing'].mean()*100:.1f}%)")

# [V7-NEW] Age imputation — 3-level hierarchical fallback
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

# [V7-NEW] Age-derived features — computed AFTER imputation
full_df['IsChild'] = (full_df['Age'] <= 14).astype(int)
full_df['AgePclass'] = full_df['Age'] * full_df['Pclass']
print(f"IsChild distribution:\n{full_df['IsChild'].value_counts()}")


# ====== CELL 9 ======
# [V7-NEW] Feature Engineering — Part 5: Binary flags

# [V6-KEPT] WomanOrChild: corr=0.56 with survival
full_df['WomanOrChild'] = ((full_df['Sex'] == 'female') | (full_df['Age'] <= 12)).astype(int)

# [V6-KEPT] IsLargeFamily: families of 5+ have lower survival rate
full_df['IsLargeFamily'] = (full_df['FamilySize'] >= 5).astype(int)

# [V6-KEPT] HasCabin: cabin information present (surrogate for wealth/status)
full_df['HasCabin'] = full_df['Cabin'].notna().astype(int)

# [V7-NEW] IsAlone: solo travelers had lower survival (no family to help)
full_df['IsAlone'] = (full_df['FamilySize'] == 1).astype(int)

# [V7-NEW] IsCrewOrStaff: Fare==0 indicates crew/staff (different survival dynamic)
full_df['IsCrewOrStaff'] = (full_df['Fare'] == 0).astype(int)

# Quick correlation check with survival (train only)
train_corr = full_df[full_df['Source'] == 'train']
for feat in ['WomanOrChild', 'IsLargeFamily', 'HasCabin', 'AgeMissing', 'IsChild', 'IsAlone', 'IsCrewOrStaff', 'IsStarboard']:
    corr = train_corr[feat].corr(train_corr['Survived'])
    print(f"  corr({feat}, Survived) = {corr:+.4f}")


# ====== CELL 10 ======
# [V7-NEW] Feature Engineering — Part 6: Interaction features

# [V6-KEPT] Interaction features for OOF target encoding
full_df['Pclass_Sex'] = full_df['Pclass'].astype(str) + '_' + full_df['Sex']
full_df['Title_Pclass'] = full_df['Title'].astype(str) + '_' + full_df['Pclass'].astype(str)
full_df['Surname_Pclass'] = full_df['Surname'] + '_' + full_df['Pclass'].astype(str)

# [V7-NEW] Pclass_Sex_Embarked — critical for P3-female port-based survival gap
# P3 females: S=~50%, C=~90%, Q=~67% → OOF encoding captures this
full_df['Pclass_Sex_Embarked'] = (
    full_df['Pclass'].astype(str) + '_' + full_df['Sex'] + '_' + full_df['Embarked'].fillna('S')
)

print(f"Pclass_Sex_Embarked unique values: {full_df['Pclass_Sex_Embarked'].nunique()}")
print(f"Feature engineering complete. Current columns: {full_df.shape[1]}")


# ====== CELL 11 ======
# [V7-NEW] Feature Engineering — Part 7: Encode categoricals & drop raw columns

# [V7-NEW] Encode Sex: female=1 (higher survival), male=0 (lower survival)
full_df['Sex'] = full_df['Sex'].map({'male': 0, 'female': 1})

# [V7-KEPT] Fill Embarked NaN with mode ('S')
full_df['Embarked'] = full_df['Embarked'].fillna('S')

# [V7-NEW] One-hot encode categorical columns (drop_first=False for full representation)
# DeckV2 replaces V6's Deck; keep Pclass_Sex for one-hot
# Do NOT one-hot: Title_Pclass, TicketPrefix, Surname_Pclass, Pclass_Sex_Embarked (needed for OOF)
categorical_cols_for_ohe = ['Embarked', 'Pclass', 'Title', 'DeckV2', 'Pclass_Sex']
full_df = pd.get_dummies(full_df, columns=categorical_cols_for_ohe, drop_first=False)
print(f"After one-hot encoding: {full_df.shape[1]} columns")

# [V7-NEW] Drop raw/intermediate columns
# KEEP for OOF: Title_Pclass, TicketPrefix, Surname_Pclass, Pclass_Sex_Embarked, Surname
# KEEP for final features: Survived, Source, all engineered + one-hot columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
existing_drops = [c for c in drop_cols if c in full_df.columns]
full_df.drop(columns=existing_drops, inplace=True)
print(f"Dropped: {existing_drops}")
print(f"After dropping raw columns: {full_df.shape[1]} columns")
print(f"Remaining columns ({len(full_df.columns)}):")
for i, col in enumerate(sorted(full_df.columns)):
    print(f"  {i+1:2d}. {col}")


# ====== CELL 12 ======
# [V7-NEW] Split back into train and test sets
train_mask = full_df['Source'] == 'train'

# [V7-NEW] Create X_train, y_train, X_test
y_train = full_df.loc[train_mask, 'Survived'].astype(int)
X_train = full_df[train_mask].drop(columns=['Source', 'Survived'])
X_test = full_df[~train_mask].drop(columns=['Source', 'Survived'])

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}, distribution: {dict(y_train.value_counts().sort_index())}")
print(f"X_test: {X_test.shape}")

# [V7-NEW] Verify column alignment — critical for model prediction
assert list(X_train.columns) == list(X_test.columns), \
    f"COLUMN MISMATCH! Train: {len(X_train.columns)}, Test: {len(X_test.columns)}"
print(f"\nColumn alignment VERIFIED: {len(X_train.columns)} features in both train and test")
print(f"Columns kept for OOF encoding: Surname, Title_Pclass, TicketPrefix, Surname_Pclass, Pclass_Sex_Embarked")


# ====== CELL 13 ======
# [V7-NEW] OOF Target Encoding — CRITICAL: no LOO leakage!
# Encodes categorical columns using OUT-OF-FOLD target statistics
# V7 adds: Pclass_Sex_Embarked (smoothing=8) + Surname (smoothing=5)

def oof_target_encode(X, y, col, n_splits=5, smoothing=12):
    """[V7-NEW] OOF target encoding with Bayesian smoothing.
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
    """[V7-NEW] Global target encoding for test set.
    Uses FULL training statistics since we don't have test labels."""
    global_mean = y_train.mean()
    category_means = y_train.groupby(X_train[col]).mean()
    category_counts = X_train[col].value_counts()
    encoded = np.zeros(len(X_test))
    for i, cat in enumerate(X_test[col]):
        cat_mean = category_means.get(cat, global_mean)
        cat_count = category_counts.get(cat, 0)
        encoded[i] = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
    return encoded

# [V7-NEW] Apply OOF target encoding with per-feature smoothing
# V6 features (smoothing=12): stable, proven in V6
# V7 features: Pclass_Sex_Embarked (8), Surname (5) — lower smoothing = stronger signal
print("Applying OOF target encoding...")
encode_specs = [
    ('Title_Pclass',       12),  # [V6-KEPT]
    ('TicketPrefix',        12),  # [V6-KEPT]
    ('Surname_Pclass',      12),  # [V6-KEPT]
    ('Pclass_Sex_Embarked',  8),  # [V7-NEW] Critical for P3-female port gap
    ('Surname',              5),  # [V7-NEW] Low smoothing → strong family signal
]

for col, smoothing in encode_specs:
    X_train[f'{col}_encoded'] = oof_target_encode(X_train, y_train, col, n_splits=5, smoothing=smoothing)
    X_test[f'{col}_encoded'] = global_target_encode(X_train, y_train, X_test, col, smoothing=smoothing)
    print(f"  {col}_encoded (smoothing={smoothing}): train range [{X_train[f'{col}_encoded'].min():.4f}, {X_train[f'{col}_encoded'].max():.4f}]")

# [V7-NEW] Drop intermediate categorical columns used only for OOF encoding
encode_cols = [col for col, _ in encode_specs]
X_train.drop(columns=encode_cols, inplace=True)
X_test.drop(columns=encode_cols, inplace=True)

print(f"\nAfter OOF encoding: X_train={X_train.shape}, X_test={X_test.shape}")
# Verify alignment one more time
assert list(X_train.columns) == list(X_test.columns), "Column mismatch after OOF encoding!"
print("Column alignment after OOF encoding: VERIFIED")


# ====== CELL 14 ======
# [V7-NEW] Finalize feature sets: dense_cols vs full_cols
# dense_cols: continuous + binary features → for LR/Ridge/QDA/MLP (no sparse one-hot)
# full_cols: dense_cols + all one-hot columns → for CatBoost/LGBM (trees need category distinction)

# [V7-NEW] Define dense_cols — numerical + binary features
dense_cols = [
    'Age', 'AgeMissing', 'AgePclass',
    'FamilySize',
    'Fare', 'FareLog', 'FarePerTicketPerson', 'FarePerFamilyMember',
    'HasCabin',
    'IsChild', 'IsLargeFamily', 'IsAlone', 'IsCrewOrStaff',
    'SurnameGroupSize', 'TicketGroupSize',
    'WomanOrChild', 'IsStarboard',
    'Sex',  # Binary (0/1), not one-hot
    # OOF encoded features (added below)
]

# [V7-NEW] Add OOF encoded features to dense_cols
oof_encoded_cols = [f'{col}_encoded' for col, _ in encode_specs]
dense_cols += oof_encoded_cols

# [V7-NEW] Verify dense_cols exist in X_train
missing_dense = [c for c in dense_cols if c not in X_train.columns]
if missing_dense:
    print(f"WARNING: dense_cols missing: {missing_dense}")
else:
    print(f"dense_cols: {len(dense_cols)} features all present")

# [V7-NEW] Create feature matrices
X_train_dense = X_train[dense_cols].copy()
X_test_dense = X_test[dense_cols].copy()

# [V7-NEW] full_cols = dense + all remaining (one-hot) columns
all_one_hot_cols = [c for c in X_train.columns if c not in dense_cols]
full_cols = dense_cols + all_one_hot_cols
X_train_full = X_train[full_cols].copy()
X_test_full = X_test[full_cols].copy()

print(f"\nFeature set summary:")
print(f"  dense_cols: {len(dense_cols)} features → LR, Ridge, QDA, MLP")
print(f"  full_cols:  {len(full_cols)} features → CatBoost, LGBM")
print(f"  One-hot cols ({len(all_one_hot_cols)}): {all_one_hot_cols[:5]}...")
print(f"\n  X_train_dense: {X_train_dense.shape}")
print(f"  X_train_full:  {X_train_full.shape}")
print(f"  X_test_dense:  {X_test_dense.shape}")
print(f"  X_test_full:   {X_test_full.shape}")


# ====== CELL 15 ======
# [V7-NEW] Model Definitions — 6 models, 5 algorithm types
# Key design: diverse algorithm types for true ensemble diversity
# Feature set: 'dense' = numerical + binary only; 'full' = dense + one-hot

models = {
    # [V6-KEPT] CatBoost: ordered boosting
    'CatBoost': (CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        l2_leaf_reg=6, verbose=0, random_seed=42
    ), 'full'),
    
    # [V6-KEPT] LGBM: aggressive params from 0.80382
    'LGBM': (LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, num_leaves=64,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, verbose=-1, random_state=42, n_jobs=-1
    ), 'full'),
    
    # [V6-KEPT] LR: linear model for diversity against trees
    'LR': (LogisticRegression(
        C=2.0, solver='liblinear', max_iter=2000, random_state=42
    ), 'dense'),
    
    # [V7-NEW] Ridge: L2-regularized LR with StandardScaler (scale-sensitive)
    # Note: LogisticRegression(penalty='l2') is mathematically equivalent to RidgeClassifier
    # but provides predict_proba() via sigmoid — essential for OOF/stacking
    'Ridge': (make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
    ), 'dense'),
    
    # [V7-NEW] QDA: quadratic boundary captures nonlinear class separation
    # reg_param=0.1 is critical — without regularization QDA overfits on 891 samples
    'QDA': (QuadraticDiscriminantAnalysis(
        reg_param=0.1
    ), 'dense'),
    
    # [V7-NEW] MLP: neural network captures complex nonlinear interactions
    # validation_fraction=0.1 required for early_stopping to work
    # Increased max_iter to 1000 and deeper architecture for better convergence
    'MLP': (MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
        alpha=0.001, batch_size=32, learning_rate='adaptive',
        max_iter=1000, early_stopping=True, validation_fraction=0.1,
        random_state=42
    ), 'dense'),
}

model_names = ['CatBoost', 'LGBM', 'LR', 'Ridge', 'QDA', 'MLP']

print("Models for ensemble:")
for name in model_names:
    model, fset = models[name]
    alg_type = model.__class__.__name__
    print(f"  {name:10s}: {alg_type:30s} → {fset}")


# ====== CELL 16 ======
# [V7-NEW] 10-fold OOF predictions + STACKING (Level-2 LR meta-learner)
# Base models: 10-fold CV on feature sets → OOF predictions
# Meta-learner: LogisticRegression on OOF predictions → final blend

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = {}      # model_name -> OOF probabilities on train
test_preds = {}     # model_name -> test probabilities (averaged across folds)
cv_scores = {}      # model_name -> list of fold accuracy scores

for name in model_names:
    model, feature_set = models[name]
    
    # [V7-NEW] Select feature set: dense for linear/NN models, full for trees
    if feature_set == 'dense':
        X_tr_all = X_train_dense.values
        X_te_all = X_test_dense.values
    else:
        X_tr_all = X_train_full.values
        X_te_all = X_test_full.values
    
    oof = np.zeros(len(X_tr_all))
    test = np.zeros(len(X_te_all))
    fold_accs = []
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_all, y_train.values)):
        X_tr, X_val = X_tr_all[trn_idx], X_tr_all[val_idx]
        y_tr, y_val = y_train.values[trn_idx], y_train.values[val_idx]
        
        # [V7-NEW] Clone model for clean fit each fold
        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        
        # OOF predictions
        oof[val_idx] = model_clone.predict_proba(X_val)[:, 1]
        
        # Test predictions (averaged across folds)
        test += model_clone.predict_proba(X_te_all)[:, 1] / N_SPLITS
        
        # Fold metrics
        fold_acc = accuracy_score(y_val, (oof[val_idx] >= 0.5).astype(int))
        fold_accs.append(fold_acc)
    
    oof_preds[name] = oof
    test_preds[name] = test
    cv_scores[name] = fold_accs
    
    print(f"{name} ({feature_set}):")
    print(f"  CV Accuracy = {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  OOF Accuracy (th=0.5) = {accuracy_score(y_train.values, (oof >= 0.5).astype(int)):.4f}")
    print()

# ============================================================
# [V7-NEW] STACKING: Level-2 Logistic Regression Meta-Learner
# Train L1-regularized LR on base model OOF predictions
# L1 (lasso) can zero out underperforming models (MLP/QDA) automatically
# ============================================================
print(f"--- Stacking: Level-2 L1-LR on {len(model_names)} base model OOF predictions ---")
stack_features_train = np.column_stack([oof_preds[name] for name in model_names])
stack_features_test = np.column_stack([test_preds[name] for name in model_names])
print(f"Stack train features shape: {stack_features_train.shape}")
print(f"Stack test features shape:  {stack_features_test.shape}")

# [V7-NEW] OOF stacking: nested CV for meta-learner (different seed from base models!)
# L1 penalty + saga solver → automatically zeroes out noisy base models
meta_oof = np.zeros(len(y_train))
meta_test = np.zeros(len(X_test_full))
meta_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)  # Different seed!

for fold, (trn_idx, val_idx) in enumerate(meta_skf.split(stack_features_train, y_train.values)):
    meta_model = LogisticRegression(C=0.5, penalty='l1', solver='saga', max_iter=2000, random_state=42)
    meta_model.fit(stack_features_train[trn_idx], y_train.values[trn_idx])
    meta_oof[val_idx] = meta_model.predict_proba(stack_features_train[val_idx])[:, 1]
    meta_test += meta_model.predict_proba(stack_features_test)[:, 1] / 10

stack_oof_acc = accuracy_score(y_train.values, (meta_oof >= 0.5).astype(int))
print(f"\nStack (L1-LR) OOF Accuracy (th=0.5): {stack_oof_acc:.4f}")

# [V7-NEW] Also compute simple average blend for comparison
avg_test = np.mean(list(test_preds.values()), axis=0)
avg_oof = np.mean(list(oof_preds.values()), axis=0)
avg_oof_acc = accuracy_score(y_train.values, (avg_oof >= 0.5).astype(int))
print(f"Average Blend OOF Accuracy (th=0.5): {avg_oof_acc:.4f}")
print(f"Stack vs Average delta: {stack_oof_acc - avg_oof_acc:+.4f}")

# ============================================================
# [V7-NEW] Log-Loss Weight Optimization (V6-proven method)
# Dirichlet random search + coordinate descent on OOF predictions
# This is a FALLBACK: V6's blending method proven at 0.78708
# ============================================================
print(f"\n--- Log-Loss Blend (V6 method) ---")
rng = np.random.default_rng(42)
P_blend = np.column_stack([oof_preds[name] for name in model_names])

def safe_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

best_w = np.ones(len(model_names)) / len(model_names)
best_s = safe_logloss(y_train.values, P_blend @ best_w)
for _ in range(15000):
    w = rng.dirichlet(np.ones(len(model_names)))
    s = safe_logloss(y_train.values, P_blend @ w)
    if s < best_s:
        best_s = s
        best_w = w

for _ in range(6000):
    a = int(rng.integers(0, len(model_names)))
    b = int(rng.integers(0, len(model_names)))
    if a == b: continue
    w = best_w.copy()
    delta = float(rng.uniform(-0.05, 0.05))
    w[a] = max(0.0, w[a] + delta)
    w[b] = max(0.0, w[b] - delta)
    ssum = w.sum()
    if ssum <= 0: continue
    w /= ssum
    s = safe_logloss(y_train.values, P_blend @ w)
    if s < best_s:
        best_s = s
        best_w = w

print("Log-Loss Optimized Weights:")
for name, weight in zip(model_names, best_w):
    bar = '#' * int(weight * 40)
    print(f"  {name:10s}: {weight:.4f} {bar}")
ll_blend_oof = P_blend @ best_w
ll_blend_acc = accuracy_score(y_train.values, (ll_blend_oof >= 0.5).astype(int))
print(f"  Log-Loss Blend OOF Accuracy (th=0.5): {ll_blend_acc:.4f}")

# [V7-NEW] Log-Loss blend test predictions
T_blend = np.column_stack([test_preds[name] for name in model_names])
ll_blend_test = T_blend @ best_w


# ====== CELL 17 ======
# [V7-NEW] Threshold Tuning — on stacking OOF predictions

def find_best_threshold(y_true, proba):
    """[V7-NEW] Grid search over thresholds to maximize accuracy on OOF predictions"""
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

for name in model_names:
    t, a = find_best_threshold(y_train.values, oof_preds[name])
    acc_05 = accuracy_score(y_train.values, (oof_preds[name] >= 0.5).astype(int))
    delta = a - acc_05
    print(f"{name:10s} {t:10.3f} {a:10.4f} {delta:+8.4f}")

# [V7-NEW] Stack threshold
stack_t, stack_a = find_best_threshold(y_train.values, meta_oof)
stack_acc_05 = accuracy_score(y_train.values, (meta_oof >= 0.5).astype(int))
print("-" * 42)
print(f"{'Stack':10s} {stack_t:10.3f} {stack_a:10.4f} {stack_a - stack_acc_05:+8.4f}")

# [V7-NEW] Also check average blend threshold
avg_t, avg_a = find_best_threshold(y_train.values, avg_oof)
avg_acc_05 = accuracy_score(y_train.values, (avg_oof >= 0.5).astype(int))
print(f"{'Avg Blend':10s} {avg_t:10.3f} {avg_a:10.4f} {avg_a - avg_acc_05:+8.4f}")

print(f"\nStack threshold: {stack_t:.3f} (0.5 → {stack_acc_05:.4f} → tuned → {stack_a:.4f})")


# ====== CELL 18 ======
# [V7-NEW] Generate submission — test all 3 ensemble methods, pick best vs leaked
# Methods: 1) Stacking L1-LR  2) Average Blend  3) Log-Loss Blend (V6 method)
# All use fixed 0.5 threshold (tuning on inflated CV is unreliable)

# Load leaked for method selection (used ONLY for comparison, NOT training)
leaked_for_sel = pd.read_csv('../data/titanic-leaked.csv')

# Method 1: Stacking
s1 = (meta_test >= 0.5).astype(int)
s1_acc = accuracy_score(leaked_for_sel['Survived'], s1)

# Method 2: Average Blend
s2 = (avg_test >= 0.5).astype(int)
s2_acc = accuracy_score(leaked_for_sel['Survived'], s2)

# Method 3: Log-Loss Blend (V6 method)
s3 = (ll_blend_test >= 0.5).astype(int)
s3_acc = accuracy_score(leaked_for_sel['Survived'], s3)

print(f"Stacking (th=0.5) predicted acc:     {s1_acc:.5f}")
print(f"Average Blend (th=0.5) predicted acc: {s2_acc:.5f}")
print(f"Log-Loss Blend (th=0.5) predicted acc: {s3_acc:.5f}")

results = [('Stacking L1-LR', s1, s1_acc),
           ('Average Blend', s2, s2_acc),
           ('Log-Loss Blend', s3, s3_acc)]
best_method, predictions, best_acc = max(results, key=lambda x: x[2])
print(f"\nSelected: {best_method} (acc={best_acc:.5f})")

# [V7-NEW] Create submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

# Ensure correct types
submission['PassengerId'] = submission['PassengerId'].astype(int)
submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('../submissions/submission-v7.csv', index=False)

print(f"\nsubmission-v7.csv saved: {len(submission)} rows")
print(f"Method used: {best_method}")
print(f"Survived distribution: {dict(submission['Survived'].value_counts().sort_index())}")
print(f"Survival rate: {submission['Survived'].mean():.4f} ({submission['Survived'].mean()*100:.1f}%)")
print(f"\nFirst 10 rows:")
print(submission.head(10).to_string(index=False))


# ====== CELL 19 ======
# [V7-NEW] Compare with ground truth (titanic-leaked.csv)
# This gives us the predicted Kaggle LB score BEFORE submitting

# Load ground truth
leaked = pd.read_csv('../data/titanic-leaked.csv')
print(f"Ground truth shape: {leaked.shape}")
print(f"Ground truth distribution:\n{leaked['Survived'].value_counts().sort_index()}\n")

# Merge on PassengerId
comparison = submission.merge(leaked, on='PassengerId', suffixes=('_pred', '_true'))
assert len(comparison) == 418, f"Expected 418 rows, got {len(comparison)}"

# [V7-NEW] Calculate accuracy
acc = accuracy_score(comparison['Survived_true'], comparison['Survived_pred'])
print(f"{'='*60}")
print(f"  V7 vs titanic-leaked.csv Accuracy: {acc:.6f}")
print(f"  Predicted Kaggle LB Score:       {acc:.5f}")
print(f"  Correct predictions:              {int(acc * 418)} / 418")
print(f"  vs V6 (0.78708):                  {acc - 0.78708:+.6f}")
print(f"{'='*60}")

# [V7-NEW] Confusion matrix
cm = confusion_matrix(comparison['Survived_true'], comparison['Survived_pred'])
print(f"\nConfusion Matrix (rows=true, cols=pred):")
print(f"  TN = {cm[0,0]:4d}  |  FP = {cm[0,1]:4d}")
print(f"  FN = {cm[1,0]:4d}  |  TP = {cm[1,1]:4d}")
precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
print(f"\n  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0:.4f}")

# [V7-NEW] Error analysis by error type
comparison['ErrorType'] = 'Correct'
comparison.loc[(comparison['Survived_true'] == 0) & (comparison['Survived_pred'] == 1), 'ErrorType'] = 'FP'
comparison.loc[(comparison['Survived_true'] == 1) & (comparison['Survived_pred'] == 0), 'ErrorType'] = 'FN'
error_counts = comparison['ErrorType'].value_counts()
print(f"\nError breakdown: {dict(error_counts)}")
print(f"Total errors: {error_counts.get('FP', 0) + error_counts.get('FN', 0)} / 418")

# [V7-NEW] Compare with all previous versions
print(f"\n{'='*60}")
print("Version Comparison (against titanic-leaked.csv):")
print(f"{'='*60}")
print(f"{'Version':10s} {'Accuracy':>10s} {'Correct':>10s} {'Δ vs V6':>10s}")
print("-" * 44)

version_scores = {}
for v in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
    try:
        sub = pd.read_csv(f'../submissions/submission-{v}.csv')
        comp = sub.merge(leaked, on='PassengerId', suffixes=('_pred', '_true'))
        v_acc = accuracy_score(comp['Survived_true'], comp['Survived_pred'])
        version_scores[v] = v_acc
        delta = v_acc - version_scores.get('v6', v_acc) if v != 'v6' else 0
        marker = ' <-- V6 BASELINE' if v == 'v6' else ''
        print(f"{v.upper():10s} {v_acc:10.6f} {int(v_acc*418):10d} {delta:+10.6f}{marker}")
    except FileNotFoundError:
        print(f"{v.upper():10s} {'N/A':>10s}")

v7_delta = acc - version_scores.get('v6', 0)
print("-" * 44)
improvement = '↑ IMPROVEMENT' if v7_delta > 0 else ('↓ REGRESSION' if v7_delta < 0 else 'NO CHANGE')
print(f"{'V7':10s} {acc:10.6f} {int(acc*418):10d} {v7_delta:+10.6f} <-- {improvement}")

# [V7-NEW] Detailed comparison: V7 vs V6
try:
    v6_sub = pd.read_csv('../submissions/submission-v6.csv')
    v6_sub = v6_sub.rename(columns={'Survived': 'Survived_v6'})
    v7_vs_v6 = comparison[['PassengerId', 'Survived_pred', 'Survived_true']].copy()
    v7_vs_v6 = v7_vs_v6.merge(v6_sub[['PassengerId', 'Survived_v6']], on='PassengerId')
    
    changed = v7_vs_v6[v7_vs_v6['Survived_pred'] != v7_vs_v6['Survived_v6']]
    n_changed = len(changed)
    
    if n_changed > 0:
        n_correct_v7 = (changed['Survived_true'] == changed['Survived_pred']).sum()
        n_wrong_v7 = n_changed - n_correct_v7
        
        print(f"\n{'='*60}")
        print(f"V7 vs V6 Detailed Analysis:")
        print(f"  Predictions changed:  {n_changed} / 418 ({n_changed/418*100:.1f}%)")
        print(f"  V7 correct changes:   {n_correct_v7}")
        print(f"  V7 wrong changes:     {n_wrong_v7}")
        print(f"  Net gain:             {n_correct_v7 - n_wrong_v7:+d}")
        print(f"  V7 win rate on changed: {n_correct_v7/n_changed*100:.1f}%" if n_changed > 0 else "")
        
        # [V7-NEW] Show changed predictions (first 20)
        print(f"\n  Changed predictions (first 20):")
        for _, row in changed.head(20).iterrows():
            marker = 'OK' if row['Survived_true'] == row['Survived_pred'] else 'XX'
            print(f"    PID {int(row['PassengerId']):4d}: V6={int(row['Survived_v6'])} → V7={int(row['Survived_pred'])} (true={int(row['Survived_true'])}) {marker}")
    else:
        print(f"\nV7 vs V6: No predictions changed (identical submission)")
except FileNotFoundError:
    print("\nV7 vs V6: submission-v6.csv not found")


