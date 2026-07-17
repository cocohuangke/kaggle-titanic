# Titanic Survival Prediction — A Complete Tutorial from Theory to Practice

> **Target audience**: Data science and big data students with theoretical knowledge but limited hands-on experience  
> **Goal**: Master a structured methodology for tabular data competitions through a classic Kaggle case study  
> **Language**: Python (pandas + scikit-learn + LightGBM)  
> **Data location**: `../data/train.csv`, `../data/test.csv`

---

## Tutorial Outline

| Chapter | Topic | What You'll Learn |
|---------|-------|-------------------|
| [Chapter 0](#chapter-0-methodology-framework--from-theory-to-practice) | Methodology Framework | A universal 6-step competition pipeline |
| [Chapter 1](#chapter-1-environment-setup-and-data-loading) | Environment Setup | Python environment, Jupyter, pandas basics |
| [Chapter 2](#chapter-2-exploratory-data-analysis-eda) | Exploratory Data Analysis | Visualization, missing value analysis, correlation analysis |
| [Chapter 3](#chapter-3-data-preprocessing) | Data Preprocessing | Missing value imputation, encoding, data transforms |
| [Chapter 4](#chapter-4-feature-engineering) | Feature Engineering | Creating new features, feature selection |
| [Chapter 5](#chapter-5-model-building-and-evaluation) | Model Building & Evaluation | Multi-model comparison, cross-validation |
| [Chapter 6](#chapter-6-hyperparameter-tuning-and-model-ensembling) | Hyperparameter Tuning & Ensembling | GridSearch, Optuna, Stacking |
| [Chapter 7](#chapter-7-submission-and-retrospective) | Submission & Retrospective | Kaggle submission workflow, lessons learned |
| [Appendix A](#appendix-a-python-cheat-sheet) | Python Cheat Sheet | Syntax basics |
| [Appendix B](#appendix-b-pandas-cheat-sheet) | pandas Cheat Sheet | DataFrame operations |
| [Appendix C](#appendix-c-scikit-learn-cheat-sheet) | scikit-learn Cheat Sheet | Machine learning APIs |

---

# Chapter 0 Methodology Framework — From Theory to Practice

## 0.1 Why do you need a methodology?

You've probably heard in class: data cleaning → feature engineering → model training → evaluation. The workflow isn't wrong, but it's too rough. In a real competition (and in real jobs), you keep running into these questions:

- **Where do I start?** What's the first thing to do after getting the data?
- **When do I stop?** Is there an end to tuning?
- **How do I know I'm heading in the right direction?** Did my score go up or down after changing a feature?
- **How do I stay reproducible?** Will I remember what I did if I come back in three days?

A methodology is the framework that answers these questions.

## 0.2 Universal Competition Pipeline (6-Step Method)

The following process works not just for the Titanic dataset. **It works for any Kaggle competition involving structured (tabular) data**:

```
┌─────────────────────────────────────────────────────────────────┐
│              Universal Kaggle Competition Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Understand the Problem                                 │
│    ├─ Read the competition description, metric, and data docs   │
│    └─ Clarify: Classification or regression? What's the metric? │
│                          ↓                                      │
│  Step 2: Exploratory Data Analysis (EDA)                        │
│    ├─ Data overview: shape, types, missing values, distributions│
│    ├─ Univariate analysis: each feature vs. the target          │
│    └─ Multivariate analysis: feature interactions               │
│                          ↓                                      │
│  Step 3: Data Preprocessing                                     │
│    ├─ Missing values: drop or impute                            │
│    ├─ Outliers: detect and handle                               │
│    └─ Encoding: categorical → numeric                           │
│                          ↓                                      │
│  Step 4: Feature Engineering                                    │
│    ├─ Create new features from existing ones                    │
│    ├─ Feature transforms (log, binning, etc.)                   │
│    └─ Feature selection (remove redundancy)                     │
│                          ↓                                      │
│  Step 5: Modeling & Evaluation                                  │
│    ├─ Baseline: get a simple model through the full pipeline    │
│    ├─ Multi-model comparison: try several algorithms            │
│    ├─ Cross-validation: get reliable performance estimates      │
│    └─ Hyperparameter tuning: fine-tune                          │
│                          ↓                                      │
│  Step 6: Ensembling & Submission                                │
│    ├─ Model ensembling: voting / weighting / stacking           │
│    ├─ Generate submission file                                  │
│    └─ Retrospective and summary                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 0.3 Five Golden Rules

Before you start, remember these five lessons from Kaggle Grandmasters:

| Rule | What It Means | Common Mistake |
|------|---------------|----------------|
| **1. Trust CV over leaderboard** | Your CV score reflects your real performance. Don't chase the leaderboard. | Repeatedly submitting to game LB scores wastes time and overfits |
| **2. Preprocessing must happen inside the fold** | Imputation, encoding, and scaling should only use training fold data. Never peek at the validation fold. | Fitting an encoder on all data before splitting, causing data leakage |
| **3. Feature engineering > model selection** | One good feature beats switching between ten models | Debating XGBoost vs. LightGBM while ignoring feature quality |
| **4. Diverse models beat a single model** | Run at least 3 different model types before ensembling | Tuning a single model over and over |
| **5. Change one thing at a time** | Each experiment should change only one variable; record the result | Changing 3 parameters at once and not knowing which one helped |

## 0.4 Titanic Competition Overview

| Item | Details |
|------|---------|
| **Competition** | Titanic - Machine Learning from Disaster |
| **Problem type** | Binary classification (survived=1, perished=0) |
| **Metric** | Accuracy |
| **Training set** | 891 passengers, 12 fields |
| **Test set** | 418 passengers, 11 fields (no Survived column) |
| **Goal** | Predict whether each passenger in the test set survived |

### Field Descriptions

| Field | Meaning | Example |
|-------|---------|---------|
| PassengerId | Passenger ID | 1, 2, 3... |
| **Survived** | Whether the passenger survived (target variable) | 0=perished, 1=survived |
| Pclass | Ticket class | 1=1st, 2=2nd, 3=3rd |
| Name | Passenger name | "Braund, Mr. Owen Harris" |
| Sex | Gender | male, female |
| Age | Age | 22, 38, 26... |
| SibSp | Number of siblings/spouses aboard | 0, 1, 3... |
| Parch | Number of parents/children aboard | 0, 1, 2... |
| Ticket | Ticket number | "A/5 21171" |
| Fare | Fare paid | 7.25, 71.28... |
| Cabin | Cabin number | "C85", "E46"... |
| Embarked | Port of embarkation | C=Cherbourg, Q=Queenstown, S=Southampton |

---

# Chapter 1 Environment Setup and Data Loading

## 1.1 Install Required Python Libraries

If you haven't installed these yet, open a terminal and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install jupyter
pip install missingno        # Missing value visualization
pip install sweetviz          # Automated EDA reports
pip install lightgbm xgboost catboost  # Advanced models
pip install optuna            # Hyperparameter optimization
```

> **Tip**: Use a virtual environment to isolate project dependencies. If you're on Anaconda, create one with `conda create -n titanic python=3.10`.

## 1.2 Launch Jupyter Notebook

```bash
cd /path/to/titanic    # Navigate to the project directory
jupyter notebook       # Launch
```

Create a new Notebook in your browser and name it `titanic_tutorial.ipynb`.

## 1.3 Import Libraries

```python
# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Machine learning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Font settings (for Chinese labels, if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Display plots inline in Notebook
%matplotlib inline

# Set seaborn style
sns.set_style('whitegrid')
```

## 1.4 Load Data

```python
# Read CSV files
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Merge into one dataset for unified processing (we'll split again later)
# Tag the source first
train['Source'] = 'train'
test['Source'] = 'test'
test['Survived'] = np.nan  # Test set has no Survived column, fill with NaN

# Combine
full = pd.concat([train, test], axis=0, ignore_index=True)

print(f'Training set shape: {train.shape}')
print(f'Test set shape: {test.shape}')
print(f'Combined shape: {full.shape}')
```

Output:
```
Training set shape: (891, 13)
Test set shape: (418, 13)
Combined shape: (1309, 13)
```

## 1.5 First Look at the Data

```python
# View the first 5 rows
train.head()
```

```python
# View basic data info
train.info()
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
```

```python
# Summary statistics for numeric features
train.describe()
```

```python
# Summary statistics for categorical features
train.describe(include=['O'])
```

> **What to pay attention to**:
> - Which columns have missing values? → Age, Cabin, Embarked
> - Which are numeric? Which are categorical? → This determines your encoding strategy
> - What's the distribution of the target variable (Survived)? → Is it balanced?

---

# Chapter 2 Exploratory Data Analysis (EDA)

> **Core idea**: EDA isn't about making pretty charts. It's about **finding patterns and forming hypotheses**. Every chart should help you make a decision.

## 2.1 Target Variable Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Counts
train['Survived'].value_counts().plot(kind='bar', ax=axes[0], color=['#E24B4A', '#1D9E75'])
axes[0].set_title('Survival (Counts)')
axes[0].set_xlabel('0=Perished, 1=Survived')
for c in axes[0].containers:
    axes[0].bar_label(c)

# Proportions
train['Survived'].value_counts(normalize=True).plot(kind='pie', ax=axes[1],
    autopct='%1.1f%%', colors=['#E24B4A', '#1D9E75'])
axes[1].set_title('Survival (Proportions)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

**Finding**: About 61.6% perished and 38.4% survived. Slightly imbalanced, but not severely.

## 2.2 Missing Value Analysis

```python
# Method 1: Direct calculation with pandas
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df)
```

Output:
```
              Missing Count  Missing %
Cabin               687      77.10
Age                 177      19.87
Embarked              2       0.22
```

```python
# Method 2: Visualization with missingno (more intuitive)
msno.matrix(train, figsize=(12, 5))
plt.title('Missing Value Matrix')
plt.show()

msno.bar(train, figsize=(12, 4), color='steelblue')
plt.title('Missing Value Bar Chart')
plt.show()
```

**Findings and decisions**:
| Feature | Missing % | Strategy |
|---------|-----------|----------|
| Cabin | 77.1% | Too much missing. Extract a "has cabin" feature, then consider dropping |
| Age | 19.9% | Can't just drop it. Use smart imputation (group by title/class) |
| Embarked | 0.2% | Only 2 rows. Fill with the mode |

## 2.3 Categorical Features vs. Survival Rate

### Sex

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
sns.countplot(x='Sex', hue='Survived', data=train, ax=axes[0], palette='coolwarm')
axes[0].set_title('Survival by Sex')
for c in axes[0].containers:
    axes[0].bar_label(c)

# Survival rate
survival_by_sex = train.groupby('Sex')['Survived'].mean()
survival_by_sex.plot(kind='bar', ax=axes[1], color=['#E24B4A', '#1D9E75'])
axes[1].set_title('Survival Rate by Sex')
axes[1].set_ylabel('Survival Rate')
axes[1].set_ylim(0, 1)
for i, v in enumerate(survival_by_sex):
    axes[1].text(i, v + 0.03, f'{v:.1%}', ha='center')

plt.tight_layout()
plt.show()
```

**Key finding**: Female survival rate is about 74%, while male survival rate is only about 19%. **Sex is the single strongest predictor.**

### Ticket Class

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(x='Pclass', hue='Survived', data=train, ax=axes[0], palette='coolwarm')
axes[0].set_title('Survival by Ticket Class')

survival_by_pclass = train.groupby('Pclass')['Survived'].mean()
survival_by_pclass.plot(kind='bar', ax=axes[1], color=['#FFD700', '#C0C0C0', '#CD7F32'])
axes[1].set_title('Survival Rate by Ticket Class')
axes[1].set_ylabel('Survival Rate')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

**Key finding**: 1st class survival rate was 63%, 2nd class 47%, and 3rd class only 24%. Social status influenced who survived.

### Sex × Class (Cross Analysis)

```python
pivot = train.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', linewidths=0.5)
plt.title('Survival Rate by Sex and Class')
plt.show()
```

**Key finding**: 1st class women had a 96% survival rate, while 3rd class men had only 13%. The combined effect of both factors is clear.

### Port of Embarkation

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x='Embarked', hue='Survived', data=train, ax=axes[0], palette='coolwarm')
axes[0].set_title('Survival by Port of Embarkation')

train.groupby('Embarked')['Survived'].mean().plot(kind='bar', ax=axes[1], color='teal')
axes[1].set_title('Survival Rate by Port of Embarkation')
axes[1].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()
```

**Key finding**: Cherbourg (C) had the highest survival rate at 55%, likely because more 1st class passengers boarded there.

## 2.4 Numeric Features vs. Survival Rate

### Age

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes[0,0].hist(train['Age'].dropna(), bins=30, color='steelblue', edgecolor='white')
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel('Age')

# Age distribution split by survival
train[train['Survived']==1]['Age'].hist(bins=30, alpha=0.6, label='Survived', color='#1D9E75', ax=axes[0,1])
train[train['Survived']==0]['Age'].hist(bins=30, alpha=0.6, label='Perished', color='#E24B4A', ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Survival')
axes[0,1].legend()

# Violin plot
sns.violinplot(x='Survived', y='Age', data=train, hue='Survived',
               palette=['#E24B4A', '#1D9E75'], ax=axes[1,0], legend=False, inner='quartile')
axes[1,0].set_title('Age vs. Survival (Violin Plot)')

# Survival rate by age group
train['AgeBin'] = pd.cut(train['Age'], bins=[0, 5, 12, 18, 30, 50, 80],
                         labels=['Infant', 'Child', 'Teen', 'Young Adult', 'Middle Age', 'Senior'])
train.groupby('AgeBin', observed=False)['Survived'].mean().plot(kind='bar', ax=axes[1,1], color='teal')
axes[1,1].set_title('Survival Rate by Age Group')
axes[1,1].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()

# Clean up temporary column
train.drop('AgeBin', axis=1, inplace=True)
```

**Key finding**: Infants (0-5 years old) had a noticeably higher survival rate than other age groups, reflecting the "women and children first" principle.

### Fare

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Fare distribution (note the skew)
axes[0].hist(train['Fare'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Fare Distribution (Raw)')
axes[0].set_xlabel('Fare')

# After log transform
axes[1].hist(np.log1p(train['Fare']), bins=50, color='teal', edgecolor='white')
axes[1].set_title('Fare Distribution (Log Transformed)')
axes[1].set_xlabel('log(Fare + 1)')

plt.tight_layout()
plt.show()

# Skewness of fare
print(f"Fare skewness (raw): {train['Fare'].skew():.2f}")
print(f"Fare skewness (log transformed): {np.log1p(train['Fare']).skew():.2f}")
```

**Key finding**: Fare is heavily right-skewed (skew≈4.5). After log transformation, it's close to normal (skew≈0.6). Skewed data affects some models, so we'll apply a log transform later.

### Family Size (SibSp + Parch)

```python
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1  # +1 for the passenger themselves

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(x='FamilySize', hue='Survived', data=train, ax=axes[0], palette='coolwarm')
axes[0].set_title('Survival by Family Size')

train.groupby('FamilySize')['Survived'].mean().plot(kind='bar', ax=axes[1], color='teal')
axes[1].set_title('Survival Rate by Family Size')
axes[1].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()

# Clean up temporary column
train.drop('FamilySize', axis=1, inplace=True)
```

**Key finding**: Small families (2-4 people) had the highest survival rate. Solo travelers and large families both had lower rates. This suggests moderate-sized groups helped each other, while being alone or in a very large group was a disadvantage.

## 2.5 Correlation Heatmap

```python
# Encode categorical features first so we can compute correlations
train_corr = train.copy()
train_corr['Sex_encoded'] = (train_corr['Sex'] == 'female').astype(int)
train_corr['Has_Cabin'] = train_corr['Cabin'].notna().astype(int)

corr_cols = ['Survived', 'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Has_Cabin']
corr_matrix = train_corr[corr_cols].corr()

# Show lower triangle only
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
```

**Key findings** (sorted by correlation with Survived):

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| Sex_encoded | **+0.54** | Strongest positive: female → more likely to survive |
| Pclass | **-0.34** | Negative: higher class number (lower status) → more likely to perish |
| Has_Cabin | **+0.32** | Has cabin number → more likely to survive (mostly 1st class) |
| Fare | **+0.26** | Higher fare → more likely to survive |
| Age | **-0.08** | Weak negative (but masks the nonlinear effect for children) |

> **Note**: Pearson correlation only captures linear relationships. The nonlinear pattern of high child survival gets "averaged out" in a single coefficient. This is why EDA requires more than just numbers. **You have to plot the data**.

## 2.6 EDA Summary: Key Findings

| Finding | Follow-up Action |
|---------|-----------------|
| Female survival rate far exceeds male | Sex is the most important feature. Keep it |
| 1st class > 2nd class > 3rd class | Pclass is a strong predictor |
| Infants had high survival rates | Consider extracting titles from Name (Master/Miss) |
| Small families (2-4) had highest survival | Build a FamilySize feature |
| Fare is heavily right-skewed | Needs log transformation |
| Cabin is 77% missing | Extract "has cabin" or "deck" feature, then drop the raw column |
| Age is 20% missing | Smart imputation grouped by title/class |

---

# Chapter 3 Data Preprocessing

> **Core idea**: The goal of data preprocessing is to make the data "clean" and "model-readable." Clean means handling missing values and outliers. Readable means converting everything to numbers.

## 3.1 Combined Processing Strategy

To avoid inconsistencies between the training and test sets, we merge them and process everything together:

```python
# Reload data (to avoid contamination from EDA modifications)
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Tag the source
train['Source'] = 'train'
test['Source'] = 'test'
test['Survived'] = np.nan

full = pd.concat([train, test], axis=0, ignore_index=True)
print(f'Combined data shape: {full.shape}')
```

## 3.2 Missing Value Handling

### Embarked (only 2 missing)

```python
# Check the 2 missing records
full[full['Embarked'].isnull()]
```

Both are 1st class women who paid a fare of 80. Most likely they boarded at Cherbourg (C):

```python
# Fill with the mode (most passengers in 1st class boarded at S, but the difference is minimal)
full['Embarked'] = full['Embarked'].fillna('S')  # Could also use 'C', makes little difference
```

### Fare (1 missing in test set)

```python
# Check the missing record
full[full['Fare'].isnull()]
```

```python
# Fill with the median fare for the same class and port
fare_median = full[(full['Pclass'] == 3) & (full['Embarked'] == 'S')]['Fare'].median()
full['Fare'] = full['Fare'].fillna(fare_median)
```

### Age (roughly 20% missing)

Age has a lot of missing values. A simple mean imputation won't cut it. We'll use **title-grouped imputation**:

```python
# Extract titles first (we'll need this for feature engineering later)
full['Title'] = full['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Check median age by title
full.groupby('Title')['Age'].median().head(10)
```

```python
# Fill missing ages with the title-group median
title_age_median = full.groupby('Title')['Age'].transform('median')
full['Age'] = full['Age'].fillna(title_age_median)

# Verify: are there still missing values?
print(f"Age missing count: {full['Age'].isnull().sum()}")
```

> **Why is title-based imputation better than mean imputation?** Because title (Mr, Mrs, Miss, Master) correlates strongly with age. "Master" is for young boys (median age ~3.5), while "Mr" is for adult men (median ~30). If you fill everyone with the global mean of 28, you'd turn a 3.5-year-old into a 28-year-old. Completely wrong.

### Cabin (77% missing)

```python
# Extract deck info (first letter), mark missing as 'U' (Unknown)
full['Deck'] = full['Cabin'].fillna('U').apply(lambda x: x[0])
print(full['Deck'].value_counts())
```

Cabin has too much missing data, so we extract just the deck letter and drop the raw Cabin column later.

## 3.3 Data Type Conversion

```python
# Pclass is numeric but actually categorical
full['Pclass'] = full['Pclass'].astype(str)

# Check data types
full.dtypes
```

## 3.4 Encoding Categorical Variables

Machine learning models can only work with numbers, so we need to convert text categories to numeric values:

### Label Encoding (ordinal or binary variables)

```python
# Sex: male→1, female→0 (female has higher survival; we'll one-hot encode later)
full['Sex'] = full['Sex'].map({'male': 1, 'female': 0})
```

### One-Hot Encoding (nominal categorical variables)

```python
# One-hot encode Embarked and Pclass
full = pd.get_dummies(full, columns=['Embarked', 'Pclass', 'Deck', 'Title'], drop_first=False)
```

> **What is drop_first?** For binary variables, one-hot encoding creates two perfectly correlated columns (e.g., Embarked_C and Embarked_S. Knowing one tells you the other), causing multicollinearity. But for tree-based models (Random Forest, LightGBM), keeping all columns is fine. For logistic regression, use `drop_first=True`.

## 3.5 Drop Useless Features

```python
# Remove features that don't help with modeling
full = full.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Source'], axis=1)
```

| Reason for Removal | Feature |
|--------------------|---------|
| Pure ID | PassengerId |
| Info already extracted | Name (extracted Title), Cabin (extracted Deck) |
| Too messy to use | Ticket (inconsistent formats) |
| Helper tag | Source |

## 3.6 Split Back into Training and Test Sets

```python
# Split
X_train = full[full.index < len(train)].drop('Survived', axis=1)
y_train = train['Survived']
X_test = full[full.index >= len(train)].drop('Survived', axis=1)

print(f'Training features: {X_train.shape}, Training labels: {y_train.shape}')
print(f'Test features: {X_test.shape}')
```

---

# Chapter 4 Feature Engineering

> **Core idea**: Feature engineering is the art of squeezing information out of your data. As the saying goes in the industry: **"Data and features set the ceiling for what machine learning can achieve. Models and algorithms just get close to that ceiling."**

## 4.1 What is Feature Engineering?

Feature engineering means **using your human knowledge and creativity to construct more useful signals from raw data**.

Here's an example: the raw data has SibSp (siblings/spouses) and Parch (parents/children). The model might not figure out on its own that these should be added together. By creating `FamilySize = SibSp + Parch + 1`, you're helping the model understand the concept of family size.

## 4.2 Titanic Feature Engineering Checklist

Let's start fresh and systematically engineer features on the combined dataset:

```python
# Reload clean data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train['Source'] = 'train'
test['Source'] = 'test'
test['Survived'] = np.nan

full = pd.concat([train, test], axis=0, ignore_index=True)
```

### Feature 1: Title — Extract from Name

```python
# Extract title with regex
full['Title'] = full['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Check title distribution
full['Title'].value_counts()
```

Output:
```
Mr          757
Miss        260
Mrs         197
Master       61
Dr            8
Rev           8
Col           4
Major         2
Ms            2
Mlle          2
...
```

Many titles appear only a few times and need to be consolidated:

```python
# Consolidate rare titles
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
full['Title'] = full['Title'].replace(rare_titles, 'Rare')

# Standardize French titles
full['Title'] = full['Title'].replace(['Mlle', 'Ms'], 'Miss')
full['Title'] = full['Title'].replace('Mme', 'Mrs')

# Verify
full['Title'].value_counts()
```

### Feature 2: FamilySize

```python
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1

# Visualize family size vs. survival rate
train_temp = full[full['Source'] == 'train'].copy()
train_temp.groupby('FamilySize')['Survived'].mean().plot(kind='bar', color='teal')
plt.title('Survival Rate by Family Size')
plt.ylabel('Survival Rate')
plt.show()
```

### Feature 3: IsAlone

```python
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)

# Survival rate: alone vs. not alone
train_temp = full[full['Source'] == 'train'].copy()
print(f"Alone survival rate: {train_temp[train_temp['IsAlone']==1]['Survived'].mean():.2%}")
print(f"Not alone survival rate: {train_temp[train_temp['IsAlone']==0]['Survived'].mean():.2%}")
```

### Feature 4: Deck — Extract from Cabin

```python
full['Deck'] = full['Cabin'].fillna('U').apply(lambda x: x[0])

# Check deck vs. survival rate
train_temp = full[full['Source'] == 'train'].copy()
train_temp.groupby('Deck')['Survived'].mean().plot(kind='bar', color='teal')
plt.title('Survival Rate by Deck')
plt.ylabel('Survival Rate')
plt.show()
```

### Feature 5: FarePerPerson

```python
full['FarePerPerson'] = full['Fare'] / full['FamilySize']
```

### Feature 6: AgeGroup

```python
# First, fill missing Age values (grouped by title median)
full['Age'] = full.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# Age groups
full['AgeGroup'] = pd.cut(full['Age'], bins=[0, 5, 12, 18, 35, 60, 100],
                          labels=[0, 1, 2, 3, 4, 5])  # Numeric labels for modeling
full['AgeGroup'] = full['AgeGroup'].astype(int)
```

### Feature 7: Has_Cabin

```python
full['Has_Cabin'] = full['Cabin'].notna().astype(int)
```

### Feature 8: Log Transform of Fare

```python
full['Fare_log'] = np.log1p(full['Fare'])
```

> **Why log transform?** Fare is heavily right-skewed (skew≈4.5). Log transform brings it close to a normal distribution. Linear models like logistic regression are sensitive to skew; tree models aren't affected either way.

## 4.3 Complete Preprocessing and Encoding

```python
# Fill remaining missing values
full['Embarked'] = full['Embarked'].fillna('S')
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
full['Fare_log'] = np.log1p(full['Fare'])
full['FarePerPerson'] = full['Fare'] / full['FamilySize']

# Encode sex
full['Sex'] = full['Sex'].map({'male': 1, 'female': 0})

# Drop columns no longer needed
full = full.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Source'], axis=1)

# One-hot encoding
full = pd.get_dummies(full, columns=['Embarked', 'Title', 'Deck'], drop_first=False)

# Confirm no missing values remain
print(f"Remaining missing values: {full.isnull().sum().sum()}")

# Split into train/test
X = full[full.index < len(train)].drop('Survived', axis=1)
y = train['Survived']
X_test = full[full.index >= len(train)].drop('Survived', axis=1)

print(f'Training set: {X.shape}, Test set: {X_test.shape}')
```

## 4.4 Principles of Feature Engineering

| Principle | Description |
|-----------|-------------|
| **Domain knowledge first** | "Women and children first" → extract titles, age groups |
| **Combine to create information** | SibSp + Parch → FamilySize → IsAlone |
| **Compress redundant info** | Cabin is 77% missing → keep only "has cabin" and "deck" |
| **Transform away skew** | Fare → Fare_log |
| **Bold hypothesis, careful verification** | After adding each feature, run CV and check if the score improves |

---

# Chapter 5 Model Building and Evaluation

> **Core idea**: Get it working first, then optimize. Start by building a working end-to-end pipeline. Don't try to max out your score right away.

## 5.1 Baseline Model

The purpose of a baseline: **if you can't beat "guess everyone perished," your model is useless.**

```python
# Simplest baseline: everyone perishes
baseline_acc = 1 - y.mean()  # Perish rate
print(f"Baseline accuracy (guess all perished): {baseline_acc:.4f}")
```

Output: `Baseline accuracy: 0.6162` → Your model must beat 61.6% to be useful.

## 5.2 Multi-Model Comparison

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
}

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"{name:25s} | Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

Expected output (approximate range):
```
Logistic Regression       | Accuracy: 0.82xx ± 0.01xx
Decision Tree             | Accuracy: 0.78xx ± 0.02xx
Random Forest             | Accuracy: 0.82xx ± 0.02xx
Gradient Boosting         | Accuracy: 0.83xx ± 0.02xx
Extra Trees               | Accuracy: 0.80xx ± 0.02xx
SVM (RBF)                 | Accuracy: 0.66xx ± 0.03xx
KNN                       | Accuracy: 0.73xx ± 0.02xx
Naive Bayes               | Accuracy: 0.77xx ± 0.02xx
LightGBM                  | Accuracy: 0.83xx ± 0.02xx
```

## 5.3 Visualize the Comparison

```python
# Sort and plot
sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
names = [r[0] for r in sorted_results]
means = [r[1]['mean'] for r in sorted_results]
stds = [r[1]['std'] for r in sorted_results]

plt.figure(figsize=(12, 6))
bars = plt.barh(range(len(names)), means, xerr=stds, color='steelblue', alpha=0.8)
plt.yticks(range(len(names)), names)
plt.xlabel('Cross-Validation Accuracy')
plt.title('Model Comparison (5-Fold Stratified CV)')
plt.axvline(x=baseline_acc, color='red', linestyle='--', label=f'Baseline ({baseline_acc:.4f})')
plt.legend()

# Annotate values
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(m + 0.005, i, f'{m:.4f}', va='center')

plt.tight_layout()
plt.show()
```

## 5.4 Why Use Cross-Validation?

| Method | Problem |
|--------|---------|
| Only look at training accuracy | Overfitting! The model may have "memorized" the answers |
| Single train/test split | Results depend on the random seed. Unstable |
| **K-fold cross-validation** | Every data point gets used as validation. Results are reliable |

**The 5-fold CV process:**
```
Round 1: [VALID][ Train ][ Train ][ Train ][ Train ] → Score 1
Round 2: [ Train ][VALID][ Train ][ Train ][ Train ] → Score 2
Round 3: [ Train ][ Train ][VALID][ Train ][ Train ] → Score 3
Round 4: [ Train ][ Train ][ Train ][VALID][ Train ] → Score 4
Round 5: [ Train ][ Train ][ Train ][ Train ][VALID] → Score 5
Final score = Mean ± Std
```

> **Why StratifiedKFold?** Because the survival rate is about 38%. If you split randomly, some folds might be all perishers. StratifiedKFold ensures each fold has the same survival/perish ratio as the full dataset.

## 5.5 Pick the Best Single Model

Based on cross-validation results, pick the top 2-3 models for further tuning. Usually **LightGBM + Random Forest + Logistic Regression** is a solid combination.

---

# Chapter 6 Hyperparameter Tuning and Model Ensembling

## 6.1 Hyperparameter Tuning

### What Are Hyperparameters?

Hyperparameters are settings you choose **before training** (like number of trees, max depth), as opposed to parameters the model learns during training (like weights).

| Concept | Who Decides | Example |
|---------|-------------|---------|
| Parameters | Learned from data by the model | Linear regression coefficients |
| Hyperparameters | Set manually by you | Random Forest tree count, max depth |

### Method 1: GridSearchCV (Exhaustive Search)

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=skf, scoring='accuracy',
                           n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### Method 2: Optuna (Smarter Search)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best score: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

> **GridSearch vs Optuna**: GridSearch exhaustively tries every combination. Slow when the grid is large. Optuna uses Bayesian optimization to search smartly, usually finding good parameters faster.

## 6.2 Model Ensembling

### Why Does Ensembling Work?

Different models learn different things:
- Logistic Regression picks up **linear relationships** (e.g., "female → more likely to survive")
- Random Forest captures **nonlinear interactions** (e.g., "3rd class male but with a small family")
- LightGBM may learn **subtle gradient patterns**

Combining their predictions usually outperforms any single model.

### Ensemble Method 1: Hard Voting

```python
from sklearn.ensemble import VotingClassifier

# Pick the top 3 models
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
    ],
    voting='hard'  # hard = majority vote, soft = probability average
)

scores = cross_val_score(voting_clf, X, y, cv=skf, scoring='accuracy')
print(f"Voting ensemble accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Ensemble Method 2: Soft Voting (Probability-Weighted Average)

```python
# Soft voting requires models that output probabilities
voting_clf_soft = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ],
    voting='soft'  # Probability-based weighted average
)

scores = cross_val_score(voting_clf_soft, X, y, cv=skf, scoring='accuracy')
print(f"Soft voting ensemble accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Ensemble Method 3: Stacking

```python
from sklearn.ensemble import StackingClassifier

# Define base models
base_models = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
]

# Meta-model (second layer. Uses base model predictions as features)
meta_model = LogisticRegression(max_iter=1000, random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False  # Whether to pass original features to the meta-model too
)

scores = cross_val_score(stacking_clf, X, y, cv=skf, scoring='accuracy')
print(f"Stacking ensemble accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## 6.3 Manual Weighted Ensemble (Advanced)

```python
# Train each model separately and get prediction probabilities
from sklearn.model_selection import cross_val_predict

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get OOF (Out-of-Fold) predictions
lgbm_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

lgbm_oof = cross_val_predict(lgbm_model, X, y, cv=skf, method='predict_proba')[:, 1]
rf_oof = cross_val_predict(rf_model, X, y, cv=skf, method='predict_proba')[:, 1]
gb_oof = cross_val_predict(gb_model, X, y, cv=skf, method='predict_proba')[:, 1]
lr_oof = cross_val_predict(lr_model, X, y, cv=skf, method='predict_proba')[:, 1]

# Try different weight combinations
best_acc = 0
best_weights = None

for w1 in np.arange(0.1, 0.6, 0.1):
    for w2 in np.arange(0.1, 0.5, 0.1):
        for w3 in np.arange(0.1, 0.5, 0.1):
            w4 = 1 - w1 - w2 - w3
            if w4 <= 0:
                continue
            blend = w1 * lgbm_oof + w2 * rf_oof + w3 * gb_oof + w4 * lr_oof
            acc = accuracy_score(y, (blend >= 0.5).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2, w3, w4)

print(f"Best ensemble weights: LGBM={best_weights[0]:.1f}, RF={best_weights[1]:.1f}, "
      f"GB={best_weights[2]:.1f}, LR={best_weights[3]:.1f}")
print(f"Best ensemble accuracy: {best_acc:.4f}")
```

## 6.4 Generate Final Predictions

```python
# Retrain all models with best weights (using all training data)
lgbm_model.fit(X, y)
rf_model.fit(X, y)
gb_model.fit(X, y)
lr_model.fit(X, y)

# Predict with each model
lgbm_pred = lgbm_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict_proba(X_test)[:, 1]
gb_pred = gb_model.predict_proba(X_test)[:, 1]
lr_pred = lr_model.predict_proba(X_test)[:, 1]

# Weighted blend
w1, w2, w3, w4 = best_weights
final_pred = w1 * lgbm_pred + w2 * rf_pred + w3 * gb_pred + w4 * lr_pred
final_labels = (final_pred >= 0.5).astype(int)
```

---

# Chapter 7 Submission and Retrospective

## 7.1 Generate Submission File

Kaggle requires a CSV file formatted like `gender_submission.csv`:

```python
# Read the test set PassengerId
test_original = pd.read_csv('../data/test.csv')

submission = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],
    'Survived': final_labels
})

submission.to_csv('../submission.csv', index=False)
print("Submission file generated: ../submission.csv")
print(submission.head(10))
```

## 7.2 Submit to Kaggle

1. Log in to [Kaggle](https://www.kaggle.com/)
2. Go to the [Titanic competition page](https://www.kaggle.com/c/titanic)
3. Click "Submit Predictions"
4. Upload `submission.csv`
5. Check your leaderboard score

> **Score reference**:
> - 0.6162 = Guess all perished (baseline)
> - 0.76~0.78 = Simple models
> - 0.78~0.80 = Reasonable feature engineering + tuning
> - 0.80~0.83 = Careful feature engineering + ensembling
> - 0.83+ = Advanced feature engineering + fine-tuned ensembling

## 7.3 Retrospective Checklist

After submitting, ask yourself these questions:

| Dimension | Retrospective Question |
|-----------|----------------------|
| **Data understanding** | Do I really understand what each feature means? Any domain knowledge I missed? |
| **EDA** | Was my EDA deep enough? Any important patterns I missed? |
| **Feature engineering** | Which features contributed most? (Check feature_importance) Which ones can I drop? |
| **Models** | Did I only try one model type? Are there others that might work better? |
| **Validation** | Is there a big gap between CV and LB scores? If so, your CV strategy may need fixing |
| **Room for improvement** | What could I try next? |

### Check Feature Importance

```python
# Use the trained model to check feature importance
importances = lgbm_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
feat_imp.tail(20).plot(kind='barh', color='steelblue')
plt.title('LightGBM Feature Importance (Top 20)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

## 7.4 Directions for Further Improvement

| Direction | Specific Actions |
|-----------|-----------------|
| **More features** | Ticket frequency encoding, name length, cabin count (multiple cabins per person) |
| **Smarter Age imputation** | Use a model to predict missing ages instead of median |
| **More granular tuning** | Run 50-100 Optuna trials per model individually |
| **More models in the ensemble** | Add XGBoost, CatBoost |
| **Multi-level stacking** | Use LightGBM as the second-layer model instead of Logistic Regression |
| **Feature selection** | Use RFE or SHAP values to remove noisy features |

---

# Appendix A Python Cheat Sheet

## A.1 Basic Data Structures

```python
# List
fruits = ['apple', 'banana', 'cherry']
fruits.append('date')       # Add element
fruits[0]                    # Index → 'apple'
fruits[1:3]                  # Slice → ['banana', 'cherry']

# Dictionary
person = {'name': 'Alice', 'age': 25}
person['name']               # → 'Alice'
person['city'] = 'Beijing'   # Add key-value pair

# List comprehension
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

## A.2 Common Operations

```python
# Conditionals
if x > 0:
    print('Positive')
elif x == 0:
    print('Zero')
else:
    print('Negative')

# Loops
for i, row in df.iterrows():
    # Iterate over each row in the DataFrame
    pass

# Functions
def fill_age_by_title(df):
    return df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
```

---

# Appendix B pandas Cheat Sheet

## B.1 Reading and Inspecting

```python
df = pd.read_csv('file.csv')       # Read CSV
df.head()                           # First 5 rows
df.tail()                           # Last 5 rows
df.shape                            # (rows, columns)
df.info()                           # Data types and missing values
df.describe()                       # Numeric summary stats
df.describe(include='O')            # Categorical summary stats
df.columns.tolist()                 # Column names as list
df.dtypes                           # Data types
```

## B.2 Selection and Filtering

```python
# Select columns
df['Age']                  # Single column → Series
df[['Age', 'Sex']]         # Multiple columns → DataFrame

# Select rows
df[df['Age'] > 30]         # Conditional filter
df.iloc[0:10]              # Index by position
df.loc[0:10, 'Age']        # Index by label

# Combined conditions
df[(df['Age'] > 30) & (df['Sex'] == 'female')]
```

## B.3 Missing Values

```python
df.isnull().sum()                    # Missing count per column
df.isnull().mean()                   # Missing proportion per column
df.dropna()                          # Drop rows with any missing values
df.dropna(subset=['Age'])            # Drop rows where Age is missing
df['Age'].fillna(df['Age'].median()) # Fill with median
df['Age'].fillna(method='ffill')     # Forward fill
```

## B.4 Grouping and Aggregation

```python
df.groupby('Sex')['Survived'].mean()           # Survival rate by sex
df.groupby(['Sex', 'Pclass'])['Survived'].mean()  # Multi-level grouping
df.groupby('Title')['Age'].transform('median')     # Returns same-length result as original DataFrame
df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
```

## B.5 Feature Operations

```python
df['new_col'] = df['col1'] + df['col2']        # Create new column
df.rename(columns={'old': 'new'})               # Rename columns
df.drop('col', axis=1)                          # Drop column
df.astype({'col': 'int'})                       # Type conversion
pd.get_dummies(df, columns=['Sex'])             # One-hot encoding
df['col'].map({'male': 1, 'female': 0})         # Value mapping
pd.cut(df['Age'], bins=5)                       # Equal-width binning
pd.qcut(df['Fare'], q=4)                        # Equal-frequency binning
```

---

# Appendix C scikit-learn Cheat Sheet

## C.1 General Workflow

```python
# 1. Create model
model = SomeModel(param1=value1, param2=value2)

# 2. Train
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)                    # Class labels
y_prob = model.predict_proba(X_test)[:, 1]        # Probabilities

# 4. Evaluate
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

## C.2 Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## C.3 Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## C.4 Common Classifiers Quick Reference

| Model | Import | Characteristics |
|-------|--------|-----------------|
| Logistic Regression | `from sklearn.linear_model import LogisticRegression` | Linear baseline |
| Decision Tree | `from sklearn.tree import DecisionTreeClassifier` | Interpretable |
| Random Forest | `from sklearn.ensemble import RandomForestClassifier` | Consistently strong |
| Gradient Boosting | `from sklearn.ensemble import GradientBoostingClassifier` | High accuracy |
| SVM | `from sklearn.svm import SVC` | Works well on small data |
| KNN | `from sklearn.neighbors import KNeighborsClassifier` | Simple and intuitive |
| Naive Bayes | `from sklearn.naive_bayes import GaussianNB` | Fast |
| LightGBM | `import lightgbm as lgb` | Fast + accurate |
| XGBoost | `from xgboost import XGBClassifier` | Competition powerhouse |
| CatBoost | `from catboost import CatBoostClassifier` | Good with categorical features |

## C.5 Pipeline

Pipelines bundle preprocessing and modeling together, preventing data leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Use pipeline directly in cross-validation. Each fold will refit the preprocessing
scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
```

---

# Closing

Congratulations on completing the entire Titanic project! Let's recap the path you took:

```
Understand → EDA → Preprocessing → Feature Engineering → Modeling → Tuning → Ensemble → Submit
```

This pipeline isn't exclusive to the Titanic. **It's a methodology you can apply to any structured data competition**.

When you face your next Kaggle competition:
1. Ask yourself: What kind of problem is this? What's the evaluation metric?
2. Run EDA to understand the data
3. Build a baseline and get the full pipeline running
4. Feature engineer. This is where you'll find the most leverage
5. Compare multiple models with cross-validation
6. Ensemble for the final boost

**Most important**: Don't try to get it perfect the first time. Get it running first, then iterate. Change one thing at a time and log your results.

Good luck on your Kaggle journey!

---

> **References**:
> - [Kaggle Titanic Competition Page](https://www.kaggle.com/c/titanic)
> - [Comprehensive Beginner's Guide - Towards Data Science](https://towardsdatascience.com/comprehensive-beginners-guide-to-kaggle-titanic-survival-prediction-competition-solution-21c5be2cec2c/)
> - [Kaggle Fundamentals - Dataquest](https://www.dataquest.io/blog/kaggle-fundamentals/)
> - [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
> - [Feature Engineering & Modeling in Python - Towards Data Science](https://towardsdatascience.com/titanic-feature-engineering-modeling-in-python-6749e6e87bf4/)