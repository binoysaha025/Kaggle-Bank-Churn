# Binary Classification with a Bank Churn Dataset

**Project Link:** https://www.kaggle.com/competitions/playground-series-s4e1

---

## Define Project

This Kaggle Playground challenge tasks participants with predicting whether a bank customer will churn (leave the bank) based on demographic and account data. It is a binary classification problem where the target variable `Exited` is encoded as 1 (churned) or 0 (stayed). The dataset is synthetically generated from a real bank churn dataset and contains approximately 165,000 training samples and 110,000 test samples with 13 features including age, account balance, credit score, geography, and account activity status. The evaluation metric is ROC-AUC.

---

## Data Loading and Initial Look

- Training set: 165,034 rows, 14 columns (13 features + target)
- Test set: 110,023 rows, 13 columns (no target)
- No missing values found in any column

| Feature | Type | Values/Range | Missing | Outliers |
|---|---|---|---|---|
| CreditScore | Numerical | 350 to 850 | 0 | Scores below 400 are rare but valid |
| Age | Numerical | 18 to 92 | 0 | Values above 70 are rare |
| Tenure | Numerical | 0 to 10 | 0 | None |
| Balance | Numerical | 0 to 250,000 | 0 | Large spike at 0 |
| NumOfProducts | Numerical | 1 to 4 | 0 | Values of 3 and 4 are very rare |
| EstimatedSalary | Numerical | 0 to 200,000 | 0 | None |
| HasCrCard | Categorical | 0, 1 | 0 | N/A |
| IsActiveMember | Categorical | 0, 1 | 0 | N/A |
| Geography | Categorical | France, Germany, Spain | 0 | N/A |
| Gender | Categorical | Male, Female | 0 | N/A |

Outlier definition: a data point more than 3 standard deviations from the mean for numerical features.

Class imbalance: approximately 80% stayed (Exited=0) and 20% churned (Exited=1). The target is encoded as integers 0 and 1.

---

## Data Visualization

**Plot 1: Overlapping histograms for numerical features (2x3 grid)**
<img width="1486" height="790" alt="image" src="https://github.com/user-attachments/assets/05aad4a9-8314-4e20-be32-2337a6a3581c" />


**Plot 2: Bar charts for categorical features (1x5 grid)**
<img width="1790" height="390" alt="image" src="https://github.com/user-attachments/assets/6b14d15a-4fbd-41e8-a87a-c11f535b5245" />

Key observations:
- Age shows the strongest separation. Churned customers peak around age 40 to 50 while stayed customers peak around age 30.
- NumOfProducts shows that customers with 3 or 4 products churn at an extremely high rate while customers with 2 products almost never churn.
- IsActiveMember shows inactive members (0) churn at a disproportionately higher rate than active members.
- Geography shows Germany has a notably higher churn rate proportionally despite having fewer total customers than France.
- CreditScore, Tenure, and EstimatedSalary show nearly identical distributions between classes and are expected to contribute minimally to model performance.

---

## Data Cleaning and Preparation for Machine Learning

**Dropped columns:** `id`, `CustomerId`, `Surname` - these are row identifiers with no predictive signal.

**One-hot encoding:** Applied `pd.get_dummies` to `Geography` and `Gender`, producing binary columns `Geography_France`, `Geography_Germany`, `Geography_Spain`, `Gender_Female`, `Gender_Male`.

**Rescaling:** Applied `StandardScaler` to numerical features: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary. Random Forest does not strictly require scaling but it is applied as best practice. Scaler was fit only on training data and applied to validation, test, and Kaggle test sets to prevent data leakage.

**Visualization before and after scaling:** Plot histograms of the 6 numerical features before scaling (raw values) and after scaling (centered around 0, unit variance) side by side to confirm the transformation.

---

## Machine Learning

### Problem Formulation

- Dropped `id`, `CustomerId`, `Surname` prior to modeling
- Original categorical columns `Geography` and `Gender` dropped after one-hot encoding
- Target `Exited` is already encoded as 0 and 1
- Split: 60% train, 20% validation, 20% test using `train_test_split` with `random_state=42`

### Train ML Algorithm

Model: Random Forest Classifier with `n_estimators=100`, `random_state=42`. Chosen for strong out-of-the-box performance on tabular data and no requirement for feature scaling.

### Evaluate Performance on Validation Sample

Metric: ROC-AUC score, which is the official Kaggle evaluation metric for this competition.

| Split | ROC-AUC |
|---|---|
| Validation | 0.8800 |
| Test (held-out) | 0.8729 |

Visualization: ROC curve plotted for the validation set showing true positive rate vs false positive rate across all thresholds.

### Apply ML to the Challenge Test Set

Model was applied to `test.csv` using `predict_proba` to generate churn probabilities. Output saved as `submission.csv` with columns `id` and `Exited` and uploaded to Kaggle for leaderboard scoring.
