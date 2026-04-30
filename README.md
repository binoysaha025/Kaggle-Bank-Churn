Binary Classification with a Bank Churn Dataset
This repository holds an attempt to apply a Random Forest classifier to predict customer churn using data from the Playground Series S4E1 Kaggle challenge.
Overview
The task, as defined by the Kaggle challenge, is to predict whether a bank customer will leave (churn) based on demographic and account information. This is a binary classification problem where the target variable Exited is 1 (churned) or 0 (stayed). The approach in this repository formulates the problem using a Random Forest classifier trained on 13 features including age, balance, credit score, geography, and account activity. Our best model achieved a ROC-AUC of 0.87 on the held-out test set. At the time of writing, the top score on the Kaggle leaderboard for this metric is approximately 0.92.

Summary of Workdone
Data

Type: Input: CSV file of numerical and categorical features. Output: binary flag Exited (1 = churned, 0 = stayed)
Size: ~165,000 training rows, ~110,000 test rows, 13 features
Split: 60% train (~99,000), 20% validation (~33,000), 20% test (~33,000). Kaggle test set (~110,000) used for final submission only.

Preprocessing / Cleanup

Dropped identifier columns id, CustomerId, and Surname — no predictive signal
One-hot encoded categorical features Geography (France/Germany/Spain) and Gender (Male/Female) using pd.get_dummies
Applied StandardScaler to numerical features: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary. Scaler was fit only on training data and applied to validation, test, and Kaggle test sets to prevent data leakage.
No missing values were found in the dataset.

Data Visualization
Histograms of each feature were plotted comparing churned vs stayed customers. Key findings:

Age showed the strongest separation — churned customers tend to be older (peak 40–50) vs stayed (peak ~30)
NumOfProducts — customers with 3–4 products churned at an extremely high rate; customers with 2 products almost never churned
IsActiveMember — inactive members churned at a disproportionately higher rate
Geography — Germany showed significantly higher churn rate proportionally despite fewer total customers
CreditScore, Tenure, and EstimatedSalary showed little separation between classes and are expected to contribute minimally

Problem Formulation

Input: 13 features (after dropping identifiers and encoding categoricals: 16 columns total)
Output: Probability of Exited = 1
Model: Random Forest Classifier — chosen for its robustness to unscaled features, resistance to overfitting, and strong out-of-the-box performance on tabular data
Hyperparameters: n_estimators=100, random_state=42

Training

Software: Python 3.14, scikit-learn, pandas, matplotlib
Hardware: Local CPU (Apple Silicon), n_jobs=-1 to parallelize across cores
Training time: Under 2 minutes
Random Forest does not have training curves (loss vs epoch) as it is not an iterative gradient-based method — all 100 trees are built independently and the model stops once all trees are constructed
No convergence issues or difficulties encountered

Performance Comparison

Metric: ROC-AUC (area under the receiver operating characteristic curve) — standard for binary classification with class imbalance (~80/20 split)

SplitROC-AUCValidation0.8800Test (held-out)0.8729Kaggle leaderboardTBD after submission
Conclusions

Random Forest is an effective baseline for this task, achieving 0.87 AUC with minimal tuning
Age, NumOfProducts, and IsActiveMember are the most predictive features based on visual analysis
The ~80/20 class imbalance did not require special handling for this model but could be addressed with techniques like SMOTE or class weighting for further improvement

Future Work

Try XGBoost or LightGBM which typically outperform Random Forest on tabular Kaggle competitions
Address class imbalance explicitly using class_weight='balanced' or oversampling
Feature engineering — e.g. interaction terms between Age and IsActiveMember
Hyperparameter tuning via GridSearchCV


How to Reproduce Results

Download train.csv and test.csv from the Kaggle competition page
Place them in the same directory as the notebook
Run notebook.ipynb top to bottom
submission.csv will be generated in the working directory — upload to Kaggle for leaderboard score


Overview of Files

notebook.ipynb — single notebook containing all steps: data loading, EDA, preprocessing, training, evaluation, and submission generation
train.csv — training data with labels (download from Kaggle)
test.csv — Kaggle test set without labels (download from Kaggle)
submission.csv — generated predictions for Kaggle submission


Software Setup
pandas
numpy
scikit-learn
matplotlib
Install with:
pip install pandas numpy scikit-learn matplotlib

Citations

Kaggle Playground Series S4E1: https://www.kaggle.com/competitions/playground-series-s4e1
Scikit-learn: Pedregosa et al., JMLR 12, pp. 2825–2830, 2011
