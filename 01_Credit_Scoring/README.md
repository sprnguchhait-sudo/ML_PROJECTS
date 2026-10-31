# Task 1 — Credit Scoring Model

A reproducible binary classification demonstration. It generates synthetic financial records so it runs immediately without uploading personal financial data.

## Run

```bash
pip install -r requirements.txt
python train.py
```

The script creates a stratified train/test split, imputes missing values, scales numeric features, one-hot encodes categorical features, trains Logistic Regression and Random Forest, and prints precision, recall, F1, ROC-AUC and a confusion matrix. The best ROC-AUC pipeline is saved as `credit_scoring_model.joblib`.

## Replace the sample data

Edit `load_data()` in `train.py` to load a CSV with target column `creditworthy` (1 = creditworthy, 0 = not creditworthy). Use predictors such as annual income, debt-to-income ratio, credit history length, late payment count and employment category. Keep sensitive/protected attributes out of the model unless they are being used in a lawful fairness audit.

Synthetic data is only for demonstrating the workflow; model scores on it do not establish real creditworthiness.
