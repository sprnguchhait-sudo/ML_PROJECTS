# Task 4 — Disease Prediction from Medical Data

An educational classification workflow using scikit-learn's built-in Wisconsin Diagnostic Breast Cancer dataset. The example predicts the dataset's benign/malignant label from numeric measurements and compares Logistic Regression, SVM and Random Forest.

## Run

```bash
pip install -r requirements.txt
python train.py
```

It uses a stratified hold-out test set and reports accuracy, precision, recall, F1, ROC-AUC and a confusion matrix. The best ROC-AUC model is saved to `disease_demo_model.joblib`.

## Safety

This is a classroom demonstration on a public benchmark dataset. It is not a medical device and cannot diagnose, rule out, or predict disease for an individual. Do not use it to make clinical decisions. Real clinical applications require appropriate representative data, external validation, clinical expertise, privacy controls and regulatory review.
