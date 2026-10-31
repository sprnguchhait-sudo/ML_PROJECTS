"""Educational credit scoring classification pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def load_data(n=2500):
    """Create reproducible synthetic data; replace with an approved real dataset."""
    rng = np.random.default_rng(RANDOM_STATE)
    income = rng.lognormal(mean=10.8, sigma=0.45, size=n)
    debt = rng.uniform(0.02, 0.85, size=n)
    history = rng.integers(1, 31, size=n)
    late = rng.poisson(1.1, size=n)
    employment = rng.choice(["employed", "self_employed", "contract"], size=n,
                            p=[0.68, 0.20, 0.12])
    logit = (1.3 + 0.000018 * (income - 50000) - 3.2 * debt
             + 0.055 * history - 0.72 * late
             + np.where(employment == "employed", 0.25, 0.0)
             + rng.normal(0, 0.65, n))
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    X = pd.DataFrame({"annual_income": income, "debt_to_income": debt,
                      "credit_history_years": history, "late_payments": late,
                      "employment_type": employment})
    return X, pd.Series(y, name="creditworthy")


def main():
    X, y = load_data()
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                                  class_weight="balanced", random_state=RANDOM_STATE)
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    best_name, best_auc, best_pipeline = None, -1.0, None
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, score)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=["Not creditworthy", "Creditworthy"],
                                    zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print(f"ROC-AUC: {auc:.4f}")
        if auc > best_auc:
            best_name, best_auc, best_pipeline = name, auc, pipe
    out = Path(__file__).resolve().parent / "credit_scoring_model.joblib"
    joblib.dump(best_pipeline, out)
    print(f"\nSaved best model ({best_name}, ROC-AUC={best_auc:.4f}) to {out.name}")


if __name__ == "__main__":
    main()
