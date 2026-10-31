"""Educational disease-classification example; not for clinical use."""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # 0 = malignant, 1 = benign in this dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                 class_weight="balanced")
    }
    best_auc, best_name, best_model = -1, None, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=data.target_names, zero_division=0))
        print("Confusion matrix (rows=true, columns=predicted):\n", confusion_matrix(y_test, pred))
        print(f"Accuracy={accuracy_score(y_test, pred):.4f}  "
              f"Precision={precision_score(y_test, pred, zero_division=0):.4f}  "
              f"Recall={recall_score(y_test, pred, zero_division=0):.4f}  "
              f"F1={f1_score(y_test, pred, zero_division=0):.4f}  ROC-AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model
    path = Path(__file__).resolve().parent / "disease_demo_model.joblib"
    joblib.dump(best_model, path)
    print(f"\nSaved best demonstration model ({best_name}) to {path.name}")
    print("Reminder: educational dataset only; not for diagnosis or clinical decisions.")


if __name__ == "__main__":
    main()
