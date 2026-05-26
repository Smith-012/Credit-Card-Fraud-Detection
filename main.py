import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def infer_target(df: pd.DataFrame) -> str:
    for col in ["Class", "class", "fraud", "is_fraud", "target"]:
        if col in df.columns:
            return col
    raise ValueError("No fraud target found. Use Class, class, fraud, is_fraud, or target")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main(data_path: Path, out_dir: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    target = infer_target(df)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    minority_count = int(y_train.value_counts().min())
    majority_count = int(y_train.value_counts().max())
    current_ratio = minority_count / majority_count
    target_ratio = min(0.5, max(current_ratio + 0.05, 0.1))
    smote = SMOTE(random_state=42, sampling_strategy=target_ratio)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_pre, y_train)

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "rf": RandomForestClassifier(n_estimators=60, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    best_name = None
    best_model = None
    best_cv = -1.0

    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring="f1")
        score = float(np.mean(cv_scores))
        if score > best_cv:
            best_cv = score
            best_name = name
            best_model = model

    best_model.fit(X_train_bal, y_train_bal)
    preds = best_model.predict(X_test_pre)
    probs = best_model.predict_proba(X_test_pre)[:, 1]

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": best_model}, out_dir / "model.joblib")
    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_model: {best_name}\n")
        f.write(f"cv_f1: {best_cv}\n")
        f.write(f"test_precision: {precision}\n")
        f.write(f"test_recall: {recall}\n")
        f.write(f"test_f1: {f1}\n")
        f.write(f"test_roc_auc: {roc_auc}\n\n")
        f.write(report)

    print("Fraud portfolio pipeline complete")
    print(f"Best model: {best_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Portfolio-grade credit card fraud detection")
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    main(args.data, args.out)


if __name__ == "__main__":
    cli()
