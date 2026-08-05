"""Train baseline and final models, cross-validate, evaluate on a held-out
test set, and write results to reports/metrics.json.

Run with:  python -m src.train
"""

import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_train_test
from src.evaluate import evaluate_model, print_report
from src.preprocessing import build_scaler

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def cross_validate(pipeline, X_train, y_train, cv_folds: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    return {
        "cv_folds": cv_folds,
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "cv_accuracy_scores": [float(s) for s in scores],
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_train_test()
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    all_results = {}

    # --- Baseline #1: majority-class dummy classifier ---
    # Establishes the floor: any real model must beat "always predict the
    # most common class" to be worth reporting.
    dummy = Pipeline([("scaler", build_scaler()), ("clf", DummyClassifier(strategy="most_frequent"))])
    dummy.fit(X_train, y_train)
    dummy_results = evaluate_model(dummy, X_test, y_test)
    print_report("Baseline: Dummy (most frequent class)", dummy_results)
    all_results["baseline_dummy"] = dummy_results

    # --- Baseline #2: plain Logistic Regression ---
    logreg = Pipeline(
        [("scaler", build_scaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
    )
    logreg_cv = cross_validate(logreg, X_train, y_train)
    logreg.fit(X_train, y_train)
    logreg_results = evaluate_model(logreg, X_test, y_test)
    logreg_results["cross_validation"] = logreg_cv
    print_report("Baseline: Logistic Regression", logreg_results)
    print(f"  5-fold CV accuracy: {logreg_cv['cv_accuracy_mean']:.4f} +/- {logreg_cv['cv_accuracy_std']:.4f}")
    all_results["baseline_logistic_regression"] = logreg_results

    # --- Candidate: Random Forest ---
    rf = Pipeline(
        [
            ("scaler", build_scaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_cv = cross_validate(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test)
    rf_results["cross_validation"] = rf_cv
    print_report("Candidate: Random Forest", rf_results)
    print(f"  5-fold CV accuracy: {rf_cv['cv_accuracy_mean']:.4f} +/- {rf_cv['cv_accuracy_std']:.4f}")
    all_results["candidate_random_forest"] = rf_results

    # --- Model selection: pick whichever model the evidence supports ---
    # No model is hardcoded as "the final model." Selection is based on
    # mean cross-validated accuracy on the training set (not the test
    # set, to avoid leaking test performance into the selection decision).
    candidates = {
        "logistic_regression": (logreg, logreg_cv["cv_accuracy_mean"]),
        "random_forest": (rf, rf_cv["cv_accuracy_mean"]),
    }
    final_name, (final_model, final_cv_acc) = max(candidates.items(), key=lambda kv: kv[1][1])
    print(f"\nSelected '{final_name}' as final model (highest CV accuracy: {final_cv_acc:.4f})")
    all_results["final_model_selected"] = final_name

    # Persist model (gitignored -- regenerate by rerunning this script)
    model_path = MODELS_DIR / "burnout_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved final model ({final_name}) to {model_path}")

    # Persist metrics (checked into git as evidence for the README numbers)
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
