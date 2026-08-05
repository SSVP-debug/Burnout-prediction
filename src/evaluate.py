"""Evaluation utilities: confusion matrix + precision/recall/F1, no bare
accuracy numbers without evidence behind them."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, labels=("Low", "Moderate", "High")) -> dict:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(labels))

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(labels),
    }


def print_report(name: str, results: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    labels = results["confusion_matrix_labels"]
    print(" " * 12 + "  ".join(f"{l:>10}" for l in labels))
    for label, row in zip(labels, results["confusion_matrix"]):
        print(f"{label:>10}  " + "  ".join(f"{v:>10}" for v in row))

    print("\nPer-class precision / recall / F1:")
    for label in labels:
        m = results["classification_report"][label]
        print(
            f"  {label:>10}: precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  f1={m['f1-score']:.3f}  "
            f"support={int(m['support'])}"
        )
