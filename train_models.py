#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = Path("outputs")
DATASETS_META = OUTPUT_DIR / "datasets_meta.json"
REPORTS_DIR = OUTPUT_DIR / "reports"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"

TEST_SIZE = 0.20
RANDOM_SEED = 42
EPOCHS = 8
BATCH_SIZE = 64          # used only by the MLP (solver='adam')
HIDDEN_SIZE = 128        # size of the single hidden layer

def ensure_dirs() -> None:
    """Create all output folders if they do not exist."""
    for folder in (OUTPUT_DIR, REPORTS_DIR, CONFUSION_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset(meta: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single dataset based on a single entry from ``datasets_meta.json``.

    Returns
    -------
    X : np.ndarray
        2‑D dense feature matrix (MLPClassifier does not accept sparse
        input, so we convert the sparse case to dense here).
    y : np.ndarray
        1‑D label vector – may contain strings, therefore we load it with
        ``allow_pickle=True`` (safe because the file is created by us).
    """
    data_path = Path(meta["data_path"])
    labels_path = Path(meta["labels_path"])

    if meta["format"] == "sparse_npz":
        X = sparse.load_npz(data_path).toarray()
    else:
        X = np.load(data_path, allow_pickle=False)

    y = np.load(labels_path, allow_pickle=True)

    return X, y


def build_model() -> MLPClassifier:
    """
    Tiny dense MLP that satisfies the “perceptron‑style ANN” requirement
    while staying in pure scikit‑learn (no TensorFlow/PyTorch).
    """
    return MLPClassifier(
        hidden_layer_sizes=(HIDDEN_SIZE,),
        activation="relu",
        solver="adam",
        batch_size=BATCH_SIZE,
        max_iter=EPOCHS,
        random_state=RANDOM_SEED,
        early_stopping=False,     
    )


def macro_precision_from_report(report: Dict) -> float:
    """
    Compute “total accuracy” as the macro‑averaged precision.
    ``classification_report(..., output_dict=True)`` returns a nested
    dict where numeric keys correspond to class‑wise statistics.
    """
    precisions = [
        info["precision"]
        for key, info in report.items()
        if key.isdigit()               
    ]
    return float(np.mean(precisions)) if precisions else 0.0


def plot_confusion(cm: np.ndarray, class_names: np.ndarray,
                  title: str, filename: str) -> None:
    """Draw a coloured heat‑map of the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CONFUSION_DIR / filename)
    plt.close()


def train_and_evaluate(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Train an MLP on ``X, y`` and write out:
    • classification report (JSON)
    • one‑line summary of macro precision
    • confusion‑matrix image
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )

    model = build_model()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    report = classification_report(
        y_te,
        y_pred,
        output_dict=True,
        zero_division=0,          
    )
    macro_prec = macro_precision_from_report(report)

    report_path = REPORTS_DIR / f"{name}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    summary_path = REPORTS_DIR / "total_accuracy.txt"
    # Use *append* mode because we want one line per dataset
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"{name}: {macro_prec:.4f}\n")

    cm = confusion_matrix(y_te, y_pred)
    plot_confusion(
        cm,
        le.classes_,
        title=f"Confusion Matrix – {name}",
        filename=f"{name}_cm.png",
    )


def main() -> None:
    """Orchestrates loading, training and evaluation for all nine datasets."""
    ensure_dirs()

    if not DATASETS_META.is_file():
        sys.exit(
            "⚠️  datasets_meta.json not found. Run `data_preprocessing.py` first."
        )

    meta_list = json.loads(DATASETS_META.read_text(encoding="utf-8"))

    for meta in meta_list:
        dataset_name = meta["name"]
        print(f"▶️  Processing dataset `{dataset_name}` …")
        X, y = load_dataset(meta)
        train_and_evaluate(dataset_name, X, y)

    print("\n✅  All reports and confusion‑matrix plots are in `outputs/`.")


if __name__ == "__main__":
    main()
