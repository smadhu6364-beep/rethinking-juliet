
"""
Train Random Forest baseline
for software vulnerability detection.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

RESULTS_PATH = Path("results/rf_baseline_evaluation.AFTER_label_fix.json")
CHECKPOINT_DIR = Path("experiments/rf-full")

# -----------------------------------------
# Load splits
# -----------------------------------------

print("Loading datasets...")

train_df = pd.read_parquet("data/splits/train_dedup.parquet")
val_df = pd.read_parquet("data/splits/val_dedup.parquet")
test_df = pd.read_parquet("data/splits/test_dedup.parquet")

# -----------------------------------------
# TF-IDF vectorisation
# -----------------------------------------

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_df["code"])
X_val = vectorizer.transform(val_df["code"])
X_test = vectorizer.transform(test_df["code"])

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# -----------------------------------------
# Train Random Forest
# -----------------------------------------

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------------------
# Validation evaluation
# -----------------------------------------

print("\nValidation Results")

val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]

print(f"Accuracy : {accuracy_score(y_val, val_preds):.4f}")
print(f"Precision: {precision_score(y_val, val_preds):.4f}")
print(f"Recall   : {recall_score(y_val, val_preds):.4f}")
print(f"F1-score : {f1_score(y_val, val_preds):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, val_probs):.4f}")

# -----------------------------------------
# Test evaluation
# -----------------------------------------

print("\nTest Results")

test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, test_preds):.4f}")
print(f"Precision: {precision_score(y_test, test_preds):.4f}")
print(f"Recall   : {recall_score(y_test, test_preds):.4f}")
print(f"F1-score : {f1_score(y_test, test_preds):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, test_probs):.4f}")

# -----------------------------------------
# Full report
# -----------------------------------------

print("\nClassification Report")

print(classification_report(y_test, test_preds))

# -----------------------------------------
# Save results to file
# -----------------------------------------

results = {
    "model": "random_forest_baseline",
    "data": "train_dedup + test_dedup",
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "accuracy": accuracy_score(y_test, test_preds),
    "precision": precision_score(y_test, test_preds),
    "recall": recall_score(y_test, test_preds),
    "f1": f1_score(y_test, test_preds),
    "roc_auc": roc_auc_score(y_test, test_probs),
    "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_dir": str(CHECKPOINT_DIR),
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")

# -----------------------------------------
# Persist the fitted model + vectorizer
# -----------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, CHECKPOINT_DIR / "rf_model.joblib")
joblib.dump(vectorizer, CHECKPOINT_DIR / "tfidf_vectorizer.joblib")

print(f"Checkpoint saved to {CHECKPOINT_DIR}/ (rf_model.joblib, tfidf_vectorizer.joblib)")

