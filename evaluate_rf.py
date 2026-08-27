"""
Random Forest baseline evaluation on clean dedup test set.
Run: python evaluate_rf.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
import json

print("Loading dedup splits...")
train_df = pd.read_parquet("data/splits/train_dedup.parquet")
test_df  = pd.read_parquet("data/splits/test_dedup.parquet")

print(f"Train: {len(train_df)} rows, balance: {train_df['label'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Test:  {len(test_df)} rows, balance: {test_df['label'].value_counts(normalize=True).round(3).to_dict()}")

print("\nExtracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=5)
X_train = vectorizer.fit_transform(train_df['code'])
X_test  = vectorizer.transform(test_df['code'])
y_train = train_df['label'].values
y_test  = test_df['label'].values

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Evaluating...")
preds = rf.predict(X_test)

acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec  = recall_score(y_test, preds, zero_division=0)
f1   = f1_score(y_test, preds, zero_division=0)
cm   = confusion_matrix(y_test, preds)

print("\n" + "="*60)
print("RANDOM FOREST BASELINE RESULTS")
print("="*60)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(f"  Actual SAFE  predicted SAFE:  {cm[0][0]}")
print(f"  Actual SAFE  predicted VULN:  {cm[0][1]}")
print(f"  Actual VULN  predicted SAFE:  {cm[1][0]}")
print(f"  Actual VULN  predicted VULN:  {cm[1][1]}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["SAFE","VULNERABLE"]))

results = {
    "model": "random_forest_baseline",
    "data": "train_dedup + test_dedup",
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "confusion_matrix": cm.tolist(),
}

with open("results/rf_baseline_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/rf_baseline_evaluation.json")