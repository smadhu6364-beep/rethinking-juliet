
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)

# -----------------------------------------
# DUMMY LABELS
# -----------------------------------------

y_true = np.array([
    0, 0, 0, 1, 1, 1, 0, 1, 0, 1
])

# -----------------------------------------
# DUMMY PROBABILITIES
# -----------------------------------------

y_scores = np.array([
    0.01,
    0.10,
    0.20,
    0.90,
    0.95,
    0.99,
    0.30,
    0.85,
    0.40,
    0.92
])

# -----------------------------------------
# ROC
# -----------------------------------------

fpr, tpr, _ = roc_curve(
    y_true,
    y_scores
)

roc_auc = auc(
    fpr,
    tpr
)

# -----------------------------------------
# PLOT
# -----------------------------------------

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.tight_layout()

# -----------------------------------------
# SAVE
# -----------------------------------------

plt.savefig(
    "results/roc_curve.png"
)

print("ROC curve saved.")
