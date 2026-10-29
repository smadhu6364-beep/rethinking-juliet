
import matplotlib.pyplot as plt

# -----------------------------------------
# MODEL NAMES
# -----------------------------------------

models = [
    "Random Forest",
    "CodeBERT",
    "GraphCodeBERT",
    "Hybrid"
]

# -----------------------------------------
# F1 SCORES
# -----------------------------------------

f1_scores = [
    0.99,
    0.99,
    1.00,
    0.99
]

# -----------------------------------------
# PLOT
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    models,
    f1_scores
)

plt.ylabel("F1 Score")

plt.xlabel("Model")

plt.title(
    "Model Performance Comparison"
)

plt.ylim(0.9, 1.01)

plt.tight_layout()

# -----------------------------------------
# SAVE
# -----------------------------------------

plt.savefig(
    "results/model_comparison.png"
)

print("Plot saved successfully.")
