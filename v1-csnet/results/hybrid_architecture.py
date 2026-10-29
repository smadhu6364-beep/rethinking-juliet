
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))

ax.axis("off")

# -----------------------------------------
# COMPONENTS
# -----------------------------------------

ax.text(
    0.1,
    0.5,
    "Code Input",
    bbox=dict(boxstyle="round", fill=False),
    ha="center"
)

ax.text(
    0.35,
    0.7,
    "CodeBERT",
    bbox=dict(boxstyle="round", fill=False),
    ha="center"
)

ax.text(
    0.35,
    0.3,
    "GraphCodeBERT",
    bbox=dict(boxstyle="round", fill=False),
    ha="center"
)

ax.text(
    0.6,
    0.5,
    "Concatenation",
    bbox=dict(boxstyle="round", fill=False),
    ha="center"
)

ax.text(
    0.82,
    0.5,
    "MLP Classifier",
    bbox=dict(boxstyle="round", fill=False),
    ha="center"
)

# -----------------------------------------
# ARROWS
# -----------------------------------------

ax.annotate(
    "",
    xy=(0.28, 0.68),
    xytext=(0.15, 0.52),
    arrowprops=dict(arrowstyle="->")
)

ax.annotate(
    "",
    xy=(0.28, 0.32),
    xytext=(0.15, 0.48),
    arrowprops=dict(arrowstyle="->")
)

ax.annotate(
    "",
    xy=(0.53, 0.52),
    xytext=(0.42, 0.68),
    arrowprops=dict(arrowstyle="->")
)

ax.annotate(
    "",
    xy=(0.53, 0.48),
    xytext=(0.42, 0.32),
    arrowprops=dict(arrowstyle="->")
)

ax.annotate(
    "",
    xy=(0.75, 0.5),
    xytext=(0.67, 0.5),
    arrowprops=dict(arrowstyle="->")
)

# -----------------------------------------
# SAVE
# -----------------------------------------

plt.savefig(
    "results/hybrid_architecture.png",
    bbox_inches="tight"
)

print("Architecture diagram saved.")
