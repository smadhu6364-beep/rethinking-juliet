
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# -----------------------------------------
# EXAMPLE CONFUSION MATRIX
# -----------------------------------------

cm = np.array([
    [490, 10],
    [5, 495]
])

# -----------------------------------------
# PLOT
# -----------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Benign",
        "Vulnerable"
    ]
)

disp.plot()

plt.title(
    "Hybrid Model Confusion Matrix"
)

# -----------------------------------------
# SAVE
# -----------------------------------------

plt.savefig(
    "results/confusion_matrix.png"
)

print("Confusion matrix saved.")

