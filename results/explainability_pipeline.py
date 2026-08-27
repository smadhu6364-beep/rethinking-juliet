
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))

ax.axis("off")

# -----------------------------------------
# COMPONENTS
# -----------------------------------------

components = [
    ("Code Sample", 0.08),
    ("Transformer Model", 0.25),
    ("Attention Extraction", 0.45),
    ("Line Scoring", 0.65),
    ("Generated Explanation", 0.88)
]

for label, x in components:

    ax.text(
        x,
        0.5,
        label,
        bbox=dict(
            boxstyle="round",
            fill=False
        ),
        ha="center"
    )


# -----------------------------------------
# ARROWS
# -----------------------------------------

arrow_y = 0.5

for i in range(len(components) - 1):

    x1 = components[i][1]
    x2 = components[i + 1][1]

    ax.annotate(
        "",
        xy=(x2 - 0.07, arrow_y),
        xytext=(x1 + 0.07, arrow_y),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.5
        )
    )



# -----------------------------------------
# SAVE
# -----------------------------------------

plt.savefig(
    "results/explainability_pipeline.png",
    bbox_inches="tight"
)

print("Explainability pipeline diagram saved.")
