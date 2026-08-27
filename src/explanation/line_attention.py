
"""
Line-level attention prototype
for vulnerability explanation.
"""

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# -----------------------------------------
# MODEL
# -----------------------------------------

MODEL_NAME = "microsoft/codebert-base"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=True
)

model.eval()

# -----------------------------------------
# LOAD SAMPLE
# -----------------------------------------

print("Loading dataset...")

df = pd.read_parquet(
    "data/splits/test_sanitized.parquet"
)

sample = df.iloc[0]

code = sample["code"]

print("\nLABEL:", sample["label"])

print("\nCODE SAMPLE:\n")
print(code[:1000])

# -----------------------------------------
# SPLIT INTO LINES
# -----------------------------------------

lines = code.split("\n")

line_scores = []

# -----------------------------------------
# SCORE EACH LINE
# -----------------------------------------

for line in lines:

    clean = line.strip()

    if len(clean) == 0:
        continue

    inputs = tokenizer(
        line,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    attentions = outputs.attentions

    last_layer = attentions[-1]

    attention_scores = last_layer.mean(dim=1)

    cls_attention = attention_scores[0][0]

    score = cls_attention.mean().item()

    line_scores.append(
        (line, score)
    )

# -----------------------------------------
# SORT LINES
# -----------------------------------------

line_scores = sorted(
    line_scores,
    key=lambda x: x[1],
    reverse=True
)

# -----------------------------------------
# PRINT RESULTS
# -----------------------------------------

print("\nTOP SUSPICIOUS LINES:\n")

for line, score in line_scores[:10]:

    print(f"{score:.6f} | {line}")
