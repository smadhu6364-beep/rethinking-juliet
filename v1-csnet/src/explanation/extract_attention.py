
"""
Basic attention extraction prototype
for CodeBERT explanations.
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
# TOKENIZE
# -----------------------------------------

inputs = tokenizer(
    code,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

# -----------------------------------------
# FORWARD PASS
# -----------------------------------------

with torch.no_grad():

    outputs = model(**inputs)

# -----------------------------------------
# ATTENTION
# -----------------------------------------

attentions = outputs.attentions

print("\nNumber of attention layers:")
print(len(attentions))

print("\nAttention tensor shape:")
print(attentions[0].shape)

# -----------------------------------------
# TOKENS
# -----------------------------------------

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("\nFIRST 50 TOKENS:\n")


# -----------------------------------------
# SIMPLE TOKEN IMPORTANCE
# -----------------------------------------

last_layer = attentions[-1]

# Average across heads
attention_scores = last_layer.mean(dim=1)

# Focus on CLS token attention
cls_attention = attention_scores[0][0]

token_scores = []

for token, score in zip(tokens, cls_attention):

    token_scores.append(
        (token, score.item())
    )

# Sort descending
token_scores = sorted(
    token_scores,
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP IMPORTANT TOKENS:\n")

# -----------------------------------------
# FILTER NOISY TOKENS
# -----------------------------------------

ignore_tokens = {
    "<s>",
    "</s>",
    "Ċ",
    "{",
    "}",
    "(",
    ")",
    "()",
    ";",
    ","
}

filtered_scores = []

for token, score in token_scores:

    clean = token.strip()

    if clean in ignore_tokens:
        continue

    if len(clean) <= 1:
        continue

    filtered_scores.append(
        (token, score)
    )

print("\nTOP SEMANTIC TOKENS:\n")

for token, score in filtered_scores[:20]:

    print(f"{token:<20} {score:.6f}")




