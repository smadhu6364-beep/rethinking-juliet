"""
Real transformer inference demo
using trained CodeBERT model.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "experiments/codebert-test/checkpoint-1000"

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/codebert-base"
)

print("Loading trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# --------------------------------------------------
# SAMPLE INPUT
# --------------------------------------------------

code_sample = """
int add(int a, int b) {
    return a + b;
}
"""

print("\n" + "=" * 60)
print("REAL VULNERABILITY DETECTION DEMO")
print("=" * 60)

print("\nINPUT CODE:\n")

print(code_sample)

# --------------------------------------------------
# TOKENIZE INPUT
# --------------------------------------------------

inputs = tokenizer(
    code_sample,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=256
)

# --------------------------------------------------
# MODEL INFERENCE
# --------------------------------------------------

with torch.no_grad():

    outputs = model(**inputs)

    logits = outputs.logits

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

probabilities = torch.softmax(
    logits,
    dim=1
)

prediction_id = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = probabilities[0][prediction_id].item()

# --------------------------------------------------
# LABEL MAPPING
# --------------------------------------------------

label_map = {
    0: "SAFE",
    1: "VULNERABLE"
}

prediction = label_map[prediction_id]

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\nPREDICTION:")

print(prediction)

print("\nCONFIDENCE:")

print(round(confidence, 4))

print("\nRAW PROBABILITIES:")

print(probabilities)

print("\n" + "=" * 60)