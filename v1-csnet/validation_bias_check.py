import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("experiments/codebert-test/checkpoint-1000")
model.eval()

val_df = pd.read_parquet("data/splits/val_clean.parquet").sample(500, random_state=42)

predictions = []
for code in tqdm(val_df['code']):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    predictions.append(pred)

val_df['predicted'] = predictions
print("Predicted VULNERABLE:", sum(p == 1 for p in predictions), "/", len(predictions))
print("Actual VULNERABLE:   ", val_df['label'].sum(), "/", len(val_df))
print("Confusion:")
print(pd.crosstab(val_df['label'], val_df['predicted']))