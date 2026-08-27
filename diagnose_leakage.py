import pandas as pd
import re
from collections import Counter

train = pd.read_parquet("data/splits/train_dedup.parquet")
train['code_len'] = train['code'].str.len()

print("DIAGNOSTIC 1 — Code length by class")
print(f"VULNERABLE mean: {train[train['label']==1]['code_len'].mean():.0f}")
print(f"VULNERABLE median: {train[train['label']==1]['code_len'].median():.0f}")
print(f"SAFE mean:       {train[train['label']==0]['code_len'].mean():.0f}")
print(f"SAFE median:     {train[train['label']==0]['code_len'].median():.0f}")

vuln_names = set(train[train['label']==1]['function_name'])
safe_names = set(train[train['label']==0]['function_name'])
print(f"\nDIAGNOSTIC 2 — Function name overlap")
print(f"Function name overlap: {len(vuln_names & safe_names)}")
print(f"Total unique names: {len(vuln_names | safe_names)}")

print("\nDIAGNOSTIC 3 — Tokens exclusive to one class")
vuln_tokens = Counter()
safe_tokens = Counter()
for _, row in train.sample(5000, random_state=42).iterrows():
    toks = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', row['code']))
    if row['label'] == 1:
        for t in toks: vuln_tokens[t] += 1
    else:
        for t in toks: safe_tokens[t] += 1

print("Tokens only in VULNERABLE (count > 100):")
for tok, vc in vuln_tokens.most_common(200):
    sc = safe_tokens.get(tok, 0)
    if vc > 100 and sc == 0:
        print(f"  {tok}: vuln={vc} safe={sc}")