import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train = pd.read_parquet('data/splits/train_dedup.parquet')
test = pd.read_parquet('data/splits/test_dedup.parquet')

train['length'] = train['code'].str.len()
test['length'] = test['code'].str.len()

clf = DecisionTreeClassifier(max_depth=1, random_state=42)
clf.fit(train[['length']], train['label'])
acc = clf.score(test[['length']], test['label'])
print(f"Length classifier accuracy: {acc:.4f}")