"""
Random Forest trained on the C/C++ family-disjoint training split (five
seeds), evaluated on Devign and the 35-function diagnostic suite. Output has
the same shape as the neural runs so make_tables.py can treat them alike.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments_cpu import SEEDS, TOKEN, metrics
from ood_suite import load_devign, load_ood

OUT = Path(__file__).resolve().parent / "out"

tr = pd.read_parquet(OUT / "cpp_family_train.parquet")
ood, dv = load_ood(), load_devign()
vec = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, ngram_range=(1, 2),
                      max_features=10000, min_df=5)
X = vec.fit_transform(tr.code)
Xo, Xd = vec.transform(ood.code), vec.transform(dv.code)
runs = []
for s in SEEDS:
    rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=4).fit(X, tr.label)
    po, pdv = rf.predict(Xo), rf.predict(Xd)
    ood["pred"] = po
    runs.append(dict(seed=s, devign=metrics(dv.label.to_numpy(), pdv),
                     ood35=metrics(ood.label.to_numpy(), po),
                     ood35_by_category={c: dict(n=len(g), correct=int((g.pred == g.label).sum()))
                                        for c, g in ood.groupby("category")}))
# make_tables reads a single dict with per-metric means in the run format
agg = runs[0] | {"runs": runs}
agg["devign"] = {k: float(np.mean([r["devign"][k] for r in runs])) for k in runs[0]["devign"]}
(OUT / "rf_external_cpp.json").write_text(json.dumps(agg, indent=1, default=float))
print(agg["devign"])
