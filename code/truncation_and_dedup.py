"""
Two checks the reviewer panel asked for.

1. Truncation: for every vulnerable test function (family split), where does
   its first manifest flaw line fall in the model's token sequence? Report the
   share whose flaw line lies beyond the 256-token input limit.
2. Deduplication accounting: from labelled functions to the unique set, how
   many functions of each label leave at each step (label conflict, exact
   duplicate).
Writes out/truncation_dedup.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from build_splits import normalise
from extract_v2 import LANGS, load_manifest

OUT = Path(__file__).resolve().parent / "out"
tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
res = {}

for lang in ("cpp", "java"):
    man = load_manifest(LANGS[lang]["manifest"])
    fn = pd.read_parquet(OUT / f"functions_{lang}.parquet",
                         columns=["file", "function_name", "start_line", "end_line", "label", "code_raw"])
    te = pd.read_parquet(OUT / f"{lang}_family_test.parquet", columns=["file", "function_name", "code_raw", "label"])
    v = te[te.label == 1].merge(fn[fn.label == 1], on=["file", "function_name", "code_raw"], how="left")
    pos = []
    for r in v.itertuples():
        lines = sorted(l for l in man.get(r.file, ()) if r.start_line <= l <= r.end_line)
        if not lines:
            continue
        k = lines[0] - r.start_line                       # lines before the flaw
        prefix = "\n".join(r.code_raw.split("\n")[:k])
        n_before = len(tok.tokenize(normalise(prefix))) if prefix.strip() else 0
        pos.append(n_before)
    pos = np.array(pos)
    # 256-token window = <s> + 254 tokens + </s>
    res[f"{lang}_flaw_beyond_256"] = float((pos >= 254).mean())
    res[f"{lang}_flaw_pos_median"] = float(np.median(pos))
    res[f"{lang}_n_vuln_test_checked"] = int(len(pos))

    # dedup accounting by label
    lab = fn[fn.label.notna()].copy()
    lab["label"] = lab.label.astype(int)
    lab["code"] = lab.code_raw.map(normalise)
    per = lab.groupby("code")["label"].nunique()
    conflict = lab.code.isin(per[per > 1].index)
    kept = lab[~conflict]
    uniq = kept.drop_duplicates("code")
    for l, name in ((1, "vuln"), (0, "safe")):
        res[f"{lang}_{name}_labelled"] = int((lab.label == l).sum())
        res[f"{lang}_{name}_removed_conflict"] = int(((lab.label == l) & conflict).sum())
        res[f"{lang}_{name}_removed_exact_dup"] = int((kept.label == l).sum() - (uniq.label == l).sum())
        res[f"{lang}_{name}_unique"] = int((uniq.label == l).sum())
    print(lang, {k: v for k, v in res.items() if k.startswith(lang)}, flush=True)

(OUT / "truncation_dedup.json").write_text(json.dumps(res, indent=1))
