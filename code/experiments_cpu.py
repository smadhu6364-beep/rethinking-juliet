"""
CPU experiments for the revision (everything except transformer fine-tuning).

For each language and split scheme (random, family):
  * length-only baseline (depth-1 tree on sanitized character count)
  * TF-IDF Random Forest, 5 seeds
  * artifact ablations, each neutralising one artifact:
      raw        - names and comments left in (pre-fix text)
      sanitized  - final text (reference)
      len-matched- train and test resampled so both classes share one
                   length distribution (20 quantile bins)
      no-excl    - tokens exclusive to one class in train removed
      both       - len-matched + no-excl
  * residual near-duplication between train and test
Writes out/cpu_results_<lang>.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).resolve().parent / "out"
SEEDS = [0, 1, 2, 3, 4]
TOKEN = r"[A-Za-z_]\w*|\d+|\S"
KEYWORDS = set("""auto break case char const continue default do double else enum extern
float for goto if int long register return short signed sizeof static struct switch
typedef union unsigned void volatile while class public private protected new delete
this try catch throw throws namespace using bool true false null NULL boolean byte
final import package super interface extends implements string String var foreach in
out ref readonly override virtual abstract""".split())


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
                rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y)))


def summarise(runs):
    keys = ["acc", "prec", "rec", "f1"]
    return {k: dict(mean=float(np.mean([r[k] for r in runs])),
                    sd=float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0)
            for k in keys} | {"runs": runs}


def length_match(df, rng, bins=20):
    """Resample so vulnerable and safe have the same length histogram."""
    q = np.unique(np.quantile(df["len"], np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(q, df["len"], side="right") - 1, 0, len(q) - 2)
    keep = []
    for k in np.unique(b):
        g = df[b == k]
        v, s = g[g.label == 1], g[g.label == 0]
        m = min(len(v), len(s))
        if m == 0:
            continue
        keep += list(rng.choice(v.index, m, replace=False)) + list(rng.choice(s.index, m, replace=False))
    return df.loc[sorted(keep)]


def exclusive_tokens(train, min_count=20):
    cv = CountVectorizer(token_pattern=TOKEN, lowercase=False)
    X = cv.fit_transform(train.code)
    y = train.label.to_numpy()
    cv1 = np.asarray(X[y == 1].sum(0)).ravel()
    cv0 = np.asarray(X[y == 0].sum(0)).ravel()
    vocab = np.array(cv.get_feature_names_out())
    excl = vocab[((cv1 >= min_count) & (cv0 == 0)) | ((cv0 >= min_count) & (cv1 == 0))]
    top = sorted([(t, int(cv1[i]), int(cv0[i])) for i, t in enumerate(vocab) if t in set(excl)],
                 key=lambda r: -(r[1] + r[2]))[:25]
    return set(excl), top


def drop_tokens(code, toks):
    return " ".join(t for t in re.findall(TOKEN, code) if t not in toks)


def rf_runs(tr_text, ytr, te_text, yte):
    vec = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, ngram_range=(1, 2),
                          max_features=10000, min_df=5)
    Xtr = vec.fit_transform(tr_text)
    Xte = vec.transform(te_text)
    runs = []
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)
        rf.fit(Xtr, ytr)
        runs.append(metrics(yte, rf.predict(Xte)))
    return summarise(runs)


def length_only(tr, te, col="len"):
    t = DecisionTreeClassifier(max_depth=1, random_state=0).fit(tr[[col]], tr.label)
    r = metrics(te.label, t.predict(te[[col]]))
    r["threshold"] = float(t.tree_.threshold[0])
    return r


def template_key(code):
    toks = []
    for t in re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+\w*|\S', code):
        if t[0] in "\"'":
            toks.append("S")
        elif t[0].isdigit():
            toks.append("N")
        elif re.match(r"[A-Za-z_]", t) and t not in KEYWORDS:
            toks.append("I")
        else:
            toks.append(t)
    return " ".join(toks)


def near_dup(tr, te, rng, n=1000):
    """Share of test functions whose identifier/literal-abstracted template
    occurs in train, and max 5-gram Jaccard to any train function."""
    tk_tr = set(tr.code.map(template_key))
    tk_te = te.code.map(template_key)
    exact_template = float(tk_te.isin(tk_tr).mean())
    sub = te.sample(min(n, len(te)), random_state=0)
    cv = CountVectorizer(token_pattern=TOKEN, lowercase=False, ngram_range=(5, 5), binary=True)
    Xtr = cv.fit_transform(tr.code).astype(np.float32)
    Xte = cv.transform(sub.code).astype(np.float32)
    a = np.asarray(Xtr.sum(1)).ravel()
    b = np.asarray(Xte.sum(1)).ravel()
    best = np.zeros(Xte.shape[0])
    for i in range(0, Xte.shape[0], 100):
        inter = (Xte[i:i + 100] @ Xtr.T).toarray()
        union = b[i:i + 100, None] + a[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        best[i:i + 100] = jac.max(1)
    return dict(test_template_in_train=exact_template,
                jaccard5_max_median=float(np.median(best)),
                jaccard5_ge_0_9=float((best >= 0.9).mean()),
                jaccard5_ge_0_7=float((best >= 0.7).mean()), sample=int(len(sub)))


def run(lang):
    res = {}
    for scheme in ("random", "family"):
        tr = pd.read_parquet(OUT / f"{lang}_{scheme}_train.parquet")
        te = pd.read_parquet(OUT / f"{lang}_{scheme}_test.parquet")
        rng = np.random.default_rng(0)
        r = {}
        # descriptive length statistics (sanitized text and raw text)
        r["length_stats"] = {
            f"{'vuln' if l else 'safe'}": dict(
                mean=float(g["len"].mean()), median=float(g["len"].median()),
                mean_raw=float(g.len_raw.mean()), median_raw=float(g.len_raw.median()))
            for l, g in pd.concat([tr, te]).groupby("label")}
        r["length_only"] = length_only(tr, te)
        r["length_only_raw"] = length_only(tr, te, "len_raw")
        r["rf_sanitized"] = rf_runs(tr.code, tr.label, te.code, te.label)
        r["rf_raw"] = rf_runs(tr.code_raw, tr.label, te.code_raw, te.label)
        trm, tem = length_match(tr, rng), length_match(te, rng)
        r["len_matched_sizes"] = dict(train=len(trm), test=len(tem))
        r["length_only_on_len_matched_test"] = length_only(tr, tem)
        r["rf_full_train_on_len_matched_test"] = rf_runs(tr.code, tr.label, tem.code, tem.label)
        r["rf_len_matched"] = rf_runs(trm.code, trm.label, tem.code, tem.label)
        excl, top = exclusive_tokens(tr)
        r["exclusive_tokens_n"] = len(excl)
        r["exclusive_tokens_top"] = top
        r["rf_no_exclusive"] = rf_runs(tr.code.map(lambda c: drop_tokens(c, excl)), tr.label,
                                       te.code.map(lambda c: drop_tokens(c, excl)), te.label)
        r["rf_len_matched_no_exclusive"] = rf_runs(
            trm.code.map(lambda c: drop_tokens(c, excl)), trm.label,
            tem.code.map(lambda c: drop_tokens(c, excl)), tem.label)
        r["near_dup"] = near_dup(tr, te, rng)
        res[scheme] = r
        print(lang, scheme, "len-only acc %.4f" % r["length_only"]["acc"],
              "RF f1 %.4f" % r["rf_sanitized"]["f1"]["mean"],
              "RF len-matched f1 %.4f" % r["rf_len_matched"]["f1"]["mean"],
              "both %.4f" % r["rf_len_matched_no_exclusive"]["f1"]["mean"], flush=True)
    (OUT / f"cpu_results_{lang}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["cpp", "java", "csharp"]):
        run(lang)
