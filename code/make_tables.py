"""
Generate revised/results.tex (number macros) and revised/tab_*.tex from the
result files in out/. Any value that is not available yet is written as a
bold ?? so a missing number can never pass unnoticed into the PDF.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from extract_v2 import LANGS, load_manifest

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REV = HERE.parent / "revised"
MISSING = r"\textbf{??}"
macros: dict[str, str] = {}


def load(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else None


def f3(x):
    return MISSING if x is None else f"{x:.3f}"


def f4(x):
    return MISSING if x is None else f"{x:.4f}"


def pm(d, k="f1", nd=3):
    if d is None:
        return MISSING
    return f"{d[k]['mean']:.{nd}f} $\\pm$ {d[k]['sd']:.{nd}f}"


def pct(x, nd=1):
    return MISSING if x is None else f"{100 * x:.{nd}f}\\%"


def n(x):
    return MISSING if x is None else f"{int(x):,}".replace(",", "{,}")


def ms(vals, nd=3):
    """mean +- sd string from a list of floats."""
    if not vals:
        return MISSING
    if len(vals) == 1:          # one run: no spread to report
        return f"{vals[0]:.{nd}f}"
    sd = np.std(vals, ddof=1)
    return f"{np.mean(vals):.{nd}f} $\\pm$ {sd:.{nd}f}"


# ---------------------------------------------------------------- data
st = {l: load(f"stages_{l}.json") for l in ("cpp", "java")}
fn = {l: pd.read_parquet(OUT / f"functions_{l}.parquet",
                         columns=["cwe", "role", "label", "reason", "manifest_flaw",
                                  "file", "start_line", "end_line"])
      for l in ("cpp", "java")}

for l, key in (("cpp", "Cpp"), ("java", "Java")):
    man = load_manifest(LANGS[l]["manifest"])
    macros[f"{key}ManifestLines"] = n(sum(len(v) for v in man.values()))
    macros[f"{key}ManifestFiles"] = n(sum(1 for v in man.values() if v))
    d = fn[l]
    macros[f"{key}VulnLabelled"] = n((d.label == 1).sum())
    macros[f"{key}GoodFlaw"] = n((d.reason == "good-flow but manifest flaw line").sum())
    # share of manifest flaw lines that fall inside a scanned function
    spans = d.groupby("file").apply(lambda g: list(zip(g.start_line, g.end_line)))
    tot = cov = 0
    for fname, lines in man.items():
        sp = spans.get(fname, [])
        for ln in lines:
            tot += 1
            cov += any(a <= ln <= b for a, b in sp)
    macros[f"{key}ManifestCovered"] = pct(cov / tot if tot else None)
    s = st[l] or {}
    macros[f"{key}Conflict"] = n(s.get("removed_label_conflict"))
    macros[f"{key}CweVuln"] = n(s.get("cwes_with_vuln_after_dedup"))
    macros[f"{key}CweBoth"] = n(s.get("cwes_both_classes"))
    macros[f"{key}CweAll"] = n(len([c for c in d.cwe.unique() if c != "unknown"]))

OLD_VULN_CPP = 13893
new_v = int((fn["cpp"].label == 1).sum())
macros["RecallLossCpp"] = pct(1 - OLD_VULN_CPP / new_v)
macros["CppRandFamOverlap"] = pct((st["cpp"] or {}).get("random_test_in_train_family_pct", 0) / 100)

# audit table: role x outcome, both languages
rows = []
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    d = fn[l].copy()
    d["outcome"] = np.where(d.label.isna(), "excluded", np.where(d.label.fillna(-1) == 1, "vulnerable", "safe"))
    t = d.groupby(["role", "outcome"]).size().unstack(fill_value=0)
    for role in ["bad", "bad-helper", "good", "goodG2B/B2G", "good-helper", "main", "other"]:
        if role in t.index:
            r = t.loc[role]
            rows.append((name, role, r.get("vulnerable", 0), r.get("safe", 0), r.get("excluded", 0)))
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Label Audit: Scanned Functions by Role and Outcome (Before Deduplication)}",
         r"\label{tab:audit}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"\textbf{Lang.} & \textbf{Role} & \textbf{Vuln.} & \textbf{Safe} & \textbf{Excl.} \\", r"\midrule"]
prev = None
for lang, role, v, s, e in rows:
    if prev and lang != prev:
        lines.append(r"\midrule")
    lines.append(f"{lang if lang != prev else ''} & \\texttt{{{role}}} & {n(v)} & {n(s)} & {n(e)} \\\\")
    prev = lang
lines += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\footnotesize Vulnerable: bad-flow function containing a manifest flaw line. "
          r"Excluded bad-flow functions are wrappers or sources whose flaw lies in another function. "
          r"\texttt{other}: support code outside the bad/good naming scheme.}", r"\end{table}"]
(REV / "tab_audit.tex").write_text("\n".join(lines))

# split table
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Deduplicated Data and Splits}",
         r"\label{tab:splits}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"\textbf{Lang.} & \textbf{Split} & \textbf{Train} & \textbf{Val.} & \textbf{Test} \\", r"\midrule"]
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    s = st[l]
    lines.append(f"{name} & unique: {n(s['unique_after_dedup'])} & \\multicolumn{{3}}{{l}}"
                 f"{{({n(s['vulnerable'])} vuln., {n(s['safe'])} safe)}} \\\\")
    for sc, lab in (("random", "Random"), ("family", "Family")):
        cells = [f"{n(s[f'{sc}_{p}_n'])} ({s[f'{sc}_{p}_vuln_pct']:.1f}\\%)" for p in ("train", "val", "test")]
        lines.append(f" & {lab} & " + " & ".join(cells) + r" \\")
    if l == "cpp":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\footnotesize Percentages: share of vulnerable functions. Exact-text overlap between train and test is zero in all splits.}",
          r"\end{table}"]
(REV / "tab_splits.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- CPU results
cpu = {l: load(f"cpu_results_{l}.json") for l in ("cpp", "java")}
c = cpu["cpp"]
if c:
    r = c["random"]
    macros["CppMedVuln"] = n(r["length_stats"]["vuln"]["median"])
    macros["CppMedSafe"] = n(r["length_stats"]["safe"]["median"])
    macros["CppLenAcc"] = f3(r["length_only"]["acc"])
    lo = r["length_only"]
    maj = max(lo["tn"] + lo["fp"], lo["fn"] + lo["tp"]) / lo["n"]
    macros["CppMajority"] = f3(maj)
    macros["CppLenAccMatched"] = f3(r["length_only_on_len_matched_test"]["acc"])
    macros["CppExclusive"] = n(r["exclusive_tokens_n"])
    macros["RFcppRandF"] = f3(r["rf_sanitized"]["f1"]["mean"])
    macros["RFcppFamF"] = f3(c["family"]["rf_sanitized"]["f1"]["mean"])
j = cpu["java"]
if j:
    r = j["random"]
    macros["JavaMedVuln"] = n(r["length_stats"]["vuln"]["median"])
    macros["JavaMedSafe"] = n(r["length_stats"]["safe"]["median"])
    macros["JavaLenAcc"] = f3(r["length_only"]["acc"])
    lo = r["length_only"]
    macros["JavaMajority"] = f3(max(lo["tn"] + lo["fp"], lo["fn"] + lo["tp"]) / lo["n"])
    macros["RFjavaRandF"] = f3(r["rf_sanitized"]["f1"]["mean"])
    macros["RFjavaFamF"] = f3(j["family"]["rf_sanitized"]["f1"]["mean"])

# length table
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Function Length (Sanitised Characters) and Length-Only Baseline, Random Split}",
         r"\label{tab:length}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{\textbf{Median length}} & \multicolumn{2}{c}{\textbf{Length-only acc.}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"\textbf{Lang.} & \textbf{Vuln.} & \textbf{Safe} & \textbf{Test} & \textbf{Len.-matched} \\", r"\midrule"]
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    x = cpu[l]
    if x:
        r = x["random"]
        lines.append(f"{name} & {n(r['length_stats']['vuln']['median'])} & {n(r['length_stats']['safe']['median'])} & "
                     f"{f3(r['length_only']['acc'])} & {f3(r['length_only_on_len_matched_test']['acc'])} \\\\")
    else:
        lines.append(f"{name} & {MISSING} & {MISSING} & {MISSING} & {MISSING} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(REV / "tab_length.tex").write_text("\n".join(lines))

# ablation table (RF F1, mean +- sd over 5 seeds)
abl = [("Raw text (names, comments kept)", "rf_raw"),
       ("Sanitised (reference)", "rf_sanitized"),
       ("Sanitised, len.-matched test", "rf_full_train_on_len_matched_test"),
       ("Len.-matched train and test", "rf_len_matched"),
       ("Class-exclusive tokens removed", "rf_no_exclusive"),
       ("Len.-matched + tokens removed", "rf_len_matched_no_exclusive")]
lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Random Forest F1 Under Shortcut Controls (Mean $\pm$ SD Over Five Seeds)}",
         r"\label{tab:ablation}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{\textbf{C/C++}} & \multicolumn{2}{c}{\textbf{Java}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"\textbf{Setting} & \textbf{Random} & \textbf{Family} & \textbf{Random} & \textbf{Family} \\", r"\midrule"]
for lab, k in abl:
    cells = []
    for l in ("cpp", "java"):
        for sc in ("random", "family"):
            cells.append(pm(cpu[l][sc][k]) if cpu[l] else MISSING)
    lines.append(lab + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(REV / "tab_ablation.tex").write_text("\n".join(lines))

# near-duplicate table
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Residual Near-Duplication Between Test and Training Functions}",
         r"\label{tab:neardup}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"\textbf{Lang.} & \textbf{Split} & \textbf{Templ.\ seen} & \textbf{Med.} $J_5$ & $J_5 \geq 0.9$ \\", r"\midrule"]
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    for sc in ("random", "family"):
        x = cpu[l][sc]["near_dup"] if cpu[l] else None
        if x:
            lines.append(f"{name if sc == 'random' else ''} & {sc.capitalize()} & {pct(x['test_template_in_train'])} & "
                         f"{x['jaccard5_max_median']:.2f} & {pct(x['jaccard5_ge_0_9'])} \\\\")
        else:
            lines.append(f"{name} & {sc} & {MISSING} & {MISSING} & {MISSING} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\footnotesize Templ.\ seen: test functions whose identifier- and literal-abstracted template occurs in training. "
          r"$J_5$: highest 5-gram Jaccard similarity to any training function (1{,}000 sampled test functions).}",
          r"\end{table}"]
(REV / "tab_neardup.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- neural results
def ft_runs(model, lang, scheme):
    runs = [load(f"ft_{model}_{lang}_{scheme}_s{s}.json") for s in (0, 1, 2)]
    return [r for r in runs if r]


def hy_runs(lang, scheme):
    return load(f"hybrid_{lang}_{scheme}.json") or []


def col(runs, key, metric="f1"):
    return [r[key][metric] for r in runs if key in r]


lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Model Comparison, F1 (Mean $\pm$ SD; Random Forest Five Seeds, Neural Models Three Seeds)}",
         r"\label{tab:models}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"\textbf{Lang.} & \textbf{Model} & \textbf{Random} & \textbf{Family} & \textbf{Family, len.-matched} & \textbf{Family acc.} \\", r"\midrule"]
fam_f1 = {}
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    x = cpu[l]
    rf = (pm(x["random"]["rf_sanitized"]), pm(x["family"]["rf_sanitized"]),
          pm(x["family"]["rf_full_train_on_len_matched_test"]), pm(x["family"]["rf_sanitized"], "acc")) if x else (MISSING,) * 4
    lines.append(f"{name} & Random Forest & " + " & ".join(rf) + r" \\")
    for m, lab in (("codebert", "CodeBERT"), ("graphcodebert", "GraphCodeBERT (token-only)")):
        rr, rf_ = ft_runs(m, l, "random"), ft_runs(m, l, "family")
        fam_f1[(m, l)] = col(rf_, "test")
        lines.append(f" & {lab} & {ms(col(rr, 'test')) if rr else '--'} & {ms(col(rf_, 'test'))} & "
                     f"{ms(col(rf_, 'test_len_matched'))} & {ms(col(rf_, 'test', 'acc'))} \\\\")
    hr, hf = hy_runs(l, "random"), hy_runs(l, "family")
    fam_f1[("hybrid", l)] = col(hf, "test")
    lines.append(f" & Hybrid (frozen) & {ms(col(hr, 'test')) if hr else '--'} & {ms(col(hf, 'test'))} & "
                 f"{ms(col(hf, 'test_len_matched'))} & {ms(col(hf, 'test', 'acc'))} \\\\")
    if l == "cpp":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\footnotesize Random split: fine-tuned models one seed (no $\pm$), hybrid three seeds; Java neural models were not run on the random split (--). "
          r"Family, len.-matched: the family-disjoint test set resampled so both classes share one length distribution.}",
          r"\end{table*}"]
(REV / "tab_models.tex").write_text("\n".join(lines))
macros["CBcppFamF"] = f3(np.mean(fam_f1[("codebert", "cpp")])) if fam_f1[("codebert", "cpp")] else MISSING
macros["GCBcppFamF"] = f3(np.mean(fam_f1[("graphcodebert", "cpp")])) if fam_f1[("graphcodebert", "cpp")] else MISSING
macros["HYcppFamF"] = f3(np.mean(fam_f1[("hybrid", "cpp")])) if fam_f1[("hybrid", "cpp")] else MISSING

# truncation table
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Share of Training Inputs Truncated at 256 Tokens}",
         r"\label{tab:trunc}", r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
         r"\textbf{Lang.} & \textbf{Vulnerable} & \textbf{Safe} \\", r"\midrule"]
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    runs = ft_runs("codebert", l, "family")
    if runs:
        t = runs[0]["truncated_frac_by_label"]
        lines.append(f"{name} & {pct(t['1'])} & {pct(t['0'])} \\\\")
    else:
        lines.append(f"{name} & {MISSING} & {MISSING} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(REV / "tab_trunc.tex").write_text("\n".join(lines))

# external table (C/C++ family-split models)
from ood_suite import load_ood  # noqa: E402

ood = load_ood()
macros["OODMedLen"] = n(ood["len"].median())
if c:
    te = pd.read_parquet(OUT / "cpp_random_test.parquet", columns=["len"])
    macros["CppMedAll"] = n(te["len"].median())
cats = ["juliet_style", "real_world", "safe_lookalike", "trick"]
lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{External Evaluation of Models Trained on the C/C++ Family-Disjoint Split}",
         r"\label{tab:external}", r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{\textbf{Devign test (2{,}732)}} & \multicolumn{4}{c}{\textbf{Diagnostic suite: correct / n}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
         r"\textbf{Model} & \textbf{Acc.} & \textbf{F1} & \textbf{Pred.\ vuln.} & \textbf{Juliet-style} & \textbf{Real-world} & \textbf{Safe look-alike} & \textbf{Safe, no API} \\",
         r"\midrule"]
dev_f1 = []
rfx = load("rf_external_cpp.json")
ext_rows = [("Random Forest", rfx["runs"] if rfx else [])]
for m, lab in (("codebert", "CodeBERT"), ("graphcodebert", "GraphCodeBERT (token-only)")):
    ext_rows.append((lab, ft_runs(m, "cpp", "family")))
ext_rows.append(("Hybrid (frozen)", hy_runs("cpp", "family")))
for lab, runs in ext_rows:
    runs = [r for r in runs if "devign" in r]
    if not runs:
        lines.append(f"{lab} & " + " & ".join([MISSING] * 7) + r" \\")
        continue
    dev_f1 += col(runs, "devign")
    pv = [r["devign"]["tp"] + r["devign"]["fp"] for r in runs]
    cells = [ms(col(runs, "devign", "acc")), ms(col(runs, "devign")), f"{np.mean(pv) / 2732:.0%}".replace("%", r"\%")]
    if "ood35_by_category" not in runs[0] and "ood35" in runs[0]:
        tp_ = np.mean([r["ood35"]["tp"] for r in runs])
        tn_ = np.mean([r["ood35"]["tn"] for r in runs])
        cells.append(f"\\multicolumn{{2}}{{c}}{{{tp_:.1f} / 18 vulnerable}}")
        cells.append(f"\\multicolumn{{2}}{{c}}{{{tn_:.1f} / 17 safe}}")
    elif "ood35_by_category" in runs[0]:
        for cat in cats:
            vals = [r["ood35_by_category"].get(cat, {}).get("correct", 0) for r in runs]
            nn = runs[0]["ood35_by_category"].get(cat, {}).get("n", 0)
            cells.append(f"{np.mean(vals):.1f} / {nn}")
    else:
        cells += ["--"] * 4
    lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\footnotesize Means over seeds. Devign F1 of a constant all-vulnerable classifier: 0.630. "
          r"Diagnostic suite counts are averaged over seeds; for the hybrid only the confusion matrix was recorded, so its results are split into vulnerable and safe functions.}", r"\end{table*}"]
(REV / "tab_external.tex").write_text("\n".join(lines))
macros["DevMinF"] = f3(min(dev_f1)) if dev_f1 else MISSING
macros["DevMaxF"] = f3(max(dev_f1)) if dev_f1 else MISSING

# faithfulness table
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Masking Top-$k$ Attended Tokens Versus $k$ Random Tokens (C/C++ Family Split, Seed 0)}",
         r"\label{tab:faith}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"& & \multicolumn{2}{c}{\textbf{Mean $\Delta p$}} & \multicolumn{2}{c}{\textbf{Flip rate}} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"\textbf{Model} & $k$ & \textbf{Attn.} & \textbf{Rand.} & \textbf{Attn.} & \textbf{Rand.} \\", r"\midrule"]
cmp_words = []
for m, lab in (("codebert", "CodeBERT"), ("graphcodebert", "GraphCodeBERT (token-only)")):
    x = load(f"rq3_{m}_cpp_family_s0.json")
    for k in (1, 3, 5, 10):
        if x:
            a, rd = x[f"attn{k}"], x[f"rand{k}"]
            lines.append(f"{lab if k == 1 else ''} & {k} & {a['mean_delta']:.4f} & {rd['mean_delta']:.4f} & "
                         f"{pct(a['flip_rate'])} & {pct(rd['flip_rate'])} \\\\")
            cmp_words.append(a["flip_rate"] - rd["flip_rate"])
        else:
            lines.append(f"{lab if k == 1 else ''} & {k} & {MISSING} & {MISSING} & {MISSING} & {MISSING} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(REV / "tab_faith.tex").write_text("\n".join(lines))
macros["FaithCompare"] = (MISSING if not cmp_words else
                          ("more often" if np.mean(cmp_words) > 0 else "no more often"))

# accuracy by length quintile (fine-tuned models, family split, mean over seeds)
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Fine-Tuned Model Accuracy by Test-Length Quintile (Family Split, Mean Over Seeds)}",
         r"\label{tab:quintile}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"\textbf{Lang.} & \textbf{Model} & \textbf{Q1} & \textbf{Q2} & \textbf{Q3} & \textbf{Q4} & \textbf{Q5} \\",
         r"\midrule"]
for l, name in (("cpp", "C/C++"), ("java", "Java")):
    for m, lab in (("codebert", "CodeBERT"), ("graphcodebert", "GraphCodeBERT (token-only)")):
        runs = ft_runs(m, l, "family")
        if runs:
            cells = [f3(np.mean([r["test_by_len_quintile"][str(q)]["acc"] for r in runs])) for q in range(5)]
        else:
            cells = [MISSING] * 5
        lines.append(f"{name if m == 'codebert' else ''} & {lab} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\footnotesize Q1: shortest fifth of test functions (sanitised characters); Q5: longest.}",
          r"\end{table}"]
(REV / "tab_quintile.tex").write_text("\n".join(lines))

# diagnostic-suite length against the Juliet test distribution
if c:
    te_len = pd.read_parquet(OUT / "cpp_random_test.parquet", columns=["len"])["len"]
    p10 = np.percentile(te_len, 10)
    macros["CppPTen"] = n(p10)
    macros["OODBelowPTen"] = pct((ood["len"] < p10).mean(), 0)

# deduplication accounting and flaw-line truncation
td = load("truncation_dedup.json")
if td:
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{From Labelled Functions to the Deduplicated Set}",
             r"\label{tab:dedup}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"\textbf{Lang.} & \textbf{Label} & \textbf{Labelled} & \textbf{Conflict} & \textbf{Duplicate} & \textbf{Unique} \\",
             r"\midrule"]
    for l, name in (("cpp", "C/C++"), ("java", "Java")):
        for lb, labn in (("vuln", "Vulnerable"), ("safe", "Safe")):
            vals = [td[f"{l}_{lb}_{k}"] for k in ("labelled", "removed_conflict", "removed_exact_dup", "unique")]
            lines.append(f"{name if lb == 'vuln' else ''} & {labn} & " + " & ".join(n(v) for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"\vspace{2pt}{\footnotesize Conflict: the sanitised text also occurs with the opposite label (all copies removed). "
              r"Duplicate: further exact copies of a kept text.}", r"\end{table}"]
    (REV / "tab_dedup.tex").write_text("\n".join(lines))
    b = td["cpp_flaw_beyond_256"]
    macros["CppFlawBeyond"] = pct(b, 0)
    macros["JavaFlawBeyond"] = pct(td["java_flaw_beyond_256"], 1)
    recs = (col(ft_runs("codebert", "cpp", "family"), "test", "rec")
            + col(ft_runs("graphcodebert", "cpp", "family"), "test", "rec"))
    if recs:
        macros["CppRecall"] = f3(np.mean(recs))
        macros["CppBeyondDetected"] = pct((np.mean(recs) - (1 - b)) / b, 0)

# per-CWE appendix text: unique vulnerable/safe functions over all splits
for l in ("cpp", "java"):
    cov = pd.read_csv(OUT / f"coverage_{l}_per_cwe.csv", index_col=0)
    vc = [x for x in cov.columns if x.endswith("vuln")]
    sc = [x for x in cov.columns if x.endswith("safe")]
    keys = sorted((k for k in cov.index if k != "unknown"), key=lambda k: int(k.split("-")[1]))
    cells = [f"{k.split('-')[1]}: {int(cov.loc[k, vc].sum())}/{int(cov.loc[k, sc].sum())}" for k in keys]
    (REV / f"app_cwe_{l}.tex").write_text("; ".join(cells) + ".")

# shrink any table wider than its column (or page, for table*) to fit
for f in REV.glob("tab_*.tex"):
    t = f.read_text()
    if "adjustbox" not in t:
        t = t.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}")
        t = t.replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
        f.write_text(t)

macros["RepoURL"] ="https://github.com/smadhu6364-beep/rethinking-juliet"

missing = sorted(k for k, v in macros.items() if v == MISSING)
body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
(REV / "results.tex").write_text("% generated by analysis/make_tables.py\n" + body + "\n")
used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}|\\([A-Z][A-Za-z]+)\b", (REV / "main.tex").read_text()))
used = {a or b for a, b in used}
undefined = sorted(u for u in used if u not in macros and u not in {"PARstart", "EOD"})
print("macros:", len(macros), "| missing values:", missing, "| used but undefined:", undefined)
