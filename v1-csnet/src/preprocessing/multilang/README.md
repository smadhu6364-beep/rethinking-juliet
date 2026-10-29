# Java / C# Extraction Pipeline

## What this is

A self-contained extension to the existing Juliet preprocessing pipeline,
adding Java and C# support. Lives entirely under
`src/preprocessing/multilang/` — no existing C/C++ file is imported,
modified, or overwritten by anything here.

## Status — read this before using any output in the paper

**Updated 2026-07-08: has now been run against the real Juliet Java/C#
source trees** (`data/raw/Java`, `data/raw/csharp`). The "synthetic data
only" status below is history, not current — kept for the record of what
was verified before the real run, not as a caveat that still applies.

What was verified before the real run (see `tests/test_smoke.py`, all
passing) on hand-written synthetic snippets:

- The Java parser correctly finds methods and constructors.
- The C# parser does the same.
- Labeling correctly assigns 1/vulnerable to `bad`-named functions, 0/safe
  to `good`-named functions, extracts the CWE id from the path, and counts
  leakage tokens (FLAW/POTENTIAL/FIX/INCIDENTAL) in the function body.
- Deduplication correctly removes exact-string duplicates.
- Stratified train/val/test splitting works and produces zero overlap,
  verified programmatically, not assumed.

What the real run against the actual source trees found:

- Java: 127,822 raw extracted functions; C#: 168,809. Both far larger than
  the synthetic smoke tests, as expected.
- The `leak_token_*` distribution **does** match the C/C++ pattern:
  FIX-token frequency differs ~17x between label 0 and label 1 (0.65 vs
  0.04 mean occurrences per function, both languages). G2B/B2G substrings
  were present in 58.2% of rows in both languages, 100% of which mapped to
  label 0 — a real, quantified leak, not a hypothetical one. See
  `src/preprocessing/sanitize.py` and `apply_sanitization.py` for the fix
  that was applied in response (comment stripping + G2B/B2G/bad/good
  identifier normalisation), and the "Known limitations" section below,
  which is no longer accurate as originally written.
- Once names/comments stopped making near-duplicate rows look distinct,
  real deduplication rates were much higher than expected: Java
  127,822 → 33,510 unique rows (73.8% were duplicates), C# 168,809 →
  43,914 (73.9%). Most of the raw dataset's apparent size was superficial
  variation (different CWE-id/name/comment wrapping around the same
  underlying code template), not distinct code.

## How to run it (extraction — already done once, kept for re-runs)

```bash
pip install tree_sitter tree-sitter-java tree-sitter-c-sharp pandas scikit-learn pyarrow

python -m preprocessing.multilang.run_pipeline \
    --languages java csharp \
    --java-dir /path/to/juliet/java \
    --csharp-dir /path/to/juliet/csharp \
    --output-root data/processed/multilang
```

Or set `JULIET_JAVA_DIR` / `JULIET_CSHARP_DIR` environment variables instead
of passing `--java-dir` / `--csharp-dir` — see `config.py`. No path is ever
hardcoded.

Output lands under `data/processed/multilang/{java,csharp}/`:
- `{language}_functions_raw.parquet` — all extracted functions, pre-dedup
- `train.parquet`, `val.parquet`, `test.parquet` — post-dedup, stratified,
  overlap-verified

None of this touches the existing `data/processed/` C/C++ files.

## Known limitations

- ~~Name sanitisation removes `bad`/`good` from function names but does not
  remove `G2B`/`B2G` substrings~~ — **fixed 2026-07-08.**
  `src/preprocessing/sanitize.py` now strips `bad`/`good`/`G2B`/`B2G`
  identifiers (including the function's own name in its signature line,
  not just the `sanitized_function_name` metadata field) plus
  FLAW/POTENTIAL/FIX/INCIDENTAL comments from the actual code body used
  for training. Applied identically to C/C++ and Java/C# via
  `apply_sanitization.py`, so the fix doesn't introduce a new asymmetry
  between languages — it closes the same gap on both sides at once.
- The `assign_label` function raises an error (does not guess) if a
  function name doesn't match the bad/good convention. If real Juliet
  Java/C# code has naming exceptions not seen in C/C++, this pipeline will
  surface them as errors during the real run rather than silently
  mislabeling. That's the point — but expect some manual triage of
  edge cases on first run.

## Before training any model on this data — done, kept as the record

1. ~~Run `run_pipeline.py` against the real source trees.~~ Done — 127,822
   Java / 168,809 C# raw functions extracted.
2. ~~Check the logged extraction counts and failure rate per language.~~
   Done, logged at extraction time.
3. ~~Check the `leak_token_*` distribution — does it match the C/C++
   pattern?~~ Done — it did (see Status section above), and was fixed via
   `sanitize.py`/`apply_sanitization.py`.
4. Training scripts that actually exist and were used: RF
   (`train_random_forest_multilang.py`), CodeBERT
   (`train_codebert_multilang.py`), GraphCodeBERT
   (`train_graphcodebert_multilang.py`), Hybrid
   (`train_hybrid_multilang.py`) — all under `src/models/`, all taking
   `--language {java,csharp}`. Colab notebooks for the deep models are
   under `notebooks/07-10_*_colab.ipynb`.
