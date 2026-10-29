"""RQ3 reviewer-study analysis: Fleiss' kappa + rating summaries.

Reads ratings either directly from the study's Google Sheet (default, using the
service-account JSON in rq3_reviewer/credentials/) or from a CSV downloaded via
the app's Tab 3 (--csv path/to/ratings.csv).

Fleiss' kappa requires every item to be rated by the same number of raters, so
the kappa is computed on the subset of samples rated by ALL reviewers; the
report says how many samples that excluded. Run only after every recruited
reviewer has finished all 20 samples.

Usage:
    python analyze_rq3_ratings.py            # read live sheet
    python analyze_rq3_ratings.py --csv f.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict

SPREADSHEET_ID = "11tq2jk4qRVbEih-8KmgkQvXy-S0dKp8izKdy0FMjGe0"
WORKSHEET_NAME = "ratings_v2"
CREDENTIALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials")

# From the app's hardcoded SAMPLES: the model predicts VULNERABLE on all 20,
# so it is correct exactly on the 12 vulnerable-labelled samples (60%, the
# paper's Section V-D figure) and wrong on all 8 safe ones.
SAMPLE_META = {
    **{f"{cwe}_v{i}": {"cwe": cwe, "model_correct": True}
       for cwe in ("CWE121", "CWE78", "CWE190", "CWE191") for i in (1, 2, 3)},
    **{f"{cwe}_s{i}": {"cwe": cwe, "model_correct": False}
       for cwe in ("CWE121", "CWE78", "CWE190", "CWE191") for i in (1, 2)},
}
RATING_CATEGORIES = (1, 2, 3, 4, 5)


def load_from_sheet():
    import gspread
    from google.oauth2.service_account import Credentials

    json_files = [f for f in os.listdir(CREDENTIALS_DIR) if f.endswith(".json")]
    if not json_files:
        sys.exit(f"No service-account JSON found in {CREDENTIALS_DIR}")
    creds = Credentials.from_service_account_file(
        os.path.join(CREDENTIALS_DIR, json_files[0]),
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
    ws = gspread.authorize(creds).open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()
    return rows


def load_from_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fleiss_kappa(item_ratings):
    """item_ratings: list of lists, each inner list = the ratings one item
    received (all items must have the same number of ratings)."""
    n_raters = len(item_ratings[0])
    if any(len(r) != n_raters for r in item_ratings):
        raise ValueError("All items must have the same number of ratings")
    n_items = len(item_ratings)

    counts = [[r.count(c) for c in RATING_CATEGORIES] for r in item_ratings]
    total = n_items * n_raters
    p_j = [sum(row[j] for row in counts) / total for j in range(len(RATING_CATEGORIES))]
    p_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
           for row in counts]
    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    if p_e == 1.0:
        return 1.0  # every rating identical across the board
    return (p_bar - p_e) / (1 - p_e)


def interpret(k):
    for bound, label in ((0.80, "almost perfect"), (0.60, "substantial"),
                         (0.40, "moderate"), (0.20, "fair"), (0.00, "slight")):
        if k >= bound:
            return label
    return "poor"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="analyze a downloaded ratings CSV instead of the live sheet")
    args = ap.parse_args()

    rows = load_from_csv(args.csv) if args.csv else load_from_sheet()
    rows = [r for r in rows if r.get("sample_id") in SAMPLE_META]
    if not rows:
        sys.exit("No ratings found.")

    by_reviewer = defaultdict(dict)
    for r in rows:
        by_reviewer[r["reviewer_id"]][r["sample_id"]] = {
            "correctness": int(r["correctness"]),
            "usefulness": int(r["usefulness"]),
        }

    reviewers = sorted(by_reviewer)
    print(f"Reviewers: {len(reviewers)}")
    for rev in reviewers:
        print(f"  {rev}: {len(by_reviewer[rev])}/20 samples rated")

    complete = [s for s in SAMPLE_META
                if all(s in by_reviewer[rev] for rev in reviewers)]
    print(f"\nSamples rated by ALL reviewers: {len(complete)}/20 "
          f"(kappa computed on these only)")

    if len(reviewers) < 2:
        print("\nNeed at least 2 reviewers for inter-rater agreement; "
              "summary stats only.")
    elif len(complete) >= 2:
        for dim in ("correctness", "usefulness"):
            matrix = [[by_reviewer[rev][s][dim] for rev in reviewers] for s in complete]
            k = fleiss_kappa(matrix)
            print(f"Fleiss' kappa ({dim}): {k:.3f} ({interpret(k)})")

    all_ratings = [(s, by_reviewer[rev][s]) for rev in reviewers
                   for s in by_reviewer[rev]]
    for dim in ("correctness", "usefulness"):
        overall = mean([v[dim] for _, v in all_ratings])
        right = mean([v[dim] for s, v in all_ratings if SAMPLE_META[s]["model_correct"]])
        wrong = mean([v[dim] for s, v in all_ratings if not SAMPLE_META[s]["model_correct"]])
        print(f"\nMean {dim}: {overall:.2f} overall | "
              f"{right:.2f} on model-correct (12 vuln) | "
              f"{wrong:.2f} on model-incorrect (8 safe)")

    print("\nPer-CWE mean correctness/usefulness:")
    for cwe in ("CWE121", "CWE78", "CWE190", "CWE191"):
        cs = [v["correctness"] for s, v in all_ratings if SAMPLE_META[s]["cwe"] == cwe]
        us = [v["usefulness"] for s, v in all_ratings if SAMPLE_META[s]["cwe"] == cwe]
        print(f"  {cwe}: {mean(cs):.2f} / {mean(us):.2f}  (n={len(cs)})")

    out = {
        "n_reviewers": len(reviewers),
        "reviewers": {rev: len(by_reviewer[rev]) for rev in reviewers},
        "complete_samples": len(complete),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "rq3_analysis_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSummary written to rq3_analysis_summary.json")


if __name__ == "__main__":
    main()
