"""
Smoke test using small, hand-written Juliet-style snippets -- NOT the real
Juliet dataset (not available yet). This only proves the parsing, labeling,
dedup, and split logic execute correctly end to end. It says nothing about
what will happen on the real 100k+ function dataset, and must not be
quoted as if it does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.multilang.parsers.java_parser import JavaFunctionParser
from preprocessing.multilang.parsers.csharp_parser import CSharpFunctionParser
from preprocessing.multilang.labeling import label_function
from preprocessing.multilang.dedup_split_multilang import (
    deduplicate, stratified_split, verify_zero_overlap,
)
import pandas as pd

JAVA_SAMPLE = """
package testcases.CWE89_SQL_Injection;

public class CWE89_bad {
    public void bad() throws Throwable {
        // FLAW: this string is not sanitized
        String query = "SELECT * FROM users WHERE id = " + userId;
        statement.execute(query);
    }
}

public class CWE89_good1 {
    public void goodG2B() throws Throwable {
        String query = "SELECT * FROM users WHERE id = ?";
        PreparedStatement ps = connection.prepareStatement(query);
    }
}
"""

CSHARP_SAMPLE = """
namespace testcases.CWE89_SQL_Injection
{
    public class CWE89_bad
    {
        public void bad()
        {
            // FLAW: unsanitized input
            string query = "SELECT * FROM Users WHERE Id = " + userId;
            command.ExecuteReader(query);
        }
    }

    public class CWE89_good1
    {
        public void goodG2B()
        {
            string query = "SELECT * FROM Users WHERE Id = @id";
        }
    }
}
"""


def test_java_parser_finds_methods():
    parser = JavaFunctionParser()
    functions = parser.extract_functions(JAVA_SAMPLE)
    names = {f.function_name for f in functions}
    assert names == {"bad", "goodG2B"}, f"Unexpected function set: {names}"
    print(f"[PASS] Java parser found: {names}")


def test_csharp_parser_finds_methods():
    parser = CSharpFunctionParser()
    functions = parser.extract_functions(CSHARP_SAMPLE)
    names = {f.function_name for f in functions}
    assert names == {"bad", "goodG2B"}, f"Unexpected function set: {names}"
    print(f"[PASS] C# parser found: {names}")


def test_labeling_and_leakage_counting():
    result = label_function(
        "bad", "CWE89_SQL_Injection/CWE89_bad.java",
        "// FLAW: this string is not sanitized\nvoid bad() {}"
    )
    assert result.label == 1
    assert result.cwe_id == "CWE89"
    assert result.leakage_token_counts["FLAW"] == 1
    print(f"[PASS] Labeling: label={result.label}, cwe={result.cwe_id}, "
          f"leakage={result.leakage_token_counts}")

    safe_result = label_function("goodG2B", "CWE89/CWE89_good1.java", "void goodG2B() {}")
    assert safe_result.label == 0
    print("[PASS] Safe-label case correct")


def test_dedup_and_split_pipeline():
    # Build a small synthetic dataset spanning several fake CWE categories,
    # large enough for stratified splitting to be meaningful.
    rows = []
    for cwe_num in range(1, 11):
        for i in range(20):
            rows.append({
                "source_code": f"void bad_{cwe_num}_{i}() {{ /* CWE{cwe_num} vuln {i} */ }}",
                "cwe_id": f"CWE{cwe_num}",
                "label": 1,
            })
            rows.append({
                "source_code": f"void good_{cwe_num}_{i}() {{ /* CWE{cwe_num} safe {i} */ }}",
                "cwe_id": f"CWE{cwe_num}",
                "label": 0,
            })
    # Inject 10 exact duplicates to confirm dedup actually removes them.
    rows.extend(rows[:10])
    df = pd.DataFrame(rows)

    deduped = deduplicate(df)
    assert len(deduped) == len(df) - 10, "Dedup did not remove the expected duplicates"

    splits = stratified_split(deduped, random_state=42)
    verify_zero_overlap(splits)  # raises on failure

    total = sum(len(s) for s in splits.values())
    assert total == len(deduped)
    print(f"[PASS] Dedup+split: {len(deduped)} rows -> "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, "
          f"overlap verified clean")


if __name__ == "__main__":
    test_java_parser_finds_methods()
    test_csharp_parser_finds_methods()
    test_labeling_and_leakage_counting()
    test_dedup_and_split_pipeline()
    print("\nAll smoke tests passed. This confirms the pipeline logic is\n"
          "correct on synthetic data -- it does NOT confirm behavior on\n"
          "the real Juliet Java/C# source trees, which have not been run yet.")
