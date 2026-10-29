"""
Function-level re-extraction of the Juliet Test Suite (C/C++, Java, C#),
labelled against Juliet's own flaw metadata rather than function names alone.

Why this exists: the original extractor used a regex that only matched
function bodies with at most one level of nested braces and only labelled a
function vulnerable if its name ended in "_bad". C++ test cases put bad()
inside a namespace (no "_bad" suffix), and most bad() bodies nest deeper than
one level, so a large share of vulnerable functions were silently dropped.

Labelling rule (all three languages):
  vulnerable = a bad-flow function whose body contains a flaw line
               (C/C++, Java: a <flaw line=...> entry in manifest.xml;
                C#: no manifest ships, so a FLAW / POTENTIAL FLAW comment,
                the oracle we validate against the manifest on C/C++ and Java)
  safe       = a good-flow function (good, goodG2B*, goodB2G*, good helpers)
               containing no flaw line
  excluded   = everything else, with a recorded reason

Output: one parquet per language with every scanned function, its role,
label (or exclusion reason), manifest/comment oracle flags and line span.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(r"C:\Users\m3s36\Projects\research\csnet26-vuln-benchmark-artefacts\data\raw")
OUT = Path(__file__).resolve().parent / "out"

LANGS = {
    "cpp": dict(root=RAW / "juliet-cpp" / "C" / "testcases", exts={".c", ".cpp"},
                manifest=RAW / "juliet-cpp" / "C" / "manifest.xml"),
    "java": dict(root=RAW / "Java" / "src" / "testcases", exts={".java"},
                 manifest=RAW / "Java" / "manifest.xml"),
    "csharp": dict(root=RAW / "csharp" / "testcases", exts={".cs"}, manifest=None),
}

CONTROL = {"if", "for", "while", "switch", "catch", "return", "sizeof", "do",
           "else", "using", "lock", "foreach", "fixed", "synchronized", "new",
           "try", "finally", "checked", "unchecked", "unsafe", "get", "set",
           "delete", "throw", "case", "default", "when"}

HEADER_RE = re.compile(r"(~?[A-Za-z_]\w*)\s*\((?:[^()]|\([^()]*\))*\)\s*"
                       r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?"
                       r"(?:throws\s+[\w.\s,]+)?\s*(?::\s*base\s*\([^)]*\)\s*)?$",
                       re.DOTALL)
FLAW_COMMENT_RE = re.compile(r"(/\*|//)\s*(POTENTIAL\s+)?FLAW", re.IGNORECASE)
CWE_PREFIX_RE = re.compile(r"^CWE\d+_\w*?_\d+[a-z]?_(?=[a-z])")
FAMILY_RE = re.compile(r"_(\d{2})([a-z]?|_?bad|_?good\w*)$")


def scan(text: str):
    """Yield (name, start_offset, end_offset) for every top-level function body.

    Braces inside comments, strings and char literals are ignored. A '{' opens a
    function when the text since the previous ';', '{' or '}' (preprocessor
    lines removed) looks like `name(args) [qualifiers]` and `name` is not a
    control keyword. Functions nested inside a function are not reported
    separately (they stay part of the outer body).
    """
    n = len(text)
    i = 0
    seg_start = 0          # start of the current header candidate
    stack = []             # entries: ("fn", name, start) or ("blk",)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"' or c == "'":
            # C# verbatim strings @"..." have no backslash escapes
            verbatim = c == '"' and i > 0 and text[i - 1] == "@"
            j = i + 1
            while j < n:
                if not verbatim and text[j] == "\\":
                    j += 2
                    continue
                if text[j] == c:
                    if verbatim and j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                if text[j] == "\n" and c == "'":
                    break
                j += 1
            i = j + 1
            continue
        if c == "#" and text[text.rfind("\n", 0, i) + 1:i].strip(" \t") == "":
            # preprocessor line (handle line continuations)
            j = i
            while True:
                k = text.find("\n", j)
                if k < 0:
                    j = n
                    break
                if text[k - 1] == "\\":
                    j = k + 1
                    continue
                j = k
                break
            if not any(e[0] == "fn" for e in stack):
                seg_start = j
            i = j
            continue
        if c == "{":
            in_fn = any(e[0] == "fn" for e in stack)
            if not in_fn:
                header = text[seg_start:i]
                header_clean = re.sub(r"\[[^\]]*\]", " ", header)  # C# attributes
                m = HEADER_RE.search(header_clean.strip())
                if m and m.group(1) not in CONTROL:
                    # function start = first non-space char of the header
                    off = seg_start + (len(header) - len(header.lstrip()))
                    stack.append(("fn", m.group(1), off))
                else:
                    stack.append(("blk",))
            else:
                stack.append(("blk",))
            i += 1
            seg_start = i
            continue
        if c == "}":
            if stack:
                e = stack.pop()
                if e[0] == "fn":
                    yield e[1], e[2], i + 1
            i += 1
            seg_start = i
            continue
        if c == ";":
            seg_start = i + 1
        i += 1


def role_of(name: str) -> str:
    s = CWE_PREFIX_RE.sub("", name)
    if s[:1].isupper() and re.match(r"(Bad|Good)", s):   # C# PascalCase
        s = s[0].lower() + s[1:]
    if s == "main":
        return "main"
    if s == "bad":
        return "bad"
    if s == "good":
        return "good"
    if re.match(r"good(G2B|B2G)\d*$", s):
        return "goodG2B/B2G"
    if re.match(r"bad", s) or re.search(r"[a-z]Bad|_bad", s):
        return "bad-helper"
    if re.match(r"good", s) or re.search(r"[a-z]Good|_good", s):
        return "good-helper"
    return "other"


def family_of(path: Path) -> str:
    """Functional-variant id: file stem without the flow-variant number and
    multi-file suffix, e.g. CWE121_..._CWE129_fgets_01.c and ..._52b.c both
    map to CWE121_..._CWE129_fgets."""
    stem = path.stem
    m = re.search(r"_\d{2}[a-z]?(_bad|_good\w*)?$", stem)
    return stem[:m.start()] if m else stem


def cwe_of(path: Path) -> str:
    m = re.match(r"CWE(\d+)", path.stem)
    return f"CWE-{m.group(1)}" if m else "unknown"


SINK_CALL_RE = re.compile(r"\w*[bB]ad\w*(?:Sink|Source|Helper|Base|Class|Function)\w*\s*\(|badSink|_bad\w*\s*\("
                          r"|\w+_\d+[a-z]_\w*\(|badVaSink|->action\(|\.action\(|badObject|baseObject")


def comment_oracle(name: str, code: str) -> bool:
    """Flaw oracle for suites without a manifest (C#): a FLAW / POTENTIAL FLAW
    comment in a function that is not a data source and does not hand data
    on to another bad-flow function. Measured against the manifest:
    C/C++ precision 0.963, recall 0.914 (see audit)."""
    body = code.split("{", 1)[1] if "{" in code else ""
    return (bool(FLAW_COMMENT_RE.search(code)) and "Source" not in name
            and not SINK_CALL_RE.search(body))


def load_manifest(path: Path | None) -> dict[str, set[int]]:
    """manifest.xml -> {file name: flaw line numbers}. Parsed line by line
    because the Java manifest has unbalanced <testcase> tags."""
    if path is None:
        return {}
    flaws: dict[str, set[int]] = {}
    cur = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r'<file path="([^"]+)"', line)
        if m:
            cur = Path(m.group(1)).name
            flaws.setdefault(cur, set())
            continue
        m = re.search(r'<flaw line="(\d+)"', line)
        if m and cur:
            flaws[cur].add(int(m.group(1)))
    return flaws


def java_flaw_map() -> dict[str, set[str]]:
    """C# Juliet is a one-to-one port of Java Juliet (same test-case and
    function names). For each Java file, the lower-cased names of the
    bad-flow functions that contain a manifest flaw line."""
    j = pd.read_parquet(OUT / "functions_java.parquet",
                        columns=["file", "function_name", "role", "manifest_flaw"])
    j = j[j.manifest_flaw & j.role.isin(["bad", "bad-helper"])]
    m: dict[str, set[str]] = {}
    for f, n in zip(j.file, j.function_name):
        m.setdefault(Path(f).stem, set()).add(n.lower())
    return m


def extract(lang: str) -> pd.DataFrame:
    cfg = LANGS[lang]
    manifest = load_manifest(cfg["manifest"])
    jmap = java_flaw_map() if lang == "csharp" else None
    if jmap is not None:
        java_files = set(pd.read_parquet(OUT / "functions_java.parquet", columns=["file"])
                         .file.map(lambda f: Path(f).stem))
    rows = []
    files = [p for p in cfg["root"].rglob("*") if p.suffix in cfg["exts"]]
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        # line index for offset -> line number
        nl = [m.start() for m in re.finditer("\n", text)]
        import bisect

        def line_at(off):
            return bisect.bisect_right(nl, off - 1) + 1

        flaw_lines = manifest.get(fp.name, set())
        in_manifest = fp.name in manifest
        for name, s, e in scan(text):
            code = text[s:e]
            l0, l1 = line_at(s), line_at(e - 1)
            has_manifest_flaw = any(l0 <= ln <= l1 for ln in flaw_lines)
            has_flaw_comment = bool(FLAW_COMMENT_RE.search(code))
            oracle_c = comment_oracle(name, code)
            role = role_of(name)
            if role in ("bad", "bad-helper"):
                if jmap is None:
                    oracle = has_manifest_flaw
                elif fp.stem not in java_files:
                    oracle = None
                else:
                    oracle = name.lower() in jmap.get(fp.stem, set())
                if oracle is None:
                    label, reason = None, "bad-flow, no matching Java test case"
                elif oracle:
                    label, reason = 1, "bad-flow, contains flaw line"
                else:
                    label, reason = None, "bad-flow, no flaw line (wrapper/source)"
            elif role in ("good", "goodG2B/B2G", "good-helper"):
                if cfg["manifest"] and has_manifest_flaw:
                    label, reason = None, "good-flow but manifest flaw line"
                else:
                    label, reason = 0, "good-flow"
            elif role == "main":
                label, reason = None, "main() harness"
            else:
                label, reason = None, "support/other function"
            rows.append(dict(lang=lang, cwe=cwe_of(fp), family=family_of(fp),
                             file=fp.name, function_name=name, role=role,
                             start_line=l0, end_line=l1, label=label, reason=reason,
                             in_manifest=in_manifest, manifest_flaw=has_manifest_flaw,
                             flaw_comment=has_flaw_comment, comment_oracle=oracle_c,
                             code_raw=code))
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int64")
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for lang in (sys.argv[1:] or LANGS):
        df = extract(lang)
        df.to_parquet(OUT / f"functions_{lang}.parquet", index=False)
        print(lang, len(df), "functions;", df.label.value_counts(dropna=False).to_dict())
