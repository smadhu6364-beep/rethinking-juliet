"""
External test sets, passed through the same sanitize + whitespace
normalisation as the training text so that train and test inputs match.

  load_ood()    - the 35 hand-written C/C++ functions (external_eval_v2.py)
  load_devign() - CodeXGLUE Devign test split, 2,732 real C functions from
                  FFmpeg and QEMU commits (Zhou et al., NeurIPS 2019)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from build_splits import normalise

SUITE = Path(r"C:\Users\m3s36\Projects\research\csnet26-vuln-benchmark-artefacts\external_eval_v2.py")
DEVIGN = Path(__file__).resolve().parent.parent / "external" / "devign" / "test.parquet"


SUITE_JSON = Path(__file__).resolve().parent / "out" / "ood35.json"


def load_ood() -> pd.DataFrame:
    if SUITE_JSON.exists():
        samples = json.loads(SUITE_JSON.read_text(encoding="utf-8"))
    else:
        # read the SAMPLES literal without importing the script (it loads a model)
        tree = ast.parse(SUITE.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SAMPLES":
                samples = ast.literal_eval(node.value)
                break
        SUITE_JSON.write_text(json.dumps(samples, indent=1), encoding="utf-8")
    df = pd.DataFrame(samples)
    df["code_raw"] = df["code"]
    df["code"] = df.code_raw.map(normalise)
    df["len"] = df.code.str.len()
    return df


def load_devign() -> pd.DataFrame:
    d = pd.read_parquet(DEVIGN)
    df = pd.DataFrame({"code_raw": d.func, "label": d.target.astype(int), "project": d.project})
    df["code"] = df.code_raw.map(normalise)
    df["len"] = df.code.str.len()
    return df
