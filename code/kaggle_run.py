"""
Kaggle entry point (GPU T4 x2). Paste into one notebook cell:

    !python /kaggle/input/juliet-revision/kaggle_run.py

Copies the scripts and splits to /kaggle/working, then runs every
fine-tuning job on the FULL training partitions (3 epochs, paper settings),
two jobs at a time, one per GPU, and finally the hybrid head. Results land in
/kaggle/working/analysis/out/*.json and are zipped to /kaggle/working/results.zip.
Jobs whose result file exists are skipped, so re-running resumes.
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

SRC = Path(glob.glob("/kaggle/input/**/kaggle_run.py", recursive=True)[0]).parent
WORK = Path("/kaggle/working/analysis")
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)
(WORK.parent / "external" / "devign").mkdir(parents=True, exist_ok=True)

for f in SRC.glob("*.py"):
    shutil.copy(f, WORK / f.name)
for f in SRC.glob("*.parquet"):
    if f.name == "devign_test.parquet":
        shutil.copy(f, WORK.parent / "external" / "devign" / "test.parquet")
    else:
        shutil.copy(f, OUT / f.name)
shutil.copy(SRC / "ood35.json", OUT / "ood35.json")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.44"], check=False)

jobs = []
for s in (0, 1, 2):
    for m in ("codebert", "graphcodebert"):
        jobs.append((m, "cpp", "family", s))
for m in ("codebert", "graphcodebert"):
    jobs.append((m, "cpp", "random", 0))
for lang in ("java", "csharp"):
    for s in (0, 1, 2):
        for m in ("codebert", "graphcodebert"):
            jobs.append((m, lang, "family", s))

lock = threading.Lock()


def worker(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        with lock:
            if not jobs:
                return
            m, lang, scheme, s = jobs.pop(0)
        tag = f"{m}_{lang}_{scheme}_s{s}"
        if (OUT / f"ft_{tag}.json").exists():
            continue
        print(f"[gpu{gpu}] start {tag}", flush=True)
        with open(OUT / f"log_ft_{tag}.txt", "w") as log:
            r = subprocess.run([sys.executable, "finetune.py", m, lang, scheme, str(s),
                                "--epochs", "3"], cwd=WORK, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        print(f"[gpu{gpu}] {'done' if r.returncode == 0 else 'FAILED'} {tag}", flush=True)


threads = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
for t in threads:
    t.start()
for t in threads:
    t.join()

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HYBRID_N_TRAIN="10000000")
for lang, scheme in (("cpp", "family"), ("cpp", "random"), ("java", "family"), ("csharp", "family")):
    if not (OUT / f"hybrid_{lang}_{scheme}.json").exists():
        print("hybrid", lang, scheme, flush=True)
        with open(OUT / f"log_hybrid_{lang}_{scheme}.txt", "w") as log:
            subprocess.run([sys.executable, "hybrid.py", lang, scheme], cwd=WORK, env=env,
                           stdout=log, stderr=subprocess.STDOUT)

for tag in ("codebert_cpp_family_s0", "graphcodebert_cpp_family_s0"):
    if not (OUT / f"rq3_{tag}.json").exists():
        with open(OUT / f"log_rq3_{tag}.txt", "w") as log:
            subprocess.run([sys.executable, "rq3.py", tag], cwd=WORK, env=env,
                           stdout=log, stderr=subprocess.STDOUT)

shutil.rmtree(OUT / "emb", ignore_errors=True)
shutil.rmtree(OUT / "ckpt", ignore_errors=True)
for f in OUT.glob("*.parquet"):
    f.unlink()
shutil.make_archive("/kaggle/working/results", "zip", OUT)
print("ALL DONE", flush=True)
