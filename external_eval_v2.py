"""
External Out-of-Distribution Evaluation v2 - 35 hand-written C/C++ samples
DCU MSc Practicum 2026 - Vulnerability Detection Project

Categories:
  A. Juliet-style vulnerable (10 samples)
  B. Real-world vulnerable patterns (8 samples)
  C. Safe lookalikes - dangerous APIs used correctly (10 samples)
  D. Adversarial trick cases (7 samples)

Total: 35 samples (18 vulnerable, 17 safe)

Run: python external_eval_v2.py
Requires: CodeBERT checkpoint — pass --checkpoint <path> or set the
CODEBERT_CHECKPOINT env var. Defaults to a relative path under experiments/.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from datetime import datetime

DEFAULT_CHECKPOINT = "experiments/codebert-test/checkpoint-1000"

# ============================================================
# 35 EVALUATION SAMPLES
# label: 1 = vulnerable, 0 = safe
# ============================================================

SAMPLES = [

# ---------- CATEGORY A: Juliet-style vulnerable (10) ----------
{
    "id": "A01_strcpy_overflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void processInput(char *userInput) {
    char buffer[16];
    strcpy(buffer, userInput);
    printf("Processed: %s\\n", buffer);
}"""
},
{
    "id": "A02_gets_overflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void readName(void) {
    char name[32];
    gets(name);
    printf("Hello %s\\n", name);
}"""
},
{
    "id": "A03_sprintf_overflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void formatMessage(char *user) {
    char msg[24];
    sprintf(msg, "Welcome back, %s!", user);
    puts(msg);
}"""
},
{
    "id": "A04_system_injection",
    "category": "juliet_style",
    "label": 1,
    "code": """void runCommand(char *filename) {
    char cmd[128];
    sprintf(cmd, "cat %s", filename);
    system(cmd);
}"""
},
{
    "id": "A05_int_overflow_alloc",
    "category": "juliet_style",
    "label": 1,
    "code": """void allocateItems(int count) {
    int size = count * sizeof(int);
    int *arr = (int *)malloc(size);
    for (int i = 0; i < count; i++) arr[i] = i;
    free(arr);
}"""
},
{
    "id": "A06_strcat_overflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void appendPath(char *dir, char *file) {
    char fullpath[64];
    strcpy(fullpath, dir);
    strcat(fullpath, "/");
    strcat(fullpath, file);
    fopen(fullpath, "r");
}"""
},
{
    "id": "A07_memcpy_overflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void copyPacket(char *data, int len) {
    char packet[256];
    memcpy(packet, data, len);
    parsePacket(packet);
}"""
},
{
    "id": "A08_scanf_overflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void readInput(void) {
    char buf[20];
    scanf("%s", buf);
    process(buf);
}"""
},
{
    "id": "A09_int_underflow",
    "category": "juliet_style",
    "label": 1,
    "code": """void decrementCounter(unsigned int n) {
    unsigned int counter = 0;
    counter = counter - n;
    allocateBuffer(counter);
}"""
},
{
    "id": "A10_use_after_free",
    "category": "juliet_style",
    "label": 1,
    "code": """void handleData(void) {
    char *data = (char *)malloc(64);
    strcpy(data, "payload");
    free(data);
    printf("%s\\n", data);
}"""
},

# ---------- CATEGORY B: Real-world vulnerable (8) ----------
{
    "id": "B01_format_string",
    "category": "real_world",
    "label": 1,
    "code": """void logMessage(char *userMsg) {
    FILE *log = fopen("app.log", "a");
    fprintf(log, userMsg);
    fclose(log);
}"""
},
{
    "id": "B02_toctou_race",
    "category": "real_world",
    "label": 1,
    "code": """void writeConfig(char *path) {
    if (access(path, W_OK) == 0) {
        FILE *f = fopen(path, "w");
        fputs("config=1", f);
        fclose(f);
    }
}"""
},
{
    "id": "B03_path_traversal",
    "category": "real_world",
    "label": 1,
    "code": """void serveFile(char *requested) {
    char path[256];
    snprintf(path, sizeof(path), "/var/www/%s", requested);
    FILE *f = fopen(path, "r");
    sendContents(f);
}"""
},
{
    "id": "B04_off_by_one",
    "category": "real_world",
    "label": 1,
    "code": """void copyString(char *src) {
    char dest[16];
    int i;
    for (i = 0; i <= 16 && src[i] != '\\0'; i++)
        dest[i] = src[i];
    dest[i] = '\\0';
}"""
},
{
    "id": "B05_null_deref",
    "category": "real_world",
    "label": 1,
    "code": """void processRecord(int id) {
    Record *r = lookupRecord(id);
    printf("Name: %s\\n", r->name);
    updateTimestamp(r);
}"""
},
{
    "id": "B06_double_free",
    "category": "real_world",
    "label": 1,
    "code": """void cleanup(char *buf, int error) {
    if (error) {
        free(buf);
        logError(error);
    }
    free(buf);
}"""
},
{
    "id": "B07_signed_unsigned",
    "category": "real_world",
    "label": 1,
    "code": """void readChunk(int len, char *src) {
    char chunk[512];
    if (len < 512) {
        memcpy(chunk, src, len);
    }
}"""
},
{
    "id": "B08_unchecked_malloc",
    "category": "real_world",
    "label": 1,
    "code": """void storeValue(int val) {
    int *slot = (int *)malloc(sizeof(int));
    *slot = val;
    registerSlot(slot);
}"""
},

# ---------- CATEGORY C: Safe lookalikes (10) ----------
{
    "id": "C01_safe_fgets",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void read_line(FILE *fp) {
    char buf[64];
    if (fgets(buf, sizeof(buf), fp) != NULL)
        printf("%s", buf);
}"""
},
{
    "id": "C02_safe_strncpy",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void copyName(char *src) {
    char dest[32];
    strncpy(dest, src, sizeof(dest) - 1);
    dest[sizeof(dest) - 1] = '\\0';
    printf("%s\\n", dest);
}"""
},
{
    "id": "C03_safe_snprintf",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void buildGreeting(char *user) {
    char msg[64];
    snprintf(msg, sizeof(msg), "Hello, %s!", user);
    puts(msg);
}"""
},
{
    "id": "C04_safe_memcpy_checked",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void copyData(char *src, size_t len) {
    char dest[128];
    if (len > sizeof(dest)) len = sizeof(dest);
    memcpy(dest, src, len);
    process(dest, len);
}"""
},
{
    "id": "C05_safe_strcpy_constant",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void setDefault(void) {
    char status[16];
    strcpy(status, "OK");
    logStatus(status);
}"""
},
{
    "id": "C06_safe_malloc_checked",
    "category": "safe_lookalike",
    "label": 0,
    "code": """int *createArray(size_t n) {
    if (n == 0 || n > 1024) return NULL;
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) return NULL;
    memset(arr, 0, n * sizeof(int));
    return arr;
}"""
},
{
    "id": "C07_safe_free_once",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void releaseBuffer(char *buf) {
    if (buf != NULL) {
        free(buf);
        buf = NULL;
    }
}"""
},
{
    "id": "C08_safe_scanf_width",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void readToken(void) {
    char token[20];
    if (scanf("%19s", token) == 1)
        handleToken(token);
}"""
},
{
    "id": "C09_safe_strcat_bounded",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void joinPath(char *dir, char *file) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    if (strstr(path, "..") == NULL)
        openFile(path);
}"""
},
{
    "id": "C10_safe_fprintf_literal",
    "category": "safe_lookalike",
    "label": 0,
    "code": """void logEvent(char *event) {
    FILE *log = fopen("events.log", "a");
    if (log != NULL) {
        fprintf(log, "%s\\n", event);
        fclose(log);
    }
}"""
},

# ---------- CATEGORY D: Adversarial trick cases (7) ----------
{
    "id": "D01_empty_function",
    "category": "trick",
    "label": 0,
    "code": """void noop(void) {
}"""
},
{
    "id": "D02_pure_arithmetic",
    "category": "trick",
    "label": 0,
    "code": """int computeSum(int a, int b) {
    int result = a + b;
    return result * 2;
}"""
},
{
    "id": "D03_string_constants",
    "category": "trick",
    "label": 0,
    "code": """const char *getVersion(void) {
    return "v2.1.0-release";
}"""
},
{
    "id": "D04_simple_getter",
    "category": "trick",
    "label": 0,
    "code": """int getCount(const Counter *c) {
    return c != NULL ? c->count : 0;
}"""
},
{
    "id": "D05_boolean_logic",
    "category": "trick",
    "label": 0,
    "code": """int isValid(int age, int score) {
    return (age >= 18) && (score > 0) && (score <= 100);
}"""
},
{
    "id": "D06_loop_counter",
    "category": "trick",
    "label": 0,
    "code": """int countEven(const int *arr, int n) {
    int count = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] % 2 == 0) count++;
    return count;
}"""
},
{
    "id": "D07_comment_flaw_word",
    "category": "trick",
    "label": 0,
    "code": """int max(int a, int b) {
    /* No flaw here, just returns the larger value */
    return a > b ? a : b;
}"""
},
]

# ============================================================
# EVALUATION
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="External OOD evaluation v2")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CODEBERT_CHECKPOINT", DEFAULT_CHECKPOINT),
        help="Path to the CodeBERT checkpoint directory "
             "(default: %(default)s, or set CODEBERT_CHECKPOINT)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint = args.checkpoint

    print("Loading CodeBERT from:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    results = []
    correct_by_cat = {}
    total_by_cat = {}

    for s in SAMPLES:
        inputs = tokenizer(s["code"], return_tensors="pt",
                           truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs))
            conf = float(probs[0][pred])

        is_correct = (pred == s["label"])
        cat = s["category"]
        total_by_cat[cat] = total_by_cat.get(cat, 0) + 1
        if is_correct:
            correct_by_cat[cat] = correct_by_cat.get(cat, 0) + 1

        results.append({
            "id": s["id"],
            "category": cat,
            "true_label": s["label"],
            "predicted": pred,
            "confidence": round(conf, 4),
            "correct": is_correct
        })
        status = "CORRECT" if is_correct else "WRONG"
        true_str = "VULN" if s["label"] == 1 else "SAFE"
        pred_str = "VULN" if pred == 1 else "SAFE"
        print(f"  [{status:7s}] {s['id']:30s} true={true_str} pred={pred_str} conf={conf:.4f}")

    print()
    print("=" * 60)
    print("RESULTS BY CATEGORY")
    print("=" * 60)
    total_correct = 0
    for cat in ["juliet_style", "real_world", "safe_lookalike", "trick"]:
        c = correct_by_cat.get(cat, 0)
        t = total_by_cat.get(cat, 0)
        total_correct += c
        print(f"  {cat:20s}: {c}/{t}")
    n = len(SAMPLES)
    print("-" * 60)
    print(f"  OVERALL: {total_correct}/{n} = {100*total_correct/n:.1f}%")
    print("=" * 60)

    out = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": n,
        "overall_correct": total_correct,
        "overall_accuracy": round(total_correct/n, 4),
        "by_category": {cat: f"{correct_by_cat.get(cat,0)}/{total_by_cat.get(cat,0)}"
                        for cat in total_by_cat},
        "per_sample": results
    }
    with open("results/external_eval_v2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\\nSaved: results/external_eval_v2_results.json")

if __name__ == "__main__":
    main()
