"""
External stress test for the trained CodeBERT model.

Runs inference on hand-written C/C++ samples that are NOT from Juliet,
to test whether the model generalizes beyond the training distribution.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "experiments/sanity-check/checkpoint-625"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

LABEL_MAP = {0: "SAFE", 1: "VULNERABLE"}

# --------------------------------------------------
# TEST CASES
# --------------------------------------------------
# Format: (description, code, expected_label, category)
# Expected labels: "SAFE" (0) or "VULNERABLE" (1)
# Categories: juliet_style, real_vuln, safe_lookalike, trick

test_cases = [
    # ============== JULIET-STYLE VULNERABLE (model should get these) ==============
    (
        "Classic strcpy buffer overflow",
        """
        void copy_data(char *input) {
            char buffer[10];
            strcpy(buffer, input);
        }
        """,
        "VULNERABLE",
        "juliet_style"
    ),
    (
        "gets() — unsafe input",
        """
        void read_user() {
            char buf[64];
            gets(buf);
            printf("%s", buf);
        }
        """,
        "VULNERABLE",
        "juliet_style"
    ),
    (
        "Integer overflow in malloc",
        """
        void allocate(int count) {
            char *data = malloc(count * sizeof(int));
            for (int i = 0; i < count; i++) {
                data[i] = 0;
            }
        }
        """,
        "VULNERABLE",
        "juliet_style"
    ),
    
    # ============== JULIET-STYLE SAFE (model should get these too) ==============
    (
        "Safe strncpy with explicit null-terminator",
        """
        void copy_data(char *input) {
            char buffer[10];
            strncpy(buffer, input, sizeof(buffer) - 1);
            buffer[sizeof(buffer) - 1] = '\\0';
        }
        """,
        "SAFE",
        "juliet_style"
    ),
    (
        "Bounded fgets read",
        """
        void read_user(FILE *fp) {
            char buf[64];
            if (fgets(buf, sizeof(buf), fp) != NULL) {
                printf("%s", buf);
            }
        }
        """,
        "SAFE",
        "juliet_style"
    ),
    
    # ============== REAL-WORLD VULNERABILITY (NOT Juliet style) ==============
    (
        "TOCTOU race condition (not in Juliet)",
        """
        int access_file(const char *path) {
            if (access(path, R_OK) == 0) {
                FILE *fp = fopen(path, "r");
                return fread_data(fp);
            }
            return -1;
        }
        """,
        "VULNERABLE",
        "real_vuln"
    ),
    (
        "SQL-like injection through format string",
        """
        void log_message(const char *user_msg) {
            printf(user_msg);
        }
        """,
        "VULNERABLE",
        "real_vuln"
    ),
    (
        "Use-after-free pattern",
        """
        void process(int *data) {
            free(data);
            *data = 42;
        }
        """,
        "VULNERABLE",
        "real_vuln"
    ),
    
    # ============== SAFE CODE THAT LOOKS DANGEROUS ==============
    (
        "Looks like buffer overflow but is safe",
        """
        void copy_fixed(const char *src) {
            char dest[100];
            memcpy(dest, src, 5);
            dest[5] = '\\0';
        }
        """,
        "SAFE",
        "safe_lookalike"
    ),
    (
        "strcpy with manually-verified short constant",
        """
        void init_label(char *out) {
            const char *label = "OK";
            strcpy(out, label);
        }
        """,
        "SAFE",
        "safe_lookalike"
    ),
    
    # ============== TRICKY / OUT-OF-DISTRIBUTION ==============
    (
        "Empty function",
        """
        void noop() {
        }
        """,
        "SAFE",
        "trick"
    ),
    (
        "Pure arithmetic, no I/O",
        """
        int compute_sum(int a, int b) {
            return a + b;
        }
        """,
        "SAFE",
        "trick"
    ),
    (
        "Vulnerable but tiny",
        """
        void f(char *x) { strcpy(0, x); }
        """,
        "VULNERABLE",
        "trick"
    ),
]

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------

def predict(code):
    """Return (label_string, confidence_float) for a code snippet."""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()
    
    return LABEL_MAP[pred_id], confidence


# --------------------------------------------------
# EVALUATE EACH TEST CASE
# --------------------------------------------------

print("\n" + "=" * 80)
print("EXTERNAL STRESS TEST — model trained on Juliet only")
print("=" * 80)

results_by_category = {}
total_correct = 0
total_count = 0

for description, code, expected, category in test_cases:
    prediction, confidence = predict(code)
    correct = (prediction == expected)
    
    total_count += 1
    if correct:
        total_correct += 1
    
    results_by_category.setdefault(category, []).append(correct)
    
    status = "✓" if correct else "✗"
    print(f"\n[{category}] {description}")
    print(f"   Expected: {expected:11s} | Predicted: {prediction:11s} ({confidence:.4f}) {status}")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nOverall: {total_correct}/{total_count} correct ({100 * total_correct / total_count:.1f}%)")

print("\nBy category:")
for category, results in results_by_category.items():
    correct = sum(results)
    total = len(results)
    pct = 100 * correct / total
    print(f"  {category:18s} {correct}/{total} ({pct:.1f}%)")

print("\n" + "=" * 80)
print("Interpretation:")
print("- If juliet_style is 100% but real_vuln is low → model memorized Juliet, didn't generalize")
print("- If real_vuln is also high → model genuinely learned vulnerability patterns")
print("- If safe_lookalike is wrong → model has false-positive bias on suspicious-looking code")
print("=" * 80)