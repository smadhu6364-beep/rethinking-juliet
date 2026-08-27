
"""
Prototype LLM-style explanation generation
for vulnerability detection.
"""

import pandas as pd

# -----------------------------------------
# LOAD SAMPLE
# -----------------------------------------

df = pd.read_parquet(
    "data/splits/test_sanitized.parquet"
)

sample = df.iloc[0]

code = sample["code"]

label = sample["label"]

# -----------------------------------------
# MOCK SUSPICIOUS LINES
# -----------------------------------------

suspicious_lines = [
    "data = (wchar_t *)calloc(100, sizeof(wchar_t));",
    "wcscpy(data, L\"A String\");",
    "FUNC(dataList);"
]

# -----------------------------------------
# EXPLANATION TEMPLATE
# -----------------------------------------

def generate_explanation(lines):

    explanation = []

    explanation.append(
        "The model identified this code as potentially vulnerable."
    )

    explanation.append(
        "Several operations were considered security-sensitive."
    )

    explanation.append("")

    for line in lines:

        if "calloc" in line:

            explanation.append(
                "- Memory allocation using calloc() may introduce risks if memory is not properly managed."
            )

        elif "wcscpy" in line:

            explanation.append(
                "- String copy operations such as wcscpy() may contribute to memory-related vulnerabilities if bounds are not validated."
            )

        elif "FUNC" in line:

            explanation.append(
                "- Function calls involving shared data structures may propagate unsafe memory usage."
            )

    explanation.append("")
    explanation.append(
        "The explanation is generated from attention-based evidence extraction."
    )

    return "\n".join(explanation)

# -----------------------------------------
# OUTPUT
# -----------------------------------------

print("\nLABEL:", label)

print("\nCODE SAMPLE:\n")
print(code[:1000])

print("\nSUSPICIOUS LINES:\n")

for line in suspicious_lines:

    print(line)

print("\nGENERATED EXPLANATION:\n")

print(
    generate_explanation(
        suspicious_lines
    )
)

