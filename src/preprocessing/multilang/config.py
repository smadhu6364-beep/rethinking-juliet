"""
Configuration for the Java/C# extraction pipeline.

Design principle: no hardcoded paths anywhere. Every path is supplied via
CLI argument or environment variable at call time. This mirrors the fix
already applied to external_eval_v2.py (hardcoded checkpoint -> configurable
--checkpoint / CODEBERT_CHECKPOINT env var) so the same lesson isn't relearned
in a second file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageConfig:
    """Per-language settings needed by the extraction pipeline."""

    name: str                  # "java" or "csharp"
    source_root_env: str       # env var name for this language's Juliet root
    file_extensions: tuple[str, ...]
    tree_sitter_language: str  # key used to look up the compiled grammar


JAVA = LanguageConfig(
    name="java",
    source_root_env="JULIET_JAVA_DIR",
    file_extensions=(".java",),
    tree_sitter_language="java",
)

CSHARP = LanguageConfig(
    name="csharp",
    source_root_env="JULIET_CSHARP_DIR",
    file_extensions=(".cs",),
    tree_sitter_language="c_sharp",
)

SUPPORTED_LANGUAGES = {"java": JAVA, "csharp": CSHARP}


def resolve_source_root(language: str, cli_path: str | None) -> Path:
    """
    Resolve the Juliet source root for a given language.

    Priority: explicit CLI path > environment variable > error.
    Never falls back to a hardcoded default — if neither is supplied,
    this fails loudly instead of silently pointing at the wrong place.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {list(SUPPORTED_LANGUAGES)}"
        )

    if cli_path:
        root = Path(cli_path)
    else:
        env_name = SUPPORTED_LANGUAGES[language].source_root_env
        env_value = os.environ.get(env_name)
        if not env_value:
            raise RuntimeError(
                f"No source root provided for '{language}'. "
                f"Pass --{language}-dir explicitly or set {env_name}."
            )
        root = Path(env_value)

    if not root.exists():
        raise FileNotFoundError(
            f"Juliet {language} source root does not exist: {root}"
        )
    return root


# Output locations. Deliberately separate from the existing C/C++ output
# directories so nothing here can collide with or overwrite prior results.
DEFAULT_OUTPUT_ROOT = Path("data/processed/multilang")
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/multilang")
