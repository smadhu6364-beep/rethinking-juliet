"""
Java function extractor using tree-sitter.

Unlike the existing C/C++ pipeline (regex-based, per the repo audit), Java
needs real parsing: Juliet Java sources use nested classes, annotations,
and multi-line signatures that a brace-counting regex handles unreliably.
Tree-sitter gives a real parse tree, so function boundaries are exact.
"""

from __future__ import annotations

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from .base_parser import BaseFunctionParser, ExtractedFunction

_JAVA_LANGUAGE = Language(tsjava.language())


class JavaFunctionParser(BaseFunctionParser):
    """Extracts method declarations from Java source files."""

    language_name = "java"

    # Node types that represent something we treat as an extractable
    # "function" for this project's purposes. Constructors included
    # deliberately -- Juliet's bad()/good() style methods are ordinary
    # methods, but constructors can carry vulnerable initialization logic
    # too, and excluding them would silently shrink recall versus C/C++.
    _METHOD_NODE_TYPES = {"method_declaration", "constructor_declaration"}

    def __init__(self) -> None:
        self._parser = Parser(_JAVA_LANGUAGE)

    def extract_functions(self, source_text: str) -> list[ExtractedFunction]:
        source_bytes = source_text.encode("utf-8", errors="replace")
        tree = self._parser.parse(source_bytes)

        functions: list[ExtractedFunction] = []
        self._walk(tree.root_node, source_bytes, functions)
        return functions

    def _walk(self, node, source_bytes: bytes, out: list[ExtractedFunction]) -> None:
        if node.type in self._METHOD_NODE_TYPES:
            name_node = node.child_by_field_name("name")
            function_name = (
                source_bytes[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                if name_node is not None
                else "<anonymous>"
            )
            body_text = source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )
            out.append(
                ExtractedFunction(
                    function_name=function_name,
                    source_code=body_text,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
        for child in node.children:
            self._walk(child, source_bytes, out)
