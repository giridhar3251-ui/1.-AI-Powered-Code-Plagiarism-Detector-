"""
normalizer.py
-------------
Parses Python source into an AST and produces a "normalized" representation
that strips away everything cosmetic (variable names, literal values,
comments, formatting) while preserving program STRUCTURE and LOGIC.

This is the foundation of the whole detector: two files that are logically
identical but renamed/reformatted should normalize to the same (or very
similar) representation.
"""

import ast
from dataclasses import dataclass, field


class IdentifierRenamer(ast.NodeTransformer):
    """
    Walks an AST and replaces every variable/function/argument name with a
    placeholder (VAR0, VAR1, ...), consistently, so the same original name
    always maps to the same placeholder within one file.

    Built-in names and imported module/function names are left alone, since
    renaming `print` or `len` would destroy meaningful structure (calling a
    built-in vs. calling a user function is itself a structural signal).
    """

    BUILTIN_NAMES = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))

    def __init__(self):
        self.mapping = {}
        self.counter = 0

    def _placeholder_for(self, name: str) -> str:
        if name in self.mapping:
            return self.mapping[name]
        placeholder = f"VAR{self.counter}"
        self.mapping[name] = placeholder
        self.counter += 1
        return placeholder

    def visit_Name(self, node):
        if node.id not in self.BUILTIN_NAMES:
            node.id = self._placeholder_for(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._placeholder_for(node.arg)
        return node

    def visit_FunctionDef(self, node):
        # Rename the function itself too (so renaming a function name doesn't
        # break a match), but keep visiting children normally.
        node.name = self._placeholder_for(node.name)
        self.generic_visit(node)
        return node

    def visit_arguments(self, node):
        self.generic_visit(node)
        return node


class LiteralStripper(ast.NodeTransformer):
    """Replaces literal constants (numbers, strings) with placeholders so
    that changing a literal value (e.g. threshold = 5 -> threshold = 10)
    doesn't count as a structural difference."""

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return node  # keep True/False, they affect logic directly
        if isinstance(node.value, (int, float, complex)):
            node.value = 0
        elif isinstance(node.value, str):
            node.value = "STR"
        return node


@dataclass
class NormalizedFunction:
    name: str               # original function name (for reporting)
    placeholder_name: str   # normalized name (VARx)
    node_sequence: str      # flattened AST node-type sequence
    source_normalized: str  # unparsed normalized source (for display)
    lineno: int
    features: dict = field(default_factory=dict)


def flatten_node_types(node) -> str:
    """
    Walks a (normalized) AST node and returns a flattened string of node
    type names in traversal order. This sequence is what gets compared by
    the structural similarity engine in matcher.py.

    Example: FunctionDef,arguments,arg,If,Compare,Name,Eq,Constant,Return...
    """
    types = []

    def walk(n):
        types.append(type(n).__name__)
        for child in ast.iter_child_nodes(n):
            walk(child)

    walk(node)
    return ",".join(types)


def extract_features(node) -> dict:
    """
    Extracts simple structural/complexity features from a function's AST.
    These feed into the scoring engine as an additional signal independent
    of raw sequence matching (two functions can have very different node
    sequences but similar complexity profiles, or vice versa).
    """
    counts = {
        "if_count": 0,
        "for_count": 0,
        "while_count": 0,
        "call_count": 0,
        "return_count": 0,
        "try_count": 0,
        "max_depth": 0,
        "num_nodes": 0,
    }

    def walk(n, depth):
        counts["num_nodes"] += 1
        counts["max_depth"] = max(counts["max_depth"], depth)
        if isinstance(n, ast.If):
            counts["if_count"] += 1
        elif isinstance(n, ast.For):
            counts["for_count"] += 1
        elif isinstance(n, ast.While):
            counts["while_count"] += 1
        elif isinstance(n, ast.Call):
            counts["call_count"] += 1
        elif isinstance(n, ast.Return):
            counts["return_count"] += 1
        elif isinstance(n, ast.Try):
            counts["try_count"] += 1
        for child in ast.iter_child_nodes(n):
            walk(child, depth + 1)

    walk(node, 0)
    # Cyclomatic complexity approximation: 1 + number of decision points
    counts["cyclomatic_complexity"] = (
        1 + counts["if_count"] + counts["for_count"] + counts["while_count"] + counts["try_count"]
    )
    return counts


def normalize_source(source: str):
    """
    Parses raw Python source and returns (whole_file_normalized_tree,
    list_of_NormalizedFunction). Raises SyntaxError if the source doesn't
    parse -- caller should handle/report this per-file rather than crash
    the whole batch.
    """
    tree = ast.parse(source)

    # Extract features and originals BEFORE renaming (need original names
    # for reporting) but on a copy, since renaming mutates the tree in place.
    import copy
    original_tree = copy.deepcopy(tree)

    renamer = IdentifierRenamer()
    tree = renamer.visit(tree)
    tree = LiteralStripper().visit(tree)
    ast.fix_missing_locations(tree)

    functions = []
    orig_func_nodes = [n for n in ast.walk(original_tree) if isinstance(n, ast.FunctionDef)]
    norm_func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    for orig_node, norm_node in zip(orig_func_nodes, norm_func_nodes):
        try:
            normalized_src = ast.unparse(norm_node)
        except Exception:
            normalized_src = ""
        functions.append(
            NormalizedFunction(
                name=orig_node.name,
                placeholder_name=norm_node.name,
                node_sequence=flatten_node_types(norm_node),
                source_normalized=normalized_src,
                lineno=orig_node.lineno,
                features=extract_features(orig_node),
            )
        )

    whole_file_sequence = flatten_node_types(tree)
    return whole_file_sequence, functions
