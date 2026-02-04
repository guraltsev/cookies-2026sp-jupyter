"""
Minimal verification script for NamedFunction.

Run (from the same directory as NamedFunction.py):

    python test_namedfunction.py

This is intentionally pytest-free to keep dependencies minimal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sympy as sp


def _import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_docstring_latex_from_sympy_expr() -> None:
    mod = _import_module_from_path("NamedFunction", Path("NamedFunction.py"))
    NamedFunction = mod.NamedFunction

    @NamedFunction
    def SquarePlusOne(x):
        """Return x^2 + 1 as a NamedFunction example."""
        return x**2 + 1

    doc = SquarePlusOne.__doc__ or ""
    assert r"\mathrm{SquarePlusOne}(x) = x^{2} + 1" in doc, doc
    assert r"\mathtt{\text{x**2 + 1}}" not in doc, doc


def test_docstring_latex_from_string_expr() -> None:
    mod = _import_module_from_path("NamedFunction", Path("NamedFunction.py"))
    NamedFunction = mod.NamedFunction

    @NamedFunction
    def SquarePlusOneStr(x):
        """Return x^2 + 1, but as a string (should still render as math in the docstring)."""
        return "x**2 + 1"

    doc = SquarePlusOneStr.__doc__ or ""
    assert r"\mathrm{SquarePlusOneStr}(x) = x^{2} + 1" in doc, doc
    assert r"\mathtt{\text{x**2 + 1}}" not in doc, doc


def test_expand_definition_rewrite() -> None:
    mod = _import_module_from_path("NamedFunction", Path("NamedFunction.py"))
    NamedFunction = mod.NamedFunction

    x = sp.Symbol("x")

    @NamedFunction
    def F(x):
        return x + 1

    expr = F(x).rewrite("expand_definition")
    assert sp.simplify(expr - (x + 1)) == 0, expr


def test_opaque_definition_stays_opaque() -> None:
    mod = _import_module_from_path("NamedFunction", Path("NamedFunction.py"))
    NamedFunction = mod.NamedFunction

    x = sp.Symbol("x")

    @NamedFunction
    def G(x):
        return None

    expr = G(x).rewrite("expand_definition")
    assert expr == G(x), expr


def main() -> None:
    tests = [
        test_docstring_latex_from_sympy_expr,
        test_docstring_latex_from_string_expr,
        test_expand_definition_rewrite,
        test_opaque_definition_stays_opaque,
    ]
    for t in tests:
        t()
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
