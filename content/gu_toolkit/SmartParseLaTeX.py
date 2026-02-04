from sympy.parsing.latex import parse_latex as _sympy_parse_latex

__all__=["LatexParseError","parse_latex"]
class LatexParseError(RuntimeError):
    pass

def parse_latex(tex: str, *args, **kwargs):
    """
    Notebook-local override: defaults to lark, falls back to antlr.
    If user explicitly passes backend=..., respect it.
    """
    backend = kwargs.get("backend", None)

    if backend is not None:
        return _sympy_parse_latex(tex, *args, **kwargs)

    lark_err = None
    try:
        return _sympy_parse_latex(tex, *args, backend="lark", **kwargs)
    except Exception as e:
        lark_err = e

    try:
        return _sympy_parse_latex(tex, *args, backend="antlr", **kwargs)
    except Exception as antlr_err:
        raise LatexParseError(
            "Failed to parse LaTeX with both backends.\n"
            f"Input: {tex!r}\n"
            f"Lark error: {type(lark_err).__name__}: {lark_err}\n"
            f"ANTLR error: {type(antlr_err).__name__}: {antlr_err}"
        ) from antlr_err
