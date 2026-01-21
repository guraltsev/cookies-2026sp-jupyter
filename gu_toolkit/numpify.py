"""
numpify: Compile SymPy expressions to NumPy-callable Python functions
====================================================================

Purpose
-------
Turn a SymPy expression into a callable Python function that evaluates using NumPy.

This module is intentionally small and "batteries included" for interactive research
workflows where you want:

- explicit argument order,
- fast vectorized evaluation via NumPy broadcasting,
- support for *custom* SymPy functions that carry a NumPy implementation (``f_numpy``),
- and inspectable generated source.

Supported Python versions
-------------------------
- Python >= 3.10

Dependencies
------------
- NumPy (required)
- SymPy (required)

Public API
----------
- :func:`numpify`
- :func:`numpify_cached`

How custom functions are handled
--------------------------------
SymPy's NumPy code printer cannot natively print arbitrary user-defined SymPy Functions.
When ``expr`` contains an unknown function call such as ``G(x)``, SymPy raises a
``PrintMethodNotImplementedError`` by default.

This module enables SymPy's ``allow_unknown_functions`` option so unknown functions are
printed as plain calls (``G(x)``). We then provide runtime bindings so that the name
``G`` resolves to a callable.

Bindings are resolved in this order:

1. Explicit bindings provided via the ``f_numpy`` argument (for function classes).
2. Auto-detection: for each function ``F`` appearing in the expression, if
   ``callable(getattr(F, "f_numpy", None))`` then that callable is used.

If an unknown function remains unbound, :func:`numpify` raises a clear error before code
generation.

Examples
--------
>>> import numpy as np
>>> import sympy as sp
>>> from numpify import numpify
>>> x = sp.Symbol("x")

Constant compiled with broadcasting:
>>> f = numpify(5, args=x)
>>> float(f(0))
5.0
>>> f(np.array([1, 2, 3]))
array([5., 5., 5.])

Symbol binding (treat `a` as an injected constant):
>>> a = sp.Symbol("a")
>>> g = numpify(a * x, args=x, f_numpy={a: 2.0})
>>> g(np.array([1, 2, 3]))
array([2., 4., 6.])

Logging
-------
This module uses Python's standard :mod:`logging` library and is silent by default.
To enable debug logging in a notebook session:

>>> import logging
>>> import numpify
>>> logging.basicConfig(level=logging.DEBUG)
>>> logging.getLogger(numpify.__name__).setLevel(logging.DEBUG)

If you import this file as part of a package (e.g. ``gu_toolkit.numpify``), use that
module name instead.
"""

from __future__ import annotations

from functools import lru_cache

import logging
import time
import textwrap
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union, cast

import numpy as np
import sympy as sp
from sympy.core.function import FunctionClass
from sympy.printing.numpy import NumPyPrinter


__all__ = ["numpify", "numpify_cached"]


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_SymBindingKey = sp.Symbol
_FuncBindingKey = Union[FunctionClass, sp.Function]
_BindingKey = Union[_SymBindingKey, _FuncBindingKey]
_SymBindings = Dict[str, Any]
_FuncBindings = Dict[str, Callable[..., Any]]


def numpify(
    expr: Any,
    *,
    args: Optional[Union[sp.Symbol, Iterable[sp.Symbol]]] = None,
    f_numpy: Optional[Mapping[_BindingKey, Any]] = None,
    vectorize: bool = True,
    expand_definition: bool = True,
) -> Callable[..., Any]:
    """Compile a SymPy expression into a NumPy-evaluable Python function.

    Parameters
    ----------
    expr:
        A SymPy expression or anything convertible via :func:`sympy.sympify`.
    args:
        Symbols treated as *positional arguments* of the compiled function.

        - If None (default), uses all free symbols of ``expr`` sorted by name.
        - If a single Symbol, that symbol is the only argument.
        - If an iterable, argument order is preserved.

    f_numpy:
        Optional bindings for:
        - **Symbols** (constants/objects injected by name into the generated function body).
          Example: ``{a: 2.0}`` binds the symbol ``a`` to the value ``2.0``.
        - **SymPy function classes** (or applications) to NumPy-callable implementations.
          Example: ``{G: G.f_numpy}`` binds ``G(x)`` to the callable.

        In addition, if a function class ``F`` appears in ``expr`` and has a callable
        attribute ``F.f_numpy``, it is auto-bound.

    vectorize:
        If True, each argument is converted via ``numpy.asarray`` to enable broadcasting.
        If False, arguments are left as-is (scalar evaluation).

    expand_definition:
        If True, attempts to rewrite custom functions via
        ``expr.rewrite("expand_definition")`` (repeated to a fixed point), and then applies
        ``sympy.expand(..., deep=True)``.

        If a function is opaque (its rewrite returns itself), the function call remains
        in the expression and must be bound via ``f_numpy`` or ``F.f_numpy``.

    Returns
    -------
    Callable[..., Any]
        A generated function. The function includes its generated source in ``__doc__``.

    Raises
    ------
    TypeError
        If ``args`` is not a Symbol or an iterable of Symbols.
        If a function binding is provided but the value is not callable.
    ValueError
        If ``expr`` contains unbound symbols or unbound unknown functions.
        If symbol bindings overlap with argument symbols.

    Notes
    -----
    This function uses ``exec`` to define the generated function. Avoid calling it on
    untrusted expressions.
    """
    # 1) Normalize expr to SymPy.
    try:
        expr_sym = sp.sympify(expr)
    except Exception as e:
        raise TypeError(f"numpify expects a SymPy-compatible expression, got {type(expr)}") from e
    if not isinstance(expr_sym, sp.Basic):
        raise TypeError(f"numpify expects a SymPy expression, got {type(expr_sym)}")
    expr = cast(sp.Basic, expr_sym)

    # 2) Normalize args.
    args_tuple = _normalize_args(expr, args)

    log_debug = logger.isEnabledFor(logging.DEBUG)
    t_total0: float | None = time.perf_counter() if log_debug else None
    if log_debug:
        logger.debug("numpify: detected args=%s", [a.name for a in args_tuple])

    # 3) Optionally expand custom definitions.
    if expand_definition:
        expr = _rewrite_expand_definition(expr)
        expr = sp.expand(expr, deep=True)

    # 4) Parse bindings.
    sym_bindings, func_bindings = _parse_bindings(expr, f_numpy)

    # 5) Validate free symbols are accounted for (either args or symbol bindings).
    free_names = {s.name for s in expr.free_symbols}
    arg_names_set = {a.name for a in args_tuple}
    missing_names = free_names - arg_names_set - set(sym_bindings.keys())
    if missing_names:
        missing_str = ", ".join(sorted(missing_names))
        args_str = ", ".join(a.name for a in args_tuple)
        raise ValueError(
            "Expression contains unbound symbols: "
            f"{missing_str}. Provide them in args=({args_str}) or bind via f_numpy={{symbol: value}}."
        )

    # 6) Prevent accidental overwrites: symbol bindings cannot overlap with args.
    overlap = set(a.name for a in args_tuple) & set(sym_bindings.keys())
    if overlap:
        raise ValueError(
            "Symbol bindings overlap with args (would overwrite argument values): "
            + ", ".join(sorted(overlap))
        )

    # 7) Create printer (allow unknown functions to print as plain calls).
    printer = NumPyPrinter(settings={"user_functions": {}, "allow_unknown_functions": True})

    # 8) Preflight: any function that prints as a *bare* call must be bound.
    _require_bound_unknown_functions(expr, printer, func_bindings)

    # 9) Generate expression code and function source.
    arg_names = [a.name for a in args_tuple]

    # "Lambdification"-like code generation step: SymPy -> NumPy expression string.
    t_codegen0: float | None = time.perf_counter() if log_debug else None
    expr_code = printer.doprint(expr)
    t_codegen_s = (time.perf_counter() - t_codegen0) if t_codegen0 is not None else None
    is_constant = (len(expr.free_symbols) == 0)

    lines: list[str] = []
    lines.append("def _generated(" + ", ".join(arg_names) + "):")

    if vectorize:
        for nm in arg_names:
            lines.append(f"    {nm} = numpy.asarray({nm})")

    # Inject symbol bindings by name
    for nm in sorted(sym_bindings.keys()):
        lines.append(f"    {nm} = _sym_bindings[{nm!r}]")

    if vectorize and is_constant and len(arg_names) > 0:
        lines.append(f"    _shape = numpy.broadcast({', '.join(arg_names)}).shape")
        lines.append(f"    return ({expr_code}) + numpy.zeros(_shape)")
    else:
        lines.append(f"    return {expr_code}")

    src = "\n".join(lines)

    # Runtime globals dict compilation (kept separate for timing / debugging).
    t_dict0: float | None = time.perf_counter() if log_debug else None
    glb: Dict[str, Any] = {
        "numpy": np,
        "_sym_bindings": sym_bindings,
        **func_bindings,  # function names like "G" -> callable
    }
    t_dict_s = (time.perf_counter() - t_dict0) if t_dict0 is not None else None

    loc: Dict[str, Any] = {}

    t_exec0: float | None = time.perf_counter() if log_debug else None
    exec(src, glb, loc)
    t_exec_s = (time.perf_counter() - t_exec0) if t_exec0 is not None else None
    fn = cast(Callable[..., Any], loc["_generated"])

    fn.__doc__ = textwrap.dedent(
        f"""
        Auto-generated NumPy function from SymPy expression.

        expr: {repr(expr)}
        args: {arg_names}

        Source:
        {src}
        """
    ).strip()

    # Store generated source for inspection in interactive sessions.
    # Use setattr to avoid type-checker complaints on the Callable type.
    setattr(fn, "_generated_source", src)
    setattr(fn, "_generated_expr_code", expr_code)

    if log_debug:
        t_total_s = (time.perf_counter() - t_total0) if t_total0 is not None else None
        logger.debug(
            "numpify timings (ms): codegen=%.2f dict=%.2f exec=%.2f total=%.2f",
            1000.0 * (t_codegen_s or 0.0),
            1000.0 * (t_dict_s or 0.0),
            1000.0 * (t_exec_s or 0.0),
            1000.0 * (t_total_s or 0.0),
        )

    return fn


def _normalize_args(expr: sp.Basic, args: Optional[Union[sp.Symbol, Iterable[sp.Symbol]]]) -> Tuple[sp.Symbol, ...]:
    """Normalize args into a tuple of SymPy Symbols."""
    if args is None:
        args_tuple: Tuple[sp.Symbol, ...] = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
        return args_tuple

    if isinstance(args, sp.Symbol):
        return (args,)

    try:
        args_tuple = tuple(args)
    except TypeError as e:
        raise TypeError("args must be a SymPy Symbol or an iterable of SymPy Symbols") from e

    for a in args_tuple:
        arg_expr = sp.sympify(a)
        if not isinstance(arg_expr, sp.Symbol):
            raise TypeError(f"args must contain only SymPy Symbols, got {type(arg_expr)}")
    return cast(Tuple[sp.Symbol, ...], args_tuple)


def _rewrite_expand_definition(expr: sp.Basic, *, max_passes: int = 10) -> sp.Basic:
    """Rewrite using the 'expand_definition' target until stable (or max_passes)."""
    current = expr
    for _ in range(max_passes):
        nxt = current.rewrite("expand_definition")
        if nxt == current:
            break
        current = cast(sp.Basic, nxt)
    return current


def _parse_bindings(expr: sp.Basic, f_numpy: Optional[Mapping[_BindingKey, Any]]) -> Tuple[_SymBindings, _FuncBindings]:
    """Split user-provided bindings into symbol and function bindings, plus auto-bindings."""
    sym_bindings: _SymBindings = {}
    func_bindings: _FuncBindings = {}

    if f_numpy:
        for key, value in f_numpy.items():
            if isinstance(key, sp.Symbol):
                sym_bindings[key.name] = value
                continue

            if isinstance(key, sp.Function):
                name = key.func.__name__
                if not callable(value):
                    raise TypeError(f"Function binding for {name} must be callable, got {type(value)}")
                func_bindings[name] = cast(Callable[..., Any], value)
                continue

            if isinstance(key, FunctionClass):
                name = key.__name__
                if not callable(value):
                    raise TypeError(f"Function binding for {name} must be callable, got {type(value)}")
                func_bindings[name] = cast(Callable[..., Any], value)
                continue

            raise TypeError(
                "f_numpy keys must be SymPy Symbols or SymPy function objects/classes. "
                f"Got {type(key)}."
            )

    # Auto-bind NamedFunction-style implementations (F.f_numpy) when present.
    for app in expr.atoms(sp.Function):
        impl = getattr(app.func, "f_numpy", None)
        if callable(impl) and app.func.__name__ not in func_bindings:
            func_bindings[app.func.__name__] = cast(Callable[..., Any], impl)

    return sym_bindings, func_bindings


def _require_bound_unknown_functions(expr: sp.Basic, printer: NumPyPrinter, func_bindings: Mapping[str, Callable[..., Any]]) -> None:
    """Ensure any *bare* printed function calls have runtime bindings."""
    missing: set[str] = set()

    for app in expr.atoms(sp.Function):
        name = app.func.__name__
        try:
            code = printer.doprint(app).strip()
        except Exception:
            continue

        if code.startswith(f"{name}(") and name not in func_bindings:
            missing.add(name)

    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Expression contains unknown SymPy function(s) that require a NumPy implementation: "
            f"{missing_str}. Define `<F>.f_numpy` on the function class (e.g. via @NamedFunction), "
            "or pass `f_numpy={F: callable}` to numpify."
        )


# ---------------------------------------------------------------------------
# Cached compilation
# ---------------------------------------------------------------------------

_NUMPIFY_CACHE_MAXSIZE = 256


def _freeze_value_marker(value: Any) -> tuple[str, Any]:
    """Return a hashable marker for *value*.

    We prefer using the value itself when it is hashable (stable semantics).
    If it is unhashable (e.g. NumPy arrays, dicts), fall back to object identity.

    Notes
    -----
    - Using ``id(value)`` means cache hits are session-local and depend on the
      specific object instance. This is usually what you want for injected
      constants like arrays.
    """
    try:
        hash(value)
    except Exception:
        return ("ID", id(value))
    else:
        return ("H", value)


def _freeze_f_numpy_key(f_numpy: Optional[Mapping[_BindingKey, Any]]) -> tuple[tuple[Any, ...], ...]:
    """Normalize ``f_numpy`` to a hashable key for caching.

    The key is a sorted tuple of entries. Each entry includes:

    - a *normalized binding key* (symbol/function name identity)
    - a *value marker* (hashable value when possible, otherwise ``id(value)``)

    This function is intentionally conservative: it aims to prevent incorrect
    cache hits when bindings differ.
    """
    if not f_numpy:
        return tuple()

    frozen: list[tuple[Any, ...]] = []
    for k, v in f_numpy.items():
        if isinstance(k, sp.Symbol):
            k_norm = ("S", k.name)
        elif isinstance(k, sp.Function):
            # Bindings for applications behave like bindings for their function class.
            fc = k.func
            k_norm = ("F", getattr(fc, "__module__", ""), getattr(fc, "__qualname__", fc.__name__))
        elif isinstance(k, FunctionClass):
            k_norm = ("F", getattr(k, "__module__", ""), getattr(k, "__qualname__", k.__name__))
        else:
            # Should not happen: numpify() validates keys.
            k_norm = ("K", repr(k))

        v_mark = _freeze_value_marker(v)
        frozen.append((k_norm, v_mark))

    frozen.sort(key=lambda item: item[0])
    return tuple(tuple(x) for x in frozen)


class _FrozenFNumPy:
    """Small hashable wrapper around an ``f_numpy`` mapping.

    This exists solely so that :func:`functools.lru_cache` can cache compiled
    callables even when the mapping contains unhashable values (like NumPy arrays).

    The cache key is derived from a normalized, hashable view of the mapping.
    """

    __slots__ = ("mapping", "_key")

    def __init__(self, mapping: Optional[Mapping[_BindingKey, Any]]):
        self.mapping: dict[_BindingKey, Any] = {} if mapping is None else dict(mapping)
        self._key = _freeze_f_numpy_key(self.mapping)

    def __hash__(self) -> int:  # pragma: no cover
        return hash(self._key)

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        return isinstance(other, _FrozenFNumPy) and self._key == other._key


@lru_cache(maxsize=_NUMPIFY_CACHE_MAXSIZE)
def _numpify_cached_impl(
    expr: sp.Basic,
    args_tuple: Tuple[sp.Symbol, ...],
    frozen: _FrozenFNumPy,
    vectorize: bool,
    expand_definition: bool,
) -> Callable[..., Any]:
    # NOTE: This function body only runs on cache *misses*.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "numpify_cached: cache MISS (args=%s, vectorize=%s, expand_definition=%s)",
            [a.name for a in args_tuple],
            vectorize,
            expand_definition,
        )
    # Delegate to numpify() for actual compilation.
    return numpify(
        expr,
        args=args_tuple,
        f_numpy=frozen.mapping,
        vectorize=vectorize,
        expand_definition=expand_definition,
    )


def numpify_cached(
    expr: Any,
    *,
    args: Optional[Union[sp.Symbol, Iterable[sp.Symbol]]] = None,
    f_numpy: Optional[Mapping[_BindingKey, Any]] = None,
    vectorize: bool = True,
    expand_definition: bool = True,
) -> Callable[..., Any]:
    """Cached version of :func:`numpify`.

    This is a convenience wrapper for interactive sessions where the same SymPy
    expression is compiled repeatedly.

    Cache key
    ---------
    The cache key includes:

    - the SymPy expression (after :func:`sympy.sympify`),
    - the normalized argument tuple ``args``,
    - a normalized, hashable view of ``f_numpy``,
    - and the options ``vectorize`` / ``expand_definition``.

    Parameters
    ----------
    expr, args, f_numpy, vectorize, expand_definition:
        Same meaning as in :func:`numpify`.

    Returns
    -------
    Callable[..., Any]
        The compiled callable, reused across cache hits.

    Notes
    -----
    - If you mutate objects referenced by ``f_numpy`` (e.g. change entries of a
      NumPy array), cached callables will see the mutated object because the
      compiled function captures the object by reference.
    - If you need a fresh compile, call :func:`numpify` directly or clear the
      cache via ``numpify_cached.cache_clear()``.
    """
    # Normalize to SymPy and args tuple exactly as numpify() does.
    expr_sym = cast(sp.Basic, sp.sympify(expr))
    if not isinstance(expr_sym, sp.Basic):
        raise TypeError(f"numpify_cached expects a SymPy expression, got {type(expr_sym)}")

    args_tuple = _normalize_args(expr_sym, args)
    frozen = _FrozenFNumPy(f_numpy)

    return _numpify_cached_impl(expr_sym, args_tuple, frozen, vectorize, expand_definition)


# Expose cache controls on the public wrapper.
numpify_cached.cache_info = _numpify_cached_impl.cache_info  # type: ignore[attr-defined]
numpify_cached.cache_clear = _numpify_cached_impl.cache_clear  # type: ignore[attr-defined]
