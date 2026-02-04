import numpy as np
import sympy as sp
from scipy.integrate import simpson

from .numpify import numpify  # or: from gu_toolkit.numpify import numpify


def NIntegrate(expression, var_interval, *, samples: int = 1000):
    """
    Numerical integral of a SymPy expression over a finite interval using SciPy's Simpson rule.

    Parameters
    ----------
    expression:
        SymPy expression (or sympify-compatible).
    var_interval:
        Tuple (symbol, start, end), where symbol is a SymPy Symbol and start/end are numeric.
    samples:
        Number of sample points (>= 2) used to build the integration grid.

    Returns
    -------
    float or complex
    """
    # --- Parse / validate (symbol, start, end) ---
    if not (isinstance(var_interval, (tuple, list)) and len(var_interval) == 3):
        raise TypeError("NIntegrate expects var_interval=(symbol, start, end).")

    sym_raw, start_raw, end_raw = var_interval
    sym = sp.sympify(sym_raw)
    if not isinstance(sym, sp.Symbol):
        raise TypeError(f"Integration variable must be a SymPy Symbol, got {type(sym)}.")

    expr = sp.sympify(expression)
    if not isinstance(expr, sp.Basic):
        raise TypeError(f"expression must be SymPy-compatible, got {type(expression)}.")

    # Expression must be univariate in `sym` (or constant).
    free = set(expr.free_symbols)
    extra = {s for s in free if s != sym}
    if extra:
        extra_str = ", ".join(sorted(s.name for s in extra))
        raise ValueError(
            "NIntegrate expects an expression depending only on the integration symbol "
            f"{sym.name!r} (or a constant). Found extra free symbol(s): {extra_str}. "
            "Substitute numeric values first, e.g. expr.subs({a: 2.0}), then call NIntegrate again."
        )

    # Limits must be numeric (no symbols).
    start_expr = sp.sympify(start_raw)
    end_expr = sp.sympify(end_raw)
    if start_expr.free_symbols or end_expr.free_symbols:
        raise ValueError(
            "NIntegrate requires numeric integration limits. "
            f"Got start={start_expr} (free_symbols={start_expr.free_symbols}), "
            f"end={end_expr} (free_symbols={end_expr.free_symbols})."
        )

    a = float(sp.N(start_expr))
    b = float(sp.N(end_expr))

    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"NIntegrate currently supports only finite limits; got start={a}, end={b}.")

    if not isinstance(samples, int) or samples < 2:
        raise ValueError(f"samples must be an integer >= 2, got {samples!r}.")

    # Handle reversed limits.
    if b < a:
        return -NIntegrate(expr, (sym, b, a), samples=samples)

    # --- Compile to a NumPy-evaluable callable via numpify ---
    # numpify compiles SymPy -> NumPy code and preflights unbound symbols / unknown functions. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
    try:
        f = numpify(expr, args=sym)
    except ValueError as e:
        # Preserve numpify’s detailed diagnostics (e.g. unknown function bindings). :contentReference[oaicite:2]{index=2}
        raise ValueError(f"Failed to compile integrand via numpify: {e}") from e

    # --- Sample + integrate ---
    x = np.linspace(a, b, samples, dtype=float)
    y = f(x)

    y = np.asarray(y)
    if y.shape != x.shape:
        # numpify usually broadcasts constants to the right shape when args are present. :contentReference[oaicite:3]{index=3}
        try:
            y = np.broadcast_to(y, x.shape)
        except Exception as e:
            raise ValueError(f"Integrand evaluated to shape {y.shape}, cannot align to x.shape={x.shape}.") from e

    if not np.all(np.isfinite(y.real)) or (np.iscomplexobj(y) and not np.all(np.isfinite(y.imag))):
        raise ValueError(
            "Integrand produced NaN/Inf on the sampling grid. "
            "This fixed-sample integrator requires a finite, well-defined value at each sample point."
        )

    res = simpson(y, x=x)
    return complex(res) if np.iscomplexobj(res) else float(res)
