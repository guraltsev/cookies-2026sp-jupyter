from __future__ import annotations

__all__=[]
# --- The deliberate classroom prelude ---
import sympy as sp
__all__+=["sp"]
from sympy import *
__all__+= list(getattr(sp, "__all__", []))
import numpy as np
__all__+=["np"]

import pandas as pd
__all__+=["pd"]

# print("__all__ (from prelude):",__all__)
import sympy as sp

class SymbolFamily(sp.Symbol):
    """
    A SymPy Symbol that creates indexed children via [].
    Inherits from sp.Symbol, so all math (x**2, diff, etc.) works natively.
    """
    def __new__(cls, name, **kwargs):
        # Create the actual SymPy Symbol
        obj = super().__new__(cls, name, **kwargs)
        
        # Attach our family-specific attributes to the new instance
        # We use distinct names (e.g., _family_cache) to avoid colliding with SymPy internals
        obj._family_cache = {}
        obj._family_kwargs = kwargs
        return obj

    def __getitem__(self, k):
        if not isinstance(k, tuple):
            k = (k,)
            
        if k not in self._family_cache:
            sub = ",".join(map(str, k))
            child_name = f"{self.name}_{sub}"
            # Create a standard Symbol for the child
            self._family_cache[k] = sp.Symbol(child_name, **self._family_kwargs)
            
        return self._family_cache[k]

class FunctionFamily:
    """
    A wrapper for SymPy Functions (e.g., f(x)).
    SymPy Functions are complex to subclass directly, so this proxy 
    is the standard way to handle them.
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self._kwargs = kwargs
        # Create the base function (e.g. f)
        self._base = sp.Function(name, **kwargs)
        self._cache = {}

    def __getitem__(self, k):
        if not isinstance(k, tuple):
            k = (k,)
        if k not in self._cache:
            sub = ",".join(map(str, k))
            # Create a new Function for the child (e.g. f_1)
            self._cache[k] = sp.Function(f"{self.name}_{sub}", **self._kwargs)
        return self._cache[k]

    def __call__(self, *args):
        # Allows f(x) to work
        return self._base(*args)
    
    def _sympy_(self):
        # Allows SymPy to recognize this object
        return self._base
    
    def __str__(self):
        return str(self._base)
    
    def __repr__(self):
        return repr(self._base)

__all__+=["SymbolFamily","FunctionFamily"]
a = SymbolFamily('a')
b = SymbolFamily('b')
c = SymbolFamily('c')

A = SymbolFamily('A')
B = SymbolFamily('B')
C = SymbolFamily('C')


x = SymbolFamily('x')
y = SymbolFamily('y')
z = SymbolFamily('z')
t = SymbolFamily('t')
r = SymbolFamily('r')
u = SymbolFamily('u')
v = SymbolFamily('v')
w = SymbolFamily('w')
s = SymbolFamily('s')
alpha = SymbolFamily('α')
beta = SymbolFamily('β')
gamma = SymbolFamily('γ')
delta = SymbolFamily('δ')
phi = SymbolFamily('φ')
psi = SymbolFamily('ψ')
theta = SymbolFamily('θ')
tau = SymbolFamily('τ')
rho = SymbolFamily('ρ')
__all__+=["a","b","c","A","B","C","x","y","z","t","s","r","u","v","w","alpha","beta","delta", "phi","psi","gamma","tau","theta", "rho"]


k = SymbolFamily("k", integer=True)
l = SymbolFamily("l", integer=True)
m = SymbolFamily("m", integer=True)
n = SymbolFamily("n", integer=True)
__all__+=["k","l","m","n"]

f=FunctionFamily("f")
g=FunctionFamily("g")
h=FunctionFamily("h")
__all__+=["f","g","h"]



class Infix:
    """Generic infix operator used as: a |OP| b."""
    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func

    def __ror__(self, left):
        return _InfixPartial(self.func, left)


class _InfixPartial:
    __slots__ = ("func", "left")

    def __init__(self, func, left):
        self.func = func
        self.left = left

    def __or__(self, right):
        return self.func(self.left, right)

eq = Infix(sp.Eq)
lt = Infix(sp.Lt)
le = Infix(sp.Le)
gt = Infix(sp.Gt)
ge = Infix(sp.Ge)
__all__+=["Infix", "eq", "lt", "le", "gt", "ge"]

## Demo:
# x, y = sp.symbols("x y")
# expr1 = x + 1
# expr2 = 2*y

# print(expr1 | eq | expr2)      # Eq(x + 1, 2*y)
# print((x**2) | eq | (y**2))    # Eq(x**2, y**2)


from IPython.display import Latex 
__all__+=["Latex"]

from pprint import pprint
__all__+=["pprint"]