# Global `plot(...)` + `with SmartFigure()` / `with fig` Design Plan

## Summary of the agreed design

You want a **module-level `plot(...)` function** that is always the primary user entry point. By default (outside any figure context), each `plot(...)` call should create a **new** `SmartFigure` (“Mode B”), display it, and then add the plot to it. When a figure is explicitly selected via a context manager (`with fig:`), `plot(...)` should add to that figure instead.

You also want:

- `with SmartFigure(): ...` to create a temporary figure and route enclosed `plot(...)` calls to it.
- `with fig1: ...` where `fig1 = SmartFigure()` to add to an existing figure without calling `fig1.plot`.
- Each returned plot object to provide a method (or property) to retrieve its owning figure.
- **Callbacks “owned by a figure”** (e.g., param-change hooks) to execute under that figure’s context by default, so that calling global `plot(...)` inside the callback adds to the correct figure (rather than creating a new one).
- Display behavior for `with fig:`: display immediately **only the first time** a given figure is entered; subsequent `with fig:` entries should not redisplay.

Batching of plots is explicitly not desired.

---

## Conceptual model

### 1) Global `plot(...)` is the only plotting entry point

- Users write `plot(x, expr, ...)` rather than `fig.plot(...)`.
- The system determines *which* figure receives the plot based on an active “current figure” context.

### 2) Figures are selected by a scoped context (“current figure stack”)

- A figure context is established by `with fig:` and supports nesting (stack semantics).
- Inside a `with fig:` block, global `plot(...)` adds to that figure.
- Outside any `with` block, global `plot(...)` creates a new figure per call (Mode B).

### 3) Display is first-class but controlled

- Entering `with fig:` should **display immediately** *only if the figure has never been displayed before*.
- If the figure has already been displayed, entering `with fig:` should **not** display again (it should only set context).

This minimizes confusion from multiple live renders of the same underlying widget while still making `with fig:` convenient.

### 4) Each plot handle can recover its figure

- `plot(...)` returns a plot-handle object.
- That plot handle offers `.figure()` (or an equivalent property) to access the `SmartFigure` it belongs to.
- This ensures that even in Mode B (new figure each call) you can still do:
  - `p = plot(...)`
  - `with p.figure(): plot(...)` to add to the same figure later, without ever calling `fig.plot`.

---

## Specifications

### A. “Current figure” resolution rules

**A1. In-context behavior**

- If there is an active figure context, `plot(...)` uses the *topmost* active figure.

**A2. Out-of-context behavior (Mode B)**

- If there is no active figure context, `plot(...)`:
  1) creates a new `SmartFigure`,
  2) displays it,
  3) adds the requested plot to it,
  4) returns the plot handle.

**A3. Nested contexts**

- Nested `with` blocks behave like a stack:
  - entering a nested figure makes it current,
  - exiting restores the previous current figure.

**A4. Exception safety**

- If an exception is raised inside a `with fig:` block, the figure context must still be properly unwound (stack remains consistent).

### B. Context manager behavior

**B1. `with SmartFigure(): ...`**

- Creates a new figure instance and makes it current for the duration of the block.
- On entry, it displays immediately (subject to “display-once” below).
- All enclosed `plot(...)` calls add to that same figure.
- On exit, it stops being current.

**B2. `with fig1:` where `fig1` is preexisting**

- Makes `fig1` current for the duration of the block.
- On entry, it displays only if it has never been displayed before.
- Enclosed `plot(...)` calls add to `fig1`.
- On exit, it stops being current.

### C. Display-once policy for `with fig:`

**C1. Display only on first entry**

- Each `SmartFigure` maintains an internal “was displayed” state.
- On entering a `with fig:`:
  - If “was displayed” is false: display it and mark as displayed.
  - If true: do not display.

**C2. Manual display is still allowed**

- Separately from `with fig:`, the user can manually display a figure (e.g., `display(fig)` or a `fig.show()` API if you add one).
- The “display-once” rule applies only to automatic display triggered by entering `with fig:`.

### D. Plot-handle API

**D1. Returned value of `plot(...)`**

- `plot(...)` returns a plot-handle object representing the created/updated plot.

**D2. Recovering figure**

- The plot handle provides `figure()` (or a property) that returns the `SmartFigure` instance it belongs to.

### E. Callback execution under figure context

**E1. Default behavior**

- Any callback registered as belonging to a figure (for example, parameter-change hooks) executes with that figure set as the current figure.

**E2. No-display in callback context**

- Callback contexts must *not* trigger display.
- The callback wrapper should set the current figure only for dispatch of global `plot(...)`, then restore prior context.

**E3. Rationale**

- This prevents global `plot(...)` called inside a callback from accidentally creating a new figure (Mode B) simply because there is no user `with` block active at callback runtime.

### F. “No batching” guarantee

**F1. No deferred rendering mode**

- The system should not suppress or accumulate plot updates inside `with fig:`; each `plot(...)` call takes effect immediately.

**F2. Implication**

- Users who call `plot(...)` from frequently-fired callbacks must be encouraged (via documentation/examples) to update existing plots rather than create new ones repeatedly (e.g., via stable identifiers or returning handles). This is a usage/documentation point, not an implementation mandate.

### G. Figure lifetime / keepalive considerations

This is not yet a finalized decision but is a notable requirement for usability:

- A user may write `with SmartFigure(): ...` without assigning the figure to a variable and without storing returned plot handles.
- In that case, the figure may have no strong reference after the block ends, which can lead to garbage collection issues in widget-based systems.
- A keepalive policy (e.g., retaining the most recent figure(s)) may be needed to make “anonymous figures” robust in notebooks.

---

## User-facing usage examples implied by the spec

### Example 1: Mode B default (new figure per call)

```python
p = plot(x, x+2)              # new figure created and shown
q = plot(x, sp.sin(x**2))     # another new figure created and shown
```

### Example 2: Add to a new figure via context

```python
with SmartFigure():
    plot(x, x+2)
    plot(x, sp.sin(x**2))
```

### Example 3: Add to an existing figure via context

```python
fig1 = SmartFigure()
with fig1:                    # displays first time only
    plot(x, x+2)
    plot(x, sp.sin(x**2))

with fig1:                    # does NOT redisplay
    plot(x, sp.cos(x))
```

### Example 4: Recover figure from a plot handle

```python
p = plot(x, x+2)
with p.figure():
    plot(x, sp.sin(x**2))
```

### Example 5: Callback safety (conceptual)

- A param-change hook calls `plot(...)`.
- Because the hook executes in the figure’s context by default, `plot(...)` adds to that figure rather than creating a new one.

---

## Open items (not implementation instructions)

1) Finalize the keepalive policy for anonymous figures created in contexts or by bare `plot(...)`.
2) Decide whether to provide an explicit `show()` method for clarity (optional; `display(fig)` may suffice).
3) Decide how strongly to steer callback authors toward “update existing plots” patterns (documentation norms vs optional helper APIs).
