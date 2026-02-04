# Plan: Refactor SmartFigure to use PlotlyPane for responsive Plotly sizing

## Context and goal
SmartFigure currently relies on a custom CSS/JS “aspect-ratio + drag handle” hack to keep Plotly sized correctly inside a widget layout. This includes inline CSS for `.sf-plot-aspect`, and a global JavaScript injection that attaches a `ResizeObserver`, mutation observer, and manual drag logic. While functional, this approach is brittle and duplicates logic that already exists in `PlotlyPane.py`, which provides a reusable, tested, and resilient Plotly resize driver. The goal of this refactor is to migrate SmartFigure to the PlotlyPane pattern for all responsive layout needs and remove the custom resizing hack. The desired end state is:

- SmartFigure relies on `PlotlyPane` for sizing and reflow (including clip ancestor handling, deferred reveal, and autorange control).`PlotlyPane.py` already documents the architectural expectations and the JS resize mechanism. 【F:PlotlyPane.py†L1-L180】
- The “homebrew” aspect ratio CSS and JavaScript injection are removed or replaced with PlotlyPane’s standardized approach. The existing CSS/JS section in SmartFigureLayout is the main target for removal/replacement. 【F:SmartFigure.py†L340-L470】
- The public API should remain as stable as possible (e.g., `SmartFigure` display and plotting behavior remains identical). Any changes in configuration should be additive and feature-flagged or have sensible defaults.

## Key files and responsibilities

- **PlotlyPane.py**: Provides the `PlotlyPane`, `PlotlyPaneStyle`, and `PlotlyResizeDriver` that manage responsive plot sizing inside ipywidgets. Its architecture includes ResizeObserver/MutationObserver, clip ancestor handling, and optional autorange. 【F:PlotlyPane.py†L1-L180】
- **SmartFigure.py**: Implements SmartFigure and its layout. The current layout includes a CSS-based aspect ratio host (`.sf-plot-aspect`) and injected JS to resize Plotly and implement a drag handle. This is the “homebrew” functionality that should be replaced. 【F:SmartFigure.py†L340-L470】

## Proposed refactor strategy

### 1. Introduce PlotlyPane as the plot container
**Goal:** Replace the `plot_container` usage in `SmartFigureLayout` with a PlotlyPane-wrapped figure widget.

**Plan:**
1. **Add a PlotlyPane import** at the top of `SmartFigure.py` (from `PlotlyPane` module). The SmartFigure class will instantiate PlotlyPane with the `FigureWidget` and provide it as the plot container.
2. **Change layout structure** so that the main plot area is `PlotlyPane.widget` rather than an HBox/VBox containing the plot and aspect-ratio host. The outer SmartFigure layout (header, sidebar, etc.) remains as-is.
3. **Configure PlotlyPane** with a style that matches the current aesthetics (e.g., padding, border, overflow). This should replicate any visual framing previously provided by `.sf-plot-aspect` and surrounding containers.

### 2. Remove homebrew CSS and JS injection in SmartFigureLayout
**Goal:** Eliminate the custom `.sf-plot-aspect` CSS rules and JS resize logic.

**Plan:**
1. **Remove `_get_css` method and CSS injection** that define aspect ratio, plot fill, and drag handle. This is now handled by PlotlyPane’s host container and resize driver. 【F:SmartFigure.py†L370-L407】
2. **Remove `_inject_js` method** that attaches global observers and drag behavior. `PlotlyPane` includes its own JS sidecar logic for resize, mutation handling, and optional autorange. 【F:SmartFigure.py†L409-L470】【F:PlotlyPane.py†L66-L140】
3. **Delete any layout hooks** tied to `.sf-plot-aspect` or `.sf-aspect-handle` from SmartFigure. There should be no references to those CSS selectors after the refactor.

### 3. Preserve or replace the “drag to resize” interaction
**Goal:** Decide whether the height-adjustment handle is still needed and, if so, where it lives.

**Plan:**
1. **Evaluate requirement:** The existing handle provides manual vertical sizing by changing aspect ratio. This is UI-specific and not a core PlotlyPane feature. Decide whether this should be a separate optional feature in SmartFigure (e.g., a `resizable_height` flag).
2. **If needed, re-implement as a layout-level feature** outside PlotlyPane, using a wrapper container around the PlotlyPane widget that sets an explicit height (via Layout height or CSS variable). PlotlyPane will then react to that size change.
3. **If not needed, drop it** and rely on the parent container’s size or a default height set in `SmartFigureLayout` (e.g., `height="60vh"` or `min_height` in the plot container layout).

### 4. Integrate PlotlyPane autorange behavior
**Goal:** Provide a well-defined way to re-autorange after resizing (if desired), replacing any implicit assumptions in the custom JS.

**Plan:**
1. **Expose a `SmartFigure` configuration** that maps to PlotlyPane’s `autorange_mode` (e.g., `"none"`, `"once"`, or `"always"`). Default it to `"once"` or `"none"` to avoid surprising axis jumps.
2. **Call `pane.reflow()`** when SmartFigure updates the layout (e.g., when toggling full width) to ensure PlotlyPane re-evaluates the size.

### 5. Update SmartFigure layout initialization
**Goal:** Ensure the new pane is created early and is accessible where needed.

**Plan:**
1. **In `SmartFigure.__init__`**, after creating the `FigureWidget`, instantiate PlotlyPane and store it (e.g., `self._pane`).
2. **In SmartFigureLayout**, replace the plot area widget with `self._pane.widget`.
3. **Maintain existing sidebar and headers** to keep the UI stable. Only the plot container changes.

### 6. Verify functionality using existing notebook examples
**Goal:** Ensure behavior matches or improves on the current experience.

**Plan:**
1. **Check the existing “responsive plotly” notebook example** (or update it if missing) to use SmartFigure with PlotlyPane and validate responsive sizing.
2. **Confirm behaviors**:
   - plot fills available space and resizes when the notebook panel changes,
   - no flickering or incorrect initial size,
   - optional autorange works as expected when resizing,
   - sidebar toggle and full-width mode still behave correctly.

### 7. Backwards compatibility and migration strategy
**Goal:** Avoid breaking notebook code that uses SmartFigure today.

**Plan:**
1. **Keep public properties** like `fig.widget` or `fig.figure` unchanged.
2. **Preserve any configuration defaults** related to layout sizes; if the previous code implicitly set sizing via CSS aspect ratio, replace it with explicit widget `layout.height` or a pane wrapper.
3. **Document the change**: add a short note in SmartFigure docstring or README (if present) about PlotlyPane being the new sizing mechanism.

## Proposed phased implementation steps

1. **Phase 1: Internal refactor (no API changes)**
   - Add PlotlyPane to SmartFigure and wire up plot container.
   - Remove CSS/JS injection for resizing.
   - Ensure base layout still renders correctly.

2. **Phase 2: Optional sizing controls**
   - Decide on “resizable height” feature. If needed, implement a minimal wrapper around PlotlyPane that adjusts height via a layout control (e.g., a slider or a small drag handle attached to the wrapper).

3. **Phase 3: Docs and examples**
   - Update any notebooks or docs to highlight the new responsive behavior and configuration options.

## Risks and mitigations

- **Risk:** SmartFigure relied on an implicit aspect ratio for a reasonable default height.
  - **Mitigation:** Define a default `layout.height` or `min_height` for the PlotlyPane host container to avoid tiny plots.

- **Risk:** Layout container sizes are undefined (height: auto), leading to PlotlyPane not resolving a pixel height.
  - **Mitigation:** Ensure the immediate SmartFigure plot area has a defined height (e.g., `layout.height="60vh"` or explicit `min_height`). PlotlyPane’s documentation stresses this requirement. 【F:PlotlyPane.py†L139-L176】

- **Risk:** Autoresize or autorange behavior differs from previous implementation.
  - **Mitigation:** Provide clear defaults and allow overriding via SmartFigure parameters.

## Acceptance criteria

- SmartFigure renders with responsive Plotly plots using PlotlyPane’s resizing logic.
- No custom CSS/JS injection remains for Plotly sizing in SmartFigure.
- Plot sizing remains robust when notebook layout changes (sidebars, split panels, etc.).
- Optional autorange configuration is supported and documented.

