from __future__ import annotations

# NOTE: This file is SmartFigure.py with the Info Components API implemented.
#       It is intended as a drop-in replacement.

"""Widgets and interactive plotting helpers for math exploration in Jupyter.

This file defines two main ideas:

1) OneShotOutput
   A small safety wrapper around ``ipywidgets.Output`` that can only be displayed once.
   This prevents a common notebook confusion: accidentally displaying the *same* widget
   in multiple places and then wondering which one is “live”.

2) SmartFigure (+ SmartPlot)
   A thin, student-friendly wrapper around ``plotly.graph_objects.FigureWidget`` that:
   - plots SymPy expressions by compiling them to NumPy via ``numpify_cached``,
   - supports interactive parameter sliders (via ``SmartFloatSlider``),
   - optionally provides an *Info* area (a stack of ``ipywidgets.Output`` widgets),
   - re-renders automatically when you pan/zoom (throttled) or move a slider.

The intended workflow is:

- define symbols with SymPy (e.g. ``x, a = sp.symbols("x a")``),
- create a ``SmartFigure``,
- add one or more plots with ``SmartFigure.plot(...)``,
- optionally add parameters (sliders) explicitly by passing ``parameters=[a, ...]``.
- otherwise, parameters are autodetected from the expression (all free symbols that are not the plot variable) and added automatically.

---------------------------------------------------------------------------
Quick start (in a Jupyter notebook)
---------------------------------------------------------------------------

>>> import sympy as sp
>>> from SmartFigure import SmartFigure  # wherever this file lives
>>>
>>> x, a = sp.symbols("x a")
>>> fig = SmartFigure(x_range=(-6, 6), y_range=(-3, 3))
>>> fig.plot(x, sp.sin(x), id="sin")
>>> fig.plot(x, a*sp.cos(x), id="a_cos")  # adds a slider for a
>>> fig.title = "Sine and a·Cosine"
>>> fig  # display in the output cell (or use display(fig))

Tip: if you omit ``parameters`` when calling ``plot``, SmartFigure will infer them
from the expression and create sliders automatically. Pass ``[]`` to disable that.

Info panel
----------
The sidebar has two sections:

- **Parameters**: auto-created sliders for SymPy symbols.
- **Info**: a container that holds *Output widgets* created by
  :meth:`SmartFigure.get_info_output`. This design is deliberate: printing directly
  into a container widget is ambiguous in Jupyter, but printing into an
  ``Output`` widget is well-defined.
  Info outputs are keyed by id, so you can retrieve them via
  ``fig.info_output[id]`` or create/reuse them via ``fig.get_info_output(id)``.

Notes for students
------------------
- SymPy expressions are symbolic. They are like *formulas*.
- Plotly needs numerical values (arrays of numbers).
- ``numpify_cached`` bridges the two: it turns a SymPy expression into a NumPy-callable function.
- Sliders provide the numeric values of parameters like ``a`` in real time.

Architecture Note (For Developers)
----------------------------------
To avoid a "God Object," responsibilities are split via composition:
- SmartFigure: The main coordinator/facade.
- SmartFigureLayout: Handles all UI/Widget construction, CSS/JS injection, and layout logic.
- ParameterManager: Handles slider creation, storage, and change hooks. Acts as a dict proxy.
- InfoPanelManager: Handles the info sidebar and component registry.
- SmartPlot: Handles the specific math-to-trace rendering logic.


Logging / debugging
-------------------
This module uses the standard Python ``logging`` framework (no prints). By default it installs a
``NullHandler``, so you will see nothing unless you configure logging.

In a Jupyter/JupyterLab notebook, enable logs like this:

    import logging
    logging.basicConfig(level=logging.INFO)   # or logging.DEBUG

To limit output to just this module, set its logger level instead:

    import logging
    logging.getLogger(__name__).setLevel(logging.DEBUG)

Notes:
- INFO render messages are rate-limited to ~1.0s.
- DEBUG range messages (x_range/y_range) are rate-limited to ~0.5s.
"""

import re
import time
import warnings
import logging
from typing import Any, Callable, Hashable, Optional, Sequence, Tuple, Union, Dict, Iterator

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from IPython.display import Javascript, display
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

# Internal imports (assumed to exist in the same package)
from .InputConvert import InputConvert
from .numpify import numpify_cached
from .SmartSlider import SmartFloatSlider


# Module logger
# - Uses a NullHandler so importing this module never configures global logging.
# - Callers can enable logs via standard logging configuration.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# -----------------------------
# Small type aliases
# -----------------------------
NumberLike = Union[int, float]
NumberLikeOrStr = Union[int, float, str]
RangeLike = Tuple[NumberLikeOrStr, NumberLikeOrStr]
VisibleSpec = Union[bool, str]  # Plotly uses True/False or the string "legendonly".


# =============================================================================
# SECTION: OneShotOutput [id: OneShotOutput]
# =============================================================================

class OneShotOutput(widgets.Output):
    """
    A specialized Output widget that can only be displayed once.

    Why this exists
    ---------------
    In Jupyter, widgets are *live objects* connected to the frontend by a comm channel.
    If you display the same widget instance multiple times, it is easy to end up with
    confusing UI behavior (e.g., “Which copy should update?”, “Why did output appear
    in two places?”, etc.).

    ``OneShotOutput`` prevents accidental duplication by raising an error on the
    second display attempt.

    What counts as “display”?
    -------------------------
    Any of the following will count as displaying the widget:
    - having it be the last expression in a cell,
    - calling ``display(output)``,
    - placing it inside another widget/layout that is displayed.
   
     Attributes
    ----------
    _displayed : bool
        Internal flag tracking whether the widget has been displayed.
    
      Examples
    --------
    Basic output usage:

    >>> out = OneShotOutput()
    >>> with out:
    ...     print("Hello from inside the Output widget!")
    >>> out  # first display works

    Attempting to display again raises:

    >>> out  # doctest: +SKIP
    RuntimeError: OneShotOutput has already been displayed...

    Use case: preventing accidental double-display:

    >>> out = OneShotOutput()
    >>> with out:
    ...     print("I only want this shown once.")
    >>> display(out)  # ok
    >>> display(out)  # raises RuntimeError

    If you *really* need to display it again (advanced / use with caution),
    you can reset:

    >>> out.reset_display_state()
    >>> display(out)  # now allowed again

    (See ``reset_display_state`` for warnings.)
    """

    __slots__ = ("_displayed",)

    def __init__(self) -> None:
        """Initialize a new OneShotOutput widget."""
        super().__init__()
        self._displayed = False

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None, **kwargs: Any) -> Any:
        """
        IPython rich display hook used by ipywidgets.

        This is what gets called when the widget is displayed (including via
        `display(self)` or by being the last expression in a cell).
        """
        if self._displayed:
            raise RuntimeError(
                "OneShotOutput has already been displayed. "
                "This widget supports only one-time display."
            )
        self._displayed = True
        return super()._repr_mimebundle_(include=include, exclude=exclude, **kwargs)

    @property
    def has_been_displayed(self) -> bool:
        """
        Check if the widget has been displayed.

        Returns
        -------
        bool
            True if the widget has been displayed, False otherwise.
        """
        return self._displayed

    def reset_display_state(self) -> None:
        """
        Reset the display state to allow re-display.

        Warning
        -------
        This method should be used with caution as it bypasses the
        one-time display protection.
        """
        self._displayed = False


# =============================================================================
# SECTION: SmartFigureLayout (The View) [id: SmartFigureLayout]
# =============================================================================

class SmartFigureLayout:
    """
    Manages the visual structure, CSS/JS injection, and widget hierarchy of a SmartFigure.
    
    This class isolates all the "messy" UI code (CSS strings, JavaScript injection,
    VBox/HBox nesting) from the mathematical logic.

    Responsibilities:
    - Building the HBox/VBox structure.
    - Injecting the specific CSS/JS for aspect ratio handling.
    - Exposing containers for Plots, Parameters, and Info.
    - Handling layout toggles (e.g. full width, sidebar visibility).
    """

    def __init__(self, title: str = "") -> None:
        # 1. CSS and JS Injection
        #    Invisible widgets that carry the bootstrap code for the browser.
        self._css_widget = widgets.HTML(value=self._get_css())
        self._js_widget = widgets.Output(layout=widgets.Layout(width="0px", height="0px", display="none"))
        
        # 2. Title Bar
        #    We use HTMLMath for proper LaTeX title rendering.
        self.title_html = widgets.HTMLMath(value=title, layout=widgets.Layout(margin="0px"))
        self.full_width_checkbox = widgets.Checkbox(
            value=False,
            description="Full width plot",
            indent=False,
            layout=widgets.Layout(width="160px", margin="0px"),
        )
        self._titlebar = widgets.HBox(
            [self.title_html, self.full_width_checkbox],
            layout=widgets.Layout(
                width="100%", align_items="center", justify_content="space-between", margin="0 0 6px 0"
            ),
        )

        # 3. Plot Area (The "Left" Panel)
        #    Note: CSS aspect-ratio controls height. We do NOT set a fixed height here.
        self.plot_container = widgets.Box(
            children=(),
            layout=widgets.Layout(
                width="100%", min_width="320px", margin="0px", padding="0px", flex="1 1 560px"
            ),
        )
        self.plot_container.add_class("sf-plot-aspect")

        # 4. Controls Sidebar (The "Right" Panel)
        #    Initially hidden (display="none") until parameters or info widgets are added.
        self.params_header = widgets.HTML("<b>Parameters</b>", layout=widgets.Layout(display="none", margin="0"))
        self.params_box = widgets.VBox(layout=widgets.Layout(width="100%", display="none"))
        
        self.info_header = widgets.HTML("<b>Info</b>", layout=widgets.Layout(display="none", margin="10px 0 0 0"))
        self.info_box = widgets.VBox(layout=widgets.Layout(width="100%", display="none"))

        self.sidebar_container = widgets.VBox(
            [self.params_header, self.params_box, self.info_header, self.info_box],
            layout=widgets.Layout(
                margin="0px", padding="0px 0px 0px 10px", flex="0 1 380px",
                min_width="300px", max_width="400px", display="none"
            ),
        )

        # 5. Main Content Wrapper (Flex)
        #    Uses flex-wrap so the sidebar drops below the plot on narrow screens.
        self.content_wrapper = widgets.Box(
            [self.plot_container, self.sidebar_container],
            layout=widgets.Layout(
                display="flex", flex_flow="row wrap", align_items="flex-start",
                width="100%", gap="8px"
            ),
        )

        # 6. Root Widget
        self.root_widget = widgets.VBox(
            [self._css_widget, self._js_widget, self._titlebar, self.content_wrapper],
            layout=widgets.Layout(width="100%")
        )

        # Wire up internal logic
        self.full_width_checkbox.observe(self._on_full_width_change, names="value")
        self._inject_js()

    @property
    def output_widget(self) -> OneShotOutput:
        """Returns a OneShotOutput wrapping the layout, ready for display."""
        out = OneShotOutput()
        with out:
            display(self.root_widget)
        return out

    def set_title(self, text: str) -> None:
        self.title_html.value = text

    def get_title(self) -> str:
        return self.title_html.value

    def update_sidebar_visibility(self, has_params: bool, has_info: bool) -> None:
        """
        Updates visibility of headers and the sidebar itself based on content.
        
        This prevents empty "Parameters" or "Info" headers from cluttering the UI.
        """
        self.params_header.layout.display = "block" if has_params else "none"
        self.params_box.layout.display = "flex" if has_params else "none"
        
        self.info_header.layout.display = "block" if has_info else "none"
        self.info_box.layout.display = "flex" if has_info else "none"

        show_sidebar = has_params or has_info
        self.sidebar_container.layout.display = "flex" if show_sidebar else "none"

    def _on_full_width_change(self, change: Dict[str, Any]) -> None:
        """Toggles CSS flex properties for full-width mode."""
        is_full = change["new"]
        layout = self.content_wrapper.layout
        plot_layout = self.plot_container.layout
        sidebar_layout = self.sidebar_container.layout

        if is_full:
            # Stack vertically, full width
            layout.flex_flow = "column"
            plot_layout.flex = "0 0 auto"
            sidebar_layout.flex = "0 0 auto"
            sidebar_layout.max_width = ""
            sidebar_layout.width = "100%"
            sidebar_layout.padding = "0px"
        else:
            # Side-by-side (wrapping), restricted width for sidebar
            layout.flex_flow = "row wrap"
            plot_layout.flex = "1 1 560px"
            sidebar_layout.flex = "0 1 380px"
            sidebar_layout.max_width = "400px"
            sidebar_layout.width = "auto"
            sidebar_layout.padding = "0px 0px 0px 10px"

    def _get_css(self) -> str:
        """
        Returns the CSS needed for the aspect-ratio hack and drag handle.
        """
        return """
        <style>
        /* Host plot panel: height governed by aspect-ratio. */
        .sf-plot-aspect {
            position: relative; width: 100%;
            aspect-ratio: var(--sf-ar, 1.3333333333);
            min-height: 260px; box-sizing: border-box;
        }
        /* Force Plotly to fill the host container */
        .sf-plot-aspect .js-plotly-plot, .sf-plot-aspect .plotly-graph-div {
            width: 100% !important; height: 100% !important;
        }
        /* Drag handle (bottom grip) */
        .sf-aspect-handle {
            position: absolute; left: 0; right: 0; bottom: 0; height: 14px;
            cursor: ns-resize; z-index: 10;
        }
        .sf-aspect-handle::after {
            content: ""; display: block; width: 56px; height: 4px;
            margin: 5px auto; border-radius: 999px; background: rgba(0,0,0,0.25);
        }
        .sf-aspect-handle:hover::after { background: rgba(0,0,0,0.40); }
        </style>
        """

    def _inject_js(self) -> None:
        """
        Injects the JavaScript Logic for the resize handle and Plotly resizing.
        """
        # Kept inline to ensure the file is self-contained.
        js_code = r"""
        (function () {
            if (window.__smartfigure_plotly_aspect_resizer_installed) return;
            window.__smartfigure_plotly_aspect_resizer_installed = true;

            function safeResizePlotly(gd) {
                try { if (window.Plotly && Plotly.Plots) Plotly.Plots.resize(gd); } catch (e) {}
            }
            
            function ensureHandle(host) {
                if (host.__sf_handle_installed) return;
                host.__sf_handle_installed = true;
                const handle = document.createElement('div');
                handle.className = 'sf-aspect-handle';
                host.appendChild(handle);
                
                let dragging = false, startY = 0, startH = 0;
                
                function onMove(ev) {
                    if (!dragging) return;
                    const newH = Math.max(180, Math.min(2200, startH + (ev.clientY - startY)));
                    const rect = host.getBoundingClientRect();
                    host.style.setProperty('--sf-ar', String((rect.width || 1) / newH));
                    const gd = host.querySelector('.js-plotly-plot');
                    if (gd) safeResizePlotly(gd);
                }
                function onUp(ev) {
                    dragging = false; 
                    window.removeEventListener('pointermove', onMove, true);
                    window.removeEventListener('pointerup', onUp, true);
                }
                handle.addEventListener('pointerdown', (ev) => {
                    ev.preventDefault(); dragging = true; startY = ev.clientY;
                    startH = host.getBoundingClientRect().height || 1;
                    window.addEventListener('pointermove', onMove, true);
                    window.addEventListener('pointerup', onUp, true);
                });
            }

            function attachAll() {
                document.querySelectorAll('.sf-plot-aspect').forEach(ensureHandle);
                document.querySelectorAll('.js-plotly-plot').forEach(gd => {
                    if (!gd.__smartfigure_ro) {
                        const ro = new ResizeObserver(() => safeResizePlotly(gd));
                        ro.observe(gd);
                        gd.__smartfigure_ro = ro;
                    }
                });
            }

            const mo = new MutationObserver(() => attachAll());
            mo.observe(document.body, { childList: true, subtree: true });
            setTimeout(attachAll, 1000); 
        })();
        """
        with self._js_widget:
            display(Javascript(js_code))


# =============================================================================
# SECTION: ParameterManager (The Model for Parameters) [id: ParameterManager]
# =============================================================================

class ParameterManager:
    """
    Manages the collection of parameter sliders and change hooks.

    Responsibilities:
    - Creating and reusing SmartFloatSlider widgets.
    - Storing current parameter values.
    - Executing hooks when parameters change.
    - **Backward Compatibility:** Acts like a dictionary so `fig.params[sym]` works.

    Design Note:
    ------------
    By centralizing parameter logic here, we decouple the "state" of the math
    from the "rendering" of the figure.
    """

    def __init__(self, render_callback: Callable[[str, Any], None], layout_box: widgets.Box) -> None:
        self._sliders: Dict[Symbol, SmartFloatSlider] = {}
        self._hooks: Dict[Hashable, Callable[[Dict, Any], Any]] = {}
        self._hook_counter: int = 0
        self._render_callback = render_callback
        self._layout_box = layout_box # The VBox where sliders live

    def add_param(self, symbol: Symbol, **kwargs: Any) -> SmartFloatSlider:
        """
        Create or reuse a slider for the given symbol.

        Parameters
        ----------
        symbol : sympy.Symbol
            The parameter symbol.
        **kwargs :
            Options for the slider (min, max, value, step).
        """
        if symbol in self._sliders:
            return self._sliders[symbol]

        defaults = {'value': 0.0, 'min': -1.0, 'max': 1.0, 'step': 0.01}
        config = {**defaults, **kwargs}
        
        slider = SmartFloatSlider(
            description=f"${sp.latex(symbol)}$",
            value=float(config['value']),
            min=float(config['min']),
            max=float(config['max']),
            step=float(config['step'])
        )
        
        # Observe changes
        slider.observe(self._on_slider_change, names="value")
        
        self._sliders[symbol] = slider
        self._layout_box.children += (slider,)
        return slider

    def get_value(self, symbol: Symbol) -> float:
        """Returns the current float value of a parameter."""
        return self._sliders[symbol].value if symbol in self._sliders else 0.0

    @property
    def has_params(self) -> bool:
        """True if any parameters (sliders) have been created."""
        return len(self._sliders) > 0

    def add_hook(self, callback: Callable, hook_id: Optional[Hashable] = None, fig: Any = None) -> Hashable:
        """
        Register a parameter change hook. 
        The callback is run immediately on registration, with an empty change dict.
        
        Parameters
        ----------
        callback: Callable
            The function to call (signature: (change, fig)).
        hook_id: Hashable, optional
            Optional unique identifier.
        fig: SmartFigure
            The SmartFigure instance. Crucial for passing to the callback immediately
            so the hook can initialize.

        Returns
        -------
        Hashable
            The hook ID.
        """
        if hook_id is None:
            self._hook_counter += 1
            hook_id = f"hook:{self._hook_counter}"
        self._hooks[hook_id] = callback
        
        # Run immediately on registration
        try:
            callback({}, fig) 
        except Exception as e:
            warnings.warn(f"Hook failed on init: {e}")
        return hook_id

    def _on_slider_change(self, change: Dict[str, Any]) -> None:
        # 1. Trigger main render via the callback passed in __init__
        self._render_callback("param_change", change)
    
    def get_hooks(self) -> Dict[Hashable, Callable]:
        return self._hooks

    # --- Dict-like Interface for Backward Compatibility ---
    # This allows `fig.params[symbol]` to work in user hooks.
    
    def __getitem__(self, key: Symbol) -> SmartFloatSlider:
        return self._sliders[key]
    
    def __contains__(self, key: Symbol) -> bool:
        return key in self._sliders
    
    def items(self) -> Iterator[Tuple[Symbol, SmartFloatSlider]]:
        return self._sliders.items()
    
    def keys(self) -> Iterator[Symbol]:
        return self._sliders.keys()
    
    def values(self) -> Iterator[SmartFloatSlider]:
        return self._sliders.values()
    
    def get(self, key: Symbol, default: Any = None) -> Any:
        return self._sliders.get(key, default)


# =============================================================================
# SECTION: InfoPanelManager (The Model for Info) [id: InfoPanelManager]
# =============================================================================

class InfoPanelManager:
    """
    Manages the 'Info' section output widgets and interactive components.

    It allows adding "Output" widgets (where you can print text or display charts)
    and registering "Stateful Components" (classes that update when sliders move).
    """
    
    _ID_REGEX = re.compile(r"^info:(\d+)$")

    def __init__(self, layout_box: widgets.Box) -> None:
        self._outputs: Dict[Hashable, widgets.Output] = {}
        self._components: Dict[Hashable, Any] = {}
        self._layout_box = layout_box
        self._counter = 0

    def get_output(self, id: Optional[Hashable] = None, **layout_kwargs: Any) -> widgets.Output:
        """
        Get or create an Info Output widget.
        """
        if id is None:
            self._counter += 1
            id = f"info:{self._counter}"
        
        if id in self._outputs:
            out = self._outputs[id]
            if layout_kwargs:
                out.layout = widgets.Layout(**layout_kwargs)
            return out
        
        # Validate ID if string (avoids collision with auto-generated IDs)
        if isinstance(id, str):
            m = self._ID_REGEX.match(id)
            if m:
                self._counter = max(self._counter, int(m.group(1)))

        out = widgets.Output(layout=widgets.Layout(**layout_kwargs))
        setattr(out, 'id', id)
        
        self._outputs[id] = out
        self._layout_box.children += (out,)
        return out

    def add_component(self, id: Hashable, component_inst: Any) -> None:
        self._components[id] = component_inst

    def get_component(self, id: Hashable) -> Any:
        return self._components[id]

    @property
    def has_info(self) -> bool:
        return len(self._outputs) > 0


# =============================================================================
# SECTION: SmartPlot (The specific logic for one curve) [id: SmartPlot]
# =============================================================================

class SmartPlot:
    """
    A single plotted curve managed by a :class:`SmartFigure`.

    Conceptually, a ``SmartPlot`` is “one function on one set of axes”.
    It owns a single Plotly trace (a line plot) and knows how to:

    - compile the SymPy expression to a fast NumPy function (via ``numpify_cached``),
    - sample x-values on an appropriate domain,
    - evaluate y-values (including current slider parameter values),
    - push the sampled data into the Plotly trace.
    """

    def __init__(
        self,
        var: Symbol,
        func: Expr,
        smart_figure: "SmartFigure",
        parameters: Sequence[Symbol] = [],
        x_domain: Optional[RangeLike] = None,
        sampling_points: Optional[int,str] = None,
        label: str = "",
        visible: VisibleSpec = True,
    ) -> None:
        """
        Create a new SmartPlot instance. (Usually called by SmartFigure.plot)
        """
        self._smart_figure = smart_figure
        
        # Add trace to figure
        self._smart_figure.figure_widget.add_scatter(x=[], y=[], mode="lines", name=label, visible=visible)
        self._plot_handle = self._smart_figure.figure_widget.data[-1]

        self._suspend_render = True
        self.set_func(var, func, parameters)
        self.x_domain = x_domain
        
        if sampling_points == "figure_default":
            sampling_points = None
        self.sampling_points = sampling_points

        self._suspend_render = False
        
        self.render()

    def set_func(self, var: Symbol, func: Expr, parameters: Sequence[Symbol] = []) -> None:
        """
        Set the independent variable and symbolic function for this plot.
        Triggers recompilation via ``numpify_cached``.
        """
        parameters = list(parameters) 
        # Compile
        self._f_numpy = numpify_cached(func, args=[var] + parameters)
        # Store
        self._var = var
        self._parameters = parameters
        self._func = func

    @property
    def label(self) -> str:
        return self._plot_handle.name

    @label.setter
    def label(self, value: str) -> None:
        self._plot_handle.name = value

    @property
    def x_domain(self) -> Optional[Tuple[float, float]]:
        return self._x_domain

    @x_domain.setter
    def x_domain(self, value: Optional[RangeLike]) -> None:
        
        if value is None:
            self._x_domain = None
        elif value == "figure_default":
            self._x_domain = None
        else:
            raw_min, raw_max = value
            self._x_domain = (float(InputConvert(raw_min, float)), float(InputConvert(raw_max, float)))
            if self._x_domain[0] > self._x_domain[1]:
                raise ValueError("x_min must be <= x_max")
        self.render()

    @property
    def sampling_points(self) -> Optional[int]:
        return self._sampling_points

    @sampling_points.setter
    def sampling_points(self, value: Optional[int]) -> None:
        self._sampling_points = int(InputConvert(value, int)) if value is not None else None
        self.render()

    @property
    def visible(self) -> VisibleSpec:
        return self._plot_handle.visible

    @visible.setter
    def visible(self, value: VisibleSpec) -> None:
        self._plot_handle.visible = value
        if value is True:
            self.render()

    def render(self) -> None:
        """
        Compute (x, y) samples and update the Plotly trace.
        Skips computation if the plot is hidden.
        """
        if self._suspend_render or self.visible is not True:
            return

        # 1. Determine Range
        fig = self._smart_figure
        viewport = fig.current_x_range or fig.x_range
        
        if self.x_domain is None:
            x_min, x_max = float(viewport[0]), float(viewport[1])
        else:
            x_min = min(float(viewport[0]), float(self.x_domain[0]))
            x_max = max(float(viewport[1]), float(self.x_domain[1]))

        # 2. Determine Sampling
        num = self.sampling_points or fig.sampling_points or 500
        
        # 3. Compute
        x_values = np.linspace(x_min, x_max, num=int(num))
        args = [x_values]
        if self._parameters:
            # Retrieve values from the manager
            for p in self._parameters:
                args.append(fig.params.get_value(p))
        
        y_values = np.asarray(self._f_numpy(*args))
        
        # 4. Update Trace
        with fig.figure_widget.batch_update():
            self._plot_handle.x = x_values
            self._plot_handle.y = y_values
    
    def update(self, **kwargs: Any) -> None:
        """Convenience to update multiple attributes (function, label, domain) at once."""
        if 'label' in kwargs: 
            self.label = kwargs['label']
        
        if 'x_domain' in kwargs: 
            val = kwargs['x_domain']
            if val == "figure_default":
                self.x_domain = None
            else:
                x_min = InputConvert(val[0], float)
                x_max = InputConvert(val[1], float)
                self.x_domain = (x_min, x_max)
        
        if 'sampling_points' in kwargs:
            val = kwargs['sampling_points']
            if val == "figure_default":
                self.sampling_points = None
            else:
                self.sampling_points = InputConvert(val, int)
        
        # Function update
        if any(k in kwargs for k in ('var', 'func', 'parameters')):
            v = kwargs.get('var', self._var)
            f = kwargs.get('func', self._func)
            p = kwargs.get('parameters', self._parameters)
            self.set_func(v, f, p)
            self.render()


# =============================================================================
# SECTION: SmartFigure (The Coordinator) [id: SmartFigure]
# =============================================================================

class SmartFigure:
    """
    An interactive Plotly figure for plotting SymPy functions with slider parameters.

    What problem does this solve?
    -----------------------------
    We often want to:
    - type a symbolic function like ``sin(x)`` or ``a*x**2 + b`` (SymPy),
    - *see* it immediately (Plotly),
    - and then explore “What happens if I change a parameter?”

    ``SmartFigure`` provides a simple API that encourages experimentation.

    Key features
    ------------
    - Uses Plotly ``FigureWidget`` so it is interactive inside notebooks.
    - Uses a right-side controls panel for parameter sliders.
    - Supports plotting multiple curves identified by an ``id``.
    - Re-renders curves on:
      - slider changes,
      - pan/zoom changes (throttled to at most once every 0.5 seconds).

    Examples
    --------
    >>> import sympy as sp
    >>> x, a = sp.symbols("x a")
    >>> fig = SmartFigure()
    >>> fig.plot(x, a*sp.sin(x), parameters=[a], id="a_sin")
    >>> fig
    """
    
    __slots__ = [
        "_layout", "_params", "_info", "_figure", "plots",
        "_x_range", "_y_range", "_sampling_points", "_debug",
        "_last_relayout", "_render_info_last_log_t", "_render_debug_last_log_t"
    ]

    def __init__(
        self,
        sampling_points: int = 500,
        x_range: RangeLike = (-4, 4),
        y_range: RangeLike = (-3, 3),
        debug: bool = False,
    ) -> None:
        self._debug = debug
        self._sampling_points = sampling_points
        self.plots: Dict[str, SmartPlot] = {}

        # 1. Initialize Layout (View)
        self._layout = SmartFigureLayout()
        
        # 2. Initialize Managers
        # Note: we pass a callback for rendering so params can trigger updates
        self._params = ParameterManager(self.render, self._layout.params_box)
        self._info = InfoPanelManager(self._layout.info_box)

        # 3. Initialize Plotly Figure
        self._figure = go.FigureWidget()
        self._figure.update_layout(
            autosize=True, template="plotly_white", showlegend=True, margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black", showline=True, ticks="outside"),
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black", showline=True, ticks="outside"),
        )
        self._layout.plot_container.children = (self._figure,)

        # 4. Set Initial State
        self.x_range = x_range
        self.y_range = y_range
        
        # 5. Bind Events
        self._last_relayout = time.monotonic()
        self._render_info_last_log_t = 0.0
        self._render_debug_last_log_t = 0.0
        self._figure.layout.on_change(self._throttled_relayout, "xaxis.range", "yaxis.range")

    # --- Properties ---

    @property
    def title(self) -> str:
        """Title text shown above the figure (renders LaTeX)."""
        return self._layout.get_title()

    @title.setter
    def title(self, value: str) -> None:
        self._layout.set_title(value)
    
    @property
    def figure_widget(self) -> go.FigureWidget:
        """Access the underlying Plotly FigureWidget."""
        return self._figure
    
    @property
    def params(self) -> ParameterManager:
        """
        The ParameterManager instance.
        Acts like a dictionary of `{Symbol: Slider}` for backward compatibility.
        """
        return self._params
    
    @property
    def info_output(self) -> Dict[Hashable, widgets.Output]:
        """Dictionary of Info Output widgets indexed by id."""
        return self._info._outputs # Direct access for backward compat or advanced use

    @property
    def x_range(self) -> Tuple[float, float]:
        """Default x-axis range (restored on double-click)."""
        return self._x_range
    
    @x_range.setter
    def x_range(self, value: RangeLike) -> None:
        self._x_range = (float(InputConvert(value[0], float)), float(InputConvert(value[1], float)))
        self._figure.update_xaxes(range=self._x_range)

    @property
    def y_range(self) -> Tuple[float, float]:
        """Default y-axis range."""
        return self._y_range
    
    @y_range.setter
    def y_range(self, value: RangeLike) -> None:
        self._y_range = (float(InputConvert(value[0], float)), float(InputConvert(value[1], float)))
        self._figure.update_yaxes(range=self._y_range)

    @property
    def current_x_range(self) -> Optional[Tuple[float, float]]:
        """Current viewport x-range (read-only)."""
        return self._figure.layout.xaxis.range

    @property
    def current_y_range(self) -> Optional[Tuple[float, float]]:
        """Current viewport y-range (read-only)."""
        return self._figure.layout.yaxis.range
    
    @property
    def sampling_points(self) -> Optional[int]:
        """Default number of sampling points per plot."""
        return self._sampling_points

    @sampling_points.setter
    def sampling_points(self, val: Union[int, str, None]) -> None:
        self._sampling_points = int(InputConvert(val, int)) if isinstance(val, (int, float, str)) and val != "figure_default" else None

    # --- Public API ---

    def plot(
        self,
        var: Symbol,
        func: Expr,
        parameters: Optional[Sequence[Symbol]] = None,
        id: Optional[str] = None,
        x_domain: Optional[RangeLike] = None,
        sampling_points: Optional[Union[int, str]] = None,
    ) -> SmartPlot:
        """
        Plot a SymPy expression on the figure (and keep it “live”).

        Parameters
        ----------
        var : sympy.Symbol
            Independent variable (e.g. ``x``).
        func : sympy.Expr
            SymPy expression (e.g. ``sin(x)``).
        parameters : list[sympy.Symbol] or None, optional
            Parameter symbols. If None, they are inferred from the expression.
            If [], that means explicitly no parameters.
        x_domain : RangeLike or None, optional
            Domain of the independent variable (e.g. ``(-10, 10)``).
            If "figure_default", the figure's range is used when plotting. 
            If None, it is the same as "figure_default" for new plots while no change for existing plots.
        id : str, optional
            Unique identifier. If exists, the existing plot is updated in-place.
        """
        # ID Generation
        if id is None:
            for i in range(100):
                if f"f_{i}" not in self.plots:
                    id = f"f_{i}"
                    break
            if id is None: raise ValueError("Too many auto-generated IDs")

        # Parameter Autodetection
        if parameters is None:
            parameters = sorted([s for s in func.free_symbols if s != var], key=lambda s: s.sort_key())

        # Ensure Sliders Exist (Delegate to Manager)
        for p in parameters:
            self._params.add_param(p)
        
        # Update UI visibility
        self._layout.update_sidebar_visibility(self._params.has_params, self._info.has_info)

        # Create or Update Plot
        if id in self.plots:
            update_dont_create = True
        else: 
            update_dont_create = False

        if update_dont_create:
            self.plots[id].update(var=var, func=func, parameters=parameters, x_domain=x_domain, sampling_points=sampling_points)
            plot = self.plots[id]    
        else: 
            plot = SmartPlot(
                var=var, func=func, smart_figure=self, parameters=parameters,
                x_domain=x_domain, sampling_points=sampling_points, label=id
            )
            self.plots[id] = plot
        
        return plot
        

    def render(self, reason: str = "manual", trigger: Any = None) -> None:
        """
        Render all plots on the figure.

        This is a *hot* method: it is called during slider drags and (throttled)
        pan/zoom relayout events.
        """
        self._log_render(reason, trigger)
        
        # 1. Update all plots
        for plot in self.plots.values():
            plot.render()
        
        # 2. Run hooks (if triggered by parameter change)
        # Note: ParameterManager triggers this render, then we run hooks.
        if reason == "param_change" and trigger:
             hooks = self._params.get_hooks()
             for h_id, callback in list(hooks.items()):
                 try:
                     callback(trigger, self) # Pass self (SmartFigure) to hooks
                 except Exception as e:
                     warnings.warn(f"Hook {h_id} failed: {e}")

    def add_param(self, symbol: Symbol, **kwargs: Any) -> SmartFloatSlider:
        """
        Add a SmartFloatSlider parameter manually.
        """
        slider = self._params.add_param(symbol, **kwargs)
        self._layout.update_sidebar_visibility(self._params.has_params, self._info.has_info)
        return slider

    def get_info_output(self, id: Optional[Hashable] = None, **kwargs: Any) -> widgets.Output:
        """
        Create (or retrieve) an Output widget in the Info sidebar.
        """
        out = self._info.get_output(id, **kwargs)
        self._layout.update_sidebar_visibility(self._params.has_params, self._info.has_info)
        return out

    # Alias for backward compatibility
    new_info_output = get_info_output

    def add_info_component(self, id: Hashable, component_factory: Callable, hook_id: Optional[Hashable] = None, **kwargs: Any) -> Any:
        """
        Register (or replace) a stateful *info component*.

        An info component is a class/function that:
        1. Draws into an Info Output widget.
        2. Implements an `update(change, fig, out)` method.
        """
        out = self.get_info_output(id, **kwargs)
        inst = component_factory(out, self)
        
        if not hasattr(inst, 'update'):
            raise TypeError(f"Component {id} must have an 'update' method")
        
        self._info.add_component(id, inst)
        
        # Register hook to update component on param change
        if hook_id is None: hook_id = ("info_component", id)
        
        def _hook(change: Dict, fig: SmartFigure) -> None:
            inst.update(change, fig, out)
            
        self.add_param_change_hook(_hook, hook_id=hook_id)
        return inst

    def add_param_change_hook(self, callback: Callable[[Dict, SmartFigure], Any], hook_id: Optional[Hashable] = None) -> Hashable:
        """
        Register a callback to run when *any* parameter value changes.
        """
        return self._params.add_hook(callback, hook_id, fig=self)

    # --- Internal / Plumbing ---

    def _throttled_relayout(self, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last_relayout > 0.5:
            self._last_relayout = now
            self.render(reason="relayout")

    def _log_render(self, reason: str, trigger: Any) -> None:
        # Simple rate-limited logging implementation
        now = time.monotonic()
        if logger.isEnabledFor(logging.INFO) and (now - self._render_info_last_log_t) > 1.0:
            self._render_info_last_log_t = now
            logger.info(f"render(reason={reason}) plots={len(self.plots)}")
        
        if logger.isEnabledFor(logging.DEBUG) and (now - self._render_debug_last_log_t) > 0.5:
            self._render_debug_last_log_t = now
            logger.debug(f"ranges x={self.x_range} y={self.y_range}")

    def _ipython_display_(self, **kwargs: Any) -> None:
        """
        Special method called by IPython to display the object.
        Uses IPython.display.display() to render the underlying widget.
        """
        display(self._layout.output_widget)