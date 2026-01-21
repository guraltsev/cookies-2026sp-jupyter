import ipywidgets as widgets
import traitlets

from .InputConvert import InputConvert


class SmartFloatSlider(widgets.VBox):
    """
    A FloatSlider with:
      - a *single editable numeric field* (Text) that accepts expressions via InputConvert,
      - reset + settings buttons,
      - a small settings panel (min/max/step + live update toggle).

    Design notes
    ------------
    - The slider's built-in readout is disabled, so there is only one number field.
    - The Text field commits on Enter (continuous_update=False). This avoids fighting the
      user while they type partial expressions like "pi/2".
    - If parsing fails, the Text field reverts to the previous committed value.
    """

    value = traitlets.Float(0.0)

    def __init__(
        self,
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.1,
        description="Value:",
        **kwargs,
    ):
        # Remember defaults for reset
        self._defaults = {"value": value, "min": min, "max": max, "step": step}

        # Internal guard to prevent circular updates (slider -> text -> slider -> ...)
        self._syncing = False

        # --- Main controls ----------------------------------------------------
        self.slider = widgets.FloatSlider(
            value=value,
            min=min,
            max=max,
            step=step,
            description=description,
            continuous_update=True,
            readout=False,  # IMPORTANT: no built-in numeric field
            style={"description_width": "initial"},
            layout=widgets.Layout(width="60%"),
        )

        # The *only* numeric field (editable; accepts expressions)
        self.number = widgets.Text(
            value=str(value),
            continuous_update=False,  # commit on Enter (and typically blur)
            layout=widgets.Layout(width="90px"),
        )

        self.btn_reset = widgets.Button(
            description="↺", tooltip="Reset", layout=widgets.Layout(width="35px")
        )
        self.btn_settings = widgets.Button(
            description="⚙", tooltip="Settings", layout=widgets.Layout(width="35px")
        )

        # --- Settings panel ---------------------------------------------------
        style_args = {"style": {"description_width": "50px"}, "layout": widgets.Layout(width="100px")}
        self.set_min = widgets.FloatText(value=min, description="Min:", **style_args)
        self.set_max = widgets.FloatText(value=max, description="Max:", **style_args)
        self.set_step = widgets.FloatText(value=step, description="Step:", **style_args)
        self.set_live = widgets.Checkbox(
            value=True,
            description="Live Update",
            indent=False,
            layout=widgets.Layout(width="120px"),
        )

        self.settings_panel = widgets.VBox(
            [
                widgets.HBox([self.set_min, self.set_max, self.set_step]),
                widgets.HBox([self.set_live]),
            ],
            layout=widgets.Layout(
                display="none", border="1px solid #eee", padding="5px", margin="5px 0"
            ),
        )

        # --- Layout -----------------------------------------------------------
        top_row = widgets.HBox(
            [self.slider, self.number, self.btn_reset, self.btn_settings],
            layout=widgets.Layout(align_items="center"),
        )
        super().__init__([top_row, self.settings_panel], **kwargs)

        # --- Wiring -----------------------------------------------------------
        # Keep self.value and slider.value in sync
        traitlets.link((self, "value"), (self.slider, "value"))

        # Slider -> Text (display)
        self.slider.observe(self._sync_number_from_slider, names="value")

        # Text -> Slider (parse + clamp)
        self.number.observe(self._commit_text_value, names="value")

        # Buttons
        self.btn_reset.on_click(self._reset)
        self.btn_settings.on_click(self._toggle_settings)

        # Settings -> slider traits
        widgets.link((self.set_min, "value"), (self.slider, "min"))
        widgets.link((self.set_max, "value"), (self.slider, "max"))
        widgets.link((self.set_step, "value"), (self.slider, "step"))
        widgets.link((self.set_live, "value"), (self.slider, "continuous_update"))

        # Initialize trait (and normalize displayed text)
        self.value = value
        self._sync_number_text(self.value)

    # --- Helpers --------------------------------------------------------------

    def _sync_number_text(self, val: float) -> None:
        """Set the text field from a numeric value, without triggering parse logic."""
        self._syncing = True
        try:
            self.number.value = f"{val:.4g}"
        finally:
            self._syncing = False

    def _sync_number_from_slider(self, change) -> None:
        """When the slider moves, update the single numeric field."""
        if self._syncing:
            return
        self._sync_number_text(change.new)

    def _commit_text_value(self, change) -> None:
        """
        When the user commits text (Enter / blur):
          - parse via InputConvert,
          - clamp to [min, max],
          - update self.value,
          - normalize the displayed text.

        On any error, revert to the value *before* this edit.
        """
        if self._syncing:
            return

        raw = (change.new or "").strip()
        old_val = float(self.value)

        try:
            new_val = InputConvert(raw, dest_type=float, truncate=True)
            new_val = max(self.slider.min, min(float(new_val), self.slider.max))
            self.value = new_val
            self._sync_number_text(self.value)  # normalize formatting
        except (ValueError, TypeError, SyntaxError):
            # Revert to the value before the edit
            self._sync_number_text(old_val)

    # --- Button handlers ------------------------------------------------------

    def _reset(self, _) -> None:
        self.value = self._defaults["value"]  # slider sync + slider observer updates text

    def _toggle_settings(self, _) -> None:
        self.settings_panel.layout.display = (
            "none" if self.settings_panel.layout.display == "flex" else "flex"
        )
