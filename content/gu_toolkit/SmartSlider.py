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
            description="",
            continuous_update=True,
            readout=False,  # IMPORTANT: no built-in numeric field
            style={"description_width": "initial"},
            layout=widgets.Layout(width="55%"),
        )

        self.description_label = widgets.HTMLMath(
            value=description,
            layout=widgets.Layout(width="52px"),
        )

        #self._limit_style = widgets.HTML(
        #    "<style>"
        #    ".smart-slider-limit input{"
        #    "font-size:10px;"
        #    "color:#666;"
        #    "height:18px;"
        #    "border:none;"
        #    "box-shadow:none;"
        #    "background:transparent;"
        #    "padding:0px;"
        #   "text-align:center;"
        #    "}"
        #    "</style>"
        #)

        self._limit_style = widgets.HTML(r"""
<style>
/* ============================================================
   Min/Max look like small non-editable text until focus,
   but remain fully editable on click.
   Works without relying on a specific wrapper DOM.
   ============================================================ */

/* Shrink any wrapper(s) JupyterLab/ipywidgets might impose. */
.smart-slider-limit,
.smart-slider-limit * {
  min-height: 0 !important;
}

/* The actual editable element(s): handle both input + textarea. */
.smart-slider-limit :is(input, textarea) {
  /* Make it small */
  font-size: 10px !important;
  line-height: 1.1 !important;

  /* Force compact geometry despite JupyterLab theme defaults */
  height: 16px !important;
  min-height: 0 !important;
  padding: 0 2px !important;
  margin: 0 !important;

  /* Make it look like static text */
  background: transparent !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
  border-radius: 3px !important;

  color: var(--jp-ui-font-color2, #666) !important;
  text-align: center !important;
  box-sizing: border-box !important;
}

/* Optional: subtle “this is clickable” affordance on hover */
.smart-slider-limit :is(input, textarea):hover {
  border-bottom-color: rgba(0,0,0,0.20) !important;
}

/* On focus: show edit chrome so users realize they can type */
.smart-slider-limit :is(input, textarea):focus {
  outline: none !important;
  border-color: rgba(0,0,0,0.28) !important;
  background: rgba(0,0,0,0.04) !important;
}
</style>
""")
        # The *only* numeric field (editable; accepts expressions)
        self.number = widgets.Text(
            value=str(value),
            continuous_update=False,  # commit on Enter (and typically blur)
            layout=widgets.Layout(width="64px"),
        )

        button_layout = widgets.Layout(width="24px", height="24px", padding="0px")
        self.btn_reset = widgets.Button(
            description="↺",
            tooltip="Reset",
            layout=button_layout,
        )
        self.btn_settings = widgets.Button(
            description="⚙",
            tooltip="Settings",
            layout=button_layout,
        )
        self.btn_play = widgets.ToggleButton(
            value=False,
            description="▶",
            tooltip="Animate",
            layout=button_layout,
        )
        self.play = widgets.Play(
            value=value,
            min=min,
            max=max,
            step=step if step != 0 else 0.01,
            interval=60,
            layout=widgets.Layout(
                visibility="hidden",
                width="0px",
                height="0px",
                margin="0px",
                padding="0px",
            ),
        )
        self._play_syncing = False
        self._play_button_syncing = False
        self._play_direction = 1

        # --- Settings panel ---------------------------------------------------
        style_args = {"style": {"description_width": "50px"}, "layout": widgets.Layout(width="100px")}
        self.set_min = widgets.Text(
    value=f"{min:.4g}",
    continuous_update=False,
    layout=widgets.Layout(width="40px", height="16px"),
)
        self.set_min.add_class("smart-slider-limit")
        self.set_max = widgets.Text(
    value=f"{max:.4g}",
    continuous_update=False,
    layout=widgets.Layout(width="40px", height="16px"),
)
        self.set_max.add_class("smart-slider-limit")
        self.set_step = widgets.FloatText(value=step, description="Step:", **style_args)
        self.set_live = widgets.Checkbox(
            value=True,
            description="Live Update",
            indent=False,
            layout=widgets.Layout(width="110px"),
        )

        self.settings_panel = widgets.VBox(
            [widgets.HBox([self.set_step, self.set_live], layout=widgets.Layout(gap="8px"))],
            layout=widgets.Layout(
                display="none",
                border="1px solid #e5e7eb",
                padding="4px 6px",
                margin="4px 0 0 0",
                border_radius="6px",
            ),
        )

        # --- Layout -----------------------------------------------------------
        top_row = widgets.HBox(
            [
                self._limit_style,
                self.description_label,
                self.set_min,
                self.slider,
                self.set_max,
                self.number,
                self.btn_reset,
                self.btn_settings,
                self.btn_play,
            ],
            layout=widgets.Layout(align_items="center", gap="4px"),
        )
        super().__init__([top_row, self.settings_panel, self.play], **kwargs)

        # --- Wiring -----------------------------------------------------------
        # Keep self.value and slider.value in sync
        traitlets.link((self, "value"), (self.slider, "value"))

        # Slider -> Text (display)
        self.slider.observe(self._sync_number_from_slider, names="value")
        self.slider.observe(self._sync_limit_texts, names=["min", "max"])
        self.slider.observe(self._sync_play_bounds, names="step")
        # Text -> Slider (parse + clamp)
        self.number.observe(self._commit_text_value, names="value")
        self.set_min.observe(self._commit_min_value, names="value")
        self.set_max.observe(self._commit_max_value, names="value")

        # Buttons
        self.btn_reset.on_click(self._reset)
        self.btn_settings.on_click(self._toggle_settings)
        self.btn_play.observe(self._toggle_play, names="value")
        self.play.observe(self._sync_slider_from_play, names="value")

        # Settings -> slider traits
        widgets.link((self.set_step, "value"), (self.slider, "step"))
        widgets.link((self.set_live, "value"), (self.slider, "continuous_update"))

        # Initialize trait (and normalize displayed text)
        self.value = value
        self._sync_number_text(self.value)
        self._sync_limit_texts(None)

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
        if self.play.playing and not self._play_syncing:
            self._stop_playback()
        self._sync_number_text(change.new)

    def _sync_limit_texts(self, change) -> None:
        """Update the min/max limit text fields from the slider limits."""
        if self._syncing:
            return
        self._sync_play_bounds()
        self._syncing = True
        try:
            self.set_min.value = f"{self.slider.min:.4g}"
            self.set_max.value = f"{self.slider.max:.4g}"
        finally:
            self._syncing = False

    def _commit_limit_value(self, change, *, limit: str) -> None:
        """Parse and apply min/max limits from text inputs."""
        if self._syncing:
            return
        raw = (change.new or "").strip()
        old_min = float(self.slider.min)
        old_max = float(self.slider.max)

        try:
            new_val = float(InputConvert(raw, dest_type=float, truncate=True))
            if limit == "min":
                new_min = min(new_val, old_max)
                self.slider.min = new_min
            else:
                new_max = max(new_val, old_min)
                self.slider.max = new_max
            self._sync_limit_texts(None)
        except (ValueError, TypeError, SyntaxError):
            self._syncing = True
            try:
                self.set_min.value = f"{old_min:.4g}"
                self.set_max.value = f"{old_max:.4g}"
            finally:
                self._syncing = False

    def _commit_min_value(self, change) -> None:
        self._commit_limit_value(change, limit="min")

    def _commit_max_value(self, change) -> None:
        self._commit_limit_value(change, limit="max")

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
        self._stop_playback()

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
        self._stop_playback()
        self.value = self._defaults["value"]  # slider sync + slider observer updates text

    def _toggle_settings(self, _) -> None:
        self.settings_panel.layout.display = (
            "none" if self.settings_panel.layout.display == "flex" else "flex"
        )

    def _toggle_play(self, change) -> None:
        if self._play_button_syncing:
            return
        if change.new:
            self._start_playback()
        else:
            self._stop_playback()

    def _start_playback(self) -> None:
        self._sync_play_bounds()
        self._play_direction = 1 if self.play.step >= 0 else -1
        self.play.value = self.slider.value
        self.play.playing = True
        self._sync_play_button(True)

    def _stop_playback(self) -> None:
        if not self.play.playing and not self.btn_play.value:
            return
        self.play.playing = False
        self._sync_play_button(False)

    def _sync_play_button(self, playing: bool) -> None:
        self._play_button_syncing = True
        try:
            self.btn_play.value = playing
            self.btn_play.description = "⏸" if playing else "▶"
        finally:
            self._play_button_syncing = False

    def _sync_play_bounds(self, change=None) -> None:
        step = float(self.slider.step) if self.slider.step != 0 else 0.01
        self.play.min = float(self.slider.min)
        self.play.max = float(self.slider.max)
        self.play.step = abs(step) * (1 if self._play_direction >= 0 else -1)

    def _sync_slider_from_play(self, change) -> None:
        if self._play_syncing:
            return
        self._play_syncing = True
        try:
            self.slider.value = change.new
        finally:
            self._play_syncing = False
        self._update_play_direction(change.new)

    def _update_play_direction(self, current: float) -> None:
        if not self.play.playing:
            return
        max_val = float(self.slider.max)
        min_val = float(self.slider.min)
        if current >= max_val:
            self._play_direction = -1
        elif current <= min_val:
            self._play_direction = 1
        self.play.step = abs(float(self.slider.step) or 0.01) * self._play_direction
