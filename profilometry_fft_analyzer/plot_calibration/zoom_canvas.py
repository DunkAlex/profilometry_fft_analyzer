"""
zoom_canvas.py
--------------
Reusable tkinter image view: scroll-to-zoom (cursor-anchored), middle- or
Shift+left-drag panning, crosshair cursor readout, and efficient rendering
(only the visible crop is resampled, so zoom cost doesn't scale with image
size). Extracted from gui_calibrate's _CalibrationApp so the manual-fit GUI
and the profile-creation GUI share one implementation.

Owners bind their own pick handlers on `.canvas` (plain <Button-1> is left
free — Shift+Button-1 is claimed for panning) and may pass `on_after_redraw`
to repaint overlay items after each image redraw: the image item is recreated
every redraw, so overlay items must be redrawn afterward to stay on top.
"""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np

MAX_CANVAS_W = 1100
MAX_CANVAS_H = 750

ZOOM_STEP = 1.20        # multiplier per wheel notch
ZOOM_MIN = 0.1
ZOOM_MAX = 40.0
KEY_PAN_PX = 40


class ZoomPanImageFrame:
    """A Frame holding a toolbar (zoom buttons + cursor readout) and the
    zoomable canvas. Coordinate helpers convert between canvas and image
    space; `scale` is the current canvas-pixels-per-image-pixel factor."""

    def __init__(self, parent, image_rgb: np.ndarray,
                 max_w: int = MAX_CANVAS_W, max_h: int = MAX_CANVAS_H,
                 on_after_redraw: Optional[Callable[[], None]] = None):
        import tkinter as tk
        from PIL import Image

        self.tk = tk
        self.image_rgb = image_rgb
        self.on_after_redraw = on_after_redraw
        self.orig_h, self.orig_w = image_rgb.shape[:2]
        self.orig_pil = Image.fromarray(image_rgb)

        # transform state
        self.base_scale = min(max_w / self.orig_w, max_h / self.orig_h, 1.0)
        self.zoom = 1.0
        self.offset_x = 0.0     # canvas coord of image (0,0)
        self.offset_y = 0.0
        # True until the user explicitly zooms/pans (see zoom_by/pan_by/
        # _on_pan_drag) — while True, a canvas resize re-centers the image
        # (see _on_configure) instead of leaving it pinned wherever it was
        # for the old, image-sized canvas; once the user has taken the view
        # somewhere on purpose, resizing no longer moves it out from under
        # them.
        self._at_default_view = True

        # pan tracking
        self._pan_start_mouse = None
        self._pan_start_offset = None

        self.frame = tk.Frame(parent)

        # --- toolbar ---
        tb = tk.Frame(self.frame)
        tb.pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Button(tb, text="Zoom -", width=8,
                  command=lambda: self.zoom_by(1 / ZOOM_STEP)).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="Zoom +", width=8,
                  command=lambda: self.zoom_by(ZOOM_STEP)).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="Fit", width=6,
                  command=self.reset_view).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="1:1", width=6,
                  command=self.one_to_one).pack(side=tk.LEFT, padx=2)
        self.zoom_var = tk.StringVar(value="Zoom: 1.00x")
        tk.Label(tb, textvariable=self.zoom_var, width=14).pack(side=tk.LEFT, padx=8)
        self.cursor_var = tk.StringVar(value="Cursor: —")
        tk.Label(tb, textvariable=self.cursor_var).pack(side=tk.LEFT, padx=8)

        # --- canvas ---
        # width/height here are only the INITIAL request; the canvas fills
        # and expands with its parent (fill=BOTH, expand=True) so a wide,
        # short image no longer gets stuck in a short canvas — maximizing
        # the window gives zoom/pan the whole area to work with. base_scale
        # is unchanged, so the initial image size is exactly as before; the
        # extra room only becomes usable once the user zooms/pans.
        canvas_w = int(self.orig_w * self.base_scale)
        canvas_h = int(self.orig_h * self.base_scale)
        self.canvas = tk.Canvas(self.frame, width=canvas_w, height=canvas_h,
                                cursor="crosshair", bg="#222222",
                                highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._last_canvas_size = (canvas_w, canvas_h)

        # --- bindings (owner adds its own <Button-1> pick handler) ---
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Motion>", self._on_motion)
        # zoom: Windows/Mac use <MouseWheel>, Linux uses Button-4/5
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        # panning: middle-drag, or Shift+left-drag
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<Shift-Button-1>", self._on_pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self._on_pan_drag)

    # --------- geometry manager passthrough ---------

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    # --------- coordinate transforms ---------

    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    def canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return self.offset_x + ix * self.scale, self.offset_y + iy * self.scale

    def _center_offsets(self, zoom: float) -> tuple[float, float]:
        """Offsets that center the image (at `zoom`) within the canvas's
        CURRENT size — used for the default/fit view, which may now be
        smaller than the canvas since the canvas fills its whole parent
        frame rather than being sized exactly to the image."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1:
            cw = int(self.orig_w * self.base_scale)
        if ch <= 1:
            ch = int(self.orig_h * self.base_scale)
        disp_w = self.orig_w * self.base_scale * zoom
        disp_h = self.orig_h * self.base_scale * zoom
        return max(0.0, (cw - disp_w) / 2.0), max(0.0, (ch - disp_h) / 2.0)

    # --------- rendering (crop + resize) ---------

    def first_draw(self):
        """Deferred initial draw — call once after the window is built, so
        the canvas has a real size before rendering."""
        self.canvas.after(50, self._deferred_first_draw)

    def _deferred_first_draw(self):
        self.canvas.update_idletasks()
        # reset_view() (not a raw redraw()) so the image starts CENTERED in
        # whatever area the now-realized canvas actually has, rather than
        # pinned at the top-left corner of it.
        self.reset_view()

    def redraw(self):
        from PIL import Image, ImageTk

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.canvas.after(20, self.redraw)
            return

        s = self.scale
        # Visible image region in image coords
        ix0 = max(0.0, -self.offset_x / s)
        iy0 = max(0.0, -self.offset_y / s)
        ix1 = min(float(self.orig_w), (cw - self.offset_x) / s)
        iy1 = min(float(self.orig_h), (ch - self.offset_y) / s)

        self.canvas.delete("image")
        if ix0 >= ix1 or iy0 >= iy1:
            self._update_zoom_label()
            if self.on_after_redraw is not None:
                self.on_after_redraw()
            return

        # Crop and resample only the visible portion
        box = (int(np.floor(ix0)), int(np.floor(iy0)),
               int(np.ceil(ix1)),  int(np.ceil(iy1)))
        crop = self.orig_pil.crop(box)
        disp_w = max(1, int(round((box[2] - box[0]) * s)))
        disp_h = max(1, int(round((box[3] - box[1]) * s)))
        # LANCZOS for downscale/zoom-out; NEAREST above ~4x so pixels look crisp
        resample = Image.NEAREST if s > 4.0 else Image.LANCZOS
        resized = crop.resize((disp_w, disp_h), resample)
        self._tk_img = ImageTk.PhotoImage(resized)

        canvas_x = self.offset_x + box[0] * s
        canvas_y = self.offset_y + box[1] * s
        self.canvas.create_image(canvas_x, canvas_y, anchor=self.tk.NW,
                                 image=self._tk_img, tags="image")
        self.canvas.tag_lower("image")   # overlay items always above the image

        self._update_zoom_label()
        if self.on_after_redraw is not None:
            self.on_after_redraw()

    def _update_zoom_label(self):
        self.zoom_var.set(f"Zoom: {self.scale:.2f}x")

    # --------- zoom / pan ---------

    def zoom_by(self, factor: float,
                anchor_canvas: tuple[float, float] | None = None):
        new_zoom = self.zoom * factor
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, new_zoom))
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        self._at_default_view = False
        # anchor at canvas center by default; if a canvas point was provided,
        # keep the image point under that canvas point stationary
        if anchor_canvas is None:
            cw = self.canvas.winfo_width() or 1
            ch = self.canvas.winfo_height() or 1
            anchor_canvas = (cw / 2.0, ch / 2.0)
        cx, cy = anchor_canvas
        ix, iy = self.canvas_to_image(cx, cy)
        self.zoom = new_zoom
        # rearrange offset so (ix, iy) still lands under (cx, cy)
        self.offset_x = cx - ix * self.scale
        self.offset_y = cy - iy * self.scale
        self.redraw()

    def pan_by(self, dx: int, dy: int):
        self._at_default_view = False
        self.offset_x += dx
        self.offset_y += dy
        self.redraw()

    def reset_view(self):
        """Return to the default view: unzoomed and centered in whatever
        area the canvas currently occupies (see _center_offsets) — the
        "Fit" button, R/F keys, and the initial first_draw all land here."""
        self.zoom = 1.0
        self.offset_x, self.offset_y = self._center_offsets(1.0)
        self._at_default_view = True
        self.redraw()

    def one_to_one(self):
        # zoom such that scale == 1.0 (1 image pixel = 1 canvas pixel),
        # keeping the current view center
        cw = self.canvas.winfo_width() or 1
        ch = self.canvas.winfo_height() or 1
        target_zoom = 1.0 / self.base_scale
        self.zoom_by(target_zoom / self.zoom, (cw / 2.0, ch / 2.0))

    def bind_keyboard(self, root):
        """Standard zoom/pan keys on `root`: +/- zoom, R/F fit, arrows pan."""
        root.bind("<plus>", lambda e: self.zoom_by(ZOOM_STEP))
        root.bind("<equal>", lambda e: self.zoom_by(ZOOM_STEP))
        root.bind("<KP_Add>", lambda e: self.zoom_by(ZOOM_STEP))
        root.bind("<minus>", lambda e: self.zoom_by(1 / ZOOM_STEP))
        root.bind("<underscore>", lambda e: self.zoom_by(1 / ZOOM_STEP))
        root.bind("<KP_Subtract>", lambda e: self.zoom_by(1 / ZOOM_STEP))
        root.bind("r", lambda e: self.reset_view())
        root.bind("R", lambda e: self.reset_view())
        root.bind("f", lambda e: self.reset_view())
        root.bind("F", lambda e: self.reset_view())
        root.bind("<Left>",  lambda e: self.pan_by(+KEY_PAN_PX, 0))
        root.bind("<Right>", lambda e: self.pan_by(-KEY_PAN_PX, 0))
        root.bind("<Up>",    lambda e: self.pan_by(0, +KEY_PAN_PX))
        root.bind("<Down>",  lambda e: self.pan_by(0, -KEY_PAN_PX))

    # --------- event handlers ---------

    def _on_wheel(self, event):
        # Determine direction across platforms
        if event.num == 4:                # Linux scroll up
            factor = ZOOM_STEP
        elif event.num == 5:              # Linux scroll down
            factor = 1 / ZOOM_STEP
        elif getattr(event, "delta", 0) > 0:  # Windows/Mac up
            factor = ZOOM_STEP
        elif getattr(event, "delta", 0) < 0:  # Windows/Mac down
            factor = 1 / ZOOM_STEP
        else:
            return
        self.zoom_by(factor, anchor_canvas=(event.x, event.y))

    def _on_pan_start(self, event):
        self._pan_start_mouse = (event.x, event.y)
        self._pan_start_offset = (self.offset_x, self.offset_y)

    def _on_pan_drag(self, event):
        if self._pan_start_mouse is None:
            return
        self._at_default_view = False
        dx = event.x - self._pan_start_mouse[0]
        dy = event.y - self._pan_start_mouse[1]
        self.offset_x = self._pan_start_offset[0] + dx
        self.offset_y = self._pan_start_offset[1] + dy
        self.redraw()

    def _on_configure(self, event):
        """Redraw when the canvas is actually resized (window maximize /
        full-screen / pane drag), so the newly available area is painted
        instead of leaving the image stuck at its first-draw size. Ignores
        no-op configures (same size) and the not-yet-realized 1px state.

        Re-centers first when the view is still at its default (untouched by
        the user) — otherwise the image would stay pinned at whatever
        offset it had for the OLD, smaller canvas, landing in a corner of
        the newly enlarged one instead of staying centered in it. Once the
        user has zoomed/panned on purpose, resizing just reveals more
        canvas around their existing view instead of moving it."""
        size = (event.width, event.height)
        if size == self._last_canvas_size or event.width <= 1 or event.height <= 1:
            return
        self._last_canvas_size = size
        if self._at_default_view:
            self.offset_x, self.offset_y = self._center_offsets(self.zoom)
        self.redraw()

    def _on_motion(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        if 0 <= ix < self.orig_w and 0 <= iy < self.orig_h:
            r, g, b = self.image_rgb[int(iy), int(ix)]
            self.cursor_var.set(f"({int(ix)}, {int(iy)}) rgb=({r},{g},{b})")
        else:
            self.cursor_var.set("Cursor: —")
