"""Matplotlib-based rendering for surfaces, isocurves and rivers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def render_surface(
    normalized_surface: np.ndarray,
    rgb_palette: np.ndarray,
    path: Optional[Path | str] = None,
    isocurves: Optional[list[list[np.ndarray]]] = None,
    rivers: Optional[list[list[list[int]]]] = None,
    pixels_per_cell: float = 3.0,
    dpi: int = 100,
    mark_river_starts: bool = True,
    isocurve_linewidth: float = 1.0,
    isocurve_color=None,
    isocurve_alpha: float = 1.0,
) -> Figure:
    """Render a normalized surface through `rgb_palette`, optionally
    overlaying iso-elevation contours and/or traced rivers.

    `normalized_surface` holds integer indices into `rgb_palette`, as
    produced by `palette.normalize_surface`. Contours and rivers are
    expected in `(row, col)` array-index coordinates, as produced by
    `postprocess.compute_isocurves` and `rivers.generate_rivers`.

    `mark_river_starts` draws a magenta dot at each river's source — useful
    while testing tracing behaviour, easy to turn off (`mark_river_starts=False`)
    once it's no longer needed.

    Three knobs control how the traced isocurves look. `isocurve_linewidth`
    defaults to `1.0` -- bold enough to keep coastlines and sea-level
    banding crisp rather than aliased/"pixelish" at typical render sizes,
    tested against 0.5 (too thin, edges look pixelated) and higher values
    up to 5.0 (barely different at the default colour, see below). But
    *colour* is the knob that actually matters most: at the default
    colour, every level is `rgb_palette[level]`, so each line matches its
    own surrounding fill almost exactly and stays understated regardless
    of width.

    - `isocurve_color` (default `None`, meaning "match the palette",
      today's only prior behaviour): set an explicit colour (e.g.
      `"black"`) to make lines visible against their own fill instead of
      blending into it.
    - `isocurve_alpha` (default `1.0`): a contrasting colour at full
      opacity is not what you want -- `compute_isocurves` traces *every*
      level, sea-depth bins included, and a rough surface's sea levels
      alone produce enough overlapping lines that full-opacity black turns
      the sea into total noise. Something in the `~0.15-0.3` range lets
      individual strokes stay faint while dense overlap between many
      overlapping sea-level lines still builds up into a real woven
      texture -- the closer overlapping strokes are to how a hand-drawn
      contour map actually accumulates texture, the more it reads as
      texture and not noise.
    - `isocurve_linewidth` (default `1.0`) still matters, just less than
      colour -- once lines are visible at all, this is the finer control
      over how bold each individual stroke reads.
    """
    dim = normalized_surface.shape[0]
    size_inches = dim * pixels_per_cell / dpi
    fig, ax = plt.subplots(figsize=(size_inches, size_inches), dpi=dpi)
    ax.imshow(
        normalized_surface,
        cmap=ListedColormap(rgb_palette),
        vmin=0,
        vmax=len(rgb_palette) - 1,
        interpolation="nearest",
    )
    ax.set_axis_off()

    if isocurves is not None:
        for level, contours in enumerate(isocurves):
            color = isocurve_color if isocurve_color is not None else rgb_palette[min(level, len(rgb_palette) - 1)]
            for contour in contours:
                ax.plot(
                    contour[:, 1], contour[:, 0],
                    color=color, linewidth=isocurve_linewidth, alpha=isocurve_alpha,
                )

    if rivers is not None:
        river_color = rgb_palette[0]  # deepest sea entry, per palette.py's deep-sea-first ordering
        for river in rivers:
            rows = [p[0] for p in river]
            cols = [p[1] for p in river]
            ax.plot(cols, rows, color=river_color, linewidth=1)
            if mark_river_starts:
                ax.plot(cols[0], rows[0], "o", color="magenta", markersize=3)

    fig.tight_layout(pad=0)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    return fig
