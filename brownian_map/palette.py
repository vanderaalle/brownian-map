"""Loading cartographic colour palettes and mapping surfaces onto them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# A 29-colour bathymetric/hypsometric ramp (deep sea -> land -> peaks), from
# http://soliton.vm.bytemark.co.uk/pub/cpt-city/wkp/template/wiki-2.0.qgs
DEFAULT_PALETTE_PATH = Path(__file__).parent / "wiki-2.0.gpf"


def load_palette(path: Path | str = DEFAULT_PALETTE_PATH) -> np.ndarray:
    """Load a GIMP-style palette file (one `R,G,B[,...]` line per colour,
    0-255) as an `(n, 3)` array of floats in [0, 1]."""
    lines = Path(path).read_text().splitlines()
    rows = [[float(c) / 255 for c in line.split(",")[:3]] for line in lines if line.strip()]
    return np.array(rows)


def interpolate_rgb(color_a, color_b, t: float = 0.5) -> list[float]:
    """Blend two RGB colours; `t=0` returns `color_a`, `t=1` returns `color_b`."""
    return [a - (a - b) * t for a, b in zip(color_a, color_b)]


def normalize_surface(surface: np.ndarray, n_bins: int = 20, low_bins: int = 10) -> np.ndarray:
    """Map a surface of floats in [-1, 1] to integer palette indices.

    Values <= 0 (sea) map to `[0, low_bins)`; values > 0 (land) map to
    `[low_bins, low_bins + n_bins - 2)`. The asymmetry mirrors the palette,
    which devotes more entries to land elevation bands than to sea depth.
    """
    positive_scale = n_bins - 2
    indices = np.empty(surface.shape, dtype=int)
    is_land = surface > 0
    indices[is_land] = np.trunc(surface[is_land] * positive_scale).astype(int) + low_bins
    indices[~is_land] = np.trunc((surface[~is_land] + 1) * low_bins).astype(int)
    return indices


def colorize(normalized_surface: np.ndarray, rgb_palette: np.ndarray) -> np.ndarray:
    """Map an array of palette indices to an `(H, W, 3)` RGB image."""
    return rgb_palette[normalized_surface]
