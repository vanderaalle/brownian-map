"""Two textbook heightfield generators, kept separate from `surface.py`.

These exist to answer one question: how much of the "this looks like a
map" effect comes from the terrain generator, versus from the palette and
isocurves applied on top of it? `surface.py`'s spark-fill is a randomized
neighbour-averaging diffusion, which sits close to *value noise*; the two
generators here are the two baselines value noise is usually compared
against — Perlin's gradient noise and Fournier/Fussell/Carpenter's
diamond-square (1982) midpoint displacement. Run any of the three through
the same `palette`/`postprocess`/`render` pipeline and compare.

Unlike `surface.py`, these are not built to tile (no toroidal wraparound) —
that's irrelevant to the comparison, which only looks at a single tile.
"""

from __future__ import annotations

import math
import random

import numpy as np


# ---------------------------------------------------------------------------
# Perlin (gradient) noise
# ---------------------------------------------------------------------------


def _fade(t: np.ndarray) -> np.ndarray:
    """Ken Perlin's improved fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin_octave(dim: int, cells: int, rng: random.Random) -> np.ndarray:
    """One octave of classic 2D gradient noise: `cells` x `cells` random
    unit gradients, bilinearly interpolated (with the fade curve) up to a
    `dim` x `dim` grid."""
    angles = np.array(
        [[rng.uniform(0, 2 * math.pi) for _ in range(cells + 1)] for _ in range(cells + 1)]
    )
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    coords = np.linspace(0, cells, dim, endpoint=False)
    cell_index = coords.astype(int)
    local = (coords - cell_index)[:, None]  # fractional position within its cell

    def dot_grid_gradient(row_idx: np.ndarray, col_idx: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        g = gradients[row_idx][:, col_idx]  # (dim, dim, 2)
        return g[..., 0] * dx[:, None] + g[..., 1] * dy[None, :]

    rows = cell_index
    cols = cell_index
    dx = local[:, 0]
    dy = local[:, 0]

    n00 = dot_grid_gradient(rows, cols, dx, dy)
    n10 = dot_grid_gradient(rows + 1, cols, dx - 1, dy)
    n01 = dot_grid_gradient(rows, cols + 1, dx, dy - 1)
    n11 = dot_grid_gradient(rows + 1, cols + 1, dx - 1, dy - 1)

    u = _fade(dx)[:, None]
    v = _fade(dy)[None, :]

    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


def generate_perlin_surface(
    dim: int, base_cells: int = 6, octaves: int = 4, persistence: float = 0.5, seed: int | None = None
) -> np.ndarray:
    """Fractal (multi-octave) Perlin noise, `dim` x `dim`, roughly in
    [-1, 1]. `base_cells` sets the coarsest octave's gradient-grid
    resolution; each further octave doubles the frequency and scales
    amplitude by `persistence`."""
    rng = random.Random(seed)
    total = np.zeros((dim, dim))
    amplitude = 1.0
    max_amplitude = 0.0
    cells = base_cells
    for _ in range(octaves):
        total += amplitude * _perlin_octave(dim, cells, rng)
        max_amplitude += amplitude
        amplitude *= persistence
        cells *= 2
    return total / max_amplitude


# ---------------------------------------------------------------------------
# Diamond-square
# ---------------------------------------------------------------------------


def generate_diamond_square_surface(dim: int, roughness: float = 0.6, seed: int | None = None) -> np.ndarray:
    """Midpoint-displacement fractal terrain (Fournier, Fussell & Carpenter
    1982), cropped to `dim` x `dim`. `roughness` in (0, 1) controls how
    slowly the displacement magnitude decays each subdivision: near 0 gives
    smooth, rolling terrain; near 1 gives jagged, mountainous terrain.
    """
    rng = random.Random(seed)
    size = 1
    while size + 1 < dim:
        size *= 2
    size += 1  # smallest 2**n + 1 that covers dim

    grid = np.zeros((size, size))
    grid[0, 0] = rng.uniform(-1, 1)
    grid[0, -1] = rng.uniform(-1, 1)
    grid[-1, 0] = rng.uniform(-1, 1)
    grid[-1, -1] = rng.uniform(-1, 1)

    step = size - 1
    scale = 1.0
    while step > 1:
        half = step // 2

        # Diamond step: centre of each square = average of its 4 corners + jitter.
        for row in range(half, size, step):
            for col in range(half, size, step):
                corners = [
                    grid[row - half, col - half],
                    grid[row - half, col + half],
                    grid[row + half, col - half],
                    grid[row + half, col + half],
                ]
                grid[row, col] = sum(corners) / 4 + rng.uniform(-1, 1) * scale

        # Square step: midpoint of each edge = average of its 4 (wrapped)
        # neighbours in a diamond around it + jitter.
        for row in range(0, size, half):
            start = half if (row // half) % 2 == 0 else 0
            for col in range(start, size, step):
                neighbors = []
                if row - half >= 0:
                    neighbors.append(grid[row - half, col])
                if row + half < size:
                    neighbors.append(grid[row + half, col])
                if col - half >= 0:
                    neighbors.append(grid[row, col - half])
                if col + half < size:
                    neighbors.append(grid[row, col + half])
                grid[row, col] = sum(neighbors) / len(neighbors) + rng.uniform(-1, 1) * scale

        step = half
        scale *= 2 ** (-roughness)

    cropped = grid[:dim, :dim]
    return cropped / np.abs(cropped).max()


# ---------------------------------------------------------------------------
# Shared normalization for comparison figures
# ---------------------------------------------------------------------------


def set_sea_level(surface: np.ndarray, land_fraction: float = 0.35) -> np.ndarray:
    """Shift `surface` so that exactly `land_fraction` of its cells are > 0,
    matching the land/sea split `palette.normalize_surface` expects, then
    rescale to fill [-1, 1]. Lets heightfields with different raw
    distributions (Perlin, diamond-square, spark-fill) sit at a comparable
    "how much land" setting."""
    sea_level = np.quantile(surface, 1 - land_fraction)
    shifted = surface - sea_level
    return shifted / np.abs(shifted).max()


# ---------------------------------------------------------------------------
# Fractal-roughness measurement (ground truth comparison)
# ---------------------------------------------------------------------------


def radial_psd_slope(
    field: np.ndarray, fit_range: tuple[float, float] = (0.05, 0.4)
) -> tuple[float, float]:
    """Estimate how "realistic" a heightfield's roughness is, in the sense
    used to validate procedural terrain against real DEMs: natural terrain
    is an approximately self-affine fractal, so its 2D power spectrum falls
    off as a power law, `P(k) ~ k^-beta`, and `beta` maps to a Hurst
    exponent `H = (beta - 2) / 2`. Lower `H` = rougher at fine scales
    relative to coarse ones; higher `H` = smoother. Real terrain surveys
    typically land around `H ~ 0.7-0.8`, though this varies a lot by
    landscape type -- rugged, glaciated, or heavily eroded terrain (fjord
    coastlines, mountain ranges) measures rougher (lower `H`) than that
    average.

    Applies a 2D Hann window before the FFT (most heightfields passed here
    aren't periodic, so this avoids edge-discontinuity artifacts leaking
    into the spectrum), then radially bins the power spectrum by spatial
    frequency `k` and fits a line in log-log space over `fit_range` (as a
    fraction of the Nyquist frequency -- skips the noisy lowest bins and
    the highest ones, which are dominated by grid/Nyquist artifacts rather
    than the field's genuine fractal character).

    `field` must be square. Returns `(beta, H)`. The fit is sensitive to
    `fit_range` and to how small/non-square-cropped `field` is -- treat the
    result as an estimate to compare *between* fields generated the same
    way, not a precise absolute number from a single run.
    """
    n = field.shape[0]
    win1d = np.hanning(n)
    win2d = np.outer(win1d, win1d)
    windowed = (field - field.mean()) * win2d

    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(spectrum) ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    kx, ky = np.meshgrid(freqs, freqs)
    k = np.sqrt(kx**2 + ky**2)

    k_bins = np.linspace(0, k.max(), n // 2)
    bin_idx = np.digitize(k.ravel(), k_bins)
    radial_power = np.array(
        [
            power.ravel()[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
            for i in range(1, len(k_bins))
        ]
    )
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2

    lo, hi = fit_range
    mask = (k_centers >= lo) & (k_centers <= hi) & np.isfinite(radial_power) & (radial_power > 0)
    slope, _intercept = np.polyfit(np.log(k_centers[mask]), np.log(radial_power[mask]), 1)
    beta = -slope
    return beta, (beta - 2) / 2
