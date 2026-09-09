"""Cleanup and iso-contour extraction for normalized surfaces."""

from __future__ import annotations

from collections import Counter

import numpy as np
from skimage import measure


def most_common(values) -> int:
    return Counter(values).most_common(1)[0][0]


def clean_isolated_points(normalized_surface: np.ndarray) -> np.ndarray:
    """Replace any cell whose value differs from all four of its (wrapping)
    neighbours with the most common neighbour value. Removes single-cell
    speckle noise left by the fill algorithms."""
    dim = normalized_surface.shape[0]
    cleaned = normalized_surface.copy()
    for row in range(dim):
        for col in range(dim):
            neighbors = [
                normalized_surface[(row - 1) % dim, col],
                normalized_surface[(row + 1) % dim, col],
                normalized_surface[row, (col - 1) % dim],
                normalized_surface[row, (col + 1) % dim],
            ]
            if normalized_surface[row, col] not in neighbors:
                cleaned[row, col] = most_common(neighbors)
    return cleaned


def frame_surface(surface: np.ndarray, border_value=0) -> np.ndarray:
    """Pad `surface` with a one-cell border, so `compute_isocurves` closes
    contours that would otherwise run off the edge of the array."""
    return np.pad(surface, 1, constant_values=border_value)


def compute_isocurves(
    normalized_surface: np.ndarray, n_levels: int = 28
) -> list[list[np.ndarray]]:
    """One list of contours per elevation level `0..n_levels-1`.

    Each contour is an `(n, 2)` array of `(row, col)` float coordinates, as
    returned by `skimage.measure.find_contours`.
    """
    return [
        measure.find_contours(normalized_surface, level) for level in range(n_levels)
    ]


def make_bitonal(normalized_surface: np.ndarray, threshold: int, high: int, low: int = 0) -> np.ndarray:
    """Threshold a normalized surface to two values (e.g. land vs sea)."""
    return np.where(normalized_surface >= threshold, high, low)
