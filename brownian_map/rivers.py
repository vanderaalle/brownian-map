"""Hill-climbing river tracing over a normalized surface.

A river is a path of `[row, col]` points that starts at a low-elevation
point and repeatedly steps to a random strictly-higher 4-neighbour, until
it reaches `peak_value`, runs out of higher neighbours, or hits `max_steps`.

Not currently exercised in `notebook.ipynb` — hill-climbing over a static
surface (uphill here, or the downhill mountain-to-sea variant below) caps
out at short, unconvincing paths; see the notebook's parameter glossary.

Idea for a better approach, not yet implemented: model a river as a feedback
process rather than a one-shot trace over a frozen surface — as it flows,
it excavates, lowering the elevation values along and near its own path, so
later steps (and later rivers) see a terrain that the river itself has
carved. That's a different algorithm from anything here (closer to
hydraulic erosion simulation than hill-climbing).
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def neighbors4(point, dim: int) -> list[list[int]]:
    row, col = point
    candidates = [[row, col - 1], [row - 1, col], [row + 1, col], [row, col + 1]]
    return [p for p in candidates if 0 <= p[0] < dim and 0 <= p[1] < dim]


def neighbors8(point, dim: int) -> list[list[int]]:
    row, col = point
    candidates = [
        [row - 1, col - 1], [row, col - 1], [row + 1, col - 1],
        [row - 1, col], [row + 1, col],
        [row - 1, col + 1], [row, col + 1], [row + 1, col + 1],
    ]
    return [p for p in candidates if 0 <= p[0] < dim and 0 <= p[1] < dim]


def step_uphill(point, surface: np.ndarray, excluded=()) -> Optional[list[int]]:
    """Randomly pick a higher-elevation 4-neighbour of `point`, or `None` at a peak."""
    dim = surface.shape[0]
    here = surface[point[0], point[1]]
    higher = [
        p
        for p in neighbors4(point, dim)
        if surface[p[0], p[1]] > here and p not in excluded
    ]
    return random.choice(higher) if higher else None


def trace_river(start, surface: np.ndarray, peak_value: int, max_steps: int = 100) -> list[list[int]]:
    """Trace a path uphill from `start` until reaching `peak_value` or a local peak."""
    river = [list(start)]
    current = list(start)
    visited = [list(start)]
    for _ in range(max_steps):
        if surface[current[0], current[1]] >= peak_value:
            break
        next_point = step_uphill(current, surface, visited)
        if next_point is None:
            break
        visited.append(next_point)
        river.append(next_point)
        current = next_point
    return river


def generate_rivers(
    surface: np.ndarray,
    source_value: int,
    peak_value: int,
    n_candidates: int = 100_000,
    min_length: int = 10,
) -> list[list[list[int]]]:
    """Trace rivers uphill from `n_candidates` random points that sit at
    `source_value` elevation, keeping only those longer than `min_length`."""
    dim = surface.shape[0]
    rivers = []
    for _ in range(n_candidates):
        p = [random.randint(0, dim - 1), random.randint(0, dim - 1)]
        if surface[p[0], p[1]] == source_value:
            river = trace_river(p, surface, peak_value)
            if len(river) > min_length:
                rivers.append(river)
    return rivers


def step_downhill(point, surface: np.ndarray, excluded=()) -> Optional[list[int]]:
    """Randomly pick a 4-neighbour of `point` that is no higher (lower, or
    tied and unvisited — flat ground doesn't stop a river), or `None` if
    none qualify."""
    dim = surface.shape[0]
    here = surface[point[0], point[1]]
    lower_or_flat = [
        p
        for p in neighbors4(point, dim)
        if surface[p[0], p[1]] <= here and p not in excluded
    ]
    return random.choice(lower_or_flat) if lower_or_flat else None


def trace_river_downhill(start, surface: np.ndarray, sea_level: float, max_steps: int = 2000) -> list[list[int]]:
    """Trace a path downhill from a mountain `start` until reaching
    `sea_level` or a local basin (no lower neighbour)."""
    river = [list(start)]
    current = list(start)
    visited = [list(start)]
    for _ in range(max_steps):
        if surface[current[0], current[1]] <= sea_level:
            break
        next_point = step_downhill(current, surface, visited)
        if next_point is None:
            break
        visited.append(next_point)
        river.append(next_point)
        current = next_point
    return river


def generate_rivers_downhill(
    surface: np.ndarray,
    mountain_threshold: float,
    sea_level: float,
    n_candidates: int = 100_000,
    min_length: int = 10,
    max_steps: int = 2000,
) -> list[list[list[int]]]:
    """Trace rivers downhill from `n_candidates` random mountain points at or
    above `mountain_threshold` elevation down to `sea_level`, keeping only
    those longer than `min_length`.

    Unlike `generate_rivers`'s exact `source_value` match, this uses a `>=`
    threshold: on a continuous (pre-quantization) surface, exact float
    equality would almost never hit, and threshold-matching also picks
    candidates on a broader, more varied set of ridgelines than a single
    palette bin would.
    """
    dim = surface.shape[0]
    rivers = []
    for _ in range(n_candidates):
        p = [random.randint(0, dim - 1), random.randint(0, dim - 1)]
        if surface[p[0], p[1]] >= mountain_threshold:
            river = trace_river_downhill(p, surface, sea_level, max_steps=max_steps)
            if len(river) > min_length:
                rivers.append(river)
    return rivers
