"""2D Brownian-style surface generation for procedural map terrain.

All surfaces are `numpy` arrays of floats in [-1, 1], shape `(dim, dim)`,
indexed `[row, col]`. Two generator families are provided:

- Row-based generators (`generate_row_surface`): each row is derived from
  the previous one via a bounded random walk. Fast, but writing every row
  in the same direction leaves a visible directional drift (see `RowOrder`).
- Spark-propagation generators (`generate_spark_surface`,
  `generate_brownian_spark_surface`): a handful of seed points ("sparks")
  are placed on an otherwise-empty grid, then every remaining cell is
  filled by averaging its already-filled neighbours plus jitter, in random
  order. This has no directional bias and produces basins/ridges radiating
  from the sparks, which reads more like natural terrain.

The grid wraps at the edges (a torus) so both families tile seamlessly.
"""

from __future__ import annotations

import enum
import random
from typing import Optional

import numpy as np


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


# ---------------------------------------------------------------------------
# Row-based generators
# ---------------------------------------------------------------------------


def random_walk_row(length: int, step: float, start: float = 0.0) -> list[float]:
    """A single row: a bounded random walk of `length` values."""
    row = [clamp(start)]
    value = start
    for _ in range(length - 1):
        value = value + random.uniform(-step, step)
        row.append(clamp(value))
    return row


def next_row_forward(previous: list[float], step: float) -> list[float]:
    """Derive a row from `previous`, walking left-to-right."""
    length = len(previous)
    row = [clamp(previous[0] + random.uniform(-step, step))]
    for x in range(length - 1):
        base = (previous[x + 1] + row[x]) * 0.5
        row.append(clamp(base + random.uniform(-step, step)))
    return row


def next_row_backward(previous: list[float], step: float) -> list[float]:
    """Derive a row from `previous`, walking right-to-left."""
    length = len(previous)
    row = [0.0] * length
    row[-1] = clamp(previous[-1] + random.uniform(-step, step))
    for x in range(length - 1):
        base = (previous[-x - 1] + row[-x]) * 0.5
        row[length - x - 2] = clamp(base + random.uniform(-step, step))
    return row


class RowOrder(enum.Enum):
    FORWARD = "forward"      # every row left-to-right: visible drift
    BACKWARD = "backward"    # every row right-to-left: visible drift
    ALTERNATE = "alternate"  # boustrophedon: cancels the drift
    RANDOM = "random"        # random direction per row: noisier than ALTERNATE


def generate_row_surface(
    dim: int, step: float, order: RowOrder = RowOrder.ALTERNATE
) -> np.ndarray:
    """Row-derived surface. See `RowOrder` for the write-direction tradeoffs."""
    rows = [random_walk_row(dim, step)]
    for i in range(dim - 1):
        if order is RowOrder.FORWARD:
            forward = True
        elif order is RowOrder.BACKWARD:
            forward = False
        elif order is RowOrder.ALTERNATE:
            forward = i % 2 == 0
        else:  # RowOrder.RANDOM
            forward = random.choice([True, False])
        rows.append(
            next_row_forward(rows[-1], step)
            if forward
            else next_row_backward(rows[-1], step)
        )
    return np.array(rows)


# ---------------------------------------------------------------------------
# Spark-propagation generators
# ---------------------------------------------------------------------------


def _neighbor_indices(index: int, dim: int, size: int) -> list[int]:
    return [
        (index - dim) % size,
        (index + dim) % size,
        (index + 1) % size,
        (index - 1) % size,
    ]


def _fill_from_seeds(
    dim: int, seed_indices: list[int], seed_value: float, step: float
) -> np.ndarray:
    size = dim * dim
    flat: list[Optional[float]] = [None] * size
    for i in seed_indices:
        flat[i % size] = seed_value

    while None in flat:
        pending = [i for i, v in enumerate(flat) if v is None]
        random.shuffle(pending)  # avoids a left-to-right fill trail
        for i in pending:
            known = [
                flat[n] for n in _neighbor_indices(i, dim, size) if flat[n] is not None
            ]
            if known:
                flat[i] = clamp(sum(known) / len(known) + random.uniform(-step, step))

    return np.array(flat).reshape(dim, dim)


def generate_spark_surface(
    dim: int, sparks: int = 1, step: float = 0.1, seed_value: float = -1.0
) -> np.ndarray:
    """Fill a grid by radiating outward from `sparks` independent random seed points."""
    seed_indices = random.sample(range(dim * dim), sparks)
    return _fill_from_seeds(dim, seed_indices, seed_value, step)


def generate_brownian_spark_surface(
    dim: int, sparks: int = 100, step: float = 0.1, seed_value: float = -1.0
) -> np.ndarray:
    """Fill a grid by radiating outward from a *connected* seed path: a
    single Brownian walk (each step moves one cell down or one cell right,
    wrapping at the edges) lays down `sparks` seed points before the
    neighbour-averaging fill runs. Produces more contiguous basins/coastlines
    than `generate_spark_surface`'s independent seed points.

    On a grid much larger than `sparks`, the seed walk stays confined to one
    small, localized curve, so the whole map ends up as a single basin
    radiating from it — visually a single deep "starburst" centred on
    whichever point is farthest from that curve. See
    `generate_multi_spark_surface` to break that up.
    """
    size = dim * dim
    actual = random.randint(0, size - 1)
    seed_indices = [actual]
    for _ in range(sparks):
        actual = random.choice([actual + dim, actual + 1]) % size
        seed_indices.append(actual)
    return _fill_from_seeds(dim, seed_indices, seed_value, step)


def generate_multi_spark_surface(
    dim: int, n_walks: int = 3, sparks: int = 100, step: float = 0.1, seed_value: float = -1.0
) -> np.ndarray:
    """Combine `n_walks` independent `generate_brownian_spark_surface` walks
    by taking the cell-wise max.

    A single walk is dominated by one basin radiating from its (small,
    localized) seed curve — fine on a grid sized close to `sparks`, but on a
    much bigger grid that basin's radial gradient dominates the whole map.
    Overlapping several walks via max (not average, which just drags
    everything toward the shared `seed_value`) layers multiple basins and
    coastlines on top of each other without erasing any one walk's peaks.
    """
    layers = [
        generate_brownian_spark_surface(dim, sparks=sparks, step=step, seed_value=seed_value)
        for _ in range(n_walks)
    ]
    return np.maximum.reduce(layers)
