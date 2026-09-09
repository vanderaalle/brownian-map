"""Demo pipeline: generate a Brownian-spark surface, clean it, extract
isocurves, and render a map.

    python -m brownian_map.cli --dim 300 --sparks 100 --out map.png

Rivers are deliberately not part of this pipeline: hill-climbing over a
static surface caps out at short, unconvincing paths. `rivers.py` is still
there if you want to experiment — see its module docstring.
"""

from __future__ import annotations

import argparse

from . import palette as palette_mod
from . import postprocess
from . import surface as surface_mod
from .render import render_surface


def generate_map(dim: int, sparks: int, step: float, n_bins: int, seed: int | None = None):
    import random

    if seed is not None:
        random.seed(seed)

    raw_surface = surface_mod.generate_brownian_spark_surface(dim=dim, sparks=sparks, step=step)
    rgb_palette = palette_mod.load_palette()

    normalized = palette_mod.normalize_surface(raw_surface, n_bins=n_bins)
    clean = postprocess.clean_isolated_points(normalized)
    isocurves = postprocess.compute_isocurves(clean, n_levels=n_bins + 8)

    return clean, rgb_palette, isocurves


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=300, help="grid side length")
    parser.add_argument("--sparks", type=int, default=100, help="seed points along the spark walk")
    parser.add_argument("--step", type=float, default=0.1, help="per-cell jitter")
    parser.add_argument("--bins", type=int, default=20, help="palette bins (see palette.normalize_surface)")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducible maps")
    parser.add_argument("--out", default="map.png", help="output image path")
    args = parser.parse_args(argv)

    clean, rgb_palette, isocurves = generate_map(
        dim=args.dim,
        sparks=args.sparks,
        step=args.step,
        n_bins=args.bins,
        seed=args.seed,
    )
    render_surface(clean, rgb_palette, path=args.out, isocurves=isocurves)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
