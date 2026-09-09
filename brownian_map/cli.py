"""Demo pipeline: generate a Brownian-spark surface, clean it, extract
isocurves and rivers, and render a map.

    python -m brownian_map.cli --dim 300 --sparks 100 --out map.png

Note on rivers: `rivers.trace_river` only steps to strictly-higher
neighbours, so on jittery terrain (small `--step`, many bins) monotonic
uphill runs rarely chain past 5-6 cells. Lower `--min-length` or increase
`--step` for more/longer rivers; this is a property of hill-climbing on
noisy terrain, not a bug.
"""

from __future__ import annotations

import argparse

from . import palette as palette_mod
from . import postprocess, rivers as rivers_mod
from . import surface as surface_mod
from .render import render_surface


def generate_map(
    dim: int, sparks: int, step: float, n_bins: int, n_rivers: int, min_river_length: int, seed: int | None = None
):
    import random

    if seed is not None:
        random.seed(seed)

    raw_surface = surface_mod.generate_brownian_spark_surface(dim=dim, sparks=sparks, step=step)
    rgb_palette = palette_mod.load_palette()

    normalized = palette_mod.normalize_surface(raw_surface, n_bins=n_bins)
    clean = postprocess.clean_isolated_points(normalized)
    isocurves = postprocess.compute_isocurves(clean, n_levels=n_bins + 8)

    river_paths = rivers_mod.generate_rivers(
        clean,
        source_value=n_bins // 2 - 1,
        peak_value=n_bins + 7,
        n_candidates=n_rivers,
        min_length=min_river_length,
    )

    return clean, rgb_palette, isocurves, river_paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=300, help="grid side length")
    parser.add_argument("--sparks", type=int, default=100, help="seed points along the spark walk")
    parser.add_argument("--step", type=float, default=0.1, help="per-cell jitter")
    parser.add_argument("--bins", type=int, default=20, help="palette bins (see palette.normalize_surface)")
    parser.add_argument("--rivers", type=int, default=20_000, help="candidate points to try tracing rivers from")
    parser.add_argument("--min-length", type=int, default=3, help="shortest traced river to keep")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducible maps")
    parser.add_argument("--out", default="map.png", help="output image path")
    args = parser.parse_args(argv)

    clean, rgb_palette, isocurves, river_paths = generate_map(
        dim=args.dim,
        sparks=args.sparks,
        step=args.step,
        n_bins=args.bins,
        n_rivers=args.rivers,
        min_river_length=args.min_length,
        seed=args.seed,
    )
    render_surface(clean, rgb_palette, path=args.out, isocurves=isocurves, rivers=river_paths)
    print(f"wrote {args.out} ({len(river_paths)} rivers)")


if __name__ == "__main__":
    main()
