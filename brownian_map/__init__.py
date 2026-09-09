"""Procedural map generation from 2D Brownian-style random surfaces.

Pipeline: a `surface` (array of floats in [-1, 1]) is generated, then
`normalize_surface` maps it to palette indices, `clean_isolated_points`
removes single-cell noise, and `compute_isocurves` extracts elevation
contour lines. `render` draws the result.

`rivers` is not part of that pipeline — see its module docstring.

    from brownian_map import surface, palette, postprocess, render

    srf = surface.generate_brownian_spark_surface(dim=300, sparks=100)
    norm = palette.normalize_surface(srf)
    clean = postprocess.clean_isolated_points(norm)
    iso = postprocess.compute_isocurves(clean)
    render.render_surface(clean, palette.load_palette(), isocurves=iso, path="map.png")
"""

from . import palette, postprocess, render, rivers, surface

__all__ = ["surface", "palette", "postprocess", "rivers", "render"]
