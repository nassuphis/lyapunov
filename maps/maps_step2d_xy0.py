"""
2D map templates with per-pixel initial conditions (type="step2d_xy0").

These maps support x0/y0 as 2D arrays (noise, gradient, image, etc.)
rather than scalar initial conditions.
"""

MAPS_STEP2D_XY0: dict[str, dict] = {

    "nn2dxy0": dict(
        type="step2d_xy0",
        expr_x="x",
        expr_y="y+(params[0]-y)*0.01",
        jac_exprs=("0", "0", "0", "0"),
        domain=[-20.0, -20.0, 20.0, 20.0],
        pardict=dict(
            p="first",   # horizontal axis
            t="second",  # vertical axis
        ),
        x0="noise", y0="noise",
        trans=500, iter=500,
        eps_floor=1e-16,
    ),

}
