"""
2D AB-forced map templates (type="step2d_ab").

These maps have a single driven parameter via A/B forcing, but operate in 2D phase space.
"""

from .defaults import DEFAULT_TRANS, DEFAULT_ITER

MAPS_STEP2D_AB: dict[str, dict] = {

    "predprey": dict(
        type="step2d_ab",
        expr_x="abs_cap(r * x * (1.0 - x - y),1e6)",
        expr_y="abs_cap(b * x * y,1e6)",
        jac_exprs=(
            "r * (1.0 - 2.0 * x - y)",  # dXdx
            "-r * x",                    # dXdy
            "b * y",                     # dYdx
            "b * x",                     # dYdy
        ),
        domain=[-0.04, -0.6, 4.5, 6.6],
        pardict=dict(
            r="forced",
            b=3.569985,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
        eps_floor=1e-16,
    ),

    "kicked": dict(
        type="step2d_ab",
        expr_x=(
            "mod1("
            "x + s*(1 + ((1-exp(-b))/b)*y) "
            "+ a*s*((1-exp(-b))/b)*cos(2*pi*x)"
            ")"
        ),
        expr_y="exp(-b)*(y + a*cos(2*pi*x))",
        jac_exprs=(
            "1 - a*s*((1-exp(-b))/b)*2*pi*sin(2*pi*x)",  # dXdx
            "s*((1-exp(-b))/b)",                          # dXdy
            "-exp(-b)*a*2*pi*sin(2*pi*x)",                # dYdx
            "exp(-b)",                                    # dYdy
        ),
        domain=[-2.45, -6.35, 1.85744, 1.4325],
        pardict=dict(
            s="forced",
            a=0.3,
            b=3.0,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
    ),

    "henzi": dict(  # henon-lozi map
        type="step2d_ab",
        expr_x="1-s*abs(x)+y",
        expr_y="a*x",
        domain=[-10, -10, 10, 10],
        pardict=dict(
            s="forced",
            a=0.994,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
    ),

    "adbash": dict(  # adams-bashworth
        type="step2d_ab",
        expr_x="x+(h/2)*(3*r*x*(1-x)-r*y*(1-y))",
        expr_y="x",
        domain=[-5, -5, 5, 5],
        pardict=dict(
            r="forced",
            h=1.0,
        ),
        x0=0.5,
        y0=0.5,
        trans=200,
        iter=200,
    ),

    "eqn941_ab": dict(  # Eqs. (9.41, 9.42)
        type="step2d_ab",
        expr_x="Mod( x + 2*pi*k + b*sin(x) + r*cos(y) , 2*pi )",
        expr_y="Mod( y + 2*pi*omega, 2*pi )",
        jac_exprs=(
            "+1 + b*cos(x)",  # dXdx
            "-r * sin(y)",    # dXdy
            "0",              # dYdx
            "1",              # dYdy
        ),
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="forced",
            b=1.075,
            k=0.28,
            omega="(pow(5,0.5)-1)/2",
        ),
        x0=0.1,
        y0=0.5,
        trans=100,
        iter=300,
    ),

    "econ882": dict(  # Equations (8.19, 8.20), interdependent economies
        type="step2d_ab",
        expr_x="mu * x * (1.0 - x) + gamma * y",
        expr_y="mu * y * (1.0 - y) + gamma * x",
        jac_exprs=(
            "mu * (1.0 - 2.0 * x)",  # dXdx
            "gamma",                  # dXdy
            "gamma",                  # dYdx
            "mu * (1.0 - 2.0 * y)",  # dYdy
        ),
        domain=[2.0, 2.40625, 2.24375, 2.65, 2.40625, 2.0],
        pardict=dict(
            mu="forced",
            gamma=0.43,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=300,
        eps_floor=1e-16,
    ),

    "eq979_ab": dict(
        type="step2d_ab",
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="0",
        jac_exprs=(
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",
            "0",
            "0",
        ),
        domain=[0.0, 0.0, 4.3, 7.66],
        pardict=dict(
            r="forced",
            b=1,
        ),
        x0=1.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq980_ab": dict(
        type="step2d_ab",
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - 1"
        ),
        expr_y="y",
        jac_exprs=(
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",
            "0",
            "1",
        ),
        domain=[0.26, 1.36, 1.44, 3.85],
        pardict=dict(
            r="forced",
            b=0.9,
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

}
