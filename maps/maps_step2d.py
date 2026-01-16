"""
2D map templates (type="step2d").

These maps have two independent parameters (first, second) scanned over the domain.
No A/B forcing - both parameters vary independently across the image.
"""

from .defaults import DEFAULT_TRANS, DEFAULT_ITER

MAPS_STEP2D: dict[str, dict] = {

    "nn2d": dict(
        type="step2d",
        expr_x="t*x*(1-x)+p*y*(1-y)",
        expr_y="p*x*(1-x)+t*y*(1-y)",
        jac_exprs=("0", "0", "0", "0"),
        domain=[-20.0, -20.0, 20.0, 20.0],
        pardict=dict(
            p="first",   # horizontal axis
            t="second",  # vertical axis
        ),
        x0=0.5, y0=0.5,
        trans=5, iter=20,
        eps_floor=1e-16,
    ),

    "cardiac": dict(
        type="step2d",
        domain=[20.0, 0.0, 140.0, 150.0],
        pardict=dict(
            tmin="second",
            r_p="first",
            A=270.0,
            B1=2441,
            B2=90.02,
            tau1=19.6,
            tau2=200.5,
            r_eff="max(r_p,1e-12)",
            k="math.floor((tmin + x) / r_eff) + 1.0",
            t="k * r_eff - x",
            F1="abs_cap(-t/tau1,50)",
            e1="exp(F1)",
            F2="abs_cap(-t/tau2,50)",
            e2="exp(F2)",
        ),
        expr_x="A - B1 * e1 - B2 * e2",
        expr_y="y",
        jac_exprs=(
            "-(B1 / tau1) * e1 - (B2 / tau2) * e2",
            "0.0",
            "0.0",
            "0.0"
        ),
        x0=5.0,
        y0=0.0,
        trans=100,
        iter=200,
        eps_floor=1e-16,
    ),

    "parasite": dict(
        type="step2d",
        expr_x="abs_cap(x * exp( abs_cap( r * (1.0 - x/k) - a*y, 50.0 ) ),1e6)",
        expr_y="abs_cap(x * (1.0 - exp( abs_cap( -a*y, 50.0 ) ) ),1e6)",
        jac_exprs=(
            "exp( abs_cap( r * (1.0 - x/k) - a*y, 50.0 ) ) * (1.0 - r * x / k)",
            "-a * x * exp( abs_cap( r * (1.0 - x/k) - a*y, 50.0 ) )",
            "1.0 - exp( abs_cap( -a*y, 50.0 ) )",
            "x * a * exp( abs_cap( -a*y, 50.0 ) )",
        ),
        domain=[-0.1, -0.1, 4.0, 7.0],
        pardict=dict(
            r="first",
            a="second",
            k=3.1,
        ),
        x0=0.5,
        y0=0.5,
        trans=200,
        iter=800,
        eps_floor=1e-16,
    ),

    "degn_nn": dict(  # NN's degn's map
        type="step2d",
        expr_x="s*(x-0.5)+0.5+a*sin(2*pi*r*y) ",
        expr_y="y+s*(x-0.5)+0.5+a*sin(2*pi*r*y)*mod1(b/s)",
        domain=[-2, -5, 2, 5],
        pardict=dict(
            r="first",
            s="second",
            a=0.1,
            b=1e8,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=300,
    ),

    "degn": dict(  # Degn's map, Eqs. (9.20, 9.21)
        type="step2d",
        expr_common=dict(
            x1="c*(x - 0.5) + 0.5 + rho*sin(2*pi*r*y)"
        ),
        expr_x="{x1}",
        expr_y="Mod(y + ({x1}), k/b)",
        jac_exprs=(
            "c",                           # dXdx
            "2*pi*rho*r*cos(2*pi*r*y)",    # dXdy
            "c",                           # dYdx
            "1 + 2*pi*rho*r*cos(2*pi*r*y)", # dYdy
        ),
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="first",
            b="second",
            rho=0.1,
            k=1.0,
            c="b",
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=300,
    ),

    "eqn941": dict(  # Eqs. (9.41, 9.42)
        type="step2d",
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
            r="first",
            b="second",
            k=0.28,
            omega="(pow(5,0.5)-1)/2",
        ),
        x0=0.1,
        y0=0.5,
        trans=100,
        iter=300,
    ),

    "henon": dict(
        type="step2d",
        expr_x="1 + y - r * x * x",
        expr_y="s * x",
        domain=[1.0, 0.1, 1.4, 0.3],
        pardict=dict(
            r="second",
            s="first",
        ),
        x0=0.1,
        y0=0.1,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "henon2": dict(
        type="step2d",
        expr_x="r - x*x + s*y",
        expr_y="a*x+b*x*x",
        pardict=dict(
            r="second",
            s="first",
            a=1.0,
            b=0.0,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=2000,
    ),

    "kst2d": dict(
        type="step2d",
        expr_x="1+s*apow(x-c,r)-a*pow(abs(x),b)",
        expr_y="0",
        domain=[1, 1.33, 3.5, 2.5],
        pardict=dict(
            r="second",
            s="first",
            a=3.0,
            b=2.0,
            c=0.0,
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=500,
    ),

    "logistic2d": dict(
        type="step2d",
        expr_x="(1-s*x*x)*step(x)+(r-s*x*x)*(1-step(x))",
        expr_y="y",
        domain=[0.66, -0.05, 3, 1.66],
        pardict=dict(
            s="first",
            r="second",
        ),
        x0=0.5,
        y0=1.0,
        trans=100,
        iter=300,
    ),

    "fishery": dict(  # Equations (8.23, 8.24)
        type="step2d",
        expr_x="x * exp(ad + bd*x + cd*y)",
        expr_y="y * exp(ap + bp*y + cp*x)",
        jac_exprs=(
            "exp(ad + bd*x + cd*y) * (1 + bd*x)",
            "x * cd * exp(ad + bd*x + cd*y)",
            "y * cp * exp(ap + bp*y + cp*x)",
            "exp(ap + bp*y + cp*x) * (1 + bp*y)",
        ),
        domain=[-0.033, 0.0, 0.0, 0.105],
        pardict=dict(
            cd="first",
            cp="second",
            ad=1.0,
            bd=-0.005,
            ap=0.5,
            bp=-0.04,
        ),
        x0=0.5,
        y0=0.5,
        trans=5,
        iter=20,
        eps_floor=1e-16,
    ),

    "eq948_2d": dict(
        type="step2d",
        expr_x=(
            "b * ( sin(x + pow(r,mu) ) )**2"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        expr_y="y",
        jac_exprs=(
            "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
            "pow(r,k)*step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))",
            "0",
            "1",
        ),
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r="first",
            b=2.0,
            mu=1.0,
            alpha="second",
            beta=0.0,
            k=2.0,
            n=1,
            gamma=1.0,
        ),
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq979_old": dict(
        type="step2d",
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="y",
        jac_exprs=(
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)"
            "  *pow(cos(b*x - r*r - r), 2)"
            " - pow(sin(b*x + r*r), 2)"
            "  *sin(b*x - r*r - r)*cos(b*x - r*r - r)"
            ")",
            "0",
            "0",
            "1",
        ),
        domain=[0.0, 0.0, 4.3, 7.66],
        pardict=dict(
            b="first",
            r="second",
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq979": dict(
        type="step2d",
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
            b="first",
            r="second",
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq980": dict(
        type="step2d",
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - 1"
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
        domain=[0.26, 1.36, 1.44, 3.85],
        pardict=dict(
            r="first",
            b="second",
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

}
