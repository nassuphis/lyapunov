"""
1D map templates (type="step1d" or "step1d_x0").

These maps have a single driven parameter via A/B forcing.
"""

from .defaults import DEFAULT_TRANS, DEFAULT_ITER

MAPS_STEP1D: dict[str, dict] = {

    "logistic": dict(  # Classic logistic
        expr="r * x * (1.0 - x)",
        domain=[2.5, 2.5, 4.0, 4.0],  # A0, B0, A1, B1
        pardict=dict(r="forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "sine": dict(  # Sine map (classical Lyapunov variant: r sin(pi x))
        expr="r * sin(pi * x)",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(r="forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "tent": dict(  # Tent map
        expr="r*x*(1-step(x-0.5)) + r*(1-x)*step(x-0.5)",
        domain=[0.0, 0.0, 2.0, 2.0],
        pardict=dict(r="forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "heart": dict(  # Heart-cell map: x_{n+1} = sin(alpha x_n) + r_n
        expr="sin(a * x) + r",
        domain=[0.0, 0.0, 15.0, 15.0],
        pardict=dict(
            r="forced",
            a=1.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "cardiac1d": dict(
        # Guevara cardiac map, eq. (8.4), AB-forcing in r.
        expr=(
            "a"
            " - b*exp(-(r - Mod(x + d, r) + d)/19.6)"
            " - c*exp(-(r - Mod(x + d, r) + d)/200.5)"
        ),
        deriv_expr=(
            "-(b/19.6)*exp(-(r - Mod(x + d, r) + d)/19.6)"
            " - (c/200.5)*exp(-(r - Mod(x + d, r) + d)/200.5)"
        ),
        domain=[50, 50, 150, 150],
        pardict=dict(
            r="forced",
            a=270.0,
            b=2441.0,
            c=90.02,
            d=53.5,
        ),
        x0=5.0,
        trans=100,
        iter=200,
    ),

    "eq827": dict(  # Equation (8.27) – Angelini antiferromagnetic element
        expr=(
            "(-r/3.0) * exp(b*(x + 1.0/3.0)) * step(-1.0/3.0 - x) + "
            "( r/3.0) * exp(b*(1.0/3.0 - x)) * step(x - 1.0/3.0) + "
            "r*x * (1.0 - step(-1.0/3.0 - x) - step(x - 1.0/3.0))"
        ),
        deriv_expr=(
            "(-r/3.0) * b * exp(b*(x + 1.0/3.0)) * step(-1.0/3.0 - x) + "
            "(-r/3.0) * b * exp(b*(1.0/3.0 - x)) * step(x - 1.0/3.0) + "
            "r * (1.0 - step(-1.0/3.0 - x) - step(x - 1.0/3.0))"
        ),
        domain=[-15.0, -4.3, -15.0, 11.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=1.0,
        trans=100,
        iter=200,
    ),

    "nn1": dict(
        expr="pow(r/abs(x),a)*sign(x)+cos(2*pi*r/2)*sin(2*pi*x/5)",
        domain=[0.1, 0.1, 10, 10],
        pardict=dict(
            r="forced",
            a=0.25,
        ),
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn1a": dict(
        expr="pow(r/abs(x),a)*sign(x)+pow(abs(cos(2*pi*r/2)*sin(2*pi*x/5)),b)",
        domain=[0.1, 0.1, 10, 10],
        pardict=dict(
            r="forced",
            a=0.25,
            b=1.0,
        ),
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn2": dict(
        expr="pow(r/abs(x),a*cos(r))*sign(x)+cos(2*pi*r/2.25)*sin(2*pi*x/3)",
        domain=[0.0, 0.0, 15.0, 15.0],
        pardict=dict(
            r="forced",
            a=1.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn3": dict(
        expr="cos(2*pi*r*x*(1-x)/a)*pow(abs(x),cos(2*pi*(r+x)/10))*sign(x)-cos(2*pi*r)*step(x)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=25.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn4": dict(
        expr="pow(abs(x*x*x-r),cos(r))*sign(x)+pow(abs(r*r*r-x*x*x),sin(x))*sign(r)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn5": dict(
        expr="pow(abs(x*x*x-pow(r,a)),cos(r))*sign(x)+pow(abs(r*r*r-pow(x,b)),sin(x))*sign(r)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=3.0,
            b=5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn6": dict(
        expr="pow(abs(pow(x,b)-pow(r,a)),cos(r))*sign(x)+pow(abs(pow(r,a)-pow(x,b)),sin(x))*sign(r)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=3.0,
            b=5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn7": dict(
        expr="pow(abs(apow(x,b)-apow(r,a)),cos(r))*sign(x)+pow(abs(apow(r,a)-apow(x,b)),sin(x))*sign(r)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=3.0,
            b=5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn8": dict(
        expr="apow(apow(x,b)-apow(r,a),cos(r))+apow(apow(r,a)-apow(x,b),sin(x))",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=3.0,
            b=5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn9": dict(
        expr="r*cosh(sin(apow(x,x)))-x*sinh(cos(apow(r,r)))",
        expr1="r*apow(sin(pi*(x-b)),a)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=3.0,
            b=5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn10": dict(
        expr="c*apow(a*cos(2*pi*r*x*x),b*sin(2*pi*r*x))+d",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=1.0,
            b=1.0,
            c="x*(1-x)",
            d=0.0
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn11": dict(
        expr="a*(b+c)",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=1.0,
            b=1.0,
            c="x*(1-x)",
            d=0.0
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn12": dict(
        expr="term1+term2",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            a=0.0,
            b=0.0,
            c=0.0,
            d=1.0,
            e=0.0,
            f=0.0,
            v="cos(x-r)",
            term1="a*pow(v,5)+b*pow(v,4)+c*pow(v,3)+d*pow(v,2)+e*v+f",
            term2="x*(1-x)*cos(x)*cos(r)*sin(exp(x))",
        ),
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "nn13": dict(
        expr="final",
        deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            c8=1.0,
            c7=1.0,
            c6=1.0,
            c5=1.0,
            c4=1.0,
            c3=1.0,
            c2=1.0,
            c1=1.0,
            c0=1.0,
            v="cos(x-r)",
            poly="c8*v**8+c7*v**7+c6*v**6+c5*v**5+c4*v**4+c3*v**3+c2*v**2+c1*v+c0",
            final="exp(cos(poly)*sin(poly))",
        ),
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "nn14": dict(
        expr="np.real(m)",
        deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0],
        pardict=dict(
            r="forced",
            l="r*x*(1-x)",
            m="lgamma(l)*j1(l)*j0(l)*sin(l)*cos(l)*np.exp(x+1j*l)",
        ),
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq86": dict(
        expr="x + r*pow(abs(x),b)*sin(x)",
        domain=[2, 2, 2.75, 2.75],
        pardict=dict(
            r="forced",
            b=0.3334,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq826": dict(
        expr="x * exp((r/(1+x))-b)",
        domain=[10, 10, 40, 40],
        pardict=dict(
            r="forced",
            b=11.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq95": dict(
        expr=" (1-r*x*x)*step(x)+(a-r*x*x)*(1-step(x))",
        domain=[-0.5, -0.5, 5, 5],
        pardict=dict(
            r="forced",
            a=2.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq96": dict(
        expr=" r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+(a-1)*(r-2)/4)*(1-step(x-0.5))",
        domain=[2.5, 2.5, 4, 4],
        pardict=dict(
            r="forced",
            a=0.4,
        ),
        x0=0.6,
        trans=100,
        iter=300,
    ),

    "dlog": dict(
        expr="dlog",
        deriv_expr="r * (1.0 - 2.0 * (dlog))",
        domain=[2.5, 2.5, 4, 4],
        pardict=dict(
            r="forced",
            a=0.4,
            dlog="r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+0.25*(a-1)*(r-2))*(1-step(x-0.5))",
        ),
        x0=0.6,
        trans=100,
        iter=300,
    ),

    "eq97": dict(
        expr=" a*x*(1-step(x-1))+b*pow(x,1-r)*step(x-1)",
        domain=[2, 0.5, 10, 1.5],
        pardict=dict(
            r="forced",
            a=50,
            b=50,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq98": dict(
        expr=" 1+r*apow(x,b)-a*apow(x,d)",
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r="forced",
            a=1.0,
            b=1.0,
            d=0.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq932": dict(
        expr=" mod1(r*x)",
        deriv_expr="r",
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r="forced",
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq933": dict(
        expr=" 2*x*step(x)*(1-step(x-0.5))+((4*r-2)*x+(2-3*r))*step(x-0.5)*(1-step(x-1.0))",
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r="forced",
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq937": dict(
        expr="r * x * (1.0 - x) * step(x-0)*(1-step(x-r))+r*step(x-r)+0*(1-step(x))",
        domain=[0.0, 0.0, 5.0, 5.0],
        pardict=dict(
            r="forced",
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq947": dict(
        expr="b*(sin(x+r))**2",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r="forced",
            b=1.7,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq948": dict(
        expr=(
            "b * ( sin(x + pow(r,mu) ) )**2"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        deriv_expr="2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r="forced",
            b=2.0,
            mu=1.0,
            alpha=0.0,
            beta=0.0,
            k=2.0,
            n=1,
            gamma=1.0,
        ),
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq948a": dict(
        expr=(
            "b * mod1(x + pow(r,mu) )"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        deriv_expr="2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r="forced",
            b=2.0,
            mu=1.0,
            alpha=0.0,
            beta=0.0,
            k=2.0,
            n=1,
            gamma=1.0,
        ),
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq950": dict(
        expr="Mod(cosh(r*x), 2/b)",
        deriv_expr="r * sinh(r*x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=100,
        iter=300,
    ),

    "eq951": dict(
        expr_common=dict(
            S="sin(1-x)",
            C="cos((x-r)**2)",
        ),
        expr="b * r * exp({S}**3 * {C}) - 1",
        deriv_expr=(
            "b * r * exp({S}**3 * {C}) * ("
            " -3*{S}**2 * cos(1-x) * {C}"
            " -2*(x-r) * {S}**3 * sin((x-r)**2)"
            ")"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=100,
        iter=100,
    ),

    "eq952": dict(
        expr=(
            "b * sin(pow(x - r, 3)) * exp(-pow(x - r, 2))"
        ),
        deriv_expr=(
            "b * exp(-pow(x - r, 2)) * ("
            "  3*pow(x - r, 2)*cos(pow(x - r, 3))"
            " - 2*(x - r)*sin(pow(x - r, 3))"
            ")"
        ),
        domain=[-4.0, -4.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=3.2,
        ),
        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq953": dict(
        expr="b * pow(sin(x - r), 4)",
        deriv_expr="4 * b * pow(sin(x - r), 3) * cos(x - r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=400,
        iter=400,
    ),

    "eq954": dict(
        expr="cos(x + r) * cos(b - x)",
        deriv_expr=(
            "-sin(x + r) * cos(b - x)"
            " + cos(x + r) * sin(b - x)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=500,
        iter=1000,
    ),

    "eq955": dict(
        expr=(
            "b * pow(x - 1, 2) * pow(sin(r - x), 2)"
        ),
        deriv_expr=(
            "2*b*(x - 1)*pow(sin(r - x), 2)"
            " - 2*b*pow(x - 1, 2)*sin(r - x)*cos(r - x)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.8,
        ),
        x0=0.5,
        trans=100,
        iter=500,
    ),

    "eq959": dict(
        expr=(
            "(b + r) * exp(pow(sin(1 - x), 3) * cos(pow(x - r, 2))) - 1"
        ),
        deriv_expr=(
            "(b + r) * exp(pow(sin(1 - x), 3) * cos(pow(x - r, 2))) * ("
            " -3*pow(sin(1 - x), 2)*cos(1 - x)*cos(pow(x - r, 2))"
            " -2*(x - r)*pow(sin(1 - x), 3)*sin(pow(x - r, 2))"
            ")"
        ),
        domain=[-1.0, -1.0, 1.0, 1.0],
        pardict=dict(
            r="forced",
            b=0.6,
        ),
        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq961": dict(
        expr=(
            "b * cos(exp(-pow(x - r, 2)))"
        ),
        deriv_expr=(
            "2 * b * (x - r) * exp(-pow(x - r, 2)) "
            "* sin(exp(-pow(x - r, 2)))"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=5.0,
        ),
        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq962": dict(
        expr="b * r*r * exp( sin( pow(1 - x, 3) ) ) - 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq963": dict(
        expr="b * exp( pow( sin(1 - x), 3 ) ) + r",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq964": dict(
        expr="r * exp( -pow(x - b, 2) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq965": dict(
        expr="b * exp( sin(r * x) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq966": dict(
        expr="pow( abs(b*b - pow(x - r, 2)), 0.5 ) + 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq967": dict(
        expr="pow( b + pow( sin(r * x), 2 ), -1 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq968": dict(
        expr="b * exp( r * pow( sin(x) + cos(x), -1 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.3,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq969": dict(
        expr="b * (x - r) * exp( -pow(x - r, 3) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq970": dict(
        expr="b * exp( cos(1 - x) * sin(pi/2) + sin(r) )",
        deriv_expr=(
            "b * exp( cos(1 - x) * sin(pi/2) + sin(r) ) * sin(1 - x)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq971": dict(
        expr=(
            "b * r * exp(pow(sin(x - r), 4))"
        ),
        deriv_expr=(
            "4 * b * r * exp(pow(sin(x - r), 4))"
            " * pow(sin(x - r), 3) * cos(x - r)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq972": dict(
        expr=(
            "b * r * exp(pow(sin(1 - x), 3))"
        ),
        deriv_expr=(
            "-3 * b * r * exp(pow(sin(1 - x), 3))"
            " * pow(sin(1 - x), 2) * cos(1 - x)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=100,
        iter=300,
    ),

    "eq973": dict(
        expr=(
            "b * r * pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2)"
        ),
        deriv_expr=(
            "2*pow(b, 2)*r*("
            " sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")"
        ),
        domain=[-2.5, -2.5, 2.5, 2.5],
        pardict=dict(
            r="forced",
            b=1.1,
        ),
        x0=0.5,
        trans=125,
        iter=250,
    ),

    "eq974": dict(
        expr="pow( abs(r*r - pow(x - b, 2)), 0.5 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq975": dict(
        expr="b*cos(x-r)*sin(x+r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq976": dict(
        expr="(x-r)*sin( pow(x-b,2))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq977": dict(
        expr="r*sin(pi*r)*sin(pi*x)*step(x-0.5)+b*r*sin(pi*r)*sin(pi*x)*step(0.5-x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq978": dict(
        expr=(
            "r * sin(pi*r) * sin(pi*(x - b))"
        ),
        deriv_expr=(
            "r * sin(pi*r) * pi * cos(pi*(x - b))"
        ),
        domain=[0.0, 0.0, 2.0, 2.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=500,
        iter=1000,
    ),

    "eq981": dict(
        expr=(
            "b * pow(2 + sin(mod1(x) - r), -1)"
        ),
        deriv_expr=(
            "-b * cos(mod1(x) - r) * pow(2 + sin(mod1(x) - r), -2)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq982": dict(
        expr="b*r*exp(exp(exp(x*x*x)))",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(
            r="forced",
            b=0.1,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq983": dict(
        expr="b*r* exp(pow(sin(1-x*x),4))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq984": dict(
        expr="r*(sin(x)+b*sin(9.0*x))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=0.5,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq985": dict(
        expr="b * exp(tan(r*x) - x)",
        deriv_expr=(
            "b * exp(tan(r*x) - x) * (r * pow(sec(r*x), 2) - 1)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq986": dict(
        expr="b*exp(cos(x*x*x*r-b)-r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",
            b=1.0,
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

}
