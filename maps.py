# ---------------------------------------------------------------------------
# Map templates: add / tweak here to define all maps
# ---------------------------------------------------------------------------

# type "step1d"    : 1 parameter and 1d derivative (no jacobian!)
# type "step2d_ab" : 1 parameter and 2d derivative (with jacobian)
# type "step2d"    : 2 parameters and 2d derivative (with jacobian)

# 1-parameter maps are converted into 2d by forcing
# 2-parameter maps need no forcing

# parameters are static or scanned (scan: means scan over the domain)
# scanned parameters are changed by the field calculators
# scanned parameters always come first
# depending on the field calculator, either the first
# or both the second and the first parameters are
# scanned


# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAP_NAME = "logistic"
DEFAULT_SEQ      = "AB"
DEFAULT_TRANS    = 200
DEFAULT_ITER     = 1000
DEFAULT_X0       = 0.5
DEFAULT_EPS_LYAP = 1e-12
DEFAULT_CLIP     = None     # auto from data
DEFAULT_GAMMA    = 1.0      # colormap gamma

MAP_TEMPLATES: dict[str, dict] = {

    "cardiac": dict(
        type   = "step2d",
        domain=[20.0, 0.0, 140.0, 150.0],
        pardict=dict(
            tmin   = "second",
            r_p    = "first",
            A      = 270.0,
            B1     = 2441,
            B2     = 90.02, 
            tau1   = 19.6,
            tau2   = 200.5,
            r_eff  = "max(r_p,1e-12)",
            k      = "math.floor((tmin + x) / r_eff) + 1.0",
            t      = "k * r_eff - x",
            F1     = "abs_cap(-t/tau1,50)",
            e1     = "exp(F1)",
            F2     = "abs_cap(-t/tau2,50)", 
            e2     = "exp(F2)",
        ),
        expr_x = "A - B1 * e1 - B2 * e2",
        expr_y = "y",
        jac_exprs=(
            "-(B1 / tau1) * e1 - (B2 / tau2) * e2",
            "0.0",
            "0.0",
            "0.0"
        ),
        x0=5.0,       # initial x_n (duration)
        y0=0.0,       # dummy y
        trans=100,    # n_prev
        iter=200,     # n_max
        eps_floor=1e-16,
    ),
    
    "predprey": dict(
        type="step2d_ab",
        expr_x="abs_cap(r * x * (1.0 - x - y),1e6)",
        expr_y="abs_cap(b * x * y,1e6)",
        jac_exprs=(
            "r * (1.0 - 2.0 * x - y)", # dXdx
            "-r * x", # dXdy
            "b * y",  # dYdx
            "b * x",  # dYdy
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

    "parasite": dict(
        # parasite, k:3.1, rgb:mh:0.25:red:black:yellow
        type="step2d",
        expr_x="abs_cap(x * exp( abs_cap( r * (1.0 - x/K) - a*y, 50.0 ) ),1e6)",
        expr_y="abs_cap(x * (1.0 - exp( abs_cap( -a*y, 50.0 ) ) ),1e6)",
        jac_exprs=(
        "exp( abs_cap( r * (1.0 - x/K) - a*y, 50.0 ) ) * (1.0 - r * x / K)",
        "-a * x * exp( abs_cap( r * (1.0 - x/K) - a*y, 50.0 ) )",
        "1.0 - exp( abs_cap( -a*y, 50.0 ) )",
        "x * a * exp( abs_cap( -a*y, 50.0 ) )",
        ),
        domain=[-0.1, -0.1, 4.0, 7.0],
        pardict=dict(
            r="first",  
            a="second", 
            K=3.1,
        ),
        x0=0.5,
        y0=0.5,
        trans=200,
        iter=800,
        eps_floor=1e-16,
    ),

    "kicked": dict(
        # kicked:BA, ll:-3.215:-2.45, ul:-6.35:0.9325, lr:1.4325:1.85744, a:0.3, b:3
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
            "s*((1-exp(-b))/b)",                         # dXdy
            "-exp(-b)*a*2*pi*sin(2*pi*x)",               # dYdx
            "exp(-b)",                                   # dYdy
        ),
        domain=[-2.45, -6.35, 1.85744, 1.4325],
        pardict=dict(
            s="forced",
            a=0.3,
            b=3.0,
        ),
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 200,
    ),

    "henzi": dict( #henon-lozi map
        # fig 9.11 :  henzi:BA,ll:1.483:2.35,ul:2.15:1.794,lr:-0.35:0.15,gamma:0.25
        type="step2d_ab",
        # FIXME: add a default "seq" key
        # seq="AB" or somthing
        expr_x="1-s*abs(x)+y",
        expr_y="a*x",
        domain=[-10,-10,10,10],
        pardict=dict(
            s="forced",
            a=0.994,
        ),
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 200,
    ),

    "adbash": dict( #adams-bashworth
        type="step2d_ab",
        expr_x="x+(h/2)*(3*r*x*(1-x)-r*y*(1-y))",
        expr_y="x",
        domain=[-5,-5,5,5],
        pardict=dict(
            r  = "forced",
            h  = 1.0,
        ),
        x0 = 0.5,
        y0 = 0.5,
        trans = 200,
        iter = 200,
    ),

    "degn_nn": dict( #NN's degn's map
        type="step2d",
        expr_x="s*(x-0.5)+0.5+a*sin(2*pi*r*y) ",
        expr_y="y+s*(x-0.5)+0.5+a*sin(2*pi*r*y)*mod1(b/s)",
        domain=[-2,-5,2,5],
        pardict=dict(
            r  = "first",
            s  = "second",
            a  = 0.1,
            b  = 1e8,
        ),
        x0    = 0.4,
        y0    = 0.4,
        trans = 100,
        iter  = 300,
    ),

    "degn": dict(  # Degn's map, Eqs. (9.20, 9.21)
        type="step2d",
        expr_common=dict(
            x1="c*(x - 0.5) + 0.5 + rho*sin(2*pi*r*y)"
        ),
        expr_x="{x1}",
        expr_y="Mod(y + ({x1}), k/b)",
        jac_exprs=(
            "c",                                   # dXdx
            "2*pi*rho*r*cos(2*pi*r*y)",            # dXdy
            "c",                                   # dYdx
            "1 + 2*pi*rho*r*cos(2*pi*r*y)",        # dYdy
        ),
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="first",    # vertical axis: r
            b="second",   # horizontal axis: b
            rho=0.1,      # fixed R
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
        # Jacobian, treating 'mod' as identity for derivatives
        jac_exprs=(
            "+1 + b*cos(x)",  # dXdx
            "-r * sin(y)",    # dXdy
            "0",              # dYdx
            "1",              # dYdy
        ),
        # Default rectangle matching the caption (b, r)
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="first",     # vertical axis: r
            b="second",    # horizontal axis: b
            k=0.28,        # fixed R
            omega="(pow(5,0.5)-1)/2",
        ),
        x0=0.1,
        y0=0.5,
        trans=100,
        iter=300,
    ),

    "eqn941_ab": dict(  # Eqs. (9.41, 9.42)
        type="step2d_ab",
        expr_x="Mod( x + 2*pi*k + b*sin(x) + r*cos(y) , 2*pi )",
        expr_y="Mod( y + 2*pi*omega, 2*pi )",
        # Jacobian, treating 'mod' as identity for derivatives
        jac_exprs=(
            "+1 + b*cos(x)",  # dXdx
            "-r * sin(y)",    # dXdy
            "0",              # dYdx
            "1",              # dYdy
        ),
        # Default rectangle matching the caption (b, r)
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="forced", # vertical axis: r
            b=1.075,    # horizontal axis: b
            k=0.28,        # fixed R
            omega="(pow(5,0.5)-1)/2",
        ),
        x0=0.1,
        y0=0.5,
        trans=100,
        iter=300,
    ),

    "henon": dict(
        type="step2d",
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="1 + y - r * x * x",  # Henon-like
        expr_y="s * x",
        domain=[1.0, 0.1, 1.4, 0.3],  # r0,s0,r1,s1
        pardict=dict(
            r  = "second",
            s  = "first",
        ),
        x0=0.1,
        y0=0.1,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
        # optionally: manual Jacobian override
        # jac_exprs=(" -2*r*x", "1", "s", "0"),
    ),

    "henon2": dict(
        type="step2d",
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="r - x*x + s*y",  # Henon-like
        expr_y="a*x+b*x*x",
        pardict=dict(
            r  = "second",
            s  = "first",
            a=1.0,
            b= 0.0,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=2000,
        # optionally: manual Jacobian override
        # jac_exprs=(" -2*r*x", "1", "s", "0"),
    ),

    "kst2d": dict(
        type="step2d",
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="1+s*apow(x-c,r)-a*pow(abs(x),b)",  
        expr_y="0",
        domain=[1,1.33,3.5,2.5],  # r0,s0,r1,s1
        pardict=dict(
            r  = "second",
            s  = "first",
            a  = 3.0,
            b  = 2.0,
            c  = 0.0, 
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=500,
        # optionally: manual Jacobian override
    ),

    "logistic2d": dict(
        type="step2d",
        # text is "r vs a"
        # s is the LHS of the text 
        # r is the RHS of the text
        expr_x="(1-s*x*x)*step(x)+(r-s*x*x)*(1-step(x))",  # Henon-like
        expr_y="y",
        domain=[0.66,-0.05,3,1.66],  # r0,s0,r1,s1
        pardict=dict(
            s  = "first",
            r  = "second",
        ),
        x0 = 0.5,
        y0 = 1.0,
        trans = 100,
        iter = 300,
        # optionally: manual Jacobian override
    ),

    "logistic": dict( # Classic logistic
        expr="r * x * (1.0 - x)",
        domain=[2.5, 2.5, 4.0, 4.0],  # A0, B0, A1, B1
        pardict=dict(r  = "forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "sine": dict( # Sine map (classical Lyapunov variant: r sin(pi x))
        expr="r * sin(pi * x)",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(r  = "forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "tent": dict(  # Tent map
        expr="r*x*(1-step(x-0.5)) + r*(1-x)*step(x-0.5)",
        domain=[0.0, 0.0, 2.0, 2.0],
        pardict=dict(r  = "forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    
    "heart": dict( # Heart-cell map: x_{n+1} = sin(alpha x_n) + r_n
        expr="sin(a * x) + r",
        domain=[0.0, 0.0, 15.0, 15.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
        ),    
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "cardiac1d": dict(
        # Guevara cardiac map, eq. (8.4), AB‑forcing in r.
        #
        # Parameters (via a,b,c,d):
        #   a = A       (default 270)
        #   b = B1      (default 2441)
        #   c = B2      (default 90.02)
        #   d = t_min   (default 53.5)
        #
        # We use tau1 = 19.6, tau2 = 200.5 as in the caption.
        # t_n is encoded as:
        #   t_n = r - Mod(x + d, r) + d
        # so that k_n r - x_n > t_min with minimal integer k_n.
        expr=(
            "a"
            " - b*exp(-(r - Mod(x + d, r) + d)/19.6)"
            " - c*exp(-(r - Mod(x + d, r) + d)/200.5)"
        ),

        # Manual derivative d x_{n+1} / d x_n:
        #   = -(B1/tau1) * exp(-t/tau1) - (B2/tau2) * exp(-t/tau2)
        deriv_expr=(
            "-(b/19.6)*exp(-(r - Mod(x + d, r) + d)/19.6)"
            " - (c/200.5)*exp(-(r - Mod(x + d, r) + d)/200.5)"
        ),

        # Default AB‑plane window in r_A, r_B (you'll probably override).
        domain=[50,50,150,150],

        # [A, B1, B2, t_min] defaults from the text + t_min = 53.5
        pardict=dict(
            r  = "forced",
            a  = 270.0,
            b  = 2441.0,
            c  = 90.02, 
            d  = 53.5,
        ),

        x0=5.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "econ882": dict(  # Equations (8.19, 8.20), interdependent economies
        # x_{n+1} = μ x_n (1 - x_n) + γ y_n
        # y_{n+1} = μ y_n (1 - y_n) + γ x_n
        type="step2d_ab",
        expr_x="mu * x * (1.0 - x) + gamma * y",
        expr_y="mu * y * (1.0 - y) + gamma * x",
        jac_exprs=(
            "mu * (1.0 - 2.0 * x)",  # dXdx
            "gamma",                 # dXdy
            "gamma",                 # dYdx
            "mu * (1.0 - 2.0 * y)",  # dYdy
        ),
        # B vs A plane: (A,B) = (mu_A, mu_B)
        # LL:(2, 2.40625), UL:(2.24375, 2.65), LR:(2.40625, 2)
        domain=[2.0, 2.40625, 2.24375, 2.65, 2.40625, 2.0],
        pardict=dict(
            mu="forced",     # A/B-forced parameter μ
            gamma=0.43,      # γ₁ = γ₂ = 0.43
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=300,
        eps_floor=1e-16,
    ),

    "fishery": dict(  # Equations (8.23, 8.24)
        type="step2d",

        # x = D (crab biomass), y = P (pots)
        expr_x="x * exp(ad + bd*x + cd*y)",
        expr_y="y * exp(ap + bp*y + cp*x)",

        # Jacobian:
        # let g1 = aD + bD x + cD y; f1 = x e^{g1}
        # dXdx = e^{g1} + x e^{g1} bD = e^{g1} * (1 + bD x)
        # dXdy = x e^{g1} cD
        # let g2 = aP + bP y + cP x; f2 = y e^{g2}
        # dYdx = y e^{g2} cP
        # dYdy = e^{g2} + y e^{g2} bP = e^{g2} * (1 + bP y)
        jac_exprs=(
            "exp(ad + bd*x + cd*y) * (1 + bd*x)",       # dXdx
            "x * cd * exp(ad + bd*x + cd*y)",           # dXdy
            "y * cp * exp(ap + bp*y + cp*x)",           # dYdx
            "exp(ap + bp*y + cp*x) * (1 + bp*y)",       # dYdy
        ),

        # c_P vs c_D plane: (first, second) = (cD, cP)
        # LL:(-0.033, 0), UL:(-0.033, 0.105), LR:(0, 0)
        # → domain = [A0,B0,A1,B1] = [ -0.033, 0, 0, 0.105 ]
        domain=[-0.033, 0.0, 0.0, 0.105],

        pardict=dict(
            cd="first",   # horizontal axis
            cp="second",  # vertical axis
            ad=1.0,
            bd=-0.005,
            ap=0.5,
            bp=-0.04,
        ),

        x0=0.5,
        y0=0.5,
        trans=5,    # n_prev
        iter=20,    # n_max
        eps_floor=1e-16,
    ),

    "eq827": dict(  # Equation (8.27) – Angelini antiferromagnetic element
        # piecewise:
        # region1: x < -1/3
        # region2: -1/3 <= x <= 1/3
        # region3: x > 1/3
        expr=(
            # region1: x < -1/3  ->  step(-1/3 - x)
            "(-r/3.0) * exp(b*(x + 1.0/3.0)) * step(-1.0/3.0 - x) + "
            # region3: x > 1/3   ->  step(x - 1/3)
            "( r/3.0) * exp(b*(1.0/3.0 - x)) * step(x - 1.0/3.0) + "
            # region2: middle, complement of the two above
            "r*x * (1.0 - step(-1.0/3.0 - x) - step(x - 1.0/3.0))"
        ),

        # derivative df/dx (same partition, ignoring derivative of step discontinuities)
        deriv_expr=(
            # d/dx of region1 part
            "(-r/3.0) * b * exp(b*(x + 1.0/3.0)) * step(-1.0/3.0 - x) + "
            # d/dx of region3 part
            "(-r/3.0) * b * exp(b*(1.0/3.0 - x)) * step(x - 1.0/3.0) + "
            # d/dx of region2: r
            "r * (1.0 - step(-1.0/3.0 - x) - step(x - 1.0/3.0))"
        ),

        # Fig. 8.17 caption:
        # "Equation (8.27): b = 1. B versus A. r:B5AA B5AA..., nprev = 100, nmax = 200, x0 = 1.
        #  D-shading. LL:(−15, −4.3), UL:(−15, 11.), LR:(35, −4.3)"
        domain=[-15.0, -4.3, -15.0, 11.0],  # will be overridden by ll/ul/lr in spec

        pardict=dict(
            r="forced",  # A/B-forced parameter
            b=1.0,       # fixed b = 1 for this figure
        ),

        x0=1.0,
        trans=100,
        iter=200,
    ),

    "nn1": dict(
        expr="pow(r/abs(x),a)*sign(x)+cos(2*pi*r/2)*sin(2*pi*x/5)",
        #deriv_expr="0",
        domain=[0.1, 0.1, 10, 10], 
        pardict=dict(
            r  = "forced",
            a  = 0.25,
        ),
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn1a": dict(
        expr="pow(r/abs(x),a)*sign(x)+pow(abs(cos(2*pi*r/2)*sin(2*pi*x/5)),b)",
        #deriv_expr="0",
        domain=[0.1, 0.1, 10, 10],  
        pardict=dict(
            r  = "forced",
            a  = 0.25,
            b  = 1.0,
        ),
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn2": dict(
        expr="pow(r/abs(x),a*cos(r))*sign(x)+cos(2*pi*r/2.25)*sin(2*pi*x/3)",
        #deriv_expr="0",
        domain=[0.0, 0.0, 15.0, 15.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
        ),   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn3": dict(
        expr="cos(2*pi*r*x*(1-x)/a)*pow(abs(x),cos(2*pi*(r+x)/10))*sign(x)-cos(2*pi*r)*step(x)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 25.0,
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn4": dict(
        expr="pow(abs(x*x*x-r),cos(r))*sign(x)+pow(abs(r*r*r-x*x*x),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
        ),   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn5": dict(
        expr="pow(abs(x*x*x-pow(r,a)),cos(r))*sign(x)+pow(abs(r*r*r-pow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ), 
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn6": dict(
        expr="pow(abs(pow(x,b)-pow(r,a)),cos(r))*sign(x)+pow(abs(pow(r,a)-pow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn7": dict(
        expr="pow(abs(apow(x,b)-apow(r,a)),cos(r))*sign(x)+pow(abs(apow(r,a)-apow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn8": dict(
        expr="apow(apow(x,b)-apow(r,a),cos(r))+apow(apow(r,a)-apow(x,b),sin(x))",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ), 
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn9": dict(
        expr="r*cosh(sin(apow(x,x)))-x*sinh(cos(apow(r,r)))",
        expr1="r*apow(sin(pi*(x-b)),a)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

     "nn10": dict(
        expr="c*apow(a*cos(2*pi*r*x*x),b*sin(2*pi*r*x))+d",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
            b  = 1.0,
            c  = "x*(1-x)",
            d = 0.0
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

# python lyapunov1.py --spec \
# 'map:nn11:AB:-10:-10:10:10,
# c:x*(1-x)*cos(x)*cos(r)*sin(exp(x)),
# b:1*(cos(x-r))**2+0*cos(x-r),
# x0:0.5,trans:200,iter:200,
# rgb:mh_eq:1:seagreen:black:copper,
# hist:5:5:128' \           
# --pix 2000 --out tst7.png

    "nn11": dict(
        expr="a*(b+c)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
            b  = 1.0,
            c  = "x*(1-x)",
            d = 0.0
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

    # a:cos(2*pi*{0.1:0.4:4}),b:sin(2*pi*${1}*${1}),c:0.2*cos(2*pi*exp(${1})),d:{0.2:0.8:4},e:-0.5*${2},f:0
     "nn12": dict(
        expr="term1+term2",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a = 0.0, # polynomial coefficients
            b = 0.0,
            c = 0.0,
            d = 1.0,
            e = 0.0,
            f = 0.0,
            v  = "cos(x-r)",
            term1  = "a*pow(v,5)+b*pow(v,4)+c*pow(v,3)+d*pow(v,2)+e*v+f",
            term2  = "x*(1-x)*cos(x)*cos(r)*sin(exp(x))",
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
            r  = "forced",
            c8 = 1.0,
            c7 = 1.0, # polynomial coefficients
            c6 = 1.0, 
            c5 = 1.0, 
            c4 = 1.0,
            c3 = 1.0,
            c2 = 1.0,
            c1 = 1.0,
            c0 = 1.0,
            v  = "cos(x-r)",
            poly  = "c8*v**8+c7*v**7+c6*v**6+c5*v**5+c4*v**4+c3*v**3+c2*v**2+c1*v+c0",
            final = "exp(cos(poly)*sin(poly))",
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
            r  = "forced",
            l  = "r*x*(1-x)",
            m = "lgamma(l)*j1(l)*j0(l)*sin(l)*cos(l)*np.exp(x+1j*l)",
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq86": dict(
        expr="x + r*pow(abs(x),b)*sin(x)",
        # A8B8
        domain=[2,2,2.75,2.75],
        pardict=dict(
            r  = "forced",
            b  = 0.3334,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq826": dict(
        expr="x * exp((r/(1+x))-b)",
        # A8B8
        domain=[10,10,40, 40],
        pardict=dict(
            r  = "forced",
            b  = 11.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq95": dict(
        expr=" (1-r*x*x)*step(x)+(a-r*x*x)*(1-step(x))",
        # A8B8
        domain=[-0.5,-0.5,5,5],
        pardict=dict(
            r  = "forced",
            a  = 2.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq96": dict(
        expr=" r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+(a-1)*(r-2)/4)*(1-step(x-0.5))",
        # A8B8
        domain=[2.5,2.5,4, 4],
        pardict=dict(
            r  = "forced",
            a  = 0.4,
        ),  
        x0=0.6,
        trans=100,
        iter=300,
    ),

    "dlog": dict( # same as eq96, but manual derivative to check sympy's derivation
        # Map step = eq. (9.6)
        expr="dlog",
        deriv_expr="r * (1.0 - 2.0 * (dlog))",
        domain=[2.5,2.5,4, 4],
        pardict=dict(
            r  = "forced",
            a  = 0.4,
            dlog = "r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+0.25*(a-1)*(r-2))*(1-step(x-0.5))",
        ),  
        x0=0.6,
        trans=100,
        iter=300,
    ),


    "eq97": dict(
        expr=" a*x*(1-step(x-1))+b*pow(x,1-r)*step(x-1)",
        # A8B8
        domain=[2,0.5,10,1.5],
        pardict=dict(
            r  = "forced",
            a  = 50,
            b  = 50,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq98": dict(
        expr=" 1+r*apow(x,b)-a*apow(x,d)",
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r  = "forced",
            a  = 1.0,
            b  = 1.0,
            d  = 0.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq932": dict(
        expr=" mod1(r*x)",
        deriv_expr="r",      # <--- add this
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r  = "forced",
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq933": dict(
        expr=" 2*x*step(x)*(1-step(x-0.5))+((4*r-2)*x+(2-3*r))*step(x-0.5)*(1-step(x-1.0))",
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r  = "forced",
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq937": dict(
        expr="r * x * (1.0 - x) * step(x-0)*(1-step(x-r))+r*step(x-r)+0*(1-step(x))",
        domain=[0.0, 0.0, 5.0, 5.0],
        pardict=dict(
            r  = "forced",
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq947": dict(
        expr="b*(sin(x+r))**2",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "forced",
            b  = 1.7,
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
        deriv_expr = "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "forced",
            b  = 2.0,
            mu = 1.0,
            alpha  = 0.0,
            beta  = 0.0, 
            k  = 2.0,
            n = 1,            
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
        deriv_expr = "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "forced",
            b  = 2.0,
            mu = 1.0,
            alpha  = 0.0,
            beta  = 0.0, 
            k  = 2.0,
            n = 1,            
            gamma=1.0,
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),


    "eq948_2d": dict(
        type="step2d",
        expr_x=(
            "b * ( sin(x + pow(r,mu) ) )**2"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        expr_y = "y",
        jac_exprs=(
            "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",  # dXdx
            "pow(r,k)*step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))", # dXdy
            "0", # dYdx
            "1", # dYdy
        ),
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "first",
            b  = 2.0,
            mu = 1.0,
            alpha  = "second",
            beta  = 0.0, 
            k  = 2.0,
            n = 1,            
            gamma=1.0,
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq950": dict(
        # x_{n+1} = [cosh(r x_n)] mod (2/b)
        expr="Mod(cosh(r*x), 2/b)",

        # derivative wrt x, ignoring the outer Mod
        deriv_expr="r * sinh(r*x)",

        # default A/B window for the forced parameter r
        # (adjust from the spec to match the book's figure)
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r  = "forced",  # A/B sequence drives r
            b  = 1.0,       # override with b:... in the spec
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
        # x_{n+1} = b r exp(S^3 C) - 1
        expr="b * r * exp({S}**3 * {C}) - 1",

        # derivative f'(x) = b r exp(S^3 C) * ( -3 S^2 cos(1-x) C
        #                                     -2 (x-r) S^3 sin((x-r)^2) )
        deriv_expr=(
            "b * r * exp({S}**3 * {C}) * ("
            " -3*{S}**2 * cos(1-x) * {C}"
            " -2*(x-r) * {S}**3 * sin((x-r)**2)"
            ")"
        ),

        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced", # A/B‑forced parameter
            b  = 1.0,      # default b, override with b:... in the spec if needed
        ),
        x0=0.5,
        trans=100,
        iter=100,
    ),

    "eq952": dict(
        # x_{n+1} = b sin[(x_n - r)^3] e^{-(x_n - r)^2}
        expr=(
            "b * sin(pow(x - r, 3)) * exp(-pow(x - r, 2))"
        ),

        # derivative wrt x, ignoring any forcing / AB structure
        deriv_expr=(
            "b * exp(-pow(x - r, 2)) * ("
            "  3*pow(x - r, 2)*cos(pow(x - r, 3))"
            " - 2*(x - r)*sin(pow(x - r, 3))"
            ")"
        ),

        # choose a default (A,B) window for r_A,r_B;
        # tweak in the spec to match the book's figure
        domain=[-4.0, -4.0, 4.0, 4.0],

        pardict=dict(
            r  = "forced",   # driven parameter (A/B sequence)
            b  = 3.2,        # amplitude; override with b:...
        ),

        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq953": dict(
        # x_{n+1} = b sin^4(x_n - r)
        expr="b * pow(sin(x - r), 4)",

        # derivative wrt x (ignore any forcing structure)
        deriv_expr="4 * b * pow(sin(x - r), 3) * cos(x - r)",

        # default (A,B) window for r_A,r_B; tweak in spec if needed
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # driven parameter (A/B sequence)
            b=1.0,        # amplitude; override with b:... in spec
        ),

        x0=0.5,
        trans=400,
        iter=400,
    ),

    "eq954": dict(
        # x_{n+1} = cos(x_n + r) cos(1 - x_n)
        expr="cos(x + r) * cos(b - x)",

        # derivative wrt x
        deriv_expr=(
            "-sin(x + r) * cos(b - x)"
            " + cos(x + r) * sin(b - x)"
        ),

        # default (A,B) window for r_A, r_B – tune from spec as needed
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # r is the driven A/B parameter
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
            r="forced",   # A/B sequence drives 'r'
            b=0.8,        # amplitude parameter (override with b:…)
        ),
        x0=0.5,
        trans=100,
        iter=500,
    ),

    "eq959": dict(  # Eq. (9.59)
        # x_{n+1} = (b + r) * exp(sin(1 - x)^3 * cos((x - r)^2)) - 1
        expr=(
            "(b + r) * exp(pow(sin(1 - x), 3) * cos(pow(x - r, 2))) - 1"
        ),

        # derivative wrt x
        deriv_expr=(
            "(b + r) * exp(pow(sin(1 - x), 3) * cos(pow(x - r, 2))) * ("
            " -3*pow(sin(1 - x), 2)*cos(1 - x)*cos(pow(x - r, 2))"
            " -2*(x - r)*pow(sin(1 - x), 3)*sin(pow(x - r, 2))"
            ")"
        ),

        # default A/B window (you’ll override via ll/ul/lr for this figure)
        domain=[-1.0, -1.0, 1.0, 1.0],

        pardict=dict(
            r="forced",   # r is A/B-forced
            b=0.6,        # for Fig. 9.152
        ),

        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq961": dict(  # Eq. (9.61)
        expr=(
            "b * cos(exp(-pow(x - r, 2)))"
        ),
        deriv_expr=(
            "2 * b * (x - r) * exp(-pow(x - r, 2)) "
            "* sin(exp(-pow(x - r, 2)))"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",   # driven parameter (A/B sequence)
            b=5.0,        # amplitude; override with b:...
        ),
        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq962": dict(
        expr="b * r*r * exp( sin( pow(1 - x, 3) ) ) - 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq963": dict(
        expr="b * exp( pow( sin(1 - x), 3 ) ) + r",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq964": dict(
        expr="r * exp( -pow(x - b, 2) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq965": dict(
        expr="b * exp( sin(r * x) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq966": dict(
        expr="pow( abs(b*b - pow(x - r, 2)), 0.5 ) + 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq967": dict(
        expr="pow( b + pow( sin(r * x), 2 ), -1 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq968": dict(
        expr="b * exp( r * pow( sin(x) + cos(x), -1 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.3,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq969": dict(
        expr="b * (x - r) * exp( -pow(x - r, 3) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq970": dict(
        # x_{n+1} = b * exp(cos(1 - x) * sin(pi/2) + sin(r))
        expr="b * exp( cos(1 - x) * sin(pi/2) + sin(r) )",

        # derivative wrt x:
        # f'(x) = b * exp(cos(1-x) + sin(r)) * sin(1-x)
        deriv_expr=(
            "b * exp( cos(1 - x) * sin(pi/2) + sin(r) ) * sin(1 - x)"
        ),

        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B-driven
            b=1.0,        # override to 1.5 for this fig
        ),

        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq971": dict(  # Eq. (9.71)
        # x_{n+1} = b * r * exp( sin(x - r)^4 )
        expr=(
            "b * r * exp(pow(sin(x - r), 4))"
        ),

        # derivative wrt x:
        # let s = sin(x - r);  f(x) = b r e^{s^4}
        # f'(x) = b r e^{s^4} * 4 s^3 cos(x - r)
        deriv_expr=(
            "4 * b * r * exp(pow(sin(x - r), 4))"
            " * pow(sin(x - r), 3) * cos(x - r)"
        ),

        # default window – overridden by ll/ul/lr in specs
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B-driven parameter r
            b=0.5,        # placeholder; will be overridden to 1.5
        ),

        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq972": dict(  # Eq. (9.72)
        # x_{n+1} = b * r * exp( sin(1 - x)^3 )
        expr=(
            "b * r * exp(pow(sin(1 - x), 3))"
        ),

        # derivative wrt x
        deriv_expr=(
            "-3 * b * r * exp(pow(sin(1 - x), 3))"
            " * pow(sin(1 - x), 2) * cos(1 - x)"
        ),

        # default A/B window for r_A, r_B (tune from spec to match the plate)
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # driven parameter (A/B sequence)
            b=1.0,        # amplitude; override with b:... in spec
        ),

        x0=0.5,
        trans=100,
        iter=300,
    ),

    "eq973": dict(  # Eq. (9.73)
        # x_{n+1} = b * r * sin^2(b x + r^2) * cos^2(b x - r^2)
        expr=(
            "b * r * pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2)"
        ),

        # derivative wrt x
        deriv_expr=(
            "2*pow(b, 2)*r*("
            " sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")"
        ),

        # default (A,B) window for (r_A,r_B); you'll override via ll/ul/lr
        domain=[-2.5, -2.5, 2.5, 2.5],

        pardict=dict(
            r="forced",   # A/B-driven parameter r
            b=1.1,        # for Fig. 9.141; override with b:... if desired
        ),

        x0=0.5,
        trans=125,
        iter=250,
    ),

    "eq974": dict(
        expr="pow( abs(r*r - pow(x - b, 2)), 0.5 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq975": dict(
        expr="b*cos(x-r)*sin(x+r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq976": dict(
        expr="(x-r)*sin( pow(x-b,2))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq977": dict(
        expr="r*sin(pi*r)*sin(pi*x)*step(x-0.5)+b*r*sin(pi*r)*sin(pi*x)*step(0.5-x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq978": dict(  # Eq. (9.78)
        # x_{n+1} = r * sin(pi*r) * sin(pi*(x - b))
        expr=(
            "r * sin(pi*r) * sin(pi*(x - b))"
        ),

        # derivative wrt x (r treated as parameter)
        deriv_expr=(
            "r * sin(pi*r) * pi * cos(pi*(x - b))"
        ),

        # A/B window: A,B ∈ [0,2] (LL:(0,0), UL:(0,2), LR:(2,0))
        domain=[0.0, 0.0, 2.0, 2.0],

        pardict=dict(
            r="forced",   # A/B-driven parameter
            b=0.5,        # fixed b for Fig. 9.147 (override with b:... if you like)
        ),

        x0=0.5,
        trans=500,
        iter=1000,
    ),

    "eq979_old": dict(
        type="step2d",

        # (x, y) -> (x', y'), parameters (first, second) = (b, r)
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="y",   # dummy y-dimension

        # Jacobian: only dXdx is non-zero
        jac_exprs=(
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)"
            "  *pow(cos(b*x - r*r - r), 2)"
            " - pow(sin(b*x + r*r), 2)"
            "  *sin(b*x - r*r - r)*cos(b*x - r*r - r)"
            ")",  # dXdx
            "0",   # dXdy
            "0",   # dYdx
            "1",   # dYdy
        ),

        # (b,r) window matching the caption: LL:(0,0), UL:(0,7.66), LR:(4.3,0)
        # i.e. b in [0,4.3], r in [0,7.66]
        domain=[0.0, 0.0, 4.3, 7.66],

        pardict=dict(
            b="first",   # horizontal axis: b
            r="second",  # vertical axis: r
        ),

        x0=0.5,
        y0=0.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "eq979": dict(
        type="step2d",

        # x_{n+1} = b*r*sin^2(b*x + r^2)*cos^2(b*x - r^2) - r
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="0",  # dummy y-dimension

        jac_exprs=(
            # dXdx
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "0",  # dYdy
        ),

        # (b,r) window as before (from caption)
        # LL:(0,0), UL:(0,7.66), LR:(4.3,0)
        domain=[0.0, 0.0, 4.3, 7.66],

        pardict=dict(
            b="first",   # horizontal axis
            r="second",  # vertical axis
        ),

        x0=0.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq979_ab": dict(
        type="step2d_ab",

        # x_{n+1} = b*r*sin^2(b*x + r^2)*cos^2(b*x - r^2) - r
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="0",  # dummy y-dimension

        jac_exprs=(
            # dXdx
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "0",  # dYdy
        ),

        # (b,r) window as before (from caption)
        # LL:(0,0), UL:(0,7.66), LR:(4.3,0)
        domain=[0.0, 0.0, 4.3, 7.66],

        pardict=dict(
            r="forced",  # vertical axis
            b=1,   # horizontal axis
            
        ),

        x0=1.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq980": dict(
        type="step2d",

        # (x, y) -> (x', y'), parameters (first, second) = (r, b)
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - 1"
        ),
        expr_y="0",   # dummy y-dimension

        # Jacobian: only dXdx is non-zero; y-dimension is dead
        jac_exprs=(
            "2*pow(b, 2)*r*("  # dXdx
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "0",  # dYdy
        ),

        # default rectangle for (r,b); you’ll override via ll/ul/lr
        domain=[0.26, 1.36, 1.44, 3.85],

        pardict=dict(
            r="first",   # horizontal axis: r
            b="second",  # vertical axis: b
        ),

        x0=0.5,
        y0=0.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "eq980_ab": dict(
        type="step2d_ab",

        # (x, y) -> (x', y'), parameters (first, second) = (r, b)
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - 1"
        ),
        expr_y="y",   # dummy y-dimension

        # Jacobian: only dXdx is non-zero; y-dimension is dead
        jac_exprs=(
            "2*pow(b, 2)*r*("  # dXdx
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "1",  # dYdy
        ),

        # default rectangle for (r,b); you’ll override via ll/ul/lr
        domain=[0.26, 1.36, 1.44, 3.85],

        pardict=dict(
            r="forced",   # horizontal axis: r
            b=0.9,  # vertical axis: b
        ),

        x0=0.5,
        y0=0.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),



    "eq981": dict(
        # x_{n+1} = b / ( 2 + sin( (x mod 1) - r ) )
        expr=(
            "b * pow(2 + sin(mod1(x) - r), -1)"
        ),

        # derivative wrt x, treating mod1 as locally identity:
        # f'(x) = -b * cos((x mod 1) - r) / (2 + sin((x mod 1) - r))^2
        deriv_expr=(
            "-b * cos(mod1(x) - r) * pow(2 + sin(mod1(x) - r), -2)"
        ),

        # default A/B window (you'll override per-figure)
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B sequence drives r
            b=1.0,        # will override to b=2 for Fig. 9.153
        ),

        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq982": dict(
        expr="b*r*exp(exp(exp(x*x*x)))",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(
            r  = "forced",
            b  = 0.1,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq983": dict(
        expr="b*r* exp(pow(sin(1-x*x),4))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
  
    "eq984": dict(
        expr="r*(sin(x)+b*sin(9.0*x))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq985": dict(
        # x_{n+1} = b * exp(tan(r*x) - x)
        expr="b * exp(tan(r*x) - x)",

        # derivative wrt x:
        # f'(x) = b * exp(tan(r*x) - x) * (r * sec^2(r*x) - 1)
        deriv_expr=(
            "b * exp(tan(r*x) - x) * (r * pow(sec(r*x), 2) - 1)"
        ),

        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B-forced parameter
            b=1.0,        # override via b:...
        ),

        x0=0.5,
        trans=100,      # caption uses n_prev = 100
        iter=200,       # caption uses n_max = 200
    ),

    "eq986": dict(
        expr="b*exp(cos(x*x*x*r-b)-r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

}


