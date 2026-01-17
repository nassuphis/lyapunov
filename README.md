# lyapunov

A computational art framework for rendering Lyapunov fractal visualizations from chaotic dynamical systems.

## Overview

This project generates fractal images by computing Lyapunov exponents across 2D parameter spaces. The Lyapunov exponent measures sensitivity to initial conditions in dynamical systems - regions of chaos vs. stability create intricate visual patterns.

The pipeline:
```
Spec String → Macro Expansion → Map Config → Field Computation (Numba) → Colormapping → Image Output
```

## Installation

Requires Python 3.13+

```bash
uv sync
```

Dependencies on local packages:
- `specparser` - spec string parsing and macro expansion
- `rasterizer` - image output with metadata embedding

## Usage

```bash
# Render a single fractal
python lyapunov_cli.py "map:logistic:AB:2:4:2:4,iter:1000,rgb:mh:1.0:FF0000:000000:00FF00"

# Render with macros
python lyapunov_cli.py "@MYSPEC" --macros macros6.txt

# Batch rendering
python lyapunov_cli.py "@BATCH_TEMPLATE" --slots 100 --outdir tst
```

## Project Structure

### Core Modules

| File | Purpose |
|------|---------|
| `lyapunov_cli.py` | Main CLI entry point, spec parsing, batch processing |
| `maps.py` | 30+ dynamical system definitions (logistic, Henon, cardiac, etc.) |
| `fields.py` | Numba-accelerated Lyapunov field computation |
| `functions.py` | 100+ mathematical functions (Bessel, wavelets, special functions) |
| `field_color.py` | RGB colormapping schemes (Markus-Hess, HSV, palettes) |
| `affine.py` | Parameter space transformations and domain mapping |

### Image Processing

| File | Purpose |
|------|---------|
| `autolevels.py` | Percentile-based auto-levels enhancement |
| `equalize.py` | Histogram equalization and CLAHE |
| `stats.py` | Per-channel histogram analysis |
| `wavelet.py` | Wavelet functions (Mexican hat, Morlet) |

### Quality Analysis ("Boredom" Detection)

| File | Purpose |
|------|---------|
| `boring_spectrum.py` | Spectral complexity analysis |
| `boring_shapes.py` | Morphological shape analysis |
| `boring_edges.py` | Edge-based complexity scoring |
| `boring_helpers.py` | Shared utilities for boredom metrics |
| `delete_boring.py` | Automated filtering of uninteresting outputs |
| `dedupe_top.py` | Similarity-based deduplication |
| `rank_boring.py` | Scoring and ranking by interest metrics |

### AI Integration (Google Gemini)

| File | Purpose |
|------|---------|
| `ocr.py` | Extract spec strings from rendered image footers |
| `img2expr2.py` | Style transfer and artistic variants |
| `img_evaluate.py` | Aesthetic complexity scoring |

### Utilities

| File | Purpose |
|------|---------|
| `spec.py` | Spec string utilities |
| `make_specs.py` | Batch spec generation |
| `make_manifest.py` | Output manifest creation |
| `repaletter.py` | Color palette manipulation |

## Spec Format

Human-readable parameter strings define renders:

```
map:logistic:AB:2:4:2:4,x0:0.5,iter:1000,trans:100,rgb:mh:1.0:FF0000:000000:00FF00
```

Components:
- `map:<name>:<forcing>:<domain>` - dynamical system and parameter bounds
- `x0`, `y0` - initial conditions
- `iter`, `trans` - iteration counts (total, transient)
- `rgb:<scheme>:<params>` - colormapping configuration
- `eps` - epsilon floor for Lyapunov computation

### Forcing Patterns

For 1D maps, the forcing pattern (e.g., `AB`, `AABB`, `ABAB`) determines how parameters alternate during iteration.

### Colormapping Schemes

- `mh` - Markus-Hess linear (positive/zero/negative colors with gamma)
- `mh_eq` - Histogram-equalized Markus-Hess
- `hsveq` - HSV interpolation with equalization
- `palette` - Tri-palette colorization

## Macro System

Macros define reusable spec fragments in a macro file (e.g., `macros.txt`):

```
@MYSPEC=@LOGISTIC,@COLORS,iter:1000
@LOGISTIC=map:logistic:AB:2:4:2:4
@COLORS=rgb:mh:1.0:FF0000:000000:00FF00
```

### Single-Pass Rolling Substitution

The macro processor uses **single-pass substitution** that iterates through macros in file order, replacing each macro key with its value in the **current result string**. This creates a "rolling" effect where earlier macro substitutions can introduce new macro references that get expanded by later iterations.

**Example macro file:**
```
@RUN=@CONFIG
@CONFIG=map:@MAP,color:@RGB
@MAP=logistic
@RGB=mh
```

**Expansion of `@RUN`:**
```
Start:     @RUN
→ @RUN:    @CONFIG                     (replaced @RUN)
→ @CONFIG: map:@MAP,color:@RGB         (replaced @CONFIG in result)
→ @MAP:    map:logistic,color:@RGB     (replaced @MAP in result)
→ @RGB:    map:logistic,color:mh       (replaced @RGB in result)
```

**Key insight:** Macros defined **earlier** in the file are processed **first**. When `@RUN` is replaced with `@CONFIG`, the result now contains `@CONFIG`. Since `@CONFIG` comes later in the iteration, it will be replaced in the same pass.

**Important ordering rule:** If macro `@A` references `@B`, then `@A` must be defined **before** `@B`:
```
# CORRECT - @SPEC defined before its dependencies
@SPEC=map:@MAP,rgb:@RGB
@MAP=logistic
@RGB=mh

# WRONG - @MAP and @RGB already processed when @SPEC is reached
@MAP=logistic
@RGB=mh
@SPEC=map:@MAP,rgb:@RGB   # Won't expand! @MAP/@RGB already passed
```

**Real-world pattern (from macros6.txt):**
```
@RUN0=@LYAP0                              # Entry point first
@LYAP0=slot:{1:@SLOTS0},@MAP14,@RGB0      # References later macros
@SLOTS0=10                                 # Leaf values defined last
@MAP14=map:nn14:AB:-40:-40:40:40
@RGB0=rgb:pfm:@PAL0:1.2
@PAL0=#{r2line("tri_warm.txt")}
```

Entry points are first, dependencies are defined afterward (deepest dependencies last).

### Expansion Syntax

Within macro values, you can use:

- `#{expr}` - Python expression evaluated at render time (e.g., `#{row}`, `#{2+slot*0.1}`)
- `${expr}` - Python expression for list expansion (e.g., `${[1,2,3]}` expands to three specs)
- `{a:b}` - Numeric range (e.g., `{2:4}` expands to `2,3,4`)
- `{a:b|N}` - Linspace (e.g., `{0:1|5}` expands to `0,0.25,0.5,0.75,1`)
- `[a,b,c]` - Choice list (cartesian product with other dimensions)

### Random Functions

Macros support randomization via `#{...}`:
- `#{choose("a","b","c")}` - Random choice from list
- `#{rfloat3(0.1,2.0)}` - Random float with 3 decimal places
- `#{rline("file.txt")}` - Random line from file

## Supported Maps

### 1D Maps
- `logistic` - Classic logistic map
- `sine` - Sine map
- `tent` - Tent map
- `kicked` - Periodically forced map
- `nn14` - Neural network polynomial

### 2D Maps
- `henon` - Henon map
- `cardiac` - Heart cell dynamics
- `predprey` - Lotka-Volterra predator-prey
- `degn` - Biochemical oscillator
- `fishery` - Population dynamics
- And 20+ others

## Output

Rendered images include:
- Footer with spec string for reproducibility
- EXIF/XMP metadata embedding
- Batch outputs in numbered directories (`tst_00001.jpg`, etc.)

## Development

The codebase uses:
- **Numba** for JIT compilation and parallel computation
- **SymPy** for symbolic math and automatic differentiation
- **pyvips** for fast image I/O
- **SciPy** for special functions and FFT
