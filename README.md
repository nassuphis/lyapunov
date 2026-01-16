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

Macros in `macros6.txt` define reusable templates:

```
@LOGISTIC=map:logistic:AB:2:4:2:4
@COLORS=rgb:mh:1.0:FF0000:000000:00FF00
@MYSPEC=@LOGISTIC,@COLORS,iter:1000
```

Template expansion with Python expressions:
```
@BATCH=map:logistic:AB:#{2+slot*0.1}:4:2:4
```

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
