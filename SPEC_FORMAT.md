# Lyapunov Spec Format

A spec string is a comma-separated list of key:value directives that fully describe a Lyapunov fractal render.

## Basic Structure

```
map:<name>:<seq>:<a0>:<b0>:<a1>:<b1>,iter:<n>,rgb:<colormap>,...
```

## Directives

### `map` (required)

Specifies the dynamical system and parameter domain.

```
map:<name>:<sequence>:<a0>:<b0>:<a1>:<b1>
```

| Field | Description |
|-------|-------------|
| `name` | Map name from MAP_TEMPLATES (e.g., `logistic`, `nn2dxy0`) |
| `sequence` | A/B forcing sequence (e.g., `AB`, `AABB`, `ABBA`) |
| `a0, b0` | Lower-left corner of parameter domain |
| `a1, b1` | Upper-right corner of parameter domain |

**Examples:**
```
map:logistic:AB:2:4:2:4
map:henon:AABB:0:1.4:0:0.3
map:nn2dxy0:0:0:1:1
```

For non-forced 2D maps, omit the sequence:
```
map:nn2dxy0:0:0:1:1
```

### `iter`

Number of iterations for Lyapunov exponent calculation.

```
iter:<n>
```

Default: 1000

### `trans`

Number of transient iterations (discarded before measurement).

```
trans:<n>
```

Default: 100

### `x0`, `y0`

Initial conditions. For standard maps, these are scalar values.

```
x0:0.5
y0:0.5
```

For `_x0` or `_xy0` map types, these can be field specifications:

| Spec | Description |
|------|-------------|
| `x0:noise` | Uniform random in [0, 1] |
| `x0:noise:<hi>` | Uniform random in [0, hi] |
| `x0:noise:<lo>:<hi>` | Uniform random in [lo, hi] |
| `x0:grad:x` | Horizontal gradient [0, 1] |
| `x0:grad:y` | Vertical gradient [0, 1] |
| `x0:grad:x:<lo>:<hi>` | Horizontal gradient [lo, hi] |
| `x0:image:<path>` | Load grayscale image |
| `x0:<value>` | Constant value |

### `entropy`

Enable spectral entropy mode instead of Lyapunov exponent.

```
entropy:<sign>
```

| Sign | Effect |
|------|--------|
| `1` | Normal entropy (positive = more complex) |
| `-1` | Inverted entropy |

Additional entropy parameters:
- `k:<n>` - Number of frequency bins (default: 32)
- `w0:<freq>` - Lowest frequency (default: 0.1)
- `w1:<freq>` - Highest frequency (default: π)

### `hist`

Enable histogram-based field computation.

```
hist:<vcalc>:<hcalc>:<bins>
```

| Parameter | Description |
|-----------|-------------|
| `vcalc` | Value transform (0-15) applied to orbit values |
| `hcalc` | Histogram transform (0-15) applied to binned distribution |
| `bins` | Number of histogram bins (default: iter) |

**vcalc transforms:**
| ID | Transform |
|----|-----------|
| 0 | Identity (x) |
| 1 | sin(2πx) |
| 2 | cos(2πx) |
| 3 | Absolute value |
| 4 | Square |
| 5 | Sign |
| 6 | Difference (x[n] - x[n-1]) |
| 7 | Absolute difference |
| 8 | Second difference |
| ... | Additional transforms |

**hcalc transforms:**
| ID | Transform |
|----|-----------|
| 0 | Shannon entropy |
| 1 | Max bin count |
| 2 | Min bin count |
| 3 | Range (max - min) |
| 4 | Variance |
| 5 | Skewness |
| 6 | Kurtosis |
| 7 | Gini coefficient |
| ... | Additional statistics |
| 14 | Last value in orbit |

### `rgb`

Colormap specification.

```
rgb:<colormap>
```

Common colormaps:
- `mh` - Markus-Hess (classic blue/yellow Lyapunov colors)
- `viridis`, `plasma`, `inferno`, `magma`
- `hot`, `cool`, `jet`

### `eps`

Epsilon floor for Lyapunov derivative calculation.

```
eps:<value>
```

Default: 1e-12

## Affine Domain Transforms

The parameter domain can be an arbitrary parallelogram using affine coordinates:

```
ll:<x>:<y>    Lower-left corner
ul:<x>:<y>    Upper-left corner
lr:<x>:<y>    Lower-right corner
```

Or use explicit domain bounds:
```
a0:<val>  b0:<val>  a1:<val>  b1:<val>
```

## Parameter Overrides

Map-specific parameters can be overridden:

```
alpha:<value>
beta:<value>
delta:<value>
epsilon:<value>
```

These are defined per-map in MAP_TEMPLATES `pardict`.

## Examples

**Classic Lyapunov fractal:**
```
map:logistic:AB:2:4:2:4,iter:1000,rgb:mh
```

**2D map with noise initial conditions:**
```
map:nn2dxy0:0:0:1:1,x0:noise,y0:noise,hist:0:0:32,iter:500
```

**Entropy field:**
```
map:logistic:AABB:2:4:2:4,entropy:1,k:64,iter:2000
```

**Histogram with transformed values:**
```
map:henon:AB:0:1.4:0:0.3,hist:6:0:64,iter:1000
```

## Output and Slot System

Results are saved to a directory with a consistent naming scheme:

```
<directory>/<stem>_<slot><suffix>
```

For example, with `--out lyapunov/render.jpg`:
- Directory: `lyapunov/`
- Stem: `render`
- Suffix: `.jpg`

Files are named:
```
lyapunov/render_00001.jpg
lyapunov/render_00002.jpg
lyapunov/render_00003.jpg
...
```

Each image also gets a companion `.spec` file containing the full spec string:
```
lyapunov/render_00001.spec
lyapunov/render_00002.spec
...
```

### Slot Allocation

The `slot` directive is automatically appended to each expanded spec:
```
slot:<n>
```

When a spec expands to multiple renders (via expandspec lists/ranges), each gets a unique slot number. The slot is zero-padded to 5 digits in filenames.

**Single spec:**
```bash
lyapunov_cli.py 'map:logistic:AB:2:4:2:4' --out out/test.jpg
# Creates: out/test_00001.jpg
```

**Expanded spec with range:**
```bash
lyapunov_cli.py 'map:logistic:AB:2:4:2:{3.5,3.7,3.9,4.0}' --out out/sweep.jpg
# Creates: out/sweep_00001.jpg (b1=3.5)
#          out/sweep_00002.jpg (b1=3.7)
#          out/sweep_00003.jpg (b1=3.9)
#          out/sweep_00004.jpg (b1=4.0)
```

The `--overwrite` flag controls whether existing files are replaced or skipped.
