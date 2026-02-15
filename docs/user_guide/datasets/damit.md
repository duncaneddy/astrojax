# DAMIT Asteroid Models

The `astrojax.datasets` module provides access to the
[DAMIT](https://damit.cuni.cz/) (Database of Asteroid Models from
Inversion Techniques) dataset.  It handles downloading, caching, and
parsing asteroid 3D shape models, spin parameters, and metadata for
10,000+ asteroids.

## Loading DAMIT Data

### Asteroid Identity Table

The asteroids table maps DAMIT internal IDs to asteroid numbers, names,
and designations:

```python
from astrojax.datasets import load_damit_asteroids

df = load_damit_asteroids()
print(df.shape)
print(df.head())
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | Int64 | DAMIT internal asteroid ID |
| `number` | Int64 | IAU asteroid number |
| `name` | Utf8 | Asteroid name |
| `designation` | Utf8 | Provisional designation |
| `comment` | Utf8 | Additional notes |

### Models & Spin Parameters Table

The models table contains spin parameters and physical properties for
all principal-axis rotator models.  Each asteroid may have multiple
models (mirror solutions, different datasets, etc.):

```python
from astrojax.datasets import load_damit_models

models = load_damit_models()
print(models.columns)
```

Key spin parameter columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Int64 | DAMIT model ID |
| `asteroid_id` | Int64 | DAMIT asteroid ID (foreign key) |
| `lambda` | Float64 | Ecliptic longitude of spin axis [deg] |
| `beta` | Float64 | Ecliptic latitude of spin axis [deg] |
| `period` | Float64 | Rotation period [hours] |
| `yorp` | Float64 | YORP acceleration [deg/day^2] |
| `jd0` | Float64 | Reference epoch [JD] |
| `phi0` | Float64 | Initial rotation phase [deg] |
| `quality_flag` | Int64 | Model quality (1 = best) |
| `nonconvex` | Boolean | Whether the shape model is nonconvex |

Additional physical property columns include `equiv_diameter`,
`visual_albedo`, `thermal_inertia`, and their uncertainties.

### Caching Behavior

Both loading functions use a 30-day cache by default.  The DAMIT
complete export (~1.3 GB) is downloaded once and stored as a
compressed tar.gz archive:

| Scenario | Behaviour |
|----------|-----------|
| File missing, download succeeds | Load from fresh download |
| File missing, download fails | Raise `RuntimeError` |
| File stale, download succeeds | Load from fresh download |
| File stale, download fails | Fall back to cached file |
| File fresh | Load from cached file |

The cache location defaults to `~/.cache/astrojax/datasets/damit/` and
can be overridden with the `ASTROJAX_CACHE` environment variable.

## Spin Parameter Lookup

Use `get_damit_spin` to look up spin parameters for a specific asteroid
from a pre-loaded models DataFrame:

```python
from astrojax.datasets import load_damit_models, get_damit_spin

models = load_damit_models()
spin = get_damit_spin(models, asteroid_id=1)
print(f"lambda={spin['lambda']}, beta={spin['beta']}, P={spin['period']} h")

# Access the second model (mirror solution)
spin2 = get_damit_spin(models, asteroid_id=1, model_index=1)
```

## Computing Rotation Matrices

The `damit_spin_to_rotation` function computes the body-fixed-to-ecliptic
rotation matrix at a given Julian date using the DAMIT spin model:

```python
import jax.numpy as jnp
from astrojax.datasets import load_damit_models, get_damit_spin, damit_spin_to_rotation

models = load_damit_models()
spin = get_damit_spin(models, asteroid_id=1)

# Pack spin parameters into a 6-element array
spin_params = jnp.array([
    spin["lambda"], spin["beta"], spin["period"],
    spin["jd0"], spin["phi0"], spin["yorp"] or 0.0,
])

# Compute rotation at a specific epoch
R = damit_spin_to_rotation(spin_params, 2460000.5)
print(R)  # 3x3 rotation matrix
```

The function is fully JAX-compatible and works with `jax.jit`,
`jax.vmap`, and `jax.grad`:

```python
import jax

# JIT compile for performance
R_jit = jax.jit(damit_spin_to_rotation)(spin_params, 2460000.5)

# Vectorize over multiple epochs
times = jnp.linspace(2460000.5, 2460001.5, 100)
Rs = jax.vmap(damit_spin_to_rotation, in_axes=(None, 0))(spin_params, times)
print(Rs.shape)  # (100, 3, 3)
```

## Shape Models

### Loading Shapes

Load the 3D shape mesh (vertices and triangular facets) for a specific
model:

```python
from astrojax.datasets import get_damit_shape

vertices, facets = get_damit_shape(model_id=42)
print(f"Vertices: {vertices.shape}")  # (N, 3) float32
print(f"Facets: {facets.shape}")      # (M, 3) int32
```

### Scaling Shapes

DAMIT shapes use arbitrary units.  Use `scale_shape_vertices` to
rescale to physical dimensions:

```python
from astrojax.datasets import get_damit_shape, scale_shape_vertices

vertices, facets = get_damit_shape(model_id=42)
# Scale so max extent is 500 meters
scaled = scale_shape_vertices(vertices, 500.0)
```

### Rotating Shapes

Apply the spin-state rotation to transform shape vertices from the
body-fixed frame to the ecliptic frame at a given epoch:

```python
from astrojax.datasets import get_damit_shape, rotate_shape_points
import jax.numpy as jnp

vertices, facets = get_damit_shape(model_id=42)
spin_params = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
rotated = rotate_shape_points(spin_params, 2460001.5, vertices)
```

### Exporting Meshes

Export shapes to standard 3D formats (requires `trimesh`, included in
the `extras` dependency group):

```python
from astrojax.datasets import get_damit_shape, export_shape_glb, export_shape_stl

vertices, facets = get_damit_shape(model_id=42)

# Export to GLB (binary glTF)
export_shape_glb(vertices, facets, "asteroid.glb")

# Export to STL
export_shape_stl(vertices, facets, "asteroid.stl")
```

Install the extras group if trimesh is not available:

```bash
pip install 'astrojax[extras]'
```
