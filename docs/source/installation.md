# Installation

`augax` supports Python >= 3.11. We recommend using [miniforge] or [pixi].

## Installing a released version

`augax` is available on PyPI and can be installed with:

```bash
pip install augax
```

JAX is a core dependency and is always installed.

## Installing from source

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/knaaptime/augax.git
cd augax
conda env create -f environment.yml
conda activate augax
pip install -e . --no-deps
```

## Verifying the installation

```python
import jax
import augax

key = jax.random.key(0)
print(augax.pg_sample(1.0, jax.numpy.zeros(5), key))
```

```{note}
Draws take the precision JAX is running in: `float64` under `jax_enable_x64`, and
`float32` otherwise. The samplers compute in double precision either way, so a
`float32` draw is the `float64` draw, rounded.
```

[miniforge]: https://github.com/conda-forge/miniforge
[pixi]: https://pixi.sh
