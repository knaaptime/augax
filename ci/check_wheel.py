"""Post-build checks on an installed augax wheel (cibuildwheel test step).

augax is self-contained C++, so its wheel should bundle no shared library but the
MSVC C++ runtime, which delvewheel vendors on Windows under a private name, as it
does for numpy.  Anything else fails: a bundled OpenMP runtime in particular can
abort a process that loads another one.  Then check that every sampler loads, in
both precisions, and that Pólya-Gamma draws from the right law.
"""

import pathlib
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

import augax


def check_samplers(dtype):
    key = jax.random.key(0)
    draws = [
        augax.pg_sample(jnp.ones(4), jnp.zeros(4), key),
        augax.ks_sample(jnp.ones(4), key),
        augax.crt_sample(jnp.full(4, 7.0), 1.5, key),
        augax.gig_sample(0.5, 1.0, jnp.ones(4), key),
    ]
    for x in draws:
        assert x.dtype == dtype and x.shape == (4,) and bool(jnp.all(x > 0))


check_samplers(jnp.float32)  # the default, without x64
jax.config.update("jax_enable_x64", True)
check_samplers(jnp.float64)

root = pathlib.Path(augax.__file__).resolve().parent
cxx_runtime = re.compile(r"msvcp140(-[0-9a-f]+)?\.dll", re.IGNORECASE)
bundled = [
    p.name
    for d in (root / ".dylibs", root.parent / "augax.libs")
    if d.is_dir()
    for p in d.iterdir()
    if not cxx_runtime.fullmatch(p.name)
]
if bundled:
    sys.exit(f"augax wheels should bundle only the C++ runtime, found: {bundled}")

# E[PG(h, z)] = h / (2 z) * tanh(z / 2), which is h / 4 at z = 0.
n = 200_000
for h, z in ((1.0, 0.0), (1.0, 2.0), (3.0, -1.5)):
    draws = np.asarray(
        augax.pg_sample(jnp.full(n, h), jnp.full(n, z), jax.random.key(0))
    )
    mean = h / 4 if z == 0 else h / (2 * z) * np.tanh(z / 2)
    se = draws.std() / np.sqrt(n)
    if abs(draws.mean() - mean) > 5 * se:
        sys.exit(f"PG({h}, {z}) mean {draws.mean():.5f} differs from {mean:.5f}")

# Nested vmap folds into one batched call.
keys = jax.random.split(jax.random.key(1), 6).reshape(2, 3)
out = jax.vmap(jax.vmap(lambda k: augax.pg_sample(jnp.ones(4), jnp.zeros(4), k)))(keys)
assert out.shape == (2, 3, 4) and bool(jnp.all(out > 0))
print(f"augax {augax.__version__}: bundles nothing unexpected; checks passed")
