"""augax: native draws for conditionally Gaussian augmentation, in JAX.

Each sampler is an XLA FFI custom call that draws its augmentation variable inside
``@jax.jit`` / ``lax.scan`` with no host round-trip, the native replacement for a
``jax.pure_callback`` that taxes every Gibbs sweep.

``pg_sample``
    Pólya–Gamma, ``omega ~ PG(h, z)``, for the logit and negative-binomial
    likelihoods (Polson, Scott & Windle 2013).
``ks_sample``
    The Kolmogorov scale of a logistic disturbance, ``lam | eps``, which makes the
    logistic a normal scale mixture (Stefanski 1991; Holmes & Held 2006).
``crt_sample``
    The Chinese restaurant table count, which makes a negative-binomial dispersion
    conjugate to a gamma (Zhou & Carin 2015).
``gig_sample``
    The generalized inverse Gaussian, the scale of the normal mixtures behind the
    lasso, quantile regression and the generalized hyperbolic family.

Each has a ``*_numpy`` twin that makes the same draws for NumPy callers, and
:func:`pg_mean` and :func:`pg_var` give the Pólya–Gamma moments in closed form.

Example::

    import jax, augax

    @jax.jit
    def step(z, key):
        return augax.pg_sample(1.0, z, key)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from . import _cpp

try:
    # Single source of truth: the version in the git tag, via the installed
    # distribution's metadata (setuptools-scm).
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("augax")
except Exception:  # running from a source tree that was never installed
    __version__ = "0+unknown"
__all__ = [
    "pg_sample",
    "pg_sample_numpy",
    "pg_mean",
    "pg_var",
    "ks_sample",
    "ks_sample_numpy",
    "crt_sample",
    "crt_sample_numpy",
    "gig_sample",
    "gig_sample_numpy",
]

for _name, _capsule in _cpp.ffi_targets().items():
    jax.ffi.register_ffi_target(f"augax_{_name}", _capsule, platform="cpu")


def _host_draw(name, seed, *params):
    """The native draw on the host, for any leading batch dimensions."""
    seed = np.asarray(seed, dtype=np.uint32)
    params = [np.asarray(p) for p in params]
    flat = [np.ascontiguousarray(p, dtype=np.float64).reshape(-1) for p in params]
    draw = getattr(_cpp, f"{name}_host")
    out = draw(seed.reshape(-1, seed.shape[-1]), *flat)
    return out.reshape(params[0].shape).astype(params[0].dtype)


def _native(name):
    """A sampler as one native call, whatever the depth of ``vmap`` around it.

    ``seed`` is ``(..., W)``, the words of the keys, and every parameter is
    ``(..., *event)``, with the same leading dimensions.  ``vmap`` only adds one,
    which the native call already understands.

    The FFI target exists on CPU only.  Elsewhere the draw runs on the host through
    a callback, which gives the same numbers at the price of a round-trip.
    ``platform_dependent`` compiles only the branch for the platform at hand, and the
    ``jit`` makes that hold for eager calls too.
    """

    @jax.jit
    def dispatch(seed, *params):
        result = jax.ShapeDtypeStruct(params[0].shape, params[0].dtype)

        def on_cpu(seed, *params):
            target = f"augax_{name}_{params[0].dtype.name}"
            return jax.ffi.ffi_call(target, result)(seed, *params)

        def elsewhere(seed, *params):
            return jax.pure_callback(partial(_host_draw, name), result, seed, *params)

        return jax.lax.platform_dependent(seed, *params, cpu=on_cpu, default=elsewhere)

    @jax.custom_batching.custom_vmap
    def draw(seed, *params):
        return dispatch(seed, *params)

    @draw.def_vmap
    def draw_vmap(axis_size, in_batched, *args):
        args = [
            x if batched else jnp.broadcast_to(x, (axis_size,) + x.shape)
            for x, batched in zip(args, in_batched)
        ]
        return draw(*args), True

    return draw


_pg = _native("pg")
_ks = _native("ks")
_crt = _native("crt")
_gig = _native("gig")


def _key_words(key):
    """The words of one PRNG key, for the native seed."""
    words = jax.random.key_data(key)
    if words.ndim != 1:
        raise ValueError(
            "expected a single PRNG key; use jax.vmap to draw over a batch of keys, "
            f"got key data of shape {words.shape}"
        )
    return words.astype(jnp.uint32)


def _params(shape, *params):
    """Parameters as floating arrays broadcast to one shape."""
    dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
    params = [jnp.asarray(p, dtype) for p in params]
    if shape is None:
        shape = jnp.broadcast_shapes(*(p.shape for p in params))
    return [jnp.broadcast_to(p, shape) for p in params]


def _host_sample(name, rng, shape, *params):
    """A NumPy draw: the native sampler with its seed from ``rng``."""
    params = [np.asarray(p, dtype=np.float64) for p in params]
    if shape is None:
        shape = np.broadcast_shapes(*(p.shape for p in params))
    params = [np.broadcast_to(p, shape) for p in params]
    seed = rng.integers(0, 2**32, size=(1, 2), dtype=np.uint32)
    return _host_draw(name, seed, *params)


def pg_sample(
    h: ArrayLike, z: ArrayLike, key: jax.Array, *, shape: tuple[int, ...] | None = None
) -> jax.Array:
    """Draw ``omega ~ PG(h, z)`` element-wise.

    The sampler is chosen from ``(h, z)`` element by element (see the README).  Each is
    exact or within ``1e-6`` of the exact law in Kolmogorov distance, for ``h >= 0.03``.

    Parameters
    ----------
    h : array_like
        PG shape: 1 for the logit, ``y + alpha`` for the negative binomial.
    z : array_like
        PG tilt, the linear predictor.  Broadcast against ``h``.
    key : jax.Array
        One JAX PRNG key.  To draw for several chains, ``vmap`` over a batch of keys:
        ``vmap(f)(keys)[i]`` equals ``f(keys[i])``.
    shape : tuple of int, optional
        The shape of the result, to which ``h`` and ``z`` broadcast.  Defaults to their
        broadcast shape.

    Returns
    -------
    omega : jax.Array
        The draws, floored at ``1e-12`` so that they are usable as precisions.  An
        invalid parameter (NaN, ``h < 0`` or an infinite ``h``) draws NaN.
    """
    return _pg(_key_words(key), *_params(shape, h, z))


def pg_sample_numpy(
    h: ArrayLike,
    z: ArrayLike,
    rng: np.random.Generator,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """:func:`pg_sample` on the host, with its seed drawn from ``rng``."""
    return _host_sample("pg", rng, shape, h, z)


def _pg_series_arguments(z):
    """``|z| / 2`` kept away from 0, its square, and where the series takes over."""
    r2 = jnp.square(z) / 4
    # The closed forms lose digits to cancellation at small r, the more so in float32.
    small = r2 < (0.02**2 if z.dtype == jnp.float64 else 0.2**2)
    r = jnp.sqrt(jnp.where(small, 1.0, r2))
    return r, r2, small


def pg_mean(h: ArrayLike, z: ArrayLike) -> jax.Array:
    """The mean of ``PG(h, z)``, ``h tanh(z / 2) / (2 z)``, stable at ``z = 0``.

    Differentiable, for the EM and variational updates of a logit.
    """
    dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
    h, z = jnp.asarray(h, dtype), jnp.asarray(z, dtype)
    r, r2, small = _pg_series_arguments(z)
    series = (1 + r2 * (-1 / 3 + r2 * (2 / 15 - r2 * 17 / 315))) / 4
    return h * jnp.where(small, series, jnp.tanh(r) / (4 * r))


def pg_var(h: ArrayLike, z: ArrayLike) -> jax.Array:
    """The variance of ``PG(h, z)``, stable at ``z = 0`` and at large ``|z|``.

    With ``r = |z| / 2``, it is ``h (tanh(r) - r sech^2(r)) / (16 r^3)``.
    """
    dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
    h, z = jnp.asarray(h, dtype), jnp.asarray(z, dtype)
    r, r2, small = _pg_series_arguments(z)
    e = jnp.exp(-2 * r)
    sech2 = 4 * e / jnp.square(1 + e)
    closed = (jnp.tanh(r) - r * sech2) / (16 * r**3)
    series = (1 + r2 * (-4 / 5 + r2 * (17 / 35 - r2 * 248 / 945))) / 24
    return h * jnp.where(small, series, closed)


def ks_sample(
    u: ArrayLike, key: jax.Array, *, shape: tuple[int, ...] | None = None
) -> jax.Array:
    r"""Draw the Kolmogorov scale ``lam`` given a logistic disturbance ``u``.

    A standard logistic :math:`\varepsilon` is the normal scale mixture
    :math:`\varepsilon \mid \lambda \sim N(0, \lambda^2)`, :math:`\lambda = 2K`
    with :math:`K` Kolmogorov distributed.  Given :math:`\varepsilon = u` this
    draws :math:`\lambda` from

    .. math::
        p(\lambda \mid u) \propto \lambda^{-1}\,\varphi(u/\lambda)\,f_{2K}(\lambda),

    exactly: a rejection sampler with an inverse-Gaussian proposal for
    :math:`1/\lambda^2` and the alternating-series squeeze of the Kolmogorov
    density in both of its forms (Holmes & Held 2006).

    Parameters
    ----------
    u : array_like
        The disturbances.
    key : jax.Array
        One JAX PRNG key; see :func:`pg_sample`.
    shape : tuple of int, optional
        The shape of the result, to which ``u`` broadcasts.

    Returns
    -------
    lam : jax.Array
        The scales; ``lam**2`` is the variance.  A NaN ``u`` draws NaN.
    """
    return _ks(_key_words(key), *_params(shape, u))


def ks_sample_numpy(
    u: ArrayLike, rng: np.random.Generator, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """:func:`ks_sample` on the host, with its seed drawn from ``rng``."""
    return _host_sample("ks", rng, shape, u)


def crt_sample(
    y: ArrayLike, r: ArrayLike, key: jax.Array, *, shape: tuple[int, ...] | None = None
) -> jax.Array:
    r"""Draw the Chinese restaurant table count ``L ~ CRT(y, r)``.

    The tables that :math:`y` customers occupy at concentration :math:`r`,
    :math:`L = \sum_{n=0}^{y-1} \mathrm{Bernoulli}(r / (r + n))`.  With it, a
    negative-binomial dispersion :math:`r` has the conjugate update
    :math:`r \mid L \sim \mathrm{Gamma}(a + \sum L, b - \sum \log(1 - p))`,
    in shape and rate (Zhou & Carin 2015).

    The draw is exact.  Its cost grows with ``min(y, r)``, since it thins only once the
    table probability has fallen below 0.1.

    Parameters
    ----------
    y : array_like
        The counts, non-negative integers, in floating point.
    r : array_like
        The concentrations, positive.  Broadcast against ``y``.
    key : jax.Array
        One JAX PRNG key; see :func:`pg_sample`.
    shape : tuple of int, optional
        The shape of the result, to which ``y`` and ``r`` broadcast.

    Returns
    -------
    tables : jax.Array
        The counts, integer-valued, in floating point.  An invalid parameter draws NaN.
    """
    return _crt(_key_words(key), *_params(shape, y, r))


def crt_sample_numpy(
    y: ArrayLike,
    r: ArrayLike,
    rng: np.random.Generator,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """:func:`crt_sample` on the host, with its seed drawn from ``rng``."""
    return _host_sample("crt", rng, shape, y, r)


def gig_sample(
    p: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    key: jax.Array,
    *,
    shape: tuple[int, ...] | None = None,
) -> jax.Array:
    r"""Draw from the generalized inverse Gaussian ``GIG(p, a, b)``.

    Its density is proportional to :math:`x^{p-1} \exp(-(a x + b / x) / 2)`.  The draw
    is exact, by a rejection sampler that is uniformly efficient in the parameters
    (Devroye 2014).  ``a = 0`` with ``p < 0`` gives the inverse gamma, and ``b = 0``
    with ``p > 0`` the gamma.

    Parameters
    ----------
    p : array_like
        The index.
    a, b : array_like
        The non-negative rates.  All three broadcast together.
    key : jax.Array
        One JAX PRNG key; see :func:`pg_sample`.
    shape : tuple of int, optional
        The shape of the result, to which the parameters broadcast.

    Returns
    -------
    x : jax.Array
        The draws.  Parameters that define no distribution draw NaN.
    """
    return _gig(_key_words(key), *_params(shape, p, a, b))


def gig_sample_numpy(
    p: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    rng: np.random.Generator,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """:func:`gig_sample` on the host, with its seed drawn from ``rng``."""
    return _host_sample("gig", rng, shape, p, a, b)
