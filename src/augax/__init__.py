"""augax: on-device draws for conditionally Gaussian augmentation, in JAX.

Each primitive is an XLA FFI custom call that samples one augmentation
variable inside ``@jax.jit`` / ``lax.scan``.  It ships one:

``pg_sample``
    Pólya–Gamma, ``omega ~ PG(h, z)``, for the logit and negative-binomial
    likelihoods (Polson, Scott & Windle 2013).
``ks_sample``
    The Kolmogorov scale of a logistic disturbance, ``lam | eps``, which makes
    the logistic a normal scale mixture (Stefanski 1991; Holmes & Held 2006).
    :func:`ks_sample_numpy` is the same draw for host (NumPy) callers.

Draw ``omega ~ PG(h, z)`` from inside ``@jax.jit`` / ``lax.scan`` with no host
round-trip — the native replacement for ``jax.pure_callback(random_polyagamma)``
that taxes every Gibbs sweep (and serializes under ``jax.pmap``).

Draws follow the hybrid scheme of ``polyagamma``'s ``random_polyagamma``
(``method=None``): exact Devroye draws for the Bernoulli/logit augmentation
(``h = 1``), the exact alternate sampler for other real ``h`` below the
saddlepoint region — e.g. the Negative-Binomial ``h = y + alpha`` — a
saddlepoint rejection sampler for larger ``h``, and a moment-matched normal above
``h = 50`` (see the README).

Example::

    import jax, augax
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def step(h, z, key):
        return augax.pg_sample(h, z, key)   # on-device PG draw
"""

import jax
import jax.numpy as jnp
import numpy as np

import augax_cpp as _cpp

try:
    # Single source of truth: the version in the git tag, via the installed
    # distribution's metadata (setuptools-scm).
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("augax")
except Exception:  # running from a source tree that was never installed
    __version__ = "0+unknown"
__all__ = ["pg_sample", "ks_sample", "ks_sample_numpy"]

jax.ffi.register_ffi_target(
    "augax_pg_sample_f64", _cpp.pg_sample_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "augax_pg_sample_batched_f64", _cpp.pg_sample_batched_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "augax_ks_sample_f64", _cpp.ks_sample_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "augax_ks_sample_batched_f64", _cpp.ks_sample_batched_f64_capsule(), platform="cpu"
)


def _require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            'augax requires 64-bit mode: jax.config.update("jax_enable_x64", True)'
        )


def _seed_from_key(key):
    """Derive an int64 RNG seed from a JAX PRNG key (per-key, reproducible)."""
    kd = jax.random.key_data(key).ravel().astype(jnp.int64)
    # Combine the key words for a bit more entropy; kept non-negative-ish (the
    # C++ RNG treats it as an unsigned 64-bit seed).
    seed = kd[0]
    if kd.shape[0] > 1:
        seed = seed ^ (kd[1] << jnp.int64(23))
    return seed


@jax.custom_batching.custom_vmap
def _sample_batched(h, z, seed):
    """One FFI call for a batch: ``h``, ``z`` are ``(B, ...)``, ``seed`` is ``(B,)``."""
    call = jax.ffi.ffi_call(
        "augax_pg_sample_batched_f64", jax.ShapeDtypeStruct(h.shape, h.dtype)
    )
    return call(h, z, seed)


@_sample_batched.def_vmap
def _sample_batched_vmap(axis_size, in_batched, h, z, seed):
    # A further vmap over an already-batched call: merge the outer axis into
    # the batch axis, so any depth of nesting is still one native call.
    # Without this rule the raw batched ffi_call has no batching rule and
    # vmap-of-vmap fails.
    h_b, z_b, seed_b = in_batched
    if not h_b:
        h = jnp.broadcast_to(h, (axis_size,) + h.shape)
    if not z_b:
        z = jnp.broadcast_to(z, (axis_size,) + z.shape)
    if not seed_b:
        seed = jnp.broadcast_to(seed, (axis_size,) + seed.shape)
    outer, inner = h.shape[0], h.shape[1]
    out = _sample_batched(
        h.reshape((outer * inner,) + h.shape[2:]),
        z.reshape((outer * inner,) + z.shape[2:]),
        seed.reshape(outer * inner),
    )
    return out.reshape((outer, inner) + out.shape[1:]), True


@jax.custom_batching.custom_vmap
def _dispatch(h, z, seed):
    call = jax.ffi.ffi_call(
        "augax_pg_sample_f64", jax.ShapeDtypeStruct(h.shape, h.dtype)
    )
    return call(h, z, seed)


@_dispatch.def_vmap
def _dispatch_vmap(axis_size, in_batched, h, z, seed):
    h_b, z_b, seed_b = in_batched
    if not h_b:
        h = jnp.broadcast_to(h, (axis_size,) + h.shape)
    if not z_b:
        z = jnp.broadcast_to(z, (axis_size,) + z.shape)
    if not seed_b:
        seed = jnp.broadcast_to(seed, (axis_size,))
    return _sample_batched(h, z, seed), True


def pg_sample(h, z, key):
    """Draw ``omega ~ PG(h, z)`` element-wise, on-device.

    Parameters
    ----------
    h : array_like
        Shape ``[n]`` float64. PG shape (h = 1 for Bernoulli/logit; positive
        integer for the exact sum-of-Devroye path).
    z : array_like
        Shape ``[n]`` float64. PG tilt (the linear predictor).
    key : jax.Array
        A JAX PRNG key. ``vmap`` over a batch of keys draws each element's
        batch with its own RNG stream.

    Returns
    -------
    omega : jax.Array
        Draws from ``PG(h, z)`` with the same shape as ``h``.
    """
    _require_x64()
    h = jnp.asarray(h, jnp.float64)
    z = jnp.asarray(z, jnp.float64)
    if h.shape != z.shape:
        raise ValueError(f"h and z must have the same shape, got {h.shape}, {z.shape}")
    seed = _seed_from_key(key)
    return _dispatch(h, z, seed)


# ---------------------------------------------------------------------------
# The Kolmogorov scale of the logistic
# ---------------------------------------------------------------------------


@jax.custom_batching.custom_vmap
def _ks_sample_batched(u, seed):
    """One FFI call for a batch: ``u`` is ``(B, ...)``, ``seed`` is ``(B,)``."""
    call = jax.ffi.ffi_call(
        "augax_ks_sample_batched_f64", jax.ShapeDtypeStruct(u.shape, u.dtype)
    )
    return call(u, seed)


@_ks_sample_batched.def_vmap
def _ks_sample_batched_vmap(axis_size, in_batched, u, seed):
    # As for _sample_batched: merge a further vmap axis into the batch axis.
    u_b, seed_b = in_batched
    if not u_b:
        u = jnp.broadcast_to(u, (axis_size,) + u.shape)
    if not seed_b:
        seed = jnp.broadcast_to(seed, (axis_size,) + seed.shape)
    outer, inner = u.shape[0], u.shape[1]
    out = _ks_sample_batched(
        u.reshape((outer * inner,) + u.shape[2:]), seed.reshape(outer * inner)
    )
    return out.reshape((outer, inner) + out.shape[1:]), True


@jax.custom_batching.custom_vmap
def _ks_dispatch(u, seed):
    call = jax.ffi.ffi_call(
        "augax_ks_sample_f64", jax.ShapeDtypeStruct(u.shape, u.dtype)
    )
    return call(u, seed)


@_ks_dispatch.def_vmap
def _ks_dispatch_vmap(axis_size, in_batched, u, seed):
    u_b, seed_b = in_batched
    if not u_b:
        u = jnp.broadcast_to(u, (axis_size,) + u.shape)
    if not seed_b:
        seed = jnp.broadcast_to(seed, (axis_size,))
    return _ks_sample_batched(u, seed), True


def ks_sample(u, key):
    r"""Draw the Kolmogorov scale ``lam`` given a logistic disturbance ``u``, on-device.

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
        Shape ``[n]`` float64: the disturbances.
    key : jax.Array
        A JAX PRNG key. ``vmap`` over a batch of keys draws each element's
        batch with its own RNG stream.

    Returns
    -------
    lam : jax.Array
        The scales, with the shape of ``u``; ``lam**2`` is the variance.
    """
    _require_x64()
    u = jnp.asarray(u, jnp.float64)
    return _ks_dispatch(u, _seed_from_key(key))


def ks_sample_numpy(u, rng):
    """:func:`ks_sample` on the host: the same sampler, for NumPy callers.

    Parameters
    ----------
    u : array_like
        The disturbances, any shape, as float64.
    rng : numpy.random.Generator
        Supplies the seed of the C++ stream.

    Returns
    -------
    lam : numpy.ndarray
        The scales, with the shape of ``u``.
    """
    u = np.ascontiguousarray(u, dtype=np.float64)
    seed = int(rng.integers(0, np.iinfo(np.int64).max))
    return _cpp.ks_sample_host(u.reshape(-1), seed).reshape(u.shape)
