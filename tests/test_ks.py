"""Correctness tests for ``ks_sample``, the Kolmogorov scale of the logistic.

A standard logistic eps is ``eps | lam ~ N(0, lam^2)`` with ``lam = 2K`` and
``K`` Kolmogorov distributed.  Drawn given logistic disturbances, ``lam`` must
therefore follow the prior ``2K``, which the identity
``E[exp(-t^2 lam^2 / 2)] = pi t / sinh(pi t)`` checks; drawn given a fixed
``u``, its moments must match the posterior ``p(lam | u)`` by quadrature.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

stats = pytest.importorskip("scipy.stats")
integrate = pytest.importorskip("scipy.integrate")


@pytest.fixture(scope="module")
def logistic_draws():
    """``(eps, lam)`` from the host sampler and ``lam`` from the device one."""
    rng = np.random.default_rng(1)
    eps = rng.logistic(size=1_000_000)
    lam_host = augax.ks_sample_numpy(eps, rng)
    lam_jax = np.asarray(augax.ks_sample(jnp.asarray(eps), jax.random.key(0)))
    return eps, lam_host, lam_jax


@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("t", [0.25, 0.5, 1.0, 2.0])
def test_mixing_identity(logistic_draws, which, t):
    lam = logistic_draws[which]
    g = np.exp(-0.5 * t * t * lam * lam)
    z = (g.mean() - np.pi * t / np.sinh(np.pi * t)) / (g.std() / np.sqrt(g.size))
    assert abs(z) < 4.0


@pytest.mark.parametrize("which", [1, 2])
def test_marginal_is_twice_kolmogorov(logistic_draws, which):
    lam = logistic_draws[which][:200_000]
    assert stats.kstest(lam / 2, "kstwobign").pvalue > 1e-3


def test_disturbance_is_normal_given_scale(logistic_draws):
    eps, lam, _ = logistic_draws
    assert stats.kstest((eps / lam)[:200_000], "norm").pvalue > 1e-3


def test_host_and_device_agree(logistic_draws):
    _, lam_host, lam_jax = logistic_draws
    assert stats.ks_2samp(lam_host[:200_000], lam_jax[:200_000]).pvalue > 1e-3


def _posterior_moments(u):
    """Mean and variance of ``p(lam | u)`` by quadrature in log space."""
    logf = lambda lam: (  # noqa: E731
        -np.log(lam) - 0.5 * (u / lam) ** 2 + stats.kstwobign.logpdf(lam / 2)
    )
    grid = np.linspace(0.05, max(12.0, 3 * np.sqrt(abs(u))), 4000)
    c = np.max(logf(grid))
    lo, hi = grid[0], grid[-1]
    f = lambda lam: np.exp(logf(lam) - c)  # noqa: E731
    Z = integrate.quad(f, lo, hi, limit=500)[0]
    m1 = integrate.quad(lambda x: x * f(x), lo, hi, limit=500)[0] / Z
    m2 = integrate.quad(lambda x: x * x * f(x), lo, hi, limit=500)[0] / Z
    return m1, m2 - m1 * m1


# Both density branches (v <= 4/3 is common for small |u|), the chi^2 proposal
# below |u| = 1e-4, and the inverse-Gaussian proposal above it.
@pytest.mark.parametrize("u", [0.0, 1e-6, 0.3, -1.0, 3.0, 8.0, 20.0])
def test_conditional_moments(u):
    m, v = _posterior_moments(u)
    lam = augax.ks_sample_numpy(np.full(400_000, u), np.random.default_rng(2))
    assert abs(lam.mean() - m) < 4.0 * np.sqrt(v / lam.size)
    assert lam.var() == pytest.approx(v, rel=0.02)


def test_shapes_batching_and_reproducibility():
    keys = jax.random.split(jax.random.key(3), 4)
    out = jax.vmap(lambda k: augax.ks_sample(jnp.ones(5), k))(keys)
    assert out.shape == (4, 5)
    nested = jax.jit(jax.vmap(jax.vmap(lambda k: augax.ks_sample(jnp.zeros(3), k))))(
        jax.random.split(jax.random.key(4), (2, 3))
    )
    assert nested.shape == (2, 3, 3)
    a = augax.ks_sample(jnp.ones(5), keys[0])
    assert jnp.array_equal(a, augax.ks_sample(jnp.ones(5), keys[0]))
    assert not jnp.array_equal(a, augax.ks_sample(jnp.ones(5), keys[1]))
    assert augax.ks_sample_numpy(np.zeros((3, 4)), np.random.default_rng(0)).shape == (
        3,
        4,
    )


def test_blocks_beyond_one_stream():
    """Arrays above one block (2^16) are drawn in parallel, independent blocks."""
    lam = augax.ks_sample_numpy(np.zeros(300_000), np.random.default_rng(5))
    m, v = _posterior_moments(0.0)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0)
    assert abs(lam.mean() - m) < 4.0 * np.sqrt(v / lam.size)


def test_inside_scan():
    def step(carry, key):
        return carry, augax.ks_sample(jnp.linspace(-3, 3, 7), key)

    _, out = jax.lax.scan(step, 0.0, jax.random.split(jax.random.key(6), 5))
    assert out.shape == (5, 7) and bool(jnp.all(out > 0))
