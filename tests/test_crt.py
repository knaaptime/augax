"""The Chinese restaurant table count against its exact distribution."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

stats = pytest.importorskip("scipy.stats")
special = pytest.importorskip("scipy.special")


def _pmf(y, r):
    """The Poisson-binomial law of the table count, by convolution."""
    pmf = np.array([1.0])
    for n in range(y):
        p = r / (r + n)
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf


# Below, across and well past the switch to thinning at r / (r + n) = 0.1.
@pytest.mark.parametrize(
    "y, r",
    [(1, 1.0), (2, 0.5), (10, 1.0), (10, 0.1), (50, 2.0), (200, 1.0), (200, 30.0)],
)
def test_matches_exact_pmf(y, r):
    n = 1_000_000
    draws = np.asarray(augax.crt_sample(float(y), r, jax.random.key(y), shape=(n,)))
    assert np.array_equal(draws, np.floor(draws))
    pmf = _pmf(y, r)
    observed = np.bincount(draws.astype(int), minlength=pmf.size)
    keep = pmf * n > 20
    obs = np.append(observed[keep], observed[~keep].sum())
    exp = np.append(pmf[keep] * n, pmf[~keep].sum() * n)
    if exp[-1] < 1:
        obs, exp = obs[:-1], exp[:-1]
    if obs.size > 1:
        assert stats.chisquare(obs, exp * obs.sum() / exp.sum()).pvalue > 1e-3


@pytest.mark.parametrize(
    "y, r",
    [(1e6, 1.0), (1e6, 50.0), (1e9, 3.0), (1e12, 0.5), (1e4, 1e4), (2.0**53, 1.0)],
)
def test_large_counts(y, r):
    """Thinning reaches these at no cost in y, and keeps the exact moments."""
    n = 200_000
    draws = np.asarray(augax.crt_sample(y, r, jax.random.key(1), shape=(n,)))
    mean = r * (special.digamma(r + y) - special.digamma(r))
    var = mean - r * r * (special.polygamma(1, r) - special.polygamma(1, r + y))
    assert abs(draws.mean() - mean) < 5 * np.sqrt(var / n)
    assert draws.var() / var == pytest.approx(1.0, abs=0.03)


def test_edges():
    key = jax.random.key(2)
    assert float(augax.crt_sample(0.0, 1.0, key)) == 0.0
    assert float(augax.crt_sample(1.0, 1e-300, key)) == 1.0  # the first customer sits
    bad = jnp.array([-1.0, 2.5, float("nan"), float("inf"), 3.0, 3.0, 3.0, 2.0**54])
    r = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, -1.0, float("inf"), 1.0])
    assert np.all(np.isnan(augax.crt_sample(bad, r, key)))
