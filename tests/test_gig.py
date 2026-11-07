"""The generalized inverse Gaussian against its density, by quadrature on the log scale.

``scipy.stats.geninvgauss`` integrates on the original scale and gives up for the
heavy-tailed parameters this sampler has to handle, so the reference here is the same
log-concave density the sampler targets, integrated numerically.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

stats = pytest.importorskip("scipy.stats")

PARAMETERS = [
    (0.5, 1.0, 1.0),
    (-0.5, 2.0, 3.0),
    (0.0, 1.0, 1.0),
    (3.0, 0.1, 10.0),
    (-3.0, 10.0, 0.1),
    (50.0, 1.0, 1.0),
    (-50.0, 1.0, 1.0),
    (0.01, 1e-4, 1e-4),
    (1.0, 1e4, 1e4),
    (0.5, 1e-8, 1.0),
    (2.0, 1.0, 1e-8),
    (-1.5, 1e-8, 1.0),
    (0.0, 1e-6, 1e-6),
]


def _centred(x, p, a, b):
    """The draws as y, the centred log scale of the sampler, with its A and B."""
    q, omega2 = abs(p), a * b
    A = (np.sqrt(omega2 + q * q) + q) / 2
    y = np.log(x) - np.log(2 * A / a) if p >= 0 else np.log(b / (2 * A)) - np.log(x)
    return y, A, omega2 / (4 * A)


def _cdf(A, B):
    drop = lambda y: A * (np.expm1(y) - y) + B * (np.expm1(-y) + y)  # noqa: E731
    hi = 1e-3
    while drop(hi) < 45:
        hi *= 1.5
    lo = 1e-3
    while drop(-lo) < 45:
        lo *= 1.5
    grid = np.linspace(-lo, hi, 400_001)
    density = np.exp(-drop(grid))
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2)])
    return lambda y: np.interp(y, grid, cdf / cdf[-1])


@pytest.mark.parametrize("p, a, b", PARAMETERS)
def test_matches_density(p, a, b):
    x = np.asarray(augax.gig_sample(p, a, b, jax.random.key(0), shape=(400_000,)))
    y, A, B = _centred(x, p, a, b)
    assert stats.kstest(y, _cdf(A, B)).pvalue > 1e-3


@pytest.mark.parametrize(
    "p, a, b", [(0.5, 1.0, 1.0), (-1.5, 2.0, 3.0), (3.0, 0.5, 4.0)]
)
def test_matches_scipy_where_it_is_sound(p, a, b):
    x = np.asarray(augax.gig_sample(p, a, b, jax.random.key(1), shape=(100_000,)))
    law = stats.geninvgauss(p, np.sqrt(a * b), scale=np.sqrt(b / a))
    assert stats.kstest(x, law.cdf).pvalue > 1e-3


def test_limits():
    """b = 0 is a gamma with rate a / 2; a = 0 an inverse gamma with scale b / 2."""
    key = jax.random.key(2)
    n = 400_000
    x = np.asarray(augax.gig_sample(2.5, 2.0, 0.0, key, shape=(n,)))
    assert stats.kstest(x, stats.gamma(2.5).cdf).pvalue > 1e-3
    x = np.asarray(augax.gig_sample(-2.5, 0.0, 2.0, key, shape=(n,)))
    assert stats.kstest(x, stats.invgamma(2.5).cdf).pvalue > 1e-3
    x = np.asarray(augax.gig_sample(0.001, 2.0, 0.0, key, shape=(n,)))
    assert np.all(np.isfinite(x)) and np.all(x >= 0)
    assert x.mean() == pytest.approx(0.001, rel=0.2)


def test_extreme_parameters_terminate():
    key = jax.random.key(3)
    for p, a, b in [
        (0.0, 1e-300, 1e-300),
        (1e6, 1.0, 1.0),
        (-1e6, 1.0, 1.0),
        (1.0, 1e300, 1e300),
        (0.5, 1e-300, 1e300),
        (1e-300, 1.0, 1.0),
        (1e-3, 2.0, 0.0),  # gamma and inverse-gamma limits whose log-scale
        (-1e-3, 0.0, 2.0),  # tails reach past where exp(-y) overflows
        (1e-300, 1.0, 0.0),
    ]:
        x = np.asarray(augax.gig_sample(p, a, b, key, shape=(2000,)))
        assert not np.any(np.isnan(x)), (p, a, b)


@pytest.mark.parametrize(
    "p, a, b",
    [
        (float("nan"), 1.0, 1.0),
        (float("inf"), 1.0, 1.0),
        (0.5, -1.0, 1.0),
        (0.5, 1.0, -1.0),
        (0.5, float("nan"), 1.0),
        (0.5, float("inf"), 1.0),
        (0.5, 0.0, 1.0),  # p > 0 needs a > 0
        (-0.5, 1.0, 0.0),  # p < 0 needs b > 0
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
    ],
)
def test_undefined_parameters_draw_nan(p, a, b):
    assert np.isnan(augax.gig_sample(p, a, b, jax.random.key(4)))
    assert jnp.isnan(augax.gig_sample(jnp.full(3, p), a, b, jax.random.key(4))).all()
