"""Parameters a Gibbs sampler can actually produce: none may hang or mask a failure.

A NaN linear predictor once hung the native sampler uninterruptibly for h = 1 and drew
from PG(h, 0) for others, so a diverged chain carried on looking healthy.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

NAN, INF = float("nan"), float("inf")
EVERY_H = [0.05, 0.5, 1.0, 2.0, 2.5, 4.0, 6.5, 10.0, 29.0, 30.0, 100.0]
FLOOR = 1e-12


def _pg(h, z, n=2000):
    key = jax.random.key(0)
    return np.asarray(augax.pg_sample(jnp.full(n, h), jnp.full(n, z), key))


@pytest.mark.parametrize("h", EVERY_H)
def test_nan_tilt_draws_nan(h):
    assert np.all(np.isnan(_pg(h, NAN)))


@pytest.mark.parametrize("z", [0.0, 3.0, 100.0, NAN])
@pytest.mark.parametrize("h", [NAN, INF, -INF, -1.0, -1.5, -3.0, -1e-3])
def test_invalid_shape_draws_nan(h, z):
    assert np.all(np.isnan(_pg(h, z)))


@pytest.mark.parametrize("h", EVERY_H)
@pytest.mark.parametrize("z", [INF, -INF])
def test_infinite_tilt_is_the_limit(h, z):
    np.testing.assert_array_equal(_pg(h, z), FLOOR)


@pytest.mark.parametrize("z", [0.0, 3.0, 1e6, INF])
def test_zero_shape_is_the_floor(z):
    np.testing.assert_array_equal(_pg(0.0, z), FLOOR)


@pytest.mark.parametrize("h", [*EVERY_H, 1e-120, 1e-3, 1e8, 1e300])
@pytest.mark.parametrize("z", [1e2, 1e6, 1e12, 1e20, 1e50, 1e150, 1e300, 1.7e308])
def test_extreme_tilt_is_finite_and_near_the_mean(h, z):
    x = _pg(h, z, n=4000)
    assert np.all(np.isfinite(x)) and np.all(x >= FLOOR)
    expected = max(h / (2 * z), FLOOR)
    if h * z > 1e3:  # the law has concentrated; below this it is too skewed to compare
        assert x.mean() == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("h", [1e-300, 1e-120, 1e-10])
@pytest.mark.parametrize("z", [0.0, 1.0, 20.0])
def test_tiny_shape_terminates(h, z):
    x = _pg(h, z)
    assert np.all(np.isfinite(x)) and np.all(x >= FLOOR)


def test_one_bad_element_does_not_spoil_the_rest():
    h = jnp.array([1.0, 1.0, -1.0, 1.0])
    z = jnp.array([0.0, NAN, 0.0, 0.0])
    x = np.asarray(augax.pg_sample(h, z, jax.random.key(1)))
    np.testing.assert_array_equal(np.isnan(x), [False, True, True, False])


def test_ks_invalid_and_extreme():
    key = jax.random.key(2)
    u = jnp.array([NAN, INF, -INF, 1e308, -1e308, 1e200, 0.0])
    lam = np.asarray(augax.ks_sample(u, key))
    assert np.isnan(lam[0]) and np.all(np.isinf(lam[1:3]))
    np.testing.assert_allclose(lam[3:6], np.sqrt(np.abs(np.asarray(u[3:6]))))
    assert np.isfinite(lam[6]) and lam[6] > 0


def test_host_shares_the_guards():
    rng = np.random.default_rng(0)
    assert np.isnan(augax.pg_sample_numpy(1.0, NAN, rng))
    assert np.isnan(augax.ks_sample_numpy(NAN, rng))
