"""PG(h, z) against its closed forms; needs no reference sampler, so it runs everywhere.

The Laplace transform of PG(h, z) is known exactly,

    E[exp(-t omega)] = [cosh(z / 2) / cosh(sqrt(z^2 / 4 + t / 2))]^h,

which checks the whole distribution rather than a moment or two: the third cumulant of
a remainder, for one, is invisible to tests of the mean and variance.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

REGIMES = [
    1.0, 2.0, 3.0, 4.0,  # Devroye, summed
    0.05, 0.1, 0.5, 0.9, 1.05, 1.5, 1.99, 2.5, 3.9, 4.01, 4.5, 5.99,  # sum of gammas,
    6.0, 6.5, 7.9, 8.0, 10.0, 19.9, 20.0, 50.0, 64.9, 65.0,  # with fewer terms
    100.0, 199.0, 200.0, 399.0, 400.0, 1e4,  # as h rises
]  # fmt: skip
# Either side of the truncated inverse Gaussian's branch at |z| = 3.125, and of the
# switch to the inverse Gaussian at |z| = 15 (later for h > 1).
TILTS = [0.0, 0.5, -1.0, 3.0, 3.25, -10.0, 14.9, 15.0, 19.9, -40.0, 500.0]


def _draw(h, z, n, seed):
    return np.asarray(
        augax.pg_sample(
            jnp.full(n, float(h)), jnp.full(n, float(z)), jax.random.key(seed)
        )
    )


def _analytic_mean(h, z):
    return h * 0.25 if abs(z) < 1e-9 else h * np.tanh(z / 2) / (2 * z)


def _analytic_var(h, z):
    if abs(z) < 1e-9:
        return h / 24.0
    r = abs(z) / 2
    e = np.exp(-2 * r)
    return h * (np.tanh(r) - r * 4 * e / (1 + e) ** 2) / (16 * r**3)


def _log_transform(h, z, t):
    """log E[exp(-t omega)], with nothing to overflow or cancel."""
    r = abs(z) / 2
    rt = np.sqrt(r * r + t / 2)
    return h * (
        t / 2 / -(r + rt) + np.log1p(np.exp(-2 * r)) - np.log1p(np.exp(-2 * rt))
    )


def _laplace_zscore(omega, h, z):
    """The largest z-score of the mean of exp(-t omega) over a wide range of t.

    Large t probes the law at scales far below its standard deviation, where a remainder
    of the wrong shape misplaces mass that no test at t ~ 1 / sigma can see.
    """
    m, v = _analytic_mean(h, z), _analytic_var(h, z)
    worst = 0.0
    for scaled in (0.5, 1.0, 2.0):  # centred, which keeps large h in range
        t = scaled / np.sqrt(v)
        g = np.exp(-t * (omega - m))
        exact = np.exp(t * m + _log_transform(h, z, t))
        worst = max(worst, abs(g.mean() - exact) / (g.std() / np.sqrt(omega.size)))
    for scaled in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
        t = scaled / np.sqrt(v)
        exact = np.exp(_log_transform(h, z, t))
        if exact < 1e-3:  # the law has no mass left at this scale
            break
        g = np.exp(-t * omega)
        worst = max(worst, abs(g.mean() - exact) / (g.std() / np.sqrt(omega.size)))
    return worst


@pytest.mark.parametrize("h", REGIMES)
@pytest.mark.parametrize("z", TILTS)
def test_matches_exact_laplace_transform(h, z):
    omega = _draw(h, z, 1_000_000, seed=int(h * 1000) + int(abs(z) * 10))
    assert _laplace_zscore(omega, h, z) < 5


@pytest.mark.parametrize("z", [0.0, 1.0, 3.0, 3.25, 6.0, 12.0])
def test_devroye_at_high_power(z):
    """Its proposal is accepted over 99.9% of the time, so an error in the acceptance
    step moves the law by under 0.1%, which a million draws cannot see."""
    omega = _draw(1.0, z, 64_000_000, seed=int(z * 100) + 3)
    assert _laplace_zscore(omega, 1.0, z) < 5


@pytest.mark.parametrize("h", [0.05, 0.5])
@pytest.mark.parametrize("z", [12.0, 14.9, 15.0, 40.0, 200.0, 1000.0])
def test_small_h_at_large_tilt(h, z):
    """A gamma remainder matched in two cumulants fails here by 70 standard errors."""
    omega = _draw(h, z, 8_000_000, seed=int(h * 100) + int(z))
    assert _laplace_zscore(omega, h, z) < 5


@pytest.mark.parametrize(
    "h", [0.2, 0.5, 0.9, 1.05, 1.3, 2.0, 2.7, 3.0, 3.9, 4.5, 5.0, 6.5, 7.9, 10.0, 40.0]
)
@pytest.mark.parametrize("z", [0.0, 1.0, -3.0, 6.0])
def test_moments_match_analytic(h, z):
    """The mean and variance, to the resolution of 400k draws.

    Checked against the exact moments rather than ``polyagamma``: its hybrid sampler is
    itself off by up to 5% in the variance for real h < 8.
    """
    n = 400_000
    x = _draw(h, z, n, seed=int(h * 1000) + int(abs(z) * 10))
    mean, var = _analytic_mean(h, z), _analytic_var(h, z)
    z_mean = (x.mean() - mean) / np.sqrt(x.var() / n)
    z_var = (x.var() - var) / np.sqrt(np.var((x - x.mean()) ** 2) / n)
    assert abs(z_mean) < 5, f"mean {x.mean()} vs {mean} (z = {z_mean:.1f})"
    assert abs(z_var) < 5, f"var {x.var()} vs {var} (z = {z_var:.1f})"


def _series_distance(h, z):
    """A bound on the Kolmogorov distance of the sum-of-gammas sampler from the exact law.

    Both characteristic functions are in closed form, and

        sup_x |F(x) - G(x)| <= (1 / pi) int_0^inf |phi_F(u) - phi_G(u)| / u du,

    so this carries no sampling noise, and sees errors far too small to simulate.  The
    remainder's parameters are the sampler's own, from the extension.
    """
    K, shift, mu, psi = augax._cpp.pg_series_parameters(h, z)
    assert shift >= 0
    r = abs(z) / 2
    sigma = np.sqrt(_analytic_var(h, z))
    u = np.logspace(-7, 70, 6000) / sigma
    w = np.sqrt(r * r - 0.5j * u)
    log_exact = h * (
        0.5j * u / (r + w) + np.log1p(np.exp(-2 * r)) - np.log1p(np.exp(-2 * w))
    )
    k = np.arange(1, K + 1) - 0.5
    weights = 1 / (2 * np.pi**2 * (k * k + r * r / np.pi**2))
    log_approx = 1j * u * shift + 2j * mu * u / (1 + np.sqrt(1 - 2j * psi * u))
    for weight in weights:
        log_approx = log_approx - h * np.log1p(-1j * u * weight)
    delta = log_approx - log_exact
    with np.errstate(all="ignore"):
        gap = np.where(
            np.abs(delta) < 1,
            np.abs(np.exp(log_exact) * np.expm1(delta)),
            np.abs(np.exp(log_exact) - np.exp(log_approx)),
        )
    return np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(np.log(u))) / np.pi


@pytest.mark.parametrize(
    "h",
    [0.0326, 0.05, 0.1, 0.2, 0.5, 0.9, 0.999, 1.0001, 1.33, 1.34, 1.99, 2.01, 3.99]
    + [4.01, 5.99, 6.0, 8.0, 19.99, 20.0, 64.99, 65.0, 199.99, 200.0, 399.99, 400.0]
    + [1e4, 1e6],
)
def test_series_is_within_tolerance_of_the_exact_law(h):
    """Every K in the schedule, at every |z| that reaches this sampler."""
    switch = 15 + np.log(max(h, 1))
    tilts = np.linspace(0, switch, 13)
    assert max(_series_distance(h, z) for z in tilts) <= 1e-6


@pytest.mark.parametrize("h", [0.0326, 0.05, 0.2, 0.5, 0.999])
@pytest.mark.parametrize("z", [0.0, 0.03, 1.0, 5.0, 10.0, 14.9])
def test_direct_tail_keeps_the_exact_mean(h, z):
    """At the many terms of small h the tail is direct, and its mean has a closed form."""
    K, shift, mu, _ = augax._cpp.pg_series_parameters(h, z)
    k = np.arange(1, K + 1) - 0.5
    drawn = (1 / (2 * np.pi**2 * (k * k + (z / (2 * np.pi)) ** 2))).sum()
    assert shift + mu == pytest.approx(_analytic_mean(h, z) - h * drawn, rel=1e-7)


@pytest.mark.parametrize("h", [0.05, 0.5, 1.0, 2.5, 8.0, 30.0, 1000.0, 1e6])
def test_inverse_gaussian_switch_is_within_tolerance(h):
    """The characteristic functions differ by log1p(e^(-2 r)) - log1p(e^(-2 w)) alone."""
    r = (15 + np.log(max(h, 1))) / 2
    u = np.logspace(-7, 70, 6000) / np.sqrt(_analytic_var(h, 2 * r))
    w = np.sqrt(r * r - 0.5j * u)
    extra = h * (np.log1p(np.exp(-2 * r)) - np.log1p(np.exp(-2 * w)))
    gap = np.abs(np.exp(h * 0.5j * u / (r + w) + extra) * np.expm1(-extra))
    assert np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(np.log(u))) / np.pi <= 3.5e-7


@pytest.mark.parametrize("h", [51.0, 100.0, 1000.0])
def test_large_h_has_the_right_skewness(h):
    """A normal has none; PG(h, 0) has 1.96 / sqrt(h)."""
    x = _draw(h, 0.0, 4_000_000, seed=int(h))
    c = x - x.mean()
    assert (c**3).mean() / x.var() ** 1.5 == pytest.approx(1.9596 / np.sqrt(h), rel=0.1)


@pytest.mark.parametrize("h", [8.0, 10.0, 29.0, 30.0, 50.0, 51.0, 100.0, 1000.0])
@pytest.mark.parametrize("z", [30.0, 37.0, 40.0, 100.0, 300.0, 1000.0, 3000.0])
def test_large_h_at_large_tilt(h, z):
    """Saddlepoint and normal samplers once ran 2.75x high and lost their variance here."""
    x = _draw(h, z, 400_000, seed=int(h) + int(z))
    assert x.mean() / _analytic_mean(h, z) == pytest.approx(1.0, abs=0.01)
    assert x.var() / _analytic_var(h, z) == pytest.approx(1.0, abs=0.03)


def test_jit_scan_and_mixed_parameters():
    """Elements of every regime side by side, inside jit and scan."""
    h = jnp.array([0.5, 1.0, 2.5, 10.0, 40.0, 1.0, 0.5])
    z = jnp.array([1.0, -2.0, 0.0, 3.0, -1.0, 100.0, 1e6])

    @jax.jit
    def run(key):
        def step(carry, k):
            return carry, augax.pg_sample(h, z, k)

        return jax.lax.scan(step, 0.0, jax.random.split(key, 20_000))[1]

    out = np.asarray(run(jax.random.key(0)))
    assert out.shape == (20_000, 7) and np.all(out > 0)
    expected = np.array([augax.pg_mean(a, b) for a, b in zip(h, z)])
    np.testing.assert_allclose(out.mean(0), expected, rtol=0.03)


def test_multi_block_draw_is_reproducible_and_exact_in_mean():
    """Arrays over one block (2**16) draw in parallel, one stream per block."""
    n = 5 * 2**16 + 123  # several blocks and a ragged last one
    h = jnp.full(n, 0.6)
    z = jnp.full(n, 1.5)
    key = jax.random.key(7)
    a = np.asarray(augax.pg_sample(h, z, key))
    np.testing.assert_array_equal(a, np.asarray(augax.pg_sample(h, z, key)))
    blocks = a[: 5 * 2**16].reshape(5, 2**16)
    assert not np.array_equal(blocks[0], blocks[1])
    se = np.sqrt(_analytic_var(0.6, 1.5) / n)
    assert abs(a.mean() - _analytic_mean(0.6, 1.5)) < 5 * se


@pytest.mark.parametrize(
    "z", [0.0, 1e-6, 0.01, 0.039, 0.041, 0.5, -3.0, 40.0, 800.0, 1e6]
)
def test_pg_mean_and_var(z):
    """Against the sum representation, whose weights are 1 / (2 pi^2 d_k)."""
    k = np.arange(1, 4_000_001) - 0.5
    d = k * k + (z / (2 * np.pi)) ** 2
    variance = np.sum(1 / d**2) / (4 * np.pi**4)
    mean = 0.25 if z == 0 else np.tanh(abs(z) / 2) / (2 * abs(z))
    assert float(augax.pg_mean(3.0, z)) == pytest.approx(3 * mean, rel=1e-12)
    assert float(augax.pg_var(3.0, z)) == pytest.approx(
        3 * variance, rel=1e-3 if abs(z) > 1e5 else 1e-9
    )


def test_pg_moments_are_differentiable_at_zero():
    for moment in (augax.pg_mean, augax.pg_var):
        assert float(jax.grad(lambda z: moment(1.0, z))(0.0)) == 0.0
    # d/dz of tanh(z / 2) / (2 z) at 1.
    expected = 1 / (4 * np.cosh(0.5) ** 2) - np.tanh(0.5) / 2
    assert float(jax.grad(lambda z: augax.pg_mean(1.0, z))(1.0)) == pytest.approx(
        expected
    )
