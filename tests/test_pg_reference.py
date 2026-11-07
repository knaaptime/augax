"""PG(h, z) against the reference ``polyagamma``, where that reference is sound.

Its Devroye path (h = 1) is exact.  At larger h it is a close approximation that 200k
draws cannot tell from the exact law, so this checks only for a gross departure.
Elsewhere it is biased or a plain normal, so ``test_pg_exact.py`` tests every regime
against closed forms instead.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

pa = pytest.importorskip("polyagamma")
stats = pytest.importorskip("scipy.stats")

N = 200_000


def _compare(h, z, seed):
    ours = augax.pg_sample(
        jnp.full(N, float(h)), jnp.full(N, float(z)), jax.random.key(seed)
    )
    theirs = pa.random_polyagamma(
        h, z, size=N, random_state=np.random.default_rng(seed)
    )
    return stats.ks_2samp(np.asarray(ours), theirs).pvalue


@pytest.mark.parametrize("z", [0.0, 0.5, 2.0, 5.0, -3.0, 12.0])
def test_devroye_matches_reference(z):
    assert _compare(1.0, z, seed=int(abs(z) * 100) + 1) > 1e-3


@pytest.mark.parametrize("h", [8.0, 10.0, 15.0, 20.0])
@pytest.mark.parametrize("z", [0.0, 1.0, -4.0, 5.0])
def test_large_h_matches_reference(h, z):
    assert _compare(h, z, seed=int(h * 100) + int(abs(z)) + 1) > 1e-3
