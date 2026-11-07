"""vmap, keys, dtypes and threads; needs no reference sampler, so it runs everywhere."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import augax  # noqa: E402

SAMPLERS = {
    "pg": (augax.pg_sample, (1.0, 0.5)),
    "ks": (augax.ks_sample, (0.5,)),
    "crt": (augax.crt_sample, (7.0, 1.5)),
    "gig": (augax.gig_sample, (0.5, 1.0, 2.0)),
}


def _draw(key, n=4):
    return augax.pg_sample(jnp.ones(n), jnp.zeros(n), key)


def test_vmap_over_keys_gives_independent_streams():
    keys = jax.random.split(jax.random.key(0), 5)
    out = jax.vmap(_draw)(keys)
    assert out.shape == (5, 4)
    assert len({tuple(np.asarray(row)) for row in out}) == 5


@pytest.mark.parametrize("name", SAMPLERS)
@pytest.mark.parametrize("n", [4, 70_000])  # within one block, and across two
def test_vmap_row_equals_the_plain_call(name, n):
    sampler, params = SAMPLERS[name]
    keys = jax.random.split(jax.random.key(1), 3)
    f = lambda k: sampler(*params, k, shape=(n,))  # noqa: E731
    rows = np.asarray(jax.vmap(f)(keys))
    assert rows.shape == (3, n)
    for row, key in zip(rows, keys):
        np.testing.assert_array_equal(row, np.asarray(f(key)))


@pytest.mark.parametrize("jit", [False, True])
def test_nested_vmap_matches_flat_vmap(jit):
    keys = jax.random.split(jax.random.key(1), 6)
    nested = jax.vmap(jax.vmap(_draw))
    if jit:
        nested = jax.jit(nested)
    out = nested(keys.reshape(2, 3))
    flat = jax.vmap(_draw)(keys)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(np.asarray(out).reshape(6, 4), np.asarray(flat))


@pytest.mark.parametrize("name", SAMPLERS)
def test_vmap_over_scalars(name):
    sampler, params = SAMPLERS[name]
    keys = jax.random.split(jax.random.key(2), 3)
    out = jax.vmap(sampler)(*(jnp.full(3, p) for p in params), keys)
    assert out.shape == (3,) and len(set(np.asarray(out).tolist())) == 3


def test_vmap_with_unbatched_arguments():
    """Only z batched: h broadcasts, and one key means one stream, as in jax.random."""
    z = jnp.zeros((3, 4))
    key = jax.random.key(2)
    out = np.asarray(jax.vmap(lambda zz: augax.pg_sample(jnp.ones(4), zz, key))(z))
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[0], np.asarray(_draw(key)))


def test_zero_sized():
    key = jax.random.key(3)
    assert augax.pg_sample(jnp.ones(0), jnp.zeros(0), key).shape == (0,)
    keys = jax.random.split(key, 3)
    assert jax.vmap(_draw, (0, None))(keys, 0).shape == (3, 0)
    empty = jax.random.split(key, 0)
    assert jax.vmap(lambda k: _draw(k))(empty).shape == (0, 4)


def test_broadcasting_and_shape():
    key = jax.random.key(4)
    assert augax.pg_sample(1.0, jnp.zeros(4), key).shape == (4,)
    assert augax.pg_sample(jnp.ones((3, 1)), jnp.zeros(4), key).shape == (3, 4)
    assert augax.pg_sample(1.0, 0.0, key).shape == ()
    assert augax.pg_sample(1.0, 0.0, key, shape=(2, 3)).shape == (2, 3)
    assert augax.ks_sample(0.5, key, shape=(5,)).shape == (5,)
    assert augax.gig_sample(0.5, 1.0, jnp.ones(3), key).shape == (3,)
    with pytest.raises(ValueError):
        augax.pg_sample(jnp.ones(3), jnp.zeros(4), key)
    with pytest.raises(ValueError):
        augax.pg_sample(jnp.ones(3), 0.0, key, shape=(4,))


def test_a_batch_of_keys_needs_vmap():
    keys = jax.random.split(jax.random.key(5), 3)
    with pytest.raises(ValueError, match="single PRNG key"):
        augax.pg_sample(jnp.ones(3), jnp.zeros(3), keys)
    with pytest.raises(ValueError, match="single PRNG key"):
        augax.ks_sample(jnp.ones(3), keys)


def test_every_key_word_counts():
    """Keys that the 55-bit seed of old could not tell apart."""
    make = lambda a, b: jax.random.wrap_key_data(jnp.array([a, b], dtype=jnp.uint32))  # noqa: E731
    assert not jnp.array_equal(_draw(make(2**23, 0)), _draw(make(0, 1)))
    assert not jnp.array_equal(_draw(make(0, 1)), _draw(make(1, 0)))


@pytest.mark.parametrize("impl", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_key_implementations(impl):
    first, second = jax.random.split(jax.random.key(6, impl=impl))
    assert not jnp.array_equal(_draw(first), _draw(second))
    assert jnp.array_equal(_draw(first), _draw(first))


def test_legacy_raw_key():
    key = jax.random.PRNGKey(0)
    assert jnp.array_equal(_draw(key), _draw(jax.random.wrap_key_data(key)))


def test_jit_equals_eager():
    key = jax.random.key(7)
    assert jnp.array_equal(jax.jit(_draw)(key), _draw(key))


@pytest.mark.parametrize("name", SAMPLERS)
def test_host_fallback_makes_the_same_draws(name):
    """The callback a device without the FFI target runs."""
    sampler, params = SAMPLERS[name]
    key = jax.random.key(8)
    arrays = [np.full(70_000, p) for p in params]
    words = np.asarray(jax.random.key_data(key))[None]
    host = augax._host_draw(name, words, *arrays)
    np.testing.assert_array_equal(
        host, np.asarray(sampler(*params, key, shape=(70_000,)))
    )


def test_numpy_twins():
    rng = np.random.default_rng(0)
    assert augax.pg_sample_numpy(1.0, np.zeros((2, 3)), rng).shape == (2, 3)
    assert augax.ks_sample_numpy(np.zeros(3), rng, shape=(2, 3)).shape == (2, 3)
    assert augax.crt_sample_numpy(np.full(3, 5.0), 1.0, rng).shape == (3,)
    assert augax.gig_sample_numpy(0.5, 1.0, 1.0, rng).shape == ()
    draws = augax.pg_sample_numpy(1.0, 0.0, np.random.default_rng(1), shape=(200_000,))
    assert draws.mean() == pytest.approx(0.25, abs=0.002)
    np.testing.assert_array_equal(
        draws,
        augax.pg_sample_numpy(1.0, 0.0, np.random.default_rng(1), shape=(200_000,)),
    )


def _run(script, tmp_path, **env):
    """Run ``script`` in a fresh process, with the x64 flag and threads it asks for."""
    full = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    full.update(env)
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        env=full,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_draws_do_not_depend_on_the_thread_count(tmp_path):
    script = """
        import os, numpy as np, jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp, augax
        k = jax.random.key(9)
        z = jnp.asarray(np.random.default_rng(0).normal(0, 2, 200_000))
        chains = jax.vmap(lambda kk: augax.pg_sample(0.5, z[:70_000], kk))(jax.random.split(k, 4))
        np.save(f"out{os.environ['AUGAX_NUM_THREADS']}.npy", np.concatenate([
            np.asarray(augax.pg_sample(2.5, z, k)),
            np.asarray(augax.ks_sample(z, k)),
            np.asarray(chains).ravel(),
            augax.pg_sample_numpy(2.5, np.asarray(z), np.random.default_rng(1)),
        ]))
    """
    for threads in ("1", "3", "8"):
        _run(script, tmp_path, AUGAX_NUM_THREADS=threads)
    base = np.load(tmp_path / "out1.npy")
    np.testing.assert_array_equal(base, np.load(tmp_path / "out3.npy"))
    np.testing.assert_array_equal(base, np.load(tmp_path / "out8.npy"))


def test_float32_without_x64(tmp_path):
    script = """
        import numpy as np, jax
        assert not jax.config.jax_enable_x64
        import jax.numpy as jnp, augax
        k = jax.random.key(10)
        n = 200_000
        out = {
            "pg": augax.pg_sample(1.0, 0.5, k, shape=(n,)),
            "ks": augax.ks_sample(0.5, k, shape=(n,)),
            "crt": augax.crt_sample(7.0, 1.5, k, shape=(n,)),
            "gig": augax.gig_sample(0.5, 1.0, 2.0, k, shape=(n,)),
        }
        for x in out.values():
            assert x.dtype == jnp.float32
        keys = jax.random.split(k, 3)
        assert jax.jit(jax.vmap(augax.pg_sample))(jnp.ones(3), jnp.zeros(3), keys).dtype == jnp.float32
        assert jnp.isfinite(augax.pg_mean(1.0, 0.0)) and jnp.isfinite(augax.pg_var(1.0, 0.01))
        np.savez("x32.npz", **{name: np.asarray(x) for name, x in out.items()})
    """
    _run(script, tmp_path)
    saved = np.load(tmp_path / "x32.npz")
    key = jax.random.key(10)
    for name, (sampler, params) in SAMPLERS.items():
        double = np.asarray(sampler(*params, key, shape=(200_000,)))
        np.testing.assert_array_equal(saved[name], double.astype(np.float32))


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_other_platforms_compile_no_cpu_target(platform, monkeypatch):
    """Without its branch pruned, a device with no FFI target could not compile.

    The callback itself lowers only on a backend of its own platform, so a stub stands
    in for it, in a sampler traced afresh.
    """
    export = pytest.importorskip("jax.export")
    monkeypatch.setattr(
        jax,
        "pure_callback",
        lambda f, result, *a, **k: jnp.zeros(result.shape, result.dtype),
    )
    sampler = augax._native("pg")
    args = (jnp.zeros(2, jnp.uint32), jnp.ones(3), jnp.zeros(3))
    f = jax.jit(lambda *a: sampler(*a))
    assert "augax_pg" not in export.export(f, platforms=[platform])(*args).mlir_module()
    on_cpu = export.export(
        f,
        platforms=["cpu"],
        disabled_checks=[export.DisabledSafetyCheck.custom_call("augax_pg_float64")],
    )(*args).mlir_module()
    assert "augax_pg_float64" in on_cpu
