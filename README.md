# augax

On-device draws for **conditionally Gaussian augmentation** in JAX, each an XLA
FFI custom call: **Pólya-Gamma** sampling (`pg_sample`) and the **Kolmogorov
scale** of the logistic's normal scale mixture (`ks_sample`).

`augax.pg_sample(h, z, key)` draws `omega ~ PG(h, z)` from *inside* a JIT'd /
`lax.scan`-ed / `pmap`-ed program with **no host round-trip** — the native
replacement for `jax.pure_callback(random_polyagamma, ...)`, which pays a
device↔host round-trip on every Pólya-Gamma Gibbs sweep and serializes under
`jax.pmap`.

```python
import jax, augax
jax.config.update("jax_enable_x64", True)

@jax.jit
def omega_update(h, z, key):
    return augax.pg_sample(h, z, key)
```

## Method

`pg_sample` follows the hybrid dispatch of
[`zoj613/polyagamma`](https://github.com/zoj613/polyagamma) (`method=None`),
changed where that scheme is measurably biased:

- **`h > 50`**: a normal with the exact Pólya-Gamma mean and variance.
- **`8 <= h <= 50`**: the **saddlepoint** rejection sampler (Windle, Polson &
  Scott 2014, arXiv:1405.0506), O(1) per accepted draw.
- **`h = 1`** (Bernoulli/logit), **or integer `h <= 4` with `|z| <= 1`**: the
  exact **Devroye** sampler for `PG(1, z)` (Polson, Scott & Windle 2013), summed
  `h` times.
- **Otherwise, `1 < h < 8`** — e.g. Negative-Binomial `h = y + alpha`: the
  **alternate** sampler (Windle, Polson & Scott 2014, Section 4).
- **`h < 1`**: the Gamma-series representation — 20 terms plus a remainder
  matched in mean and variance — exact in both moments.

Each departure from `polyagamma` 2.0.2 removes a bias measured with 2M draws
against the exact mean and variance:

- **Alternate sampler, `1 < h < 4`, small `|z|`**: `polyagamma` runs +0.4–0.9% in
  the mean and +1–5% in the variance. Its right-hand envelope uses the constant
  `(π/2)^(h/2)`, where both its Gamma proposal and the density's tail carry
  `(π/2)^h`; augax uses the latter.
- **`h < 1`**: the alternate method assumes `h >= 1`; below it `polyagamma` runs
  −0.4–0.8% in the mean. augax uses the Gamma series.
- **`4 < h < 8` with `|z| <= 4`**: `polyagamma` uses the saddlepoint
  approximation, 1–2% low in the variance. augax uses the exact alternate
  sampler.

The saddlepoint and alternate samplers are ported from `polyagamma`
(BSD-3-Clause). Every regime is tested against the closed-form mean and
variance.

The RNG is a per-call `xoshiro256**` seeded (via `splitmix64`) from the JAX PRNG
key, so draws are reproducible and each call owns its stream — safe when several
threads call at once. Normals and exponentials come from ziggurat samplers.

An array longer than 65,536 elements is split into blocks of that size, each
with its own stream, and the blocks are drawn in parallel on every hardware
thread (set `AUGAX_NUM_THREADS` to cap it). The blocks depend only on the array
length, so a draw is the same on any machine and any thread count. On a large
Negative-Binomial model (8.1M cells) this takes the draw from 3.6 s to 0.33 s on
16 threads.

## The Kolmogorov scale

A standard logistic disturbance is a normal scale mixture,
`eps | lam ~ N(0, lam^2)` with `lam = 2K` and `K` Kolmogorov distributed
(Stefanski 1991). `ks_sample(u, key)` draws `lam` given `eps = u`, which makes a
logistic latent variable conditionally Gaussian: the propagated disturbance of a
latent SAR logit, for one. The draw is exact: `1/lam^2` is proposed from an
inverse Gaussian, `IG(1/|u|, 1)` (a chi-square at `|u| < 1e-4`), and accepted
with probability `S(v) = f(v) e^{v/2} <= 1`, `f` the density of `(2K)^2`,
evaluated lazily by the squeeze of an alternating series — the Kolmogorov
series above `v = 4/3` and its Jacobi transform below, where the first's
partial sums are not bounds (Holmes & Held 2006). It is checked against the
identity `E[exp(-t^2 lam^2 / 2)] = pi t / sinh(pi t)` and against the
posterior moments by quadrature. `ks_sample_numpy(u, rng)` is the same C++
sampler for host callers.

## Status

- ✅ All `h > 0`: the logit (`h = 1`), Negative-Binomial (`h = y + alpha`) and
  large-shape regimes.
- Dispatch is automatic: the sampler is selected from `(h, z)` with no API
  change.
- ✅ `ks_sample`: exact for every `u`, on the device and on the host.

CPU-only (the sampler is cheap scalar work; the point is to keep it on-device
next to the rest of a JIT'd Gibbs sweep).
