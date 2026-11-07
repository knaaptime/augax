# augax

Native draws for conditionally Gaussian augmentation in JAX. Each sampler runs as an
XLA FFI custom call inside `jit`, `vmap`, `lax.scan` and `pmap`, with no host round
trip. It replaces `jax.pure_callback(random_polyagamma, ...)`, which pays that round
trip on every Gibbs sweep and serializes under `pmap`.

```python
import jax, augax

@jax.jit
def omega_update(z, key):
    return augax.pg_sample(1.0, z, key)
```

| Sampler | Draws | Augments |
|---|---|---|
| `pg_sample(h, z, key)` | Pólya-Gamma, `PG(h, z)` | logit and negative-binomial likelihoods |
| `ks_sample(u, key)` | the Kolmogorov scale given a logistic disturbance | a logistic latent variable |
| `crt_sample(y, r, key)` | the Chinese restaurant table count | a negative-binomial dispersion |
| `gig_sample(p, a, b, key)` | the generalized inverse Gaussian | the lasso, quantile regression, generalized hyperbolic models |

Each has a `*_numpy` twin that takes a `numpy.random.Generator` and makes the same
draws for host callers. `pg_mean` and `pg_var` give the Pólya-Gamma moments in closed
form, stable at `z = 0` and differentiable, for EM and variational updates.

Parameters broadcast against one another, and `shape=` sets the shape of the result.
An invalid parameter draws NaN, so a diverged chain stays visible. Draws take the
precision JAX is running in: `float64` under `jax_enable_x64`, and `float32`
otherwise. The samplers compute in double precision either way.

## Pólya-Gamma

`pg_sample` chooses a sampler for each element from `(h, z)`. The first row that
applies wins:

| Where | Sampler | Accuracy |
|---|---|---|
| `\|z\| >= 15 + log(max(h, 1))` | inverse Gaussian | Kolmogorov distance below `3.4e-7` |
| integer `h <= 4` | Devroye (Polson, Scott & Windle 2013), summed `h` times | exact |
| otherwise | sum of gammas | Kolmogorov distance below `1e-6` for `h >= 0.03` |

Draws are floored at `1e-12`, so that they are usable as precisions.

At large `|z|`, `PG(h, z)` is an inverse Gaussian. Its cumulant generating function is
`h (r - r') + h (log1p(exp(-2 r)) - log1p(exp(-2 r')))`, with `r = |z| / 2` and
`r' = sqrt(r^2 - w / 2)`. The first term is that of `IG(h / (2 |z|), h^2 / 4)`, and the
second is at most `h exp(-|z|)`. One normal and one uniform therefore draw it for any
`h`, and the other samplers need not handle large `|z|` at all.

Elsewhere `pg_sample` uses the representation of `PG` as an infinite sum of gammas,
`omega = sum_k w_k G_k` with `G_k ~ Gamma(h, 1)` (Polson, Scott & Windle 2013). It draws the
first `K` terms and replaces the rest with a shifted inverse Gaussian matched to their
first three cumulants. The inverse Gaussian suits the remainder because the weights fall
as `1 / k^2`, which gives its Levy density the same `x^(-3/2)` behaviour near zero, and it
is the same law the whole sum tends to at large `|z|`.

The characteristic functions of this approximation and of `PG` are both in closed form,
which bounds the Kolmogorov distance between them with no sampling noise. `K` is set from
`h` to hold that bound below `1e-6` at every `|z|`, and the test suite recomputes it. A
sample of `n` draws detects a distance near `1.6 / sqrt(n)`, so `1e-6` needs about
`3e12` draws at a single `(h, z)`.

| `h` | 0.05 | 0.1 | 0.2 | 0.5 | 0.9 | 1.5 | 2.5 | 4 to 6 | 6 to 20 | 20 to 65 | 65 to 200 | 200 to 400 | above 400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `K` | 85 | 45 | 25 | 13 | 10 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |

Small `h` is the costly end, because the remainder then has to span many scales. `K` stops
at 128, near `h = 0.03`, and the bound no longer holds below that. Against a 512-term
series, 16M draws detect no difference at `h = 0.01`. At `h = 0.005` the distance is
`2e-3`, at `omega` near `1e-6`, against a standard deviation of `0.01`.

The test suite also checks every regime against the exact Laplace transform,
`E[exp(-t omega)] = [cosh(z / 2) / cosh(sqrt(z^2 / 4 + t / 2))]^h`, on real draws. It
reaches scales of `omega` far below the standard deviation, and needs no reference
sampler.

The widely used [`polyagamma`](https://github.com/zoj613/polyagamma) 2.0.2 is measurably
biased in several regimes, as checked with 8M draws against the exact mean and variance:

- Its alternate sampler (Windle, Polson & Scott 2014), `1 < h < 4`, small `|z|`: up to
  0.9% high in the mean and 5.5% high in the variance. Its right-hand envelope uses the
  constant `(pi / 2)^(h / 2)`, where both its gamma proposal and the density's tail carry
  `(pi / 2)^h`.
- `h < 1`: the alternate method assumes `h >= 1`, and below it `polyagamma` runs up to
  0.7% low in the mean and 1.2% low in the variance.
- `4 < h < 8` with `|z| <= 4`: its saddlepoint approximation runs 2.2% low in the variance
  at `h = 4.5` and under 0.5% low from `h = 6`.
- `h > 50`: its normal has none of the law's skewness, which is `0.27` at `h = 51`.

## The Kolmogorov scale

A standard logistic disturbance is a normal scale mixture, `eps | lam ~ N(0, lam^2)`
with `lam = 2K` and `K` Kolmogorov distributed (Stefanski 1991). `ks_sample(u, key)`
draws `lam` given `eps = u`, which makes a logistic latent variable conditionally
Gaussian: the propagated disturbance of a latent SAR logit, for one. The draw is exact.
It proposes `1/lam^2` from an inverse Gaussian, `IG(1/|u|, 1)` (a chi-square at
`|u| < 1e-4`), and accepts with probability `S(v) = f(v) e^{v/2} <= 1`, `f` the density
of `(2K)^2`, evaluated lazily by the squeeze of an alternating series: the Kolmogorov
series above `v = 4/3` and its Jacobi transform below, where the first's partial sums
are not bounds (Holmes & Held 2006). The tests check it against the identity
`E[exp(-t^2 lam^2 / 2)] = pi t / sinh(pi t)` and against the posterior moments by
quadrature.

## The Chinese restaurant table count

`crt_sample(y, r, key)` draws `L = sum_{n=0}^{y-1} Bernoulli(r / (r + n))`, the tables
that `y` customers occupy at concentration `r`. With it, a negative-binomial dispersion
has a conjugate gamma update (Zhou & Carin 2015), the companion to `pg_sample` at
`h = y + r`. The draw is exact. It draws the large probabilities one at a time and thins
the small ones by geometric jumps, so its cost grows with `min(y, r)` and hardly at all
with `y` beyond that. Counts and tables are in floating point, integer-valued.

## The generalized inverse Gaussian

`gig_sample(p, a, b, key)` draws from the density proportional to
`x^(p-1) exp(-(a x + b / x) / 2)`. On the log scale, centred at the mode, this density
is log-concave for every parameter, so a single rejection sampler serves them all
(Devroye 2014). `b = 0` with `p > 0` gives the gamma, and `a = 0` with `p < 0` the
inverse gamma.

## Threads and reproducibility

A key's elements form one stream, or one stream per block of 65,536 if there are more.
Any free thread takes the next stream, on XLA's thread pool for native calls and on
threads of their own for NumPy calls. Chains under `vmap` therefore parallelize as one
long array does. Set `AUGAX_NUM_THREADS` to cap the threads.

A draw depends on the key and the number of elements alone. The thread count does not
change it, and `vmap(f)(keys)[i]` equals `f(keys[i])`. As in `jax.random`, closing over
one key under `vmap` gives every row the same draws, so split keys per chain. augax does
not promise equal draws across builds or machines: its rejection steps compare the
results of the math library, whose last bits vary by platform.

Single-threaded cost on an Apple M-series core, against `polyagamma` 2.0.2, for `z`
drawn from `N(0, 1.5^2)`:

| `h` | 0.05 | 0.5 | 1 | 2.5 | 6.5 | 10 | 50 | 1000 |
|---|---|---|---|---|---|---|---|---|
| augax (ns/draw) | 1140 | 199 | 59 | 103 | 83 | 83 | 74 | 40 |
| `polyagamma` (ns/draw) | 193 | 116 | 68 | 183 | 490 | 487 | 458 | 16 |

The comparison flatters `polyagamma` wherever it carries the biases above: at small `h`,
and at `h = 1000`, where its draw is a normal.

## Platforms

The FFI target exists on CPU, where the point is to keep this cheap scalar work next to
the rest of a JIT'd Gibbs sweep. On any other platform the same call compiles to a host
callback instead, which makes the same draws at the price of the round trip.
