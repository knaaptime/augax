// augax: augmentation draws as XLA FFI custom calls.
//
// Pólya-Gamma (`pg_sample`), the Kolmogorov scale of the logistic's normal scale
// mixture (`ks_sample`), the Chinese restaurant table count (`crt_sample`) and the
// generalized inverse Gaussian (`gig_sample`).
//
// Replaces the host `jax.pure_callback` PG draw (a device<->host round-trip every
// Gibbs sweep, which serializes under jax.pmap and taxes every sweep) with a native
// FFI custom call inside the JIT'd program.
//
// PG sampler, by regime, the first that applies:
//   - |z| >= 15 + log(h) for h > 1, |z| >= 15 otherwise: inverse Gaussian, which
//     PG(h, z) is to within h exp(-|z|).
//   - integer h <= 4 (h = 1 is the Bernoulli/logit): the exact Devroye method for
//     PG(1, z) (Polson, Scott & Windle 2013), summed h times.
//   - otherwise (e.g. Negative-Binomial h = y + alpha): the sum-of-gammas
//     representation, K terms and an inverse Gaussian remainder, within 1e-6 of the
//     exact law in Kolmogorov distance for h >= 0.03.
//
// Invalid parameters (NaN, h < 0, infinite h) draw NaN.
//
// RNG: a per-stream splitmix64-seeded xoshiro256** — seeded from the JAX PRNG key,
// so draws are reproducible and each stream owns its generator — with ziggurat
// normals and exponentials.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <memory>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// ---------------------------------------------------------------------------
// RNG: xoshiro256** seeded via splitmix64. Small, fast, per-call (thread-safe).
// ---------------------------------------------------------------------------
namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double PI2 = PI * PI;
constexpr double TRUNC = 0.64;            // Devroye truncation point
constexpr double SQRT_TRUNC = 0.8;        // sqrt(TRUNC)
constexpr double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

// Ziggurat tables (Marsaglia & Tsang 2000; Doornik 2005), built once at load: a
// 128-block normal and a 256-block exponential.  x[i] are the block edges
// (x[0] = V / f(R) is the base block) and r[i] = x[i+1] / x[i] the fast-accept
// ratios.  Several times cheaper per variate than Box-Muller or -log(u).
struct Ziggurat {
  static constexpr int kNormBlocks = 128;
  static constexpr double kNormR = 3.442619855899;
  static constexpr double kNormV = 9.91256303526217e-3;
  static constexpr int kExpBlocks = 256;
  static constexpr double kExpR = 7.69711747013104972;
  static constexpr double kExpV = 0.0039496598225815571993;
  double nx[kNormBlocks + 1], nr[kNormBlocks];
  double ex[kExpBlocks + 1], er[kExpBlocks];
  Ziggurat() {
    double f = std::exp(-0.5 * kNormR * kNormR);
    nx[0] = kNormV / f;
    nx[1] = kNormR;
    nx[kNormBlocks] = 0.0;
    for (int i = 2; i < kNormBlocks; ++i) {
      nx[i] = std::sqrt(-2.0 * std::log(kNormV / nx[i - 1] + f));
      f = std::exp(-0.5 * nx[i] * nx[i]);
    }
    for (int i = 0; i < kNormBlocks; ++i) nr[i] = nx[i + 1] / nx[i];
    f = std::exp(-kExpR);
    ex[0] = kExpV / f;
    ex[1] = kExpR;
    ex[kExpBlocks] = 0.0;
    for (int i = 2; i < kExpBlocks; ++i) {
      ex[i] = -std::log(kExpV / ex[i - 1] + f);
      f = std::exp(-ex[i]);
    }
    for (int i = 0; i < kExpBlocks; ++i) er[i] = ex[i + 1] / ex[i];
  }
};
const Ziggurat kZig;

struct Rng {
  uint64_t s[4];
  explicit Rng(uint64_t seed) {
    // splitmix64 to seed the state.
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
      z += 0x9E3779B97F4A7C15ULL;
      uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      s[i] = x ^ (x >> 31);
    }
  }
  static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  inline uint64_t next() {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }
  // uniform in (0, 1): 53-bit mantissa, strictly > 0.
  inline double unif() {
    double u = (next() >> 11) * (1.0 / 9007199254740992.0);
    return u <= 0.0 ? 1e-300 : u;
  }
  // Exp(1) by the ziggurat; the tail beyond R is R + Exp(1) (memoryless).
  // The block index takes the low 8 bits, the uniform the high 53.
  inline double expon() {
    double extra = 0.0;
    for (;;) {
      uint64_t r = next();
      unsigned i = static_cast<unsigned>(r & 0xFF);
      double u = (r >> 11) * (1.0 / 9007199254740992.0);
      if (u < kZig.er[i]) return extra + u * kZig.ex[i];
      if (i == 0) {
        extra += Ziggurat::kExpR;
        continue;
      }
      double x = u * kZig.ex[i];
      double f0 = std::exp(-(kZig.ex[i] - x));
      double f1 = std::exp(-(kZig.ex[i + 1] - x));
      if (f1 + unif() * (f0 - f1) < 1.0) return extra + x;
    }
  }
  // N(0, 1) by the ziggurat (Doornik's ZIGNOR); the tail beyond R by
  // Marsaglia's exponential rejection.  Block index: low 7 bits; uniform: high 53.
  inline double normal() {
    for (;;) {
      uint64_t r = next();
      unsigned i = static_cast<unsigned>(r & 0x7F);
      double u = 2.0 * ((r >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
      if (std::fabs(u) < kZig.nr[i]) return u * kZig.nx[i];
      if (i == 0) {
        double x, y;
        do {
          x = std::log(unif()) / Ziggurat::kNormR;
          y = std::log(unif());
        } while (-2.0 * y < x * x);
        return u < 0.0 ? x - Ziggurat::kNormR : Ziggurat::kNormR - x;
      }
      double x = u * kZig.nx[i];
      double f0 = std::exp(-0.5 * (kZig.nx[i] * kZig.nx[i] - x * x));
      double f1 = std::exp(-0.5 * (kZig.nx[i + 1] * kZig.nx[i + 1] - x * x));
      if (f1 + unif() * (f0 - f1) < 1.0) return x;
    }
  }
};

// IG(mu, 1) / mu by Michael, Schucany & Haas (1976).  With v = mu Y^2, Y standard
// normal, the roots of their quadratic are mu w and mu / w for
//
//   w = 2 / (2 + v + sqrt(v (v + 4))),
//
// and the smaller is taken with probability 1 / (1 + w).  In this form nothing
// cancels, and Y = 0 gives 1.  An infinite v gives w = 0, the limit.
inline double inverse_gaussian_ratio(double mu, Rng& rng) {
  const double y = rng.normal();
  const double v = mu * y * y;
  const double w = 2.0 / (2.0 + v + std::sqrt(v) * std::sqrt(v + 4.0));
  return rng.unif() * (1.0 + w) <= 1.0 ? w : 1.0 / w;
}

// ---------------------------------------------------------------------------
// PG(1, z): Devroye's alternating-series rejection sampler
//
// PG(1, z) = J*(1, c) / 4 with c = |z| / 2, and J*(1, c) has the density
//
//   f(x) = cosh(c) exp(-c^2 x / 2) sum_{n >= 0} (-1)^n a_n(x),
//
// where, with t = TRUNC,
//
//   a_n(x) = pi (n + 1/2) exp(-(n + 1/2)^2 pi^2 x / 2)                  for x > t,
//   a_n(x) = pi (n + 1/2) (2 / (pi x))^(3/2) exp(-2 (n + 1/2)^2 / x)    for x <= t
//
// (Polson, Scott & Windle 2013, Section 4).  The n = 0 term bounds the sum, so it
// makes the proposal, and the rest of the series decides acceptance.
// ---------------------------------------------------------------------------

// The probability of the proposal's piece above t.  With K = pi^2 / 8 + c^2 / 2, and
// cosh(c) dropped from both, the pieces and their masses are
//
//   x >  t:  (pi / 2) exp(-K x),          mass  pi exp(-K t) / (2 K),
//   x <= t:  2 exp(-c) IG(x | 1 / c, 1),  mass  2 exp(-c) P(IG(1 / c, 1) <= t),
//
// where P(IG(1 / c, 1) <= t) = Phi((c t - 1) / sqrt(t)) + exp(2 c) Phi(-(c t + 1) / sqrt(t)).
inline double devroye_upper_probability(double c, double K) {
  // Past this the lower piece outweighs the upper by more than e^600, and returning
  // here keeps both exponentials below in range.
  const double Kt = K * TRUNC;
  if (Kt - c > 600.0) return 0.0;
  const auto norm_cdf = [](double x) {
    return 0.5 * std::erfc(-x * 0.7071067811865476);
  };
  const double lower_over_upper =
      (4.0 / PI) * K *
      (std::exp(Kt - c) * norm_cdf((c * TRUNC - 1.0) / SQRT_TRUNC) +
       std::exp(Kt + c) * norm_cdf(-(c * TRUNC + 1.0) / SQRT_TRUNC));
  return 1.0 / (1.0 + lower_over_upper);
}

// IG(1 / c, 1) truncated to x <= t, the proposal's piece below t.
//
// With the mean above t, most of the law lies past the truncation, so propose from
// its c = 0 limit, x = 1 / Y^2 for a standard normal Y with |Y| >= 1 / sqrt(t), and
// accept with probability exp(-c^2 x / 2).  The normal tail is Marsaglia's: |Y| =
// (1 + t E) / sqrt(t) for E ~ Exp(1), accepted with probability exp(-t E^2 / 2).
//
// With the mean at most t, reject from the law itself.
inline double truncated_inverse_gaussian(double c, Rng& rng) {
  if (c * TRUNC < 1.0) {
    const double half_c2 = 0.5 * c * c;
    for (;;) {
      const double e = rng.expon();
      if (TRUNC * e * e > 2.0 * rng.expon()) continue;
      const double root = 1.0 + TRUNC * e;
      const double x = TRUNC / (root * root);
      if (rng.unif() <= std::exp(-half_c2 * x)) return x;
    }
  }
  const double mu = 1.0 / c;
  double x;
  do {
    x = mu * inverse_gaussian_ratio(mu, rng);
  } while (x >= TRUNC);
  return x;
}

// Accept the proposal x, given U(0, 1), with probability sum_n (-1)^n a_n(x) / a_0(x).
// Both forms of a_n give the same series,
//
//   a_n(x) / a_0(x) = (2 n + 1) q^(n (n + 1)),
//   q = exp(-pi^2 x / 2) for x > t,   q = exp(-2 / x) for x <= t,
//
// whose partial sums bracket the probability.  Everywhere q <= exp(-2 / t), so the
// first of them is at least 1 - 3 exp(-4 / t) > 0.9942, which settles nearly every
// proposal without evaluating q at all.
inline bool accept_jacobi(double x, double u) {
  if (u <= 0.9942) return true;
  const double q = std::exp(x > TRUNC ? -0.5 * PI2 * x : -2.0 / x);
  const double q2 = q * q;
  double sum = 1.0;
  double power = q2;     // q^(n (n + 1))
  double step = q2 * q2; // q^(2 (n + 1)), the next ratio of powers
  for (int n = 1;; n += 2) {
    sum -= (2 * n + 1) * power;
    if (u <= sum) return true;
    power *= step;
    step *= q2;
    sum += (2 * n + 3) * power;
    if (u > sum) return false;
    power *= step;
    step *= q2;
  }
}

inline double sample_pg1(double z, Rng& rng) {
  const double c = 0.5 * std::fabs(z);
  const double K = 0.125 * PI2 + 0.5 * c * c;
  const double upper = devroye_upper_probability(c, K);
  for (;;) {
    const double x = rng.unif() < upper ? TRUNC + rng.expon() / K
                                        : truncated_inverse_gaussian(c, rng);
    if (accept_jacobi(x, rng.unif())) return 0.25 * x;
  }
}

// Gamma(shape, 1) for shape >= 1 (Marsaglia & Tsang 2000), set up once for draws
// that repeat a shape.
struct GammaShape {
  double d, c;
  explicit GammaShape(double shape)
      : d(shape - 1.0 / 3.0), c(1.0 / std::sqrt(9.0 * d)) {}
  double draw(Rng& rng) const {
    for (;;) {
      double x, v;
      do {
        x = rng.normal();
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      const double u = rng.unif();
      const double x2 = x * x;
      if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
      if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
  }
};

// The first three cumulants of PG(1, z), with r = |z| / 2, t = tanh(r), s = sech^2(r):
//
//   k1 = t / (4 r),
//   k2 = A / (16 r^3),   A = t - r s,
//   k3 = B / (64 r^5),   B = 3 A - 2 r^2 t s,
//
// the derivatives at 0 of log cosh(r) - log cosh(sqrt(r^2 - w / 2)).  Those of PG(h, z)
// are h times these.  Below r = 0.02, where A and B lose digits to cancellation, they
// come from their series.
struct PgCumulants {
  double k1, k2, k3;
};

inline PgCumulants pg_cumulants(double r) {
  const double r2 = r * r;
  if (r < 0.02) {
    return {0.25 * (1.0 - r2 / 3.0 + 2.0 * r2 * r2 / 15.0),
            (1.0 - 0.8 * r2 + 17.0 * r2 * r2 / 35.0) / 24.0,
            (1.0 - 17.0 * r2 / 14.0 + 62.0 * r2 * r2 / 63.0) / 60.0};
  }
  const double t = std::tanh(r);
  const double e = std::exp(-2.0 * r);
  const double s = 4.0 * e / ((1.0 + e) * (1.0 + e));
  const double A = t - r * s;
  const double B = 3.0 * A - 2.0 * r2 * t * s;
  return {0.25 * t / r, A / (16.0 * r2 * r), B / (64.0 * r2 * r2 * r)};
}

// ---------------------------------------------------------------------------
// PG(h, z) as a sum of gammas:
//
//   omega = sum_{k >= 1} w_k G_k,   G_k ~ Gamma(h, 1),
//   w_k = 1 / (2 pi^2 d_k),   d_k = (k - 1/2)^2 + a^2,   a = |z| / (2 pi),
//
// whose first three cumulants are h sum w_k, h sum w_k^2 and 2 h sum w_k^3.  The first K
// terms are drawn.  The rest becomes shift + IG(mu, lambda), matched to those cumulants.
// The inverse Gaussian is the right law for it: w_k falls as 1 / k^2, so the Levy density
// of the remainder near 0 goes as x^(-3/2), which is the inverse Gaussian's, where a
// gamma's goes as x^(-1).  It is also the limit of the whole law at large |z| (see
// sample_pg_inverse_gaussian).
//
// The characteristic function of this approximation and of PG(h, z) are both in closed
// form, which bounds the Kolmogorov distance between them with no sampling noise:
//
//   sup_x |F(x) - G(x)| <= (1 / pi) int_0^inf |phi_F(u) - phi_G(u)| / u du.
//
// The bound falls as K rises and as h rises, so K can fall as h rises.
// ---------------------------------------------------------------------------

// The terms that hold the bound above 1e-6 at every |z| this sampler sees (the test suite
// computes it).  At small h the cost is about 4 / h terms, so it stops at 128, near
// h = 0.03; below that the error grows, at scales of omega far below its standard deviation.
constexpr int PG_SERIES_MAX_TERMS = 128;

inline int pg_series_terms(double h) {
  if (h >= 400.0) return 1;
  if (h >= 200.0) return 2;
  if (h >= 65.0) return 3;
  if (h >= 20.0) return 4;
  if (h >= 6.0) return 5;
  if (h <= 4.0 / (PG_SERIES_MAX_TERMS - 5)) return PG_SERIES_MAX_TERMS;
  return static_cast<int>(std::ceil(4.0 / h)) + 5;
}

// The cumulants per unit h of sum_{k > K} w_k G_k, for K >= 10 and a^2 / K^2 well below 1.
// Subtracting the first K terms from the whole would cancel nearly every digit at large K,
// so these are direct: the midpoint Euler-Maclaurin rule,
//
//   sum_{k > K} f(k - 1/2) = int_K^inf f(s) ds + f'(K) / 24 - 7 f'''(K) / 5760 + ...,
//
// for f = d^(-j), whose integrals are series in q = a^2 / K^2.  The mean needs this
// accuracy: near |z| = 15 the standard deviation of omega is small enough for a tiny shift
// to show.
inline PgCumulants pg_tail_cumulants(double K, double a2) {
  const double K2 = K * K;
  const double K3 = K2 * K;
  const double q = a2 / K2;
  const double d = K2 + a2;
  const double d2 = d * d;
  const double d3 = d2 * d;
  const double d4 = d2 * d2;
  const double c = 7.0 / 5760.0;
  const double s1 =
      (1.0 + q * (-1.0 / 3.0 + q * (1.0 / 5.0 + q * (-1.0 / 7.0 + q / 9.0)))) / K -
      K / (12.0 * d2) - c * (24.0 * K / d3 - 48.0 * K3 / d4);
  const double s2 =
      (1.0 / 3.0 + q * (-2.0 / 5.0 + q * (3.0 / 7.0 + q * (-4.0 / 9.0 + q * 5.0 / 11.0)))) / K3 -
      K / (6.0 * d3) - c * (72.0 * K / d4 - 192.0 * K3 / (d4 * d));
  const double s3 =
      (1.0 / 5.0 + q * (-3.0 / 7.0 + q * (2.0 / 3.0 + q * (-10.0 / 11.0 + q * 15.0 / 13.0)))) /
          (K3 * K2) -
      K / (4.0 * d4) - c * (144.0 * K / (d4 * d) - 480.0 * K3 / (d3 * d3));
  const double w = 1.0 / (2.0 * PI2);
  return {w * s1, w * w * s2, 2.0 * w * w * w * s3};
}

// The remainder as shift + IG(mu, mu^2 / psi), matched to its three cumulants: an inverse
// Gaussian has k2 = mu psi and k3 = 3 mu psi^2.
struct PgRemainder {
  double shift, mu, psi;
};

inline PgRemainder pg_series_remainder(double h, double r, int K) {
  const double a2 = r * r / PI2;
  PgCumulants rest;
  if (K >= 10) {
    rest = pg_tail_cumulants(K, a2);
  } else {
    rest = pg_cumulants(r);
    for (int k = 1; k <= K; ++k) {
      const double km = k - 0.5;
      const double w = 1.0 / (2.0 * PI2 * (km * km + a2));
      rest.k1 -= w;
      rest.k2 -= w * w;
      rest.k3 -= 2.0 * w * w * w;
    }
  }
  const double psi = rest.k3 / (3.0 * rest.k2);
  const double mu = h * rest.k2 / psi;
  return {h * rest.k1 - mu, mu, psi};
}

inline double sample_pg_series(double h, double z, Rng& rng) {
  const double r = 0.5 * std::fabs(z);
  const double a2 = r * r / PI2;
  const int K = pg_series_terms(h);
  const bool boost = h < 1.0;  // Gamma(h) = Gamma(h + 1) exp(-E / h) for E ~ Exp(1)
  const GammaShape gamma(boost ? h + 1.0 : h);
  const double inv_h = 1.0 / h;
  double omega = 0.0;
  for (int k = 1; k <= K; ++k) {
    const double km = k - 0.5;
    double g = gamma.draw(rng);
    if (boost) g *= std::exp(-rng.expon() * inv_h);
    omega += g / (2.0 * PI2 * (km * km + a2));
  }
  const PgRemainder rest = pg_series_remainder(h, r, K);
  return omega + rest.shift + rest.mu * inverse_gaussian_ratio(rest.psi / rest.mu, rng);
}

// Large |z|: PG(h, z) is inverse Gaussian.  With r = |z| / 2 and r' = sqrt(r^2 - w / 2),
// the cumulant generating function of PG(h, z) is
//
//   h (r - r') + h (log1p(exp(-2 r)) - log1p(exp(-2 r'))),
//
// and the first term alone is that of IG(h / (2 |z|), h^2 / 4).  The second is at most
// h exp(-|z|) for the Laplace transform, so past PG_IG_Z + log h the two differ by less
// than 3.1e-7, far below what the samplers it replaces achieve there.  IG(mu, lambda) is
// lambda IG(mu / lambda, 1).
constexpr double PG_IG_Z = 15.0;

inline double sample_pg_inverse_gaussian(double h, double az, Rng& rng) {
  return h / (2.0 * az) * inverse_gaussian_ratio(2.0 / (h * az), rng);
}

// PG(h, z): dispatch as described in the header.  PG is even in z, so the thresholds
// use |z|.  PG(0, z) is 0, and so is the limit of an infinite |z|.  Every sampler but
// the first therefore sees only moderate |z|.
inline double sample_pg(double h, double z, Rng& rng) {
  const double az = std::fabs(z);
  if (!(h >= 0.0) || std::isinf(h) || std::isnan(z)) return NOT_A_NUMBER;
  if (h == 0.0 || std::isinf(az)) return 0.0;
  if (az >= PG_IG_Z + std::log(std::max(h, 1.0)))
    return sample_pg_inverse_gaussian(h, az, rng);
  if (h == std::floor(h) && h <= 4.0) {
    const int m = static_cast<int>(h);
    double acc = 0.0;
    for (int i = 0; i < m; ++i) acc += sample_pg1(z, rng);
    return acc;
  }
  return sample_pg_series(h, z, rng);
}

// ---------------------------------------------------------------------------
// The Kolmogorov scale of the logistic (Stefanski 1991; Holmes & Held 2006).
//
// A standard logistic eps is the normal scale mixture eps | lam ~ N(0, lam^2),
// lam = 2K with K Kolmogorov distributed.  Given an observed eps = u, the
// variance v = lam^2 has the density
//
//   p(v | u)  ∝  v^{-1/2} exp(-u^2 / (2v)) f(v),
//   f(v)  =  sum_{j>=1} (-1)^{j+1} j^2 exp(-j^2 v / 2),
//
// f the density of (2K)^2.  Factor f(v) = exp(-v/2) S(v): the proposal
// v^{-1/2} exp(-u^2/(2v) - v/2) is GIG(1/2, 1, u^2), the reciprocal of an
// inverse Gaussian IG(1/|u|, 1), and the acceptance probability is S(v) <= 1,
// so the rejection sampler is exact.  S(v) is evaluated lazily by the squeeze of
// an alternating series: the j-series above for v > 4/3, and below it the
// Jacobi-transformed series
//
//   S(v) = sqrt(2) pi^{5/2} v^{-5/2} exp(v/2 - pi^2/(2v))
//          sum_{m odd} (m^2 - v/pi^2) exp(-(m^2 - 1) pi^2 / (2v)),
//
// whose partial sums interleave around S(v) when split term by term.  Both
// branches are needed: the j-series' partial sums are not bounds below v ~ 1.
// ---------------------------------------------------------------------------

constexpr double KS_SPLIT = 4.0 / 3.0;
const double KS_LOG_LEFT = 0.5 * std::log(2.0) + 2.5 * std::log(PI);

// Accept v with probability S(v), given U ~ U(0, 1).
inline bool ks_accept(double v, double U) {
  if (v > KS_SPLIT) {
    const double X = std::exp(-0.5 * v);
    double Z = 1.0;
    for (int j = 1;; j += 2) {
      const double a = static_cast<double>((j + 1) * (j + 1));
      const double ta = a * std::pow(X, a - 1.0);
      Z -= ta;
      if (Z > U) return true;
      const double b = static_cast<double>((j + 2) * (j + 2));
      const double tb = b * std::pow(X, b - 1.0);
      Z += tb;
      if (Z < U) return false;
      if (tb == 0.0) return Z > U;  // the series has converged
    }
  }
  const double H = KS_LOG_LEFT - 2.5 * std::log(v) - PI2 / (2.0 * v) + 0.5 * v;
  const double lU = std::log(U);
  const double X = std::exp(-PI2 / (2.0 * v));
  const double K = v / PI2;
  double Z = 1.0;
  for (int m = 1;; m += 2) {
    Z -= K * std::pow(X, static_cast<double>(m * m - 1));
    if (H + std::log(Z) > lU) return true;
    const double b = static_cast<double>((m + 2) * (m + 2));
    const double tb = b * std::pow(X, b - 1.0);
    Z += tb;
    if (H + std::log(Z) < lU) return false;
    if (tb == 0.0) return H + std::log(Z) > lU;  // the series has converged
  }
}

// Below this |u| the proposal is chi^2_1 (GIG(1/2, 1, 0)) and exp(-u^2/(2v))
// joins the acceptance; above it the IG proposal's 1/|u| mean stays finite.
constexpr double KS_SMALL_U = 1e-4;

// One draw of lam = sqrt(v) given eps = u.  Past KS_HUGE_U, v is |u| to double
// precision.
constexpr double KS_HUGE_U = 1e100;

inline double sample_ks_scale(double u, Rng& rng) {
  const double r = std::fabs(u);
  if (std::isnan(r)) return NOT_A_NUMBER;
  if (r > KS_HUGE_U) return std::sqrt(r);
  for (;;) {
    double v;
    if (r < KS_SMALL_U) {
      const double y = rng.normal();
      v = y * y;
      if (rng.unif() > std::exp(-0.5 * r * r / v)) continue;
    } else {
      // 1 / v ~ IG(1 / |u|, 1)
      v = r / inverse_gaussian_ratio(1.0 / r, rng);
    }
    if (ks_accept(v, rng.unif())) return std::sqrt(v);
  }
}

// ---------------------------------------------------------------------------
// The Chinese restaurant table count (Zhou & Carin 2015): the tables y customers
// occupy at concentration r,
//
//   L = sum_{n = 0}^{y - 1} Bernoulli(r / (r + n)),
//
// which augments the Negative-Binomial dispersion to a conjugate gamma update.
//
// While the probabilities are large, draw them one at a time.  Once they are small,
// thin: since they only fall, the next success of a Bernoulli(p_n) process from
// customer n is a candidate for any customer m >= n, kept with probability
// p_m / p_n = (r + n) / (r + m), and the geometric gap skips the customers between.
// ---------------------------------------------------------------------------
inline double sample_crt(double y, double r, Rng& rng) {
  if (!(y >= 0.0) || !(r > 0.0) || std::isinf(r) || y > 9007199254740992.0 ||
      y != std::floor(y))
    return NOT_A_NUMBER;
  double tables = 0.0;
  double n = 0.0;
  for (; n < y && r > 0.1 * (r + n); n += 1.0)
    if (rng.unif() * (r + n) < r) tables += 1.0;
  while (n < y) {
    const double m = n + std::floor(std::log(rng.unif()) / std::log1p(-r / (r + n)));
    if (m >= y) break;
    if (rng.unif() * (r + m) < r + n) tables += 1.0;
    n = m + 1.0;
  }
  return tables;
}

// ---------------------------------------------------------------------------
// The generalized inverse Gaussian: density proportional to
// x^(p - 1) exp(-(a x + b / x) / 2).
//
// 1 / GIG(p, a, b) is GIG(-p, b, a), so take p >= 0.  On the log scale, centred at
// the mode, x = (2 A / a) exp(y) and y has density proportional to exp(-D(y)), where
//
//   D(y) = A phi(y) + B phi(-y),   phi(y) = exp(y) - 1 - y,
//   A = (rho + p) / 2,   B = (rho - p) / 2 = a b / (4 A),   rho = sqrt(a b + p^2).
//
// D is convex with minimum 0 at y = 0 for every parameter, so the rejection sampler for
// log-concave densities applies uniformly (Devroye 2014): the envelope is 1 in the
// middle and, on either side, the tangent to -D where D reaches about 1.  It also covers
// the gamma and inverse-gamma limits, where B = 0 and x stays finite.
// ---------------------------------------------------------------------------
struct LogGig {
  double A, B;
  static double phi(double y) {
    return std::fabs(y) < 0.01 ? y * y * (0.5 + y * (1.0 / 6.0 + y / 24.0))
                               : std::expm1(y) - y;
  }
  double drop(double y) const {
    return A * phi(y) + (B > 0.0 ? B * phi(-y) : 0.0);
  }
  double slope(double y) const {
    return A * std::expm1(y) - (B > 0.0 ? B * std::expm1(-y) : 0.0);
  }
  // The distance along `sign` at which D reaches about 1, by Newton's method inside a
  // bracket.  D need only be near 1: the point tunes efficiency, not correctness.
  double reach(double sign) const {
    double lo = 0.0, hi = std::numeric_limits<double>::infinity();
    double x = 2.0 * std::asinh(std::sqrt(0.5 / (A + B)));
    for (int i = 0; i < 64; ++i) {
      const double excess = drop(sign * x) - 1.0;
      if (std::fabs(excess) < 0.25) break;
      (excess < 0.0 ? lo : hi) = x;
      double next = x - excess / (sign * slope(sign * x));
      if (!(next > lo && next < hi))
        next = std::isinf(hi) ? 2.0 * x : 0.5 * (lo + hi);
      x = next;
    }
    return x;
  }
  double draw(Rng& rng) const {
    const double t = reach(1.0), s = reach(-1.0);
    const double right = 1.0 / slope(t), left = -1.0 / slope(-s);
    const double t0 = std::max(t - drop(t) * right, 0.0);
    const double s0 = std::max(s - drop(-s) * left, 0.0);
    const double middle = t0 + s0;
    const double total = middle + right + left;
    for (;;) {
      const double u = rng.unif() * total;
      double y, envelope = 0.0;  // envelope is the log of the envelope at y
      if (u < middle) {
        y = u - s0;
      } else {
        const double e = rng.expon();
        y = u < middle + right ? t0 + right * e : -s0 - left * e;
        envelope = -e;
      }
      if (rng.expon() >= envelope + drop(y)) return y;
    }
  }
};

inline double sample_gig(double p, double a, double b, Rng& rng) {
  if (!std::isfinite(p) || !std::isfinite(a) || !std::isfinite(b) || !(a >= 0.0) ||
      !(b >= 0.0))
    return NOT_A_NUMBER;
  if ((p >= 0.0 && a == 0.0) || (p <= 0.0 && b == 0.0)) return NOT_A_NUMBER;
  const double q = std::fabs(p);
  const double omega = std::sqrt(a) * std::sqrt(b);
  const double A = 0.5 * std::hypot(omega, q) + 0.5 * q;
  const LogGig law{A, 0.5 * omega * (0.5 * omega / A)};
  const double y = law.draw(rng);
  return p >= 0.0 ? 2.0 * (A / a) * std::exp(y) : 0.5 * (b / A) * std::exp(-y);
}

// Each sampler as the scheduler sees it: N parameters per element.
struct DrawPg {
  static constexpr int N = 2;
  // The floor keeps omega usable as a precision; a NaN passes through it.
  static double draw(const double* a, Rng& rng) {
    return std::max(sample_pg(a[0], a[1], rng), 1e-12);
  }
};

struct DrawKs {
  static constexpr int N = 1;
  static double draw(const double* a, Rng& rng) {
    return sample_ks_scale(a[0], rng);
  }
};

struct DrawCrt {
  static constexpr int N = 2;
  static double draw(const double* a, Rng& rng) {
    return sample_crt(a[0], a[1], rng);
  }
};

struct DrawGig {
  static constexpr int N = 3;
  static double draw(const double* a, Rng& rng) {
    return sample_gig(a[0], a[1], a[2], rng);
  }
};

// ---------------------------------------------------------------------------
// Scheduling
//
// A call draws for `keys` PRNG keys, each owning `n` consecutive elements: one key
// for a plain call, one per row under vmap.  A key's elements form one stream if
// there are at most kBlock of them, and otherwise one stream per block.  Streams
// therefore depend on the key and on n alone, never on the thread count or on the
// keys alongside, so vmap(f)(keys)[i] equals f(keys[i]).
//
// Those streams are the tasks, and any free thread takes the next few, so chains
// parallelize exactly as one long array does.
// ---------------------------------------------------------------------------

// Elements per independently seeded block.
constexpr int64_t kBlock = int64_t{1} << 16;

// Worker threads: AUGAX_NUM_THREADS if set, else every hardware thread.
unsigned num_threads() {
  static const unsigned n = [] {
    if (const char* env = std::getenv("AUGAX_NUM_THREADS")) {
      long v = std::strtol(env, nullptr, 10);
      if (v > 0) return static_cast<unsigned>(v);
    }
    unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1u;
  }();
  return n;
}

// One 64-bit seed from all the words of a key, whatever their number.
inline uint64_t seed_from_key(const uint32_t* words, int64_t nwords) {
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  for (int64_t i = 0; i < nwords; ++i) {
    uint64_t x = s ^ words[i];
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    s = x ^ (x >> 31);
  }
  return s;
}

struct Claims {
  std::atomic<int64_t> next{0};
  std::atomic<int64_t> done{0};
};

// Run tasks from the shared counter until none are left.
template <class Task>
void work_off(Claims& claims, int64_t tasks, int64_t grain, const Task& task) {
  for (;;) {
    const int64_t lo = claims.next.fetch_add(grain, std::memory_order_relaxed);
    if (lo >= tasks) return;
    const int64_t hi = std::min(lo + grain, tasks);
    for (int64_t i = lo; i < hi; ++i) task(i);
    claims.done.fetch_add(hi - lo, std::memory_order_release);
  }
}

// Host callers have no XLA pool, so they get threads of their own.
struct OwnThreads {
  template <class Task>
  void operator()(int64_t tasks, int64_t grain, int64_t workers,
                  const Task& task) const {
    Claims claims;
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    try {
      for (int64_t w = 1; w < workers; ++w)
        pool.emplace_back([&] { work_off(claims, tasks, grain, task); });
    } catch (const std::system_error&) {
      // Out of threads: those that started, and this one, finish the work.
    }
    work_off(claims, tasks, grain, task);
    for (auto& th : pool) th.join();
  }
};

// XLA's intra-op pool, so several devices share one pool rather than each starting
// its own.  This thread may belong to that pool, so it must not wait on queued
// helpers: it works until no task is unclaimed and then waits only for tasks a
// helper has already claimed.  A helper that starts after this returns finds no
// tasks and never touches `task`, which is why only `claims` is kept alive for it.
struct XlaPool {
  ffi::ThreadPool& pool;
  template <class Task>
  void operator()(int64_t tasks, int64_t grain, int64_t workers,
                  const Task& task) const {
    auto claims = std::make_shared<Claims>();
    const int64_t helpers = std::min<int64_t>(workers, pool.num_threads()) - 1;
    for (int64_t w = 0; w < helpers; ++w)
      pool.Schedule([claims, tasks, grain, &task] {
        work_off(*claims, tasks, grain, task);
      });
    work_off(*claims, tasks, grain, task);
    while (claims->done.load(std::memory_order_acquire) < tasks)
      std::this_thread::yield();
  }
};

template <class Draw, class T, class Run>
void draw_all(const std::array<const T*, Draw::N>& params, const uint32_t* words,
              int64_t nwords, int64_t keys, int64_t total, T* out, Run run) {
  if (total == 0) return;
  const int64_t n = total / keys;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  auto task = [&](int64_t t) {
    const int64_t key = t / blocks;
    const int64_t block = t % blocks;
    uint64_t seed = seed_from_key(words + key * nwords, nwords);
    if (blocks > 1)
      seed ^= 0xD1B54A32D192ED03ULL * static_cast<uint64_t>(block + 1);
    Rng rng(seed);
    const int64_t lo = key * n + block * kBlock;
    const int64_t hi = lo + std::min(kBlock, n - block * kBlock);
    double a[Draw::N];
    for (int64_t i = lo; i < hi; ++i) {
      for (int j = 0; j < Draw::N; ++j) a[j] = static_cast<double>(params[j][i]);
      out[i] = static_cast<T>(Draw::draw(a, rng));
    }
  };
  const int64_t tasks = keys * blocks;
  const int64_t workers =
      std::min({static_cast<int64_t>(num_threads()), tasks, (total + kBlock - 1) / kBlock});
  // Claim about a block's worth of elements at a time.
  const int64_t grain = std::max<int64_t>(kBlock / n, 1);
  if (workers <= 1) {
    for (int64_t t = 0; t < tasks; ++t) task(t);
    return;
  }
  run(tasks, grain, workers, task);
}

}  // namespace

// ---------------------------------------------------------------------------
// FFI handlers
//
//   <sampler>_<dtype>(seed[..., W], param[...], ...) -> draw[...]
//
// seed holds the words of the PRNG keys, W per key.  Any leading dimensions it has
// are vmap's, and the parameters carry the same ones, so one handler serves a plain
// call and every depth of vmap: the elements of one key are consecutive in memory.
// ---------------------------------------------------------------------------

using Seed = ffi::Buffer<ffi::U32>;

template <class Draw, ffi::DataType DT, class... Params>
static ffi::Error handle(Seed seed, ffi::ResultBuffer<DT>& out, ffi::ThreadPool& pool,
                         Params... params) {
  const auto dims = seed.dimensions();
  if (dims.size() == 0 || dims[dims.size() - 1] <= 0)
    return ffi::Error::InvalidArgument("augax: seed must end in the words of a key");
  const int64_t nwords = dims[dims.size() - 1];
  const int64_t keys = static_cast<int64_t>(seed.element_count()) / nwords;
  const int64_t total = static_cast<int64_t>(out->element_count());
  if (((static_cast<int64_t>(params.element_count()) != total) || ...))
    return ffi::Error::InvalidArgument("augax: parameters must share one shape");
  if (keys == 0) return ffi::Error::Success();
  if (total % keys != 0)
    return ffi::Error::InvalidArgument("augax: elements do not divide among the keys");
  using T = std::remove_const_t<std::remove_pointer_t<decltype(out->typed_data())>>;
  draw_all<Draw, T>({params.typed_data()...}, seed.typed_data(), nwords, keys, total,
                    out->typed_data(), XlaPool{pool});
  return ffi::Error::Success();
}

#define AUGAX_TARGET_1(NAME, DRAW, DT)                                          \
  static ffi::Error NAME##Impl(Seed seed, ffi::Buffer<DT> a,                    \
                               ffi::ResultBuffer<DT> out, ffi::ThreadPool pool) { \
    return handle<DRAW, DT>(seed, out, pool, a);                                \
  }                                                                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, NAME##Impl,                               \
                                ffi::Ffi::Bind()                                \
                                    .Arg<Seed>()                                \
                                    .Arg<ffi::Buffer<DT>>()                     \
                                    .Ret<ffi::Buffer<DT>>()                     \
                                    .Ctx<ffi::ThreadPool>());

#define AUGAX_TARGET_2(NAME, DRAW, DT)                                          \
  static ffi::Error NAME##Impl(Seed seed, ffi::Buffer<DT> a, ffi::Buffer<DT> b, \
                               ffi::ResultBuffer<DT> out, ffi::ThreadPool pool) { \
    return handle<DRAW, DT>(seed, out, pool, a, b);                             \
  }                                                                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, NAME##Impl,                               \
                                ffi::Ffi::Bind()                                \
                                    .Arg<Seed>()                                \
                                    .Arg<ffi::Buffer<DT>>()                     \
                                    .Arg<ffi::Buffer<DT>>()                     \
                                    .Ret<ffi::Buffer<DT>>()                     \
                                    .Ctx<ffi::ThreadPool>());

#define AUGAX_TARGET_3(NAME, DRAW, DT)                                          \
  static ffi::Error NAME##Impl(Seed seed, ffi::Buffer<DT> a, ffi::Buffer<DT> b, \
                               ffi::Buffer<DT> c, ffi::ResultBuffer<DT> out,    \
                               ffi::ThreadPool pool) {                          \
    return handle<DRAW, DT>(seed, out, pool, a, b, c);                          \
  }                                                                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, NAME##Impl,                               \
                                ffi::Ffi::Bind()                                \
                                    .Arg<Seed>()                                \
                                    .Arg<ffi::Buffer<DT>>()                     \
                                    .Arg<ffi::Buffer<DT>>()                     \
                                    .Arg<ffi::Buffer<DT>>()                     \
                                    .Ret<ffi::Buffer<DT>>()                     \
                                    .Ctx<ffi::ThreadPool>());

AUGAX_TARGET_2(PgF64, DrawPg, ffi::F64)
AUGAX_TARGET_2(PgF32, DrawPg, ffi::F32)
AUGAX_TARGET_1(KsF64, DrawKs, ffi::F64)
AUGAX_TARGET_1(KsF32, DrawKs, ffi::F32)
AUGAX_TARGET_2(CrtF64, DrawCrt, ffi::F64)
AUGAX_TARGET_2(CrtF32, DrawCrt, ffi::F32)
AUGAX_TARGET_3(GigF64, DrawGig, ffi::F64)
AUGAX_TARGET_3(GigF32, DrawGig, ffi::F32)

// ---------------------------------------------------------------------------
// Host entry points: the same draws for NumPy callers, one row of seed per key.
// ---------------------------------------------------------------------------

using HostSeed = nb::ndarray<const uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using HostParam = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class Draw, class... Params>
static nb::ndarray<nb::numpy, double, nb::ndim<1>> host_draw(HostSeed seed,
                                                             Params... params) {
  const std::array<HostParam, Draw::N> all{params...};
  const int64_t total = static_cast<int64_t>(all[0].shape(0));
  const int64_t keys = static_cast<int64_t>(seed.shape(0));
  const int64_t nwords = static_cast<int64_t>(seed.shape(1));
  for (const auto& p : all)
    if (static_cast<int64_t>(p.shape(0)) != total)
      throw nb::value_error("augax: parameters must share one shape");
  if (nwords == 0 || keys == 0 || total % keys != 0)
    throw nb::value_error("augax: elements do not divide among the keys");
  double* out = new double[total > 0 ? total : 1];
  std::array<const double*, Draw::N> data;
  for (int j = 0; j < Draw::N; ++j) data[j] = all[j].data();
  {
    nb::gil_scoped_release release;
    draw_all<Draw, double>(data, seed.data(), nwords, keys, total, out, OwnThreads{});
  }
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, {static_cast<size_t>(total)},
                                                     owner);
}

NB_MODULE(_cpp, m) {
  m.doc() = "Augmentation draws as XLA FFI custom calls and for host callers";
  m.def("ffi_targets", [] {
    nb::dict targets;
#define AUGAX_EXPORT(KEY, NAME) targets[KEY] = nb::capsule(reinterpret_cast<void*>(NAME))
    AUGAX_EXPORT("pg_float64", PgF64);
    AUGAX_EXPORT("pg_float32", PgF32);
    AUGAX_EXPORT("ks_float64", KsF64);
    AUGAX_EXPORT("ks_float32", KsF32);
    AUGAX_EXPORT("crt_float64", CrtF64);
    AUGAX_EXPORT("crt_float32", CrtF32);
    AUGAX_EXPORT("gig_float64", GigF64);
    AUGAX_EXPORT("gig_float32", GigF32);
#undef AUGAX_EXPORT
    return targets;
  });
  m.def("pg_host", &host_draw<DrawPg, HostParam, HostParam>);
  // The numbers behind a sum-of-gammas draw, for the test that bounds its distance from
  // the exact law.
  m.def("pg_series_parameters", [](double h, double z) {
    const int K = pg_series_terms(h);
    const PgRemainder rest = pg_series_remainder(h, 0.5 * std::fabs(z), K);
    return nb::make_tuple(K, rest.shift, rest.mu, rest.psi);
  });
  m.def("ks_host", &host_draw<DrawKs, HostParam>);
  m.def("crt_host", &host_draw<DrawCrt, HostParam, HostParam>);
  m.def("gig_host", &host_draw<DrawGig, HostParam, HostParam, HostParam>);
}
