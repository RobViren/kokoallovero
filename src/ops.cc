#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>

#include "src/kk.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/ops.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

#ifndef KK_OPS_CONSTS
#define KK_OPS_CONSTS
namespace kk {
constexpr int kNV = 3;  // vectors of time per microtile
constexpr size_t kTile = 240;
}
#endif

HWY_BEFORE_NAMESPACE();
namespace kk {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// x: row ci at x + ci*xs, sample t at [t] (caller guarantees halo). w: [Cout/kM][Cin][K][kM].
void ConvRange(const float* x, size_t xs, const float* w, const float* bias, float* y, size_t ys,
               int cin, int cout, int K, int dil, size_t t0, size_t t1, size_t tile) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d), step = kNV * N;
  const int pad = dil * (K - 1) / 2;
  for (size_t tt = t0; tt < t1; tt += tile) {
    const size_t te = std::min(tt + tile, t1);
    for (int cb = 0; cb < cout / kM; ++cb) {
      const float* wb = w + size_t(cb) * cin * K * kM;
      for (size_t t = tt; t < te; t += step) {
        auto a00 = hn::Set(d, bias[cb * kM + 0]), a01 = a00, a02 = a00;
        auto a10 = hn::Set(d, bias[cb * kM + 1]), a11 = a10, a12 = a10;
        auto a20 = hn::Set(d, bias[cb * kM + 2]), a21 = a20, a22 = a20;
        auto a30 = hn::Set(d, bias[cb * kM + 3]), a31 = a30, a32 = a30;
        const float* wp = wb;
        for (int ci = 0; ci < cin; ++ci) {
          const float* xr = x + ci * xs + t - pad;
          for (int k = 0; k < K; ++k, wp += kM) {
            const float* xp = xr + k * dil;
            const auto x0 = hn::LoadU(d, xp), x1 = hn::LoadU(d, xp + N), x2 = hn::LoadU(d, xp + 2 * N);
            auto wv = hn::Set(d, wp[0]);
            a00 = hn::MulAdd(wv, x0, a00); a01 = hn::MulAdd(wv, x1, a01); a02 = hn::MulAdd(wv, x2, a02);
            wv = hn::Set(d, wp[1]);
            a10 = hn::MulAdd(wv, x0, a10); a11 = hn::MulAdd(wv, x1, a11); a12 = hn::MulAdd(wv, x2, a12);
            wv = hn::Set(d, wp[2]);
            a20 = hn::MulAdd(wv, x0, a20); a21 = hn::MulAdd(wv, x1, a21); a22 = hn::MulAdd(wv, x2, a22);
            wv = hn::Set(d, wp[3]);
            a30 = hn::MulAdd(wv, x0, a30); a31 = hn::MulAdd(wv, x1, a31); a32 = hn::MulAdd(wv, x2, a32);
          }
        }
        float* yr = y + size_t(cb) * kM * ys + t;
        hn::StoreU(a00, d, yr); hn::StoreU(a01, d, yr + N); hn::StoreU(a02, d, yr + 2 * N); yr += ys;
        hn::StoreU(a10, d, yr); hn::StoreU(a11, d, yr + N); hn::StoreU(a12, d, yr + 2 * N); yr += ys;
        hn::StoreU(a20, d, yr); hn::StoreU(a21, d, yr + N); hn::StoreU(a22, d, yr + 2 * N); yr += ys;
        hn::StoreU(a30, d, yr); hn::StoreU(a31, d, yr + N); hn::StoreU(a32, d, yr + 2 * N);
      }
    }
  }
}

// y[t][o] = b[o] + w[o] . x[t] for o in [o0, o1); w is torch Linear layout [out][in].
void LinearRange(const float* x, int n, int in, const float* w, const float* b, float* y, int out, int o0, int o1) {
  const hn::ScalableTag<float> d;
  const int N = hn::Lanes(d);
  for (int o = o0; o < o1; ++o) {
    const float* wr = w + size_t(o) * in;
    for (int t = 0; t < n; ++t) {
      const float* xr = x + size_t(t) * in;
      auto a0 = hn::Zero(d), a1 = hn::Zero(d);
      int i = 0;
      for (; i + 2 * N <= in; i += 2 * N) {
        a0 = hn::MulAdd(hn::LoadU(d, wr + i), hn::LoadU(d, xr + i), a0);
        a1 = hn::MulAdd(hn::LoadU(d, wr + i + N), hn::LoadU(d, xr + i + N), a1);
      }
      float s = hn::ReduceSum(d, hn::Add(a0, a1));
      for (; i < in; ++i) s += wr[i] * xr[i];
      y[size_t(t) * out + o] = s + (b ? b[o] : 0.f);
    }
  }
}

// s row ci holds times [lo, hi): snake(a*x + b), zero outside [0, T) because torch zero-pads after the activation.
void SnakeTile(const float* x, size_t xs, const float* a, const float* b, const float* alpha, float* s, size_t ss,
               int cin, long lo, long hi, long T) {
  const hn::ScalableTag<float> d;
  const long N = hn::Lanes(d);
  for (int ci = 0; ci < cin; ++ci) {
    const float* xr = x + ci * xs;
    float* sr = s + ci * ss;
    const auto va = hn::Set(d, a[ci]), vb = hn::Set(d, b[ci]), al = hn::Set(d, alpha[ci]),
               ial = hn::Set(d, 1.f / alpha[ci]);
    for (long t = lo; t < hi; t += N) {
      const auto u = hn::MulAdd(va, hn::LoadU(d, xr + t), vb);
      const auto sn = hn::Sin(d, hn::Mul(al, u));
      hn::StoreU(hn::MulAdd(ial, hn::Mul(sn, sn), u), d, sr + (t - lo));
    }
    for (long t = lo; t < std::min(0L, hi); ++t) sr[t - lo] = 0.f;
    for (long t = std::max(T, lo); t < hi; ++t) sr[t - lo] = 0.f;
  }
}

// y[c][t] = (residual ? y[c][t] : 0) + o[c][t - t0] over [t0, t1), zero past T; per-row sum and sum of squares.
void ConvEpilogue(const float* o, size_t os, float* y, size_t ys, int cout, size_t t0, size_t t1, size_t T,
                  bool residual, double* sum, double* sq) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);
  for (int c = 0; c < cout; ++c) {
    const float* orow = o + c * os - t0;
    float* yr = y + c * ys;
    auto vs = hn::Zero(d), vq = hn::Zero(d);
    for (size_t t = t0; t < t1; t += N) {
      auto v = hn::LoadU(d, orow + t);
      if (residual) v = hn::Add(v, hn::LoadU(d, yr + t));
      if (t + N > T) v = hn::IfThenElseZero(hn::FirstN(d, T > t ? T - t : 0), v);
      hn::StoreU(v, d, yr + t);
      vs = hn::Add(vs, v);
      vq = hn::MulAdd(v, v, vq);
    }
    sum[c] += hn::ReduceSum(d, vs);
    sq[c] += hn::ReduceSum(d, vq);
  }
}

template <int A>
void WinoInputT(const float* s, size_t ss, int cin, const float* BT, int m, int d, int n, float* z, float* V,
                size_t vs, size_t cs) {
  const hn::ScalableTag<float> dd;
  const int N = hn::Lanes(dd), D = d * m, zlen = n + (d + d * A) / D + 2;
  for (int ci = 0; ci < cin; ++ci) {
    const float* row = s + ci * ss;
    for (int q = 0; q < D; ++q)
      for (int j = 0; j < zlen; ++j) z[q * zlen + j] = row[q + D * j];
    for (int p = 0; p < d; ++p) {
      const float* zp[A];
      for (int u = 0; u < A; ++u) zp[u] = z + ((p + d * u) % D) * zlen + (p + d * u) / D;
      float* vo = V + ci * cs + p * n;
      for (int i = 0; i < n; i += N) {
        hn::Vec<decltype(dd)> acc[A];
        for (int x = 0; x < A; ++x) acc[x] = hn::Zero(dd);
        for (int u = 0; u < A; ++u) {
          const auto xv = hn::LoadU(dd, zp[u] + i);
          for (int x = 0; x < A; ++x) acc[x] = hn::MulAdd(hn::Set(dd, BT[x * A + u]), xv, acc[x]);
        }
        for (int x = 0; x < A; ++x) hn::StoreU(acc[x], dd, vo + x * vs + i);
      }
    }
  }
}

template <int A>
void WinoOutputT(const float* M, size_t vs, size_t cs, int cout, const float* AT, int m, int d, int n,
                 const float* bias, float* o, size_t os) {
  const hn::ScalableTag<float> dd;
  const int N = hn::Lanes(dd), D = d * m;
  HWY_ALIGN float tmp[8 * 16];
  for (int co = 0; co < cout; ++co)
    for (int p = 0; p < d; ++p)
      for (int i = 0; i < n; i += N) {
        const float* mi = M + co * cs + p * n + i;
        hn::Vec<decltype(dd)> mv[A];
        for (int x = 0; x < A; ++x) mv[x] = hn::LoadU(dd, mi + x * vs);
        for (int k = 0; k < m; ++k) {
          auto r = hn::Set(dd, bias[co]);
          for (int x = 0; x < A; ++x) r = hn::MulAdd(hn::Set(dd, AT[k * A + x]), mv[x], r);
          hn::Store(r, dd, tmp + k * N);
        }
        float* orow = o + co * os + p + D * i;
        for (int l = 0; l < N; ++l)
          for (int k = 0; k < m; ++k) orow[D * l + d * k] = tmp[k * N + l];
      }
}

#define KK_WINO_SWITCH(F, A, ...)                         \
  switch (A) {                                             \
    case 6: return F<6>(__VA_ARGS__);                      \
    case 8: return F<8>(__VA_ARGS__);                      \
    case 10: return F<10>(__VA_ARGS__);                    \
    case 12: return F<12>(__VA_ARGS__);                    \
    case 13: return F<13>(__VA_ARGS__);                    \
    default: HWY_ABORT("winograd alpha %d", A);            \
  }

void WinoInput(const float* s, size_t ss, int cin, const float* BT, int A, int m, int d, int n, float* z, float* V,
               size_t vs, size_t cs) {
  KK_WINO_SWITCH(WinoInputT, A, s, ss, cin, BT, m, d, n, z, V, vs, cs)
}

void WinoOutput(const float* M, size_t vs, size_t cs, int cout, const float* AT, int A, int m, int d, int n,
                const float* bias, float* o, size_t os) {
  KK_WINO_SWITCH(WinoOutputT, A, M, vs, cs, cout, AT, m, d, n, bias, o, os)
}

size_t TimeStep() { return kNV * hn::Lanes(hn::ScalableTag<float>()); }

}  // namespace HWY_NAMESPACE
}  // namespace kk
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace kk {
HWY_EXPORT(ConvRange);
HWY_EXPORT(LinearRange);
HWY_EXPORT(TimeStep);
HWY_EXPORT(SnakeTile);
HWY_EXPORT(ConvEpilogue);
HWY_EXPORT(WinoInput);
HWY_EXPORT(WinoOutput);

static std::vector<int> FastCores() {
  cpu_set_t mask;
  sched_getaffinity(0, sizeof mask, &mask);
  struct Core {
    long freq;
    int cpu;
  };
  std::map<std::pair<int, int>, Core> cores;
  std::vector<int> spare;
  auto read = [](int cpu, const char* f) {
    std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + f);
    long v = 0;
    in >> v;
    return v;
  };
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) continue;
    const auto key = std::pair(int(read(cpu, "topology/physical_package_id")), int(read(cpu, "topology/core_id")));
    if (!cores.count(key)) cores[key] = {read(cpu, "cpufreq/cpuinfo_max_freq"), cpu};
    else spare.push_back(cpu);
  }
  std::vector<Core> order;
  for (auto& [k, c] : cores) order.push_back(c);
  std::stable_sort(order.begin(), order.end(), [](const Core& a, const Core& b) { return a.freq > b.freq; });
  std::vector<int> cpus;
  for (auto& c : order) cpus.push_back(c.cpu);
  cpus.insert(cpus.end(), spare.begin(), spare.end());
  return cpus;
}

static void PinTo(int cpu) {
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(cpu, &one);
  sched_setaffinity(0, sizeof one, &one);
}

Pool::Pool(int n) : n(n) {
  const std::vector<int> cpus = FastCores();
  auto cpu = [&](int i) { return cpus.empty() ? -1 : cpus[i % cpus.size()]; };
  if (cpu(0) >= 0) PinTo(cpu(0));
  for (int i = 1; i < n; ++i) workers_.emplace_back(&Pool::Work, this, i, cpu(i));
}

Pool::~Pool() {
  stop_ = true;
  gen_.fetch_add(1);
  gen_.notify_all();
  for (auto& t : workers_) t.join();
}

void Pool::Work(int i, int cpu) {
  if (cpu >= 0) PinTo(cpu);
  for (uint64_t seen = 0;;) {
    gen_.wait(seen);
    seen = gen_.load();
    if (stop_) return;
    call_(ctx_, i);
    if (pending_.fetch_sub(1) == 1) pending_.notify_one();
  }
}

size_t TimeStep() {
  static const size_t step = HWY_DYNAMIC_DISPATCH(TimeStep)();
  return step;
}

Act::Act(int C, int T) : C(C), T(T), stride(RoundUp(RoundUp(T, TimeStep()) + 2 * kHalo, 16)) {
  const size_t n = RoundUp(C, kM) * stride;
  buf = hwy::AllocateAligned<float>(n);
  std::memset(buf.get(), 0, n * sizeof(float));
}

Conv PackConv(int cout, int cin, int K, int dil, const float* b, const std::function<float(int, int, int)>& w) {
  Conv c{hwy::AllocateAligned<float>(RoundUp(cout, kM) * cin * K), std::vector<float>(RoundUp(cout, kM)), cin, cout,
         K, dil};
  for (int co = 0; co < int(RoundUp(cout, kM)); ++co)
    for (int ci = 0; ci < cin; ++ci)
      for (int k = 0; k < K; ++k)
        c.w[((size_t(co / kM) * cin + ci) * K + k) * kM + co % kM] = co < cout ? w(co, ci, k) : 0.f;
  if (b) std::copy_n(b, cout, c.b.begin());
  return c;
}

void ConvRaw(Pool& pool, const Conv& c, const float* x, size_t xs, float* y, size_t ys, size_t T) {
  const size_t step = TimeStep(), chunks = RoundUp(T, step) / step;
  pool.Run([&](int i) {
    const size_t a = chunks * i / pool.n * step, e = chunks * (i + 1) / pool.n * step;
    HWY_DYNAMIC_DISPATCH(ConvRange)(x, xs, c.w.get(), c.b.data(), y, ys, c.cin, int(RoundUp(c.cout, kM)), c.K,
                                    c.dil, a, e, kTile);
  });
}

void Apply(Pool& pool, const Conv& c, const Act& x, Act& y) {
  ConvRaw(pool, c, x[0], x.stride, y[0], y.stride, x.T);
  const size_t e = RoundUp(x.T, TimeStep());
  for (int co = 0; co < int(RoundUp(c.cout, kM)); ++co) std::fill(y[co] + y.T, y[co] + e, 0.f);
}

void EnableWinograd(Conv& c, int m) {
  static const double kPts[] = {0, 1, -1, 2, -2, 0.5, -0.5, 3, -3, 1. / 3, -1. / 3, 4, -4};
  const int r = c.K, A = m + r - 1;
  auto pw = [](double x, int e) { return e == 0 ? 1.0 : std::pow(x, e); };
  auto row = [&](int j, int i, int len) { return j == A - 1 ? double(i == len - 1) : pw(kPts[j], i); };
  std::vector<double> V(A * A), inv(A * A);
  for (int j = 0; j < A; ++j)
    for (int i = 0; i < A; ++i) V[j * A + i] = row(j, i, A), inv[j * A + i] = i == j;
  for (int col = 0; col < A; ++col) {
    int piv = col;
    for (int j = col + 1; j < A; ++j)
      if (std::abs(V[j * A + col]) > std::abs(V[piv * A + col])) piv = j;
    for (int i = 0; i < A; ++i) std::swap(V[col * A + i], V[piv * A + i]), std::swap(inv[col * A + i], inv[piv * A + i]);
    const double f = 1 / V[col * A + col];
    for (int i = 0; i < A; ++i) V[col * A + i] *= f, inv[col * A + i] *= f;
    for (int j = 0; j < A; ++j)
      if (j != col && V[j * A + col] != 0) {
        const double g = V[j * A + col];
        for (int i = 0; i < A; ++i) V[j * A + i] -= g * V[col * A + i], inv[j * A + i] -= g * inv[col * A + i];
      }
  }
  c.wm = m;
  c.AT.resize(m * A), c.BT.resize(A * A);
  for (int k = 0; k < m; ++k)
    for (int x = 0; x < A; ++x) c.AT[k * A + x] = float(row(x, k, m));
  for (int x = 0; x < A; ++x)
    for (int u = 0; u < A; ++u) c.BT[x * A + u] = float(inv[u * A + x]);
  const int cp = int(RoundUp(c.cout, kM));
  std::vector<double> g(A * r);
  for (int x = 0; x < A; ++x)
    for (int k = 0; k < r; ++k) g[x * r + k] = row(x, k, r);
  c.U.clear();
  for (int x = 0; x < A; ++x) {
    Conv u{hwy::AllocateAligned<float>(size_t(cp) * c.cin), std::vector<float>(cp), c.cin, c.cout, 1, 1};
    for (size_t blk = 0; blk < size_t(cp / kM) * c.cin; ++blk)
      for (int lane = 0; lane < kM; ++lane) {
        double acc = 0;
        for (int k = 0; k < r; ++k) acc += g[x * r + k] * c.w[(blk * r + k) * kM + lane];
        u.w[blk * kM + lane] = float(acc);
      }
    c.U.push_back(std::move(u));
  }
}

static void ConvSnakeWino(Pool& pool, const Conv& c, const Act& x, const float* a, const float* b, const float* alpha,
                          Act& y, bool residual, Moments& mo) {
  static const int target_cols = getenv("KK_WCOLS") ? atoi(getenv("KK_WCOLS")) : 48;
  const int m = c.wm, A = int(c.U.size()), d = c.dil, D = d * m, pad = d * (c.K - 1) / 2;
  const int cout = int(RoundUp(c.cout, kM));
  int n = int(RoundUp((target_cols + d - 1) / d, 8));
  while (d * n % 24) n += 8;
  const int cols = d * n, L = D * n, zlen = n + (d + d * A) / D + 2;
  const size_t ss = RoundUp(std::max<size_t>(L + 2 * pad + 16, size_t(D) * (zlen + 1)), 16);
  const size_t Tp = RoundUp(x.T, TimeStep()), tiles = (Tp + L - 1) / L;
  const size_t vs = size_t(c.cin) * cols, ms = size_t(cout) * cols, os = L + 16;
  const std::vector<float> zero(cout, 0.f);
  std::vector<std::vector<double>> part(pool.n, std::vector<double>(2 * cout));
  pool.Run([&](int i) {
    auto s = hwy::AllocateAligned<float>(c.cin * ss);
    auto z = hwy::AllocateAligned<float>(D * zlen + 16);
    auto V = hwy::AllocateAligned<float>(A * vs);
    auto M = hwy::AllocateAligned<float>(A * ms);
    auto o = hwy::AllocateAligned<float>(cout * os);
    double *sum = part[i].data(), *sq = sum + cout;
    for (size_t j = tiles * i / pool.n, je = tiles * (i + 1) / pool.n; j < je; ++j) {
      const size_t tt = j * L, te = std::min(tt + L, Tp);
      const long lo = long(tt) - pad, hi = long(te) + pad;
      HWY_DYNAMIC_DISPATCH(SnakeTile)(x[0], x.stride, a, b, alpha, s.get(), ss, c.cin, lo, hi, x.T);
      for (int ci = 0; ci < c.cin; ++ci) std::fill(s.get() + ci * ss + (hi - lo), s.get() + (ci + 1) * ss, 0.f);
      HWY_DYNAMIC_DISPATCH(WinoInput)(s.get(), ss, c.cin, c.BT.data(), A, m, d, n, z.get(), V.get(), vs, cols);
      for (int xi = 0; xi < A; ++xi)
        HWY_DYNAMIC_DISPATCH(ConvRange)(V.get() + xi * vs, cols, c.U[xi].w.get(), zero.data(), M.get() + xi * ms,
                                        cols, c.cin, cout, 1, 1, 0, cols, cols);
      HWY_DYNAMIC_DISPATCH(WinoOutput)(M.get(), ms, cols, c.cout, c.AT.data(), A, m, d, n, c.b.data(), o.get(), os);
      HWY_DYNAMIC_DISPATCH(ConvEpilogue)(o.get(), os, y[0], y.stride, c.cout, tt, te, x.T, residual, sum, sq);
    }
  });
  mo.sum.assign(c.cout, 0.0);
  mo.sq.assign(c.cout, 0.0);
  for (auto& p : part)
    for (int co = 0; co < c.cout; ++co) mo.sum[co] += p[co], mo.sq[co] += p[cout + co];
}

void ConvSnake(Pool& pool, const Conv& c, const Act& x, const float* a, const float* b, const float* alpha, Act& y,
               bool residual, Moments& m) {
  if (c.wm) return ConvSnakeWino(pool, c, x, a, b, alpha, y, residual, m);
  const size_t step = TimeStep(), chunks = RoundUp(x.T, step) / step;
  const int pad = c.dil * (c.K - 1) / 2, cout = int(RoundUp(c.cout, kM));
  const size_t ss = RoundUp(kTile + 2 * pad + step, 16);
  std::vector<std::vector<double>> part(pool.n, std::vector<double>(2 * cout));
  pool.Run([&](int i) {
    auto s = hwy::AllocateAligned<float>(c.cin * ss);
    auto o = hwy::AllocateAligned<float>(cout * kTile);
    double *sum = part[i].data(), *sq = sum + cout;
    for (size_t tt = chunks * i / pool.n * step, e = chunks * (i + 1) / pool.n * step; tt < e; tt += kTile) {
      const size_t te = std::min(tt + kTile, e);
      HWY_DYNAMIC_DISPATCH(SnakeTile)(x[0], x.stride, a, b, alpha, s.get(), ss, c.cin, long(tt) - pad,
                                      long(te) + pad, x.T);
      HWY_DYNAMIC_DISPATCH(ConvRange)(s.get() + pad - ptrdiff_t(tt), ss, c.w.get(), c.b.data(),
                                      o.get() - ptrdiff_t(tt), kTile, c.cin, cout, c.K, c.dil, tt, te, kTile);
      HWY_DYNAMIC_DISPATCH(ConvEpilogue)(o.get(), kTile, y[0], y.stride, c.cout, tt, te, x.T, residual, sum, sq);
    }
  });
  m.sum.assign(c.cout, 0.0);
  m.sq.assign(c.cout, 0.0);
  for (auto& p : part)
    for (int co = 0; co < c.cout; ++co) m.sum[co] += p[co], m.sq[co] += p[cout + co];
}

void Linear(Pool* pool, const float* x, int n, int in, const float* w, const float* b, float* y, int out) {
  if (!pool) return HWY_DYNAMIC_DISPATCH(LinearRange)(x, n, in, w, b, y, out, 0, out);
  pool->Run([&](int i) {
    HWY_DYNAMIC_DISPATCH(LinearRange)(x, n, in, w, b, y, out, out * i / pool->n, out * (i + 1) / pool->n);
  });
}

Weights::Weights(const float* base, const std::string& index) {
  std::stringstream in(index);
  std::string name, shape;
  size_t off;
  while (in >> name >> off >> shape) {
    Tensor& e = t[name];
    e.p = base + off;
    std::stringstream ss(shape);
    for (std::string d; std::getline(ss, d, ',');) e.shape.push_back(std::stoi(d));
  }
}

static const float* MapFloats(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  struct stat st;
  fstat(fd, &st);
  const void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  return static_cast<const float*>(p);
}

static std::string ReadText(const std::string& path) {
  std::ifstream f(path);
  return {std::istreambuf_iterator<char>(f), {}};
}

Weights::Weights(const std::string& stem) : Weights(MapFloats(stem + ".bin"), ReadText(stem + ".tsv")) {}

void Trace::Lap(const char* stage) {
  const auto now = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(now - last).count();
  laps.emplace_back(stage, ms);
  if (!quiet) std::fprintf(stderr, "%-14s %8.1f\n", stage, ms);
  last = now;
}

void Trace::Dump(const std::string& name, const float* p, std::vector<int> shape) {
  if (dump_dir.empty()) return;
  WriteNpy(dump_dir + "/" + name + ".npy", p, shape);
  last = std::chrono::steady_clock::now();
}

void Trace::Dump(const std::string& name, const Act& a) {
  if (dump_dir.empty()) return;
  std::vector<float> v(size_t(a.C) * a.T);
  for (int c = 0; c < a.C; ++c) std::copy_n(a[c], a.T, v.data() + size_t(c) * a.T);
  Dump(name, v.data(), {a.C, a.T});
}

void WriteNpy(const std::string& path, const float* p, const std::vector<int>& shape) {
  std::string dims;
  size_t n = 1;
  for (int d : shape) dims += std::to_string(d) + ",", n *= d;
  if (shape.size() > 1) dims.pop_back();
  std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  h.append(63 - (10 + h.size()) % 64, ' ') += '\n';
  const uint16_t len = h.size();
  FILE* f = std::fopen(path.c_str(), "wb");
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&len, 2, 1, f);
  std::fwrite(h.data(), 1, h.size(), f);
  std::fwrite(p, 4, n, f);
  std::fclose(f);
}

std::vector<float> ReadNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  char magic[8];
  uint16_t len;
  f.read(magic, 8).read(reinterpret_cast<char*>(&len), 2);
  std::string h(len, ' ');
  f.read(h.data(), len);
  size_t n = 1;
  std::stringstream ss(h.substr(h.find('(') + 1, h.find(')') - h.find('(') - 1));
  for (std::string d; std::getline(ss, d, ',');)
    if (d.find_first_of("0123456789") != std::string::npos) n *= std::stoul(d);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * 4);
  return v;
}

}  // namespace kk
#endif
