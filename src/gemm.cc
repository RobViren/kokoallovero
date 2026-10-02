#include <cmath>

#include "hwy/cache_control.h"
#include "src/kk.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/gemm.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

#ifndef KK_GEMM_CONSTS
#define KK_GEMM_CONSTS
namespace kk {
constexpr int kMR = 6;    // rows per microtile; with two vectors of columns that is 12 accumulators
// Depth per pass: a panel slice stays in L1 across all row tiles, and it bounds each sequential FMA chain. At 256
// the bert output drops from ~112 to ~108 dB SNR against torch.
constexpr int kKC = 128;
}
#endif

HWY_BEFORE_NAMESPACE();
namespace kk {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
using DF = hn::ScalableTag<float>;
using VF = hn::Vec<DF>;

size_t GemmPanel() { return 2 * hn::Lanes(DF()); }

HWY_INLINE VF Gelu(DF d, VF z) {
  const VF u = hn::Mul(hn::Set(d, 0.7978845608f), hn::MulAdd(hn::Mul(hn::Set(d, 0.044715f), z), hn::Mul(z, z), z));
  return hn::Mul(hn::Mul(hn::Set(d, 0.5f), z), hn::Add(hn::Set(d, 1.f), hn::Tanh(d, u)));
}

// c[r][0..cols) = (first ? bias : c[r]) + a[r][0..kc) * b[kc][2N]. last applies the epilogue.
template <int MR>
HWY_INLINE void Tile(const float* HWY_RESTRICT a, size_t lda, const float* HWY_RESTRICT b, size_t ldb, int kc,
                     float* HWY_RESTRICT c, size_t ldc, int cols, const float* HWY_RESTRICT bias, bool first,
                     bool last, const float* HWY_RESTRICT res, bool gelu) {
  const DF d;
  const int N = int(hn::Lanes(d));
  const bool full = cols == 2 * N;
  const size_t c0 = size_t(std::clamp(cols, 0, N)), c1 = size_t(std::clamp(cols - N, 0, N));
  VF acc[MR][2];
  HWY_UNROLL(kMR)
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = hn::Zero(d);
  for (int k = 0; k < kc; ++k, b += ldb) {
    const VF b0 = hn::LoadU(d, b), b1 = hn::LoadU(d, b + N);
    HWY_UNROLL(kMR)
    for (int r = 0; r < MR; ++r) {
      const VF av = hn::Set(d, a[r * lda + k]);
      acc[r][0] = hn::MulAdd(av, b0, acc[r][0]);
      acc[r][1] = hn::MulAdd(av, b1, acc[r][1]);
    }
  }
  HWY_UNROLL(kMR)
  for (int r = 0; r < MR; ++r) {
    VF v0 = acc[r][0], v1 = acc[r][1];
    if (first && bias) {
      v0 = hn::Add(v0, hn::Load(d, bias)), v1 = hn::Add(v1, hn::Load(d, bias + N));
    } else if (!first && full) {
      v0 = hn::Add(v0, hn::LoadU(d, c + r * ldc)), v1 = hn::Add(v1, hn::LoadU(d, c + r * ldc + N));
    } else if (!first) {
      v0 = hn::Add(v0, hn::LoadN(d, c + r * ldc, c0)), v1 = hn::Add(v1, hn::LoadN(d, c + r * ldc + N, c1));
    }
    if (last && gelu) v0 = Gelu(d, v0), v1 = Gelu(d, v1);
    if (last && res) {
      v0 = hn::Add(v0, hn::LoadN(d, res + r * ldc, c0));
      v1 = hn::Add(v1, hn::LoadN(d, res + r * ldc + N, c1));
    }
    if (full) {
      hn::StoreU(v0, d, c + r * ldc), hn::StoreU(v1, d, c + r * ldc + N);
    } else {
      hn::StoreN(v0, d, c + r * ldc, c0), hn::StoreN(v1, d, c + r * ldc + N, c1);
    }
  }
}

// Row tiles of a (lda) times one 2N-wide column panel of b (ldb), for all n rows. Spreads prefetches of the next
// kc x 2N slice of packed weights over the tiles, since the first tile of each slice otherwise stalls on L3.
void Panel(const float* a, size_t lda, int n, const float* b, size_t ldb, int kc, float* c, size_t ldc, int cols,
           const float* bias, bool first, bool last, const float* res, bool gelu, const float* next = nullptr) {
  const int tiles = (n + kMR - 1) / kMR, lines = int(kc * ldb / 16), per = (lines + tiles - 1) / tiles;
  for (int i = 0; i < n; i += kMR) {
    if (next)
      for (int l = i / kMR * per, e = std::min(lines, l + per); l < e; ++l) hwy::Prefetch(next + l * 16);
    const float* ai = a + i * lda;
    float* ci = c + i * ldc;
    const float* ri = res ? res + i * ldc : nullptr;
    switch (std::min(kMR, n - i)) {
      case 6: Tile<6>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu); break;
      case 5: Tile<5>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu); break;
      case 4: Tile<4>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu); break;
      case 3: Tile<3>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu); break;
      case 2: Tile<2>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu); break;
      default: Tile<1>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu); break;
    }
  }
}

// y[n][out] = x[n][in] * W + b (+ res), over column panels [p0, p1).
void GemmRange(const float* x, int n, int in, const float* w, const float* bias, float* y, int out, int p0, int p1,
               const float* res, bool gelu) {
  const int NR = int(GemmPanel());
  for (int p = p0; p < p1; ++p) {
    const int col = p * NR;
    const float* wp = w + size_t(p) * in * NR;
    for (int k0 = 0; k0 < in; k0 += kKC) {
      const int kc = std::min(kKC, in - k0);
      Panel(x + k0, in, n, wp + size_t(k0) * NR, NR, kc, y + col, out, std::min(NR, out - col), bias + col, k0 == 0,
            k0 + kc == in, res ? res + col : nullptr, gelu, wp + size_t(k0 + kc) * NR);
    }
  }
}

// qkv rows are [q | k | v], each D = heads * hd wide; hd must be a multiple of 2N.
void AttentionRange(const float* qkv, int n, int D, int heads, float* ctx, int h0, int h1) {
  const DF d;
  const int N = int(hn::Lanes(d)), hd = D / heads, np = int(RoundUp(n, 2 * N));
  const size_t ld = 3 * size_t(D);
  const VF scale = hn::Set(d, 1.f / std::sqrt(float(hd)));
  auto kt = hwy::AllocateAligned<float>(size_t(hd) * np);
  auto s = hwy::AllocateAligned<float>(size_t(n) * np);
  std::fill_n(kt.get(), size_t(hd) * np, 0.f);
  for (int h = h0; h < h1; ++h) {
    const float *q = qkv + h * hd, *k = qkv + D + h * hd, *v = qkv + 2 * D + h * hd;
    for (int j = 0; j < n; ++j)
      for (int c = 0; c < hd; ++c) kt[size_t(c) * np + j] = k[j * ld + c];
    for (int j = 0; j < np; j += 2 * N)
      Panel(q, ld, n, kt.get() + j, np, hd, s.get() + j, np, 2 * N, nullptr, true, true, nullptr, false);
    for (int i = 0; i < n; ++i) {
      float* p = s.get() + size_t(i) * np;
      std::fill(p + n, p + np, -INFINITY);
      VF mx = hn::Set(d, -INFINITY);
      for (int j = 0; j < np; j += N) mx = hn::Max(mx, hn::Load(d, p + j));
      mx = hn::Set(d, hn::ReduceMax(d, mx));
      for (int j = 0; j < np; j += N) hn::Store(hn::Exp(d, hn::Mul(hn::Sub(hn::Load(d, p + j), mx), scale)), d, p + j);
      std::fill(p + n, p + np, 0.f);
      VF sum = hn::Zero(d);
      for (int j = 0; j < np; j += N) sum = hn::Add(sum, hn::Load(d, p + j));
      const VF inv = hn::Set(d, 1.f / hn::ReduceSum(d, sum));
      for (int j = 0; j < np; j += N) hn::Store(hn::Mul(hn::Load(d, p + j), inv), d, p + j);
    }
    for (int c = 0; c < hd; c += 2 * N)
      Panel(s.get(), np, n, v + c, ld, n, ctx + h * hd + c, D, 2 * N, nullptr, true, true, nullptr, false);
  }
}

void LayerNormRange(float* x, int C, const float* g, const float* b, float eps, int r0, int r1) {
  const DF d;
  const int N = int(hn::Lanes(d));
  for (int t = r0; t < r1; ++t) {
    float* r = x + size_t(t) * C;
    VF s = hn::Zero(d);
    for (int c = 0; c < C; c += N) s = hn::Add(s, hn::LoadU(d, r + c));
    const VF m = hn::Set(d, hn::ReduceSum(d, s) / C);
    VF q = hn::Zero(d);
    for (int c = 0; c < C; c += N) {
      const VF z = hn::Sub(hn::LoadU(d, r + c), m);
      q = hn::MulAdd(z, z, q);
    }
    const VF inv = hn::Set(d, 1.f / std::sqrt(hn::ReduceSum(d, q) / C + eps));
    for (int c = 0; c < C; c += N) {
      const VF z = hn::Mul(hn::Sub(hn::LoadU(d, r + c), m), inv);
      hn::StoreU(hn::MulAdd(z, hn::LoadU(d, g + c), hn::LoadU(d, b + c)), d, r + c);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace kk
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace kk {
HWY_EXPORT(GemmPanel);
HWY_EXPORT(GemmRange);
HWY_EXPORT(AttentionRange);
HWY_EXPORT(LayerNormRange);

namespace {
int Part(int count, int part, int parts) { return int(int64_t(count) * part / parts); }
}  // namespace

Gemm PackGemm(int out, int in, const float* w, const float* b) {
  const int NR = int(HWY_DYNAMIC_DISPATCH(GemmPanel)()), cols = int(RoundUp(out, NR));
  Gemm g{hwy::AllocateAligned<float>(size_t(cols) * in), hwy::AllocateAligned<float>(cols), in, out};
  for (int o = 0; o < cols; ++o) {
    g.b[o] = b && o < out ? b[o] : 0.f;
    for (int i = 0; i < in; ++i) g.w[(size_t(o / NR) * in + i) * NR + o % NR] = o < out ? w[size_t(o) * in + i] : 0.f;
  }
  return g;
}

void MatmulPart(const Gemm& g, const float* x, int n, float* y, int part, int parts, const float* res, bool gelu) {
  const int NR = int(HWY_DYNAMIC_DISPATCH(GemmPanel)()), panels = (g.out + NR - 1) / NR;
  HWY_DYNAMIC_DISPATCH(GemmRange)(x, n, g.in, g.w.get(), g.b.get(), y, g.out, Part(panels, part, parts),
                                  Part(panels, part + 1, parts), res, gelu);
}

void Matmul(Pool& pool, const Gemm& g, const float* x, int n, float* y) {
  pool.Run([&](int i) { MatmulPart(g, x, n, y, i, pool.n); });
}

void AttentionPart(const float* qkv, int n, int D, int heads, float* ctx, int part, int parts) {
  HWY_DYNAMIC_DISPATCH(AttentionRange)(qkv, n, D, heads, ctx, Part(heads, part, parts), Part(heads, part + 1, parts));
}

void LayerNormPart(float* x, int n, int C, const float* g, const float* b, float eps, int part, int parts) {
  HWY_DYNAMIC_DISPATCH(LayerNormRange)(x, C, g, b, eps, Part(n, part, parts), Part(n, part + 1, parts));
}

}  // namespace kk
#endif
