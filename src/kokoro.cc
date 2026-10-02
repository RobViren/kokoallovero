#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>

#include "hwy/cache_control.h"
#include "src/kk.h"

namespace kk {
namespace {

using Vec = std::vector<float>;
using Vec64 = std::vector<double>;
constexpr float kPi = std::numbers::pi_v<float>;
constexpr int kStyle = 128, kHarmonics = 9, kNfft = 20, kHop = 5, kBins = kNfft / 2 + 1, kUpF0 = 300;
constexpr int kSampleRate = 24000;

struct Lin {
  const float *w = nullptr, *b = nullptr;
  int in = 0, out = 0;
};

struct Lstm {
  Gemm ih;  // both directions, [forward 4H | reverse 4H]
  const float *whh[2], *bhh[2];
  int H;
};

struct AdainResBlk1d {
  Lin norm1, norm2;
  Conv conv1, conv2, conv1x1;
  const float *pool_w = nullptr, *pool_b = nullptr;
  int cin, cout;
};

struct AdaINResBlock1 {
  Conv convs1[3], convs2[3];
  Lin adain1[3], adain2[3];
  const float *alpha1[3], *alpha2[3];
};

struct ConvT {  // stride s, kernel 2s, as s stride-1 two-tap convs, one per output phase
  std::vector<Conv> phase;
  int s, p, cout;
};

struct Strided {
  const float *w, *b;
  int cin, cout, K, stride, pad;
};

}  // namespace

struct Model {
  Weights w;
  const float *word, *pos, *tok, *ln_emb_g, *ln_emb_b, *ln_att_g, *ln_att_b, *ln_ffn_g, *ln_ffn_b;
  Gemm emb_in, qkv, dense, ffn, ffn_out, bert_encoder, duration_proj;
  Lstm dur_lstm[3], lstm, shared;
  Lin dur_norm[3];
  AdainResBlk1d F0[3], N[3];
  Conv F0_proj, N_proj;
  const float* text_emb;
  Conv text_cnn[3];
  const float *text_g[3], *text_b[3];
  Lstm text_lstm;
  Strided F0_conv, N_conv, noise_conv0;
  AdainResBlk1d encode, decode[4];
  Conv asr_res, noise_conv1, conv_post;
  ConvT ups[2];
  AdaINResBlock1 resblocks[6], noise_res[2];
  const float *l_w, *l_b;
  explicit Model(Weights w) : w(std::move(w)) {}
};

namespace {

bool Has(const Weights& w, const std::string& n) { return w.t.count(n); }

Lin MakeLin(const Weights& w, const std::string& p) {
  const auto& s = w.Shape(p + ".weight");
  return {w(p + ".weight"), Has(w, p + ".bias") ? w(p + ".bias") : nullptr, s[1], s[0]};
}

Gemm MakeGemm(const Weights& w, const std::string& p) {
  const auto& s = w.Shape(p + ".weight");
  return PackGemm(s[0], s[1], w(p + ".weight"), Has(w, p + ".bias") ? w(p + ".bias") : nullptr);
}

// Stacks same-shaped [out][in] linears along out.
Gemm StackGemm(const Weights& w, const std::vector<std::string>& ws, const std::vector<std::string>& bs) {
  const auto& s = w.Shape(ws[0]);
  const size_t size = size_t(s[0]) * s[1];
  Vec wt, b;
  for (const auto& n : ws) wt.insert(wt.end(), w(n), w(n) + size);
  for (const auto& n : bs) b.insert(b.end(), w(n), w(n) + s[0]);
  return PackGemm(s[0] * int(ws.size()), s[1], wt.data(), b.data());
}

Lstm MakeLstm(const Weights& w, const std::string& p) {
  Lstm m;
  m.ih = StackGemm(w, {p + ".weight_ih_l0", p + ".weight_ih_l0_reverse"}, {p + ".bias_ih_l0", p + ".bias_ih_l0_reverse"});
  for (int d = 0; d < 2; ++d) {
    const std::string sfx = d ? "_l0_reverse" : "_l0";
    m.whh[d] = w(p + ".weight_hh" + sfx);
    m.bhh[d] = w(p + ".bias_hh" + sfx);
  }
  m.H = m.ih.out / 8;
  return m;
}

Conv MakeConv(const Weights& w, const std::string& p, int dil = 1) {
  const auto& s = w.Shape(p + ".weight");
  const float* wp = w(p + ".weight");
  const int cin = s[1], K = s[2];
  return PackConv(s[0], cin, K, dil, Has(w, p + ".bias") ? w(p + ".bias") : nullptr,
                  [&](int co, int ci, int k) { return wp[(size_t(co) * cin + ci) * K + k]; });
}

ConvT MakeConvT(const Weights& w, const std::string& p, int s) {
  const auto& sh = w.Shape(p + ".weight");
  const float* wp = w(p + ".weight");
  const int cin = sh[0], cout = sh[1], K = sh[2];
  ConvT c{{}, s, (K - s) / 2, cout};
  for (int r = 0; r < s; ++r)
    c.phase.push_back(PackConv(cout, cin, 2, 1, w(p + ".bias"), [&](int co, int ci, int k) {
      return wp[(size_t(ci) * cout + co) * K + r + s * (1 - k)];
    }));
  return c;
}

Strided MakeStrided(const Weights& w, const std::string& p, int stride, int pad) {
  const auto& s = w.Shape(p + ".weight");
  return {w(p + ".weight"), w(p + ".bias"), s[1], s[0], s[2], stride, pad};
}

AdainResBlk1d MakeResBlk(const Weights& w, const std::string& p) {
  AdainResBlk1d b{MakeLin(w, p + ".norm1.fc"), MakeLin(w, p + ".norm2.fc"), MakeConv(w, p + ".conv1"),
           MakeConv(w, p + ".conv2")};
  b.cin = w.Shape(p + ".conv1.weight")[1];
  b.cout = w.Shape(p + ".conv1.weight")[0];
  if (Has(w, p + ".conv1x1.weight")) b.conv1x1 = MakeConv(w, p + ".conv1x1");
  if (Has(w, p + ".pool.weight")) b.pool_w = w(p + ".pool.weight"), b.pool_b = w(p + ".pool.bias");
  return b;
}

AdaINResBlock1 MakeResBlock1(const Weights& w, const std::string& p) {
  static constexpr int kDil[3] = {1, 3, 5};
  AdaINResBlock1 b;
  for (int j = 0; j < 3; ++j) {
    const std::string J = "." + std::to_string(j);
    b.convs1[j] = MakeConv(w, p + ".convs1" + J, kDil[j]);
    b.convs2[j] = MakeConv(w, p + ".convs2" + J);
    b.adain1[j] = MakeLin(w, p + ".adain1" + J + ".fc");
    b.adain2[j] = MakeLin(w, p + ".adain2" + J + ".fc");
    b.alpha1[j] = w(p + ".alpha1" + J);
    b.alpha2[j] = w(p + ".alpha2" + J);
  }
  static const std::string cfg = getenv("KK_WINO") ? getenv("KK_WINO") : "4,4,3";
  int wm[3] = {};
  std::sscanf(cfg.c_str(), "%d,%d,%d", &wm[0], &wm[1], &wm[2]);
  const int K = b.convs1[0].K, m = wm[K == 3 ? 0 : K == 7 ? 1 : 2];
  if (m)
    for (int j = 0; j < 3; ++j) EnableWinograd(b.convs1[j], m), EnableWinograd(b.convs2[j], m);
  return b;
}

Vec Run(Pool* pool, const Lin& l, const float* x, int n) {
  Vec y(size_t(n) * l.out);
  Linear(pool, x, n, l.in, l.w, l.b, y.data(), l.out);
  return y;
}

Vec Run(Pool* pool, const Lin& l, const Vec& x) { return Run(pool, l, x.data(), int(x.size() / l.in)); }

Vec Run(Pool& pool, const Gemm& g, const Vec& x) {
  const int n = int(x.size() / g.in);
  Vec y(size_t(n) * g.out);
  Matmul(pool, g, x.data(), n, y.data());
  return y;
}

struct Barrier {
  std::atomic<int> waiting{0}, gen{0};
  int n;
  explicit Barrier(int n) : n(n) {}
  void Wait() {
    const int g = gen.load(std::memory_order_acquire);
    if (waiting.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      waiting.store(0, std::memory_order_relaxed);
      gen.store(g + 1, std::memory_order_release);
    } else {
      for (int spins = 0; gen.load(std::memory_order_acquire) == g; ++spins)
        spins < 4096 ? hwy::Pause() : std::this_thread::yield();
    }
  }
};

float Sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

Vec Run(Pool& pool, const Lstm& m, const Vec& x) {
  const int H = m.H, n = int(x.size() / m.ih.in);
  Vec y(size_t(n) * 2 * H);
  const Vec g = Run(pool, m.ih, x);
  pool.For(2, [&](int d) {
    Vec h(H), c(H), r(4 * H);
    for (int s = 0; s < n; ++s) {
      const int t = d ? n - 1 - s : s;
      Linear(nullptr, h.data(), 1, H, m.whh[d], m.bhh[d], r.data(), 4 * H);
      const float* gt = g.data() + size_t(t) * 8 * H + d * 4 * H;
      for (int j = 0; j < H; ++j) {
        const float i = Sigmoid(gt[j] + r[j]), f = Sigmoid(gt[H + j] + r[H + j]);
        const float gg = std::tanh(gt[2 * H + j] + r[2 * H + j]), o = Sigmoid(gt[3 * H + j] + r[3 * H + j]);
        c[j] = f * c[j] + i * gg;
        h[j] = o * std::tanh(c[j]);
        y[size_t(t) * 2 * H + d * H + j] = h[j];
      }
    }
  });
  return y;
}

void LayerNorm(float* x, int C, const float* g, const float* b, float eps) {
  double mean = 0, var = 0;
  for (int c = 0; c < C; ++c) mean += x[c];
  mean /= C;
  for (int c = 0; c < C; ++c) var += (x[c] - mean) * (x[c] - mean);
  const float m = float(mean), inv = float(1 / std::sqrt(var / C + eps));
  for (int c = 0; c < C; ++c) x[c] = (x[c] - m) * inv * (g ? g[c] : 1.f) + (b ? b[c] : 0.f);
}

Act Copy(const Act& x) {
  Act y(x.C, x.T);
  std::copy_n(x.buf.get(), RoundUp(x.C, kM) * x.stride, y.buf.get());
  return y;
}

template <class F> void Map(Pool& pool, Act& x, F f) {
  pool.For(x.C, [&](int c) {
    float* r = x[c];
    for (int t = 0; t < x.T; ++t) r[t] = f(c, r[t]);
  });
}

void Leaky(Pool& pool, Act& x, float slope) { Map(pool, x, [=](int, float v) { return v < 0 ? v * slope : v; }); }

void Snake(Pool& pool, Act& x, const float* alpha) {
  Map(pool, x, [=](int c, float v) {
    const float s = std::sin(alpha[c] * v);
    return v + (1.f / alpha[c]) * (s * s);
  });
}

void AdaIN(Pool& pool, const Lin& fc, const float* style, const Act& x, Act& y) {
  const Vec gb = Run(nullptr, fc, style, 1);
  const int C = fc.out / 2;
  pool.For(C, [&](int c) {
    const float* r = x[c];
    double mean = 0, var = 0;
    for (int t = 0; t < x.T; ++t) mean += r[t];
    mean /= x.T;
    for (int t = 0; t < x.T; ++t) var += (r[t] - mean) * (r[t] - mean);
    const float m = float(mean), inv = float(1 / std::sqrt(var / x.T + 1e-5)), g = 1.f + gb[c], b = gb[C + c];
    float* o = y[c];
    for (int t = 0; t < x.T; ++t) o[t] = g * ((r[t] - m) * inv) + b;
  });
}

Act Run(Pool& pool, const Strided& s, const Act& x) {
  Act y(s.cout, (x.T + 2 * s.pad - s.K) / s.stride + 1);
  pool.For(s.cout, [&](int co) {
    float* yr = y[co];
    std::fill_n(yr, y.T, s.b[co]);
    for (int ci = 0; ci < s.cin; ++ci)
      for (int k = 0; k < s.K; ++k) {
        const float wv = s.w[(size_t(co) * s.cin + ci) * s.K + k];
        const float* xr = x[ci] + k - s.pad;
        for (int t = 0; t < y.T; ++t) yr[t] += wv * xr[t * s.stride];
      }
  });
  return y;
}

Act Run(Pool& pool, const ConvT& c, const Act& x) {
  Act z(c.cout, x.T + 1), y(c.cout, x.T * c.s);
  for (int r = 0; r < c.s; ++r) {
    ConvRaw(pool, c.phase[r], x[0] - 1, x.stride, z[0], z.stride, z.T);
    pool.For(c.cout, [&](int co) {
      for (int q = 0; q < z.T; ++q)
        if (const int o = q * c.s + r - c.p; o >= 0 && o < y.T) y[co][o] = z[co][q];
    });
  }
  return y;
}

Act Run(Pool& pool, const AdainResBlk1d& b, const float* style, const Act& x) {
  Act a(b.cin, x.T);
  AdaIN(pool, b.norm1, style, x, a);
  Leaky(pool, a, 0.2f);
  const int T = b.pool_w ? 2 * x.T : x.T;
  if (b.pool_w) {
    Act u(b.cin, T);
    pool.For(b.cin, [&](int c) {
      const float *w = b.pool_w + 3 * c, *r = a[c];
      for (int m = 0; m < x.T; ++m) {
        u[c][2 * m] = b.pool_b[c] + w[1] * r[m];
        u[c][2 * m + 1] = b.pool_b[c] + w[0] * r[m + 1] + w[2] * r[m];
      }
    });
    a = std::move(u);
  }
  Act h(b.cout, T), r(b.cout, T);
  Apply(pool, b.conv1, a, h);
  AdaIN(pool, b.norm2, style, h, r);
  Leaky(pool, r, 0.2f);
  Apply(pool, b.conv2, r, h);
  Act up;
  if (b.pool_w) {
    up = Act(b.cin, T);
    pool.For(b.cin, [&](int c) {
      for (int t = 0; t < T; ++t) up[c][t] = x[c][t / 2];
    });
  }
  const Act& sx = b.pool_w ? up : x;
  if (b.conv1x1.cout) Apply(pool, b.conv1x1, sx, r);
  const Act& sc = b.conv1x1.cout ? r : sx;
  const float k = float(std::sqrt(0.5));
  pool.For(b.cout, [&](int c) {
    for (int t = 0; t < T; ++t) h[c][t] = (h[c][t] + sc[c][t]) * k;
  });
  return h;
}

Moments RowMoments(Pool& pool, const Act& x) {
  Moments m{Vec64(x.C), Vec64(x.C)};
  pool.For(x.C, [&](int c) {
    for (int t = 0; t < x.T; ++t) m.sum[c] += x[c][t], m.sq[c] += double(x[c][t]) * x[c][t];
  });
  return m;
}

// InstanceNorm then (1 + gamma), beta, folded to y = a*x + b per channel.
void AdaINAffine(const Lin& fc, const float* style, const Moments& m, int T, Vec& a, Vec& b) {
  const Vec gb = Run(nullptr, fc, style, 1);
  const int C = fc.out / 2;
  a.resize(C), b.resize(C);
  for (int c = 0; c < C; ++c) {
    const double mean = m.sum[c] / T, inv = 1 / std::sqrt(std::max(m.sq[c] / T - mean * mean, 0.0) + 1e-5),
                 g = 1.0 + gb[c];
    a[c] = float(g * inv), b[c] = float(gb[C + c] - g * mean * inv);
  }
}

void Run(Pool& pool, const AdaINResBlock1& b, const float* style, Act& x) {
  Act u(x.C, x.T);
  Moments mx = RowMoments(pool, x), mu;
  Vec sa, sb;
  for (int j = 0; j < 3; ++j) {
    AdaINAffine(b.adain1[j], style, mx, x.T, sa, sb);
    ConvSnake(pool, b.convs1[j], x, sa.data(), sb.data(), b.alpha1[j], u, false, mu);
    AdaINAffine(b.adain2[j], style, mu, x.T, sa, sb);
    ConvSnake(pool, b.convs2[j], u, sa.data(), sb.data(), b.alpha2[j], x, true, mx);
  }
}

// F.interpolate(mode="linear") as torch's AVX2 build contracts it. The source phase reaches 1e5 rad, so the
// rounding of every step shows up in the harmonics.
Vec Interp(const Vec& x, int nout, float scale) {
  const int nin = int(x.size());
  Vec y(nout);
  for (int i = 0; i < nout; ++i) {
    const float src = std::max(std::fma(scale, float(i) + 0.5f, -0.5f), 0.f);
    const int i0 = std::min(int(src), nin - 1), i1 = i0 + (i0 < nin - 1);
    const float l = std::clamp(src - float(i0), 0.f, 1.f);
    y[i] = std::fma(x[i0], 1.f - l, x[i1] * l);
  }
  return y;
}

struct Twiddle {
  double c[kNfft], s[kNfft];
  Twiddle() {
    for (int j = 0; j < kNfft; ++j) {
      c[j] = std::cos(2 * std::numbers::pi * j / kNfft), s[j] = std::sin(2 * std::numbers::pi * j / kNfft);
      if (j % (kNfft / 4) == 0) c[j] = std::round(c[j]), s[j] = std::round(s[j]);
    }
  }
};
const Twiddle kTw;

const Vec& Hann() {
  static const Vec w = [] {
    Vec w(kNfft);
    for (int n = 0; n < kNfft; ++n) w[n] = float(0.5 - 0.5 * std::cos(2 * std::numbers::pi * n / kNfft));
    return w;
  }();
  return w;
}

Vec Source(Pool& pool, const Model& m, const Vec& f0, Noise& nz) {
  const int F = int(f0.size()), L = F * kUpF0;
  std::mt19937 rng(nz.seed);
  if (nz.initial_phase.empty()) {
    std::uniform_real_distribution<float> u;
    std::normal_distribution<float> g;
    nz.initial_phase.resize(kHarmonics);
    for (float& v : nz.initial_phase) v = u(rng);
    nz.randn.resize(size_t(L) * kHarmonics);
    for (float& v : nz.randn) v = g(rng);
  }
  std::vector<Vec> sines(kHarmonics);
  pool.For(kHarmonics, [&](int h) {
    Vec rad(L);
    for (int t = 0; t < L; ++t) {
      const float a = f0[t / kUpF0] * float(h + 1) / float(kSampleRate);
      rad[t] = a - std::floor(a);
    }
    if (h) rad[0] += nz.initial_phase[h];
    Vec ph = Interp(rad, F, float(kUpF0));
    double acc = 0;
    for (float& v : ph) v = float(acc += v) * 2.f * kPi * float(kUpF0);
    sines[h] = Interp(ph, L, float(1.0 / kUpF0));
    for (float& v : sines[h]) v = std::sin(v) * 0.1f;
  });
  Vec out(L);
  for (int t = 0; t < L; ++t) {
    const float uv = f0[t / kUpF0] > 10.f ? 1.f : 0.f, amp = uv * 0.003f + (1.f - uv) * 0.1f / 3.f;
    float s = m.l_b[0];
    for (int h = 0; h < kHarmonics; ++h) s += m.l_w[h] * (sines[h][t] * uv + amp * nz.randn[size_t(t) * kHarmonics + h]);
    out[t] = std::tanh(s);
  }
  return out;
}

Act Stft(Pool& pool, const Vec& x) {
  const int L = int(x.size()), F = L / kHop + 1;
  Act y(2 * kBins, F);
  const Vec& w = Hann();
  pool.For(F, [&](int f) {
    float fr[kNfft];
    for (int n = 0; n < kNfft; ++n) {
      int j = f * kHop + n - kNfft / 2;
      j = j < 0 ? -j : j >= L ? 2 * (L - 1) - j : j;
      fr[n] = w[n] * x[j];
    }
    for (int k = 0; k < kBins; ++k) {
      double re = 0, im = 0;
      for (int n = 0; n < kNfft; ++n) {
        re += fr[n] * kTw.c[k * n % kNfft];
        im -= fr[n] * kTw.s[k * n % kNfft];
      }
      y[k][f] = float(std::sqrt(re * re + im * im));
      y[kBins + k][f] = float(std::atan2(im, re));
    }
  });
  return y;
}

Vec Istft(const Act& x) {
  const int F = x.T, n = kHop * (F - 1) + kNfft;
  const Vec& w = Hann();
  Vec ola(n), env(n);
  for (int f = 0; f < F; ++f) {
    double re[kBins], im[kBins];
    for (int k = 0; k < kBins; ++k) {
      const float mag = std::exp(x[k][f]), ph = std::sin(x[kBins + k][f]);
      re[k] = mag * std::cos(ph);
      im[k] = mag * std::sin(ph);
    }
    for (int j = 0; j < kNfft; ++j) {
      double s = re[0] + (j % 2 ? -re[kBins - 1] : re[kBins - 1]);
      for (int k = 1; k < kBins - 1; ++k) {
        s += 2 * (re[k] * kTw.c[k * j % kNfft] - im[k] * kTw.s[k * j % kNfft]);
      }
      ola[f * kHop + j] += float(s / kNfft) * w[j];
      env[f * kHop + j] += w[j] * w[j];
    }
  }
  Vec y(kHop * (F - 1));
  for (size_t j = 0; j < y.size(); ++j) y[j] = ola[j + kNfft / 2] / env[j + kNfft / 2];
  return y;
}

Act Cat(std::initializer_list<const Act*> parts) {
  int C = 0;
  for (const Act* p : parts) C += p->C;
  Act y(C, (*parts.begin())->T);
  int c = 0;
  for (const Act* p : parts)
    for (int i = 0; i < p->C; ++i, ++c) std::copy_n((*p)[i], y.T, y[c]);
  return y;
}

Act Curve(const Vec& v) {
  Act a(1, int(v.size()));
  std::copy(v.begin(), v.end(), a[0]);
  return a;
}

Vec Row(const Act& a, int c) { return Vec(a[c], a[c] + a.T); }

Vec Bert(Pool& pool, const Model& m, const std::vector<int>& ids) {
  const int n = int(ids.size()), E = 128, D = m.dense.out, heads = 12;
  Vec e(size_t(n) * E);
  for (int t = 0; t < n; ++t) {
    float* r = e.data() + size_t(t) * E;
    for (int c = 0; c < E; ++c) r[c] = m.word[ids[t] * E + c] + m.tok[c] + m.pos[t * E + c];
    LayerNorm(r, E, m.ln_emb_g, m.ln_emb_b, 1e-12f);
  }
  Vec x(size_t(n) * D), qkv(size_t(n) * 3 * D), ctx(size_t(n) * D), a(size_t(n) * D), f(size_t(n) * m.ffn.out);
  Barrier bar(pool.n);
  pool.Run([&](int i) {
    const int P = pool.n;
    MatmulPart(m.emb_in, e.data(), n, x.data(), i, P);
    bar.Wait();
    for (int layer = 0; layer < 12; ++layer) {
      MatmulPart(m.qkv, x.data(), n, qkv.data(), i, P);
      bar.Wait();
      AttentionPart(qkv.data(), n, D, heads, ctx.data(), i, P);
      bar.Wait();
      MatmulPart(m.dense, ctx.data(), n, a.data(), i, P, x.data());
      bar.Wait();
      LayerNormPart(a.data(), n, D, m.ln_att_g, m.ln_att_b, 1e-12f, i, P);
      bar.Wait();
      MatmulPart(m.ffn, a.data(), n, f.data(), i, P, nullptr, true);
      bar.Wait();
      MatmulPart(m.ffn_out, f.data(), n, x.data(), i, P, a.data());
      bar.Wait();
      LayerNormPart(x.data(), n, D, m.ln_ffn_g, m.ln_ffn_b, 1e-12f, i, P);
      bar.Wait();
    }
  });
  return x;
}

Vec WithStyle(const Vec& x, int C, const float* style) {
  const int n = int(x.size() / C);
  Vec y(size_t(n) * (C + kStyle));
  for (int t = 0; t < n; ++t) {
    std::copy_n(x.data() + size_t(t) * C, C, y.data() + size_t(t) * (C + kStyle));
    std::copy_n(style, kStyle, y.data() + size_t(t) * (C + kStyle) + C);
  }
  return y;
}

Vec DurationEncoder(Pool& pool, const Model& m, const Vec& x, const float* prosody) {
  Vec d = WithStyle(x, 512, prosody);
  for (int l = 0; l < 3; ++l) {
    Vec h = Run(pool, m.dur_lstm[l], d);
    const Vec gb = Run(nullptr, m.dur_norm[l], prosody, 1);
    const int C = m.dur_norm[l].out / 2;
    for (size_t t = 0; t < h.size() / C; ++t) {
      float* r = h.data() + t * C;
      LayerNorm(r, C, nullptr, nullptr, 1e-5f);
      for (int c = 0; c < C; ++c) r[c] = (1.f + gb[c]) * r[c] + gb[C + c];
    }
    d = WithStyle(h, C, prosody);
  }
  return d;
}

Act TextEncoder(Pool& pool, const Model& m, const std::vector<int>& ids) {
  const int n = int(ids.size()), C = 512;
  Act x(C, n), y(C, n);
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < n; ++t) x[c][t] = m.text_emb[ids[t] * C + c];
  for (int l = 0; l < 3; ++l) {
    Apply(pool, m.text_cnn[l], x, y);
    for (int t = 0; t < n; ++t) {
      float col[512];
      for (int c = 0; c < C; ++c) col[c] = y[c][t];
      LayerNorm(col, C, m.text_g[l], m.text_b[l], 1e-5f);
      for (int c = 0; c < C; ++c) x[c][t] = col[c] < 0 ? 0.2f * col[c] : col[c];
    }
  }
  Vec xt(size_t(n) * C);
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < n; ++t) xt[size_t(t) * C + c] = x[c][t];
  const Vec h = Run(pool, m.text_lstm, xt);
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < n; ++t) x[c][t] = h[size_t(t) * C + c];
  return x;
}

}  // namespace

Model* LoadModel(const std::string& dir) { return LoadModel(Weights(dir + "/kokoro")); }

Model* LoadModel(Weights weights) {
  auto* m = new Model(std::move(weights));
  const Weights& w = m->w;
  const std::string B = "bert.", L = "bert.layer.", P = "predictor.", G = "decoder.generator.";
  m->word = w(B + "embeddings.word_embeddings.weight");
  m->pos = w(B + "embeddings.position_embeddings.weight");
  m->tok = w(B + "embeddings.token_type_embeddings.weight");
  m->ln_emb_g = w(B + "embeddings.LayerNorm.weight");
  m->ln_emb_b = w(B + "embeddings.LayerNorm.bias");
  m->emb_in = MakeGemm(w, B + "embedding_hidden_mapping_in");
  m->qkv = StackGemm(w, {L + "attention.query.weight", L + "attention.key.weight", L + "attention.value.weight"},
                     {L + "attention.query.bias", L + "attention.key.bias", L + "attention.value.bias"});
  m->dense = MakeGemm(w, L + "attention.dense");
  m->ln_att_g = w(L + "attention.LayerNorm.weight");
  m->ln_att_b = w(L + "attention.LayerNorm.bias");
  m->ffn = MakeGemm(w, L + "ffn");
  m->ffn_out = MakeGemm(w, L + "ffn_output");
  m->ln_ffn_g = w(L + "full_layer_layer_norm.weight");
  m->ln_ffn_b = w(L + "full_layer_layer_norm.bias");
  m->bert_encoder = MakeGemm(w, "bert_encoder");
  for (int l = 0; l < 3; ++l) {
    m->dur_lstm[l] = MakeLstm(w, P + "text_encoder.lstms." + std::to_string(2 * l));
    m->dur_norm[l] = MakeLin(w, P + "text_encoder.lstms." + std::to_string(2 * l + 1) + ".fc");
    m->F0[l] = MakeResBlk(w, P + "F0." + std::to_string(l));
    m->N[l] = MakeResBlk(w, P + "N." + std::to_string(l));
    m->text_cnn[l] = MakeConv(w, "text_encoder.cnn." + std::to_string(l) + ".0");
    m->text_g[l] = w("text_encoder.cnn." + std::to_string(l) + ".1.gamma");
    m->text_b[l] = w("text_encoder.cnn." + std::to_string(l) + ".1.beta");
  }
  m->lstm = MakeLstm(w, P + "lstm");
  m->shared = MakeLstm(w, P + "shared");
  m->duration_proj = MakeGemm(w, P + "duration_proj");
  m->F0_proj = MakeConv(w, P + "F0_proj");
  m->N_proj = MakeConv(w, P + "N_proj");
  m->text_emb = w("text_encoder.embedding.weight");
  m->text_lstm = MakeLstm(w, "text_encoder.lstm");
  m->F0_conv = MakeStrided(w, "decoder.F0_conv", 2, 1);
  m->N_conv = MakeStrided(w, "decoder.N_conv", 2, 1);
  std::vector<std::thread> build;
  build.emplace_back([&] { m->encode = MakeResBlk(w, "decoder.encode"); });
  for (int i = 0; i < 4; ++i)
    build.emplace_back([&, i] { m->decode[i] = MakeResBlk(w, "decoder.decode." + std::to_string(i)); });
  m->asr_res = MakeConv(w, "decoder.asr_res.0");
  m->ups[0] = MakeConvT(w, G + "ups.0", 10);
  m->ups[1] = MakeConvT(w, G + "ups.1", 6);
  for (int i = 0; i < 6; ++i)
    build.emplace_back([&, i] { m->resblocks[i] = MakeResBlock1(w, G + "resblocks." + std::to_string(i)); });
  for (int i = 0; i < 2; ++i)
    build.emplace_back([&, i] { m->noise_res[i] = MakeResBlock1(w, G + "noise_res." + std::to_string(i)); });
  m->noise_conv0 = MakeStrided(w, G + "noise_convs.0", 6, 3);
  m->noise_conv1 = MakeConv(w, G + "noise_convs.1");
  m->conv_post = MakeConv(w, G + "conv_post");
  m->l_w = w(G + "m_source.l_linear.weight");
  m->l_b = w(G + "m_source.l_linear.bias");
  for (auto& t : build) t.join();
  return m;
}

std::vector<float> Synthesize(Model& m, Pool& pool, const std::vector<int>& ids, const float* style, float speed,
                              Noise& noise, Trace& tr) {
  const float *timbre = style, *prosody = style + kStyle;
  const int n = int(ids.size());
  const Vec bert = Bert(pool, m, ids);
  tr.Lap("bert");
  tr.Dump("bert", bert.data(), {n, m.dense.out});

  const Vec bert_enc = Run(pool, m.bert_encoder, bert);
  const Vec d = DurationEncoder(pool, m, bert_enc, prosody);
  const Vec logits = Run(pool, m.duration_proj, Run(pool, m.lstm, d));
  Vec dur(n), frames(n);
  int T = 0;
  for (int t = 0; t < n; ++t) {
    float s = 0;
    for (int j = 0; j < m.duration_proj.out; ++j) s += Sigmoid(logits[size_t(t) * m.duration_proj.out + j]);
    dur[t] = s / speed;
    T += int(frames[t] = std::max(1.f, std::nearbyint(dur[t])));
  }
  tr.Lap("duration");
  tr.Dump("bert_enc", bert_enc.data(), {n, 512});
  tr.Dump("dur_enc", d.data(), {n, 640});
  tr.Dump("durations", dur.data(), {n});
  tr.Dump("frames", frames.data(), {n});

  std::vector<int> src(T);
  for (int t = 0, j = 0; t < n; ++t)
    for (int k = 0; k < int(frames[t]); ++k) src[j++] = t;
  Vec en(size_t(T) * 640);
  for (int j = 0; j < T; ++j) std::copy_n(d.data() + size_t(src[j]) * 640, 640, en.data() + size_t(j) * 640);
  const Vec sh = Run(pool, m.shared, en);
  Act x(512, T);
  for (int c = 0; c < 512; ++c)
    for (int j = 0; j < T; ++j) x[c][j] = sh[size_t(j) * 512 + c];
  Vec curves[2];
  for (int k = 0; k < 2; ++k) {
    const AdainResBlk1d* blocks = k ? m.N : m.F0;
    Act y = Run(pool, blocks[0], prosody, x);
    y = Run(pool, blocks[1], prosody, y);
    y = Run(pool, blocks[2], prosody, y);
    Act p(1, y.T);
    Apply(pool, k ? m.N_proj : m.F0_proj, y, p);
    curves[k] = Row(p, 0);
  }
  const Vec &f0 = curves[0], &nc = curves[1];
  tr.Lap("f0n");
  tr.Dump("f0", f0.data(), {int(f0.size())});
  tr.Dump("n", nc.data(), {int(nc.size())});

  const Act te = TextEncoder(pool, m, ids);
  Act asr(512, T);
  for (int c = 0; c < 512; ++c)
    for (int j = 0; j < T; ++j) asr[c][j] = te[c][src[j]];
  tr.Lap("text_enc");
  tr.Dump("text_enc", te);

  const Act f0c = Run(pool, m.F0_conv, Curve(f0)), ncc = Run(pool, m.N_conv, Curve(nc));
  x = Run(pool, m.encode, timbre, Cat({&asr, &f0c, &ncc}));
  tr.Lap("dec.encode");
  tr.Dump("dec.encode", x);
  Act ar(64, T);
  Apply(pool, m.asr_res, asr, ar);
  for (int i = 0; i < 4; ++i) {
    x = Run(pool, m.decode[i], timbre, Cat({&x, &ar, &f0c, &ncc}));
    const std::string name = "dec.decode" + std::to_string(i);
    tr.Lap(name.c_str());
    tr.Dump(name, x);
  }

  const Vec source = Source(pool, m, f0, noise);
  const Act har = Stft(pool, source);
  tr.Lap("source");
  tr.Dump("source", source.data(), {int(source.size())});
  tr.Dump("har", har);

  for (int i = 0; i < 2; ++i) {
    const std::string I = std::to_string(i);
    Leaky(pool, x, 0.1f);
    x = Run(pool, m.ups[i], x);
    if (i == 1) {
      Act p(x.C, x.T + 1);
      pool.For(x.C, [&](int c) {
        p[c][0] = x[c][1];
        std::copy_n(x[c], x.T, p[c] + 1);
      });
      x = std::move(p);
    }
    tr.Lap(("gen.up" + I).c_str());
    tr.Dump("gen.up" + I, x);
    Act ns = i == 0 ? Run(pool, m.noise_conv0, har) : Act(x.C, x.T);
    if (i == 1) Apply(pool, m.noise_conv1, har, ns);
    tr.Dump("gen.noiseconv" + I, ns);
    Run(pool, m.noise_res[i], timbre, ns);
    pool.For(x.C, [&](int c) {
      for (int t = 0; t < x.T; ++t) x[c][t] += ns[c][t];
    });
    tr.Lap(("gen.noise" + I).c_str());
    tr.Dump("gen.noise" + I, ns);
    Act sum[3] = {Copy(x), Copy(x), Copy(x)};
    for (int j = 0; j < 3; ++j) Run(pool, m.resblocks[3 * i + j], timbre, sum[j]);
    pool.For(x.C, [&](int c) {
      for (int t = 0; t < x.T; ++t) x[c][t] = (sum[0][c][t] + sum[1][c][t] + sum[2][c][t]) / 3.f;
    });
    tr.Lap(("gen.res" + I).c_str());
    tr.Dump("gen.res" + I, x);
  }
  Leaky(pool, x, 0.01f);
  Act post(2 * kBins, x.T);
  Apply(pool, m.conv_post, x, post);
  tr.Lap("conv_post");
  tr.Dump("conv_post", post);
  Vec audio = Istft(post);
  tr.Lap("istft");
  tr.Dump("audio", audio.data(), {int(audio.size())});
  return audio;
}

}  // namespace kk
