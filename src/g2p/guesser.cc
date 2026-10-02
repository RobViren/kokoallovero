// Unknown-word guesser: char-to-phone pre-LN transformer (tools/g2p/train.py), greedy, one thread. Vowel tokens carry
// their stress; Guess moves it back to the syllable onset by maximal onset, giving ag2p's lexicon dialect.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/g2p/guesser.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace kk::g2p {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
using DF = hn::ScalableTag<float>;
using VF = hn::Vec<DF>;

size_t Panel() { return 2 * hn::Lanes(DF()); }

HWY_INLINE VF Load16(const hwy::float16_t* p) { return hn::PromoteTo(DF(), hn::LoadU(hn::Rebind<hwy::float16_t, DF>(), p)); }

HWY_INLINE void Store(VF v, const float* bias, const float* res, bool relu, float* y, size_t n) {
  const DF d;
  v = hn::Add(v, hn::LoadU(d, bias));
  if (res) v = hn::Add(v, hn::LoadN(d, res, n));
  if (relu) v = hn::Max(v, hn::Zero(d));
  hn::StoreN(v, d, y, n);
}

// R rows of x against one 2N-wide panel of f16 weights. One row splits k into four phases to keep eight FMA chains.
template <int R>
HWY_INLINE void Tile(const hwy::float16_t* HWY_RESTRICT w, const float* HWY_RESTRICT bias, const float* HWY_RESTRICT x,
                     int in, int out, int col, float* HWY_RESTRICT y, const float* HWY_RESTRICT res, bool relu) {
  const DF d;
  const int N = int(hn::Lanes(d)), NR = 2 * N;
  constexpr int P = R == 1 ? 4 : 1;
  VF a[R * P][2];
  for (auto& r : a) r[0] = r[1] = hn::Zero(d);
  int k = 0;
  for (; k + P <= in; k += P, w += P * NR) {
    HWY_UNROLL(4)
    for (int j = 0; j < R * P; ++j) {
      const int r = R == 1 ? 0 : j, kk = R == 1 ? j : 0;
      const VF xv = hn::Set(d, x[r * in + k + kk]);
      a[j][0] = hn::MulAdd(xv, Load16(w + kk * NR), a[j][0]);
      a[j][1] = hn::MulAdd(xv, Load16(w + kk * NR + N), a[j][1]);
    }
  }
  for (; k < in; ++k, w += NR) {
    const VF xv = hn::Set(d, x[k]);
    a[0][0] = hn::MulAdd(xv, Load16(w), a[0][0]);
    a[0][1] = hn::MulAdd(xv, Load16(w + N), a[0][1]);
  }
  if constexpr (R == 1)
    for (int h = 0; h < 2; ++h) a[0][h] = hn::Add(hn::Add(a[0][h], a[1][h]), hn::Add(a[2][h], a[3][h]));
  for (int r = 0; r < R; ++r)
    for (int h = 0; h < 2; ++h) {
      const int c = col + h * N;
      Store(a[r][h], bias + c, res ? res + r * out + c : nullptr, relu, y + r * out + c,
            size_t(std::clamp(out - c, 0, N)));
    }
}

// y[n][out] = x[n][in] W + b (+ res) (relu), W packed as f16 [out/2N][in][2N]. Each panel stays in L1 across row tiles.
void Dense16(const hwy::float16_t* w, const float* bias, const float* x, int n, int in, int out, float* y,
             const float* res, bool relu) {
  const int NR = int(Panel());
  for (int col = 0; col < out; col += NR, w += size_t(in) * NR)
    for (int i = 0; i < n;) {
      const size_t o = size_t(i) * in, oy = size_t(i) * out;
      const float* ri = res ? res + oy : nullptr;
      switch (std::min(4, n - i)) {
        case 4: Tile<4>(w, bias, x + o, in, out, col, y + oy, ri, relu), i += 4; break;
        case 3: Tile<3>(w, bias, x + o, in, out, col, y + oy, ri, relu), i += 3; break;
        case 2: Tile<2>(w, bias, x + o, in, out, col, y + oy, ri, relu), i += 2; break;
        default: Tile<1>(w, bias, x + o, in, out, col, y + oy, ri, relu), i += 1; break;
      }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace kk::g2p
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace kk::g2p {
HWY_EXPORT(Panel);
HWY_EXPORT(Dense16);

struct Dense {
  hwy::AlignedFreeUniquePtr<hwy::float16_t[]> w;
  hwy::AlignedFreeUniquePtr<float[]> b;
  int in = 0, out = 0;
};

struct Guesser {
  struct Norm {
    std::vector<float> g, b;
  };
  struct Layer {
    Norm n1, n2, n3;
    Dense qkv, o, cq, ckv, co, f1, f2;
  };
  struct Phone {
    int kind = 0, stress = 0;
    std::string ipa;
  };
  int d = 0, heads = 0, ff = 0, max_src = 0, max_tgt = 0, tag = -1;
  int16_t char_id[256];
  std::vector<Phone> tgt;
  std::set<std::vector<int>> onsets;
  std::vector<float> src_emb, src_pos, tgt_emb, tgt_pos;
  std::vector<Layer> enc, dec;
  Norm enc_norm, dec_norm;
  Dense out;
};

Guesser* LoadGuesser(const unsigned char* data, size_t size);
std::string Guess(const Guesser&, const std::string& word);

namespace {

struct Reader {
  const unsigned char *p, *end;
  template <class T> T Get() {
    if (p + sizeof(T) > end) throw std::runtime_error("guesser: truncated");
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string Str(size_t n) {
    if (p + n > end) throw std::runtime_error("guesser: truncated");
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
  }
};

void LayerNorm(const Guesser::Norm& n, const float* x, float* y, int d) {
  float m = 0, v = 0;
  for (int i = 0; i < d; ++i) m += x[i];
  m /= d;
  for (int i = 0; i < d; ++i) v += (x[i] - m) * (x[i] - m);
  const float inv = 1.f / std::sqrt(v / d + 1e-5f);
  for (int i = 0; i < d; ++i) y[i] = (x[i] - m) * inv * n.g[i] + n.b[i];
}

void Lin(const Dense& g, const float* x, int n, float* y, const float* res = nullptr, bool relu = false) {
  HWY_DYNAMIC_DISPATCH(Dense16)(g.w.get(), g.b.get(), x, n, g.in, g.out, y, res, relu);
}

Dense Pack(int out, int in, const float* w, const float* b) {
  const int NR = int(HWY_DYNAMIC_DISPATCH(Panel)()), cols = (out + NR - 1) / NR * NR;
  Dense h{hwy::AllocateAligned<hwy::float16_t>(size_t(cols) * in), hwy::AllocateAligned<float>(cols), in, out};
  for (int o = 0; o < cols; ++o) {
    h.b[o] = o < out ? b[o] : 0.f;
    for (int i = 0; i < in; ++i)
      h.w[(size_t(o / NR) * in + i) * NR + o % NR] = hwy::F16FromF32(o < out ? w[size_t(o) * in + i] : 0.f);
  }
  return h;
}

// One query row against n keys/values (row strides ldk, ldv), all heads.
void Attend(const float* q, const float* k, size_t ldk, const float* v, size_t ldv, int n, int d, int heads,
            float* ctx, float* s) {
  const int hd = d / heads;
  const float scale = 1.f / std::sqrt(float(hd));
  for (int h = 0; h < heads; ++h) {
    const float* qh = q + h * hd;
    float mx = -INFINITY;
    for (int j = 0; j < n; ++j) {
      const float* kj = k + j * ldk + h * hd;
      float a = 0;
      for (int c = 0; c < hd; ++c) a += qh[c] * kj[c];
      s[j] = a * scale;
      mx = std::max(mx, s[j]);
    }
    float sum = 0;
    for (int j = 0; j < n; ++j) sum += s[j] = std::exp(s[j] - mx);
    float* o = ctx + h * hd;
    std::fill(o, o + hd, 0.f);
    for (int j = 0; j < n; ++j) {
      const float pj = s[j] / sum, *vj = v + j * ldv + h * hd;
      for (int c = 0; c < hd; ++c) o[c] += pj * vj[c];
    }
  }
}

}  // namespace

Guesser* LoadGuesser(const unsigned char* data, size_t size) {
  Reader r{data, data + size};
  if (r.Str(8) != "KKG2P001") throw std::runtime_error("guesser: bad magic");
  auto g = std::make_unique<Guesser>();
  int ne, nd;
  g->d = r.Get<int32_t>(), g->heads = r.Get<int32_t>(), ne = r.Get<int32_t>(), nd = r.Get<int32_t>();
  g->ff = r.Get<int32_t>(), g->max_src = r.Get<int32_t>(), g->max_tgt = r.Get<int32_t>(), g->tag = r.Get<int32_t>();
  std::fill(std::begin(g->char_id), std::end(g->char_id), -1);
  for (uint32_t i = 0, n = r.Get<uint32_t>(); i < n; ++i) {
    const std::string c = r.Str(r.Get<uint8_t>());
    if (c.size() == 1) g->char_id[uint8_t(c[0])] = int16_t(i);
  }
  g->tgt.resize(r.Get<uint32_t>());
  for (auto& t : g->tgt) {
    t.kind = r.Get<uint8_t>(), t.stress = r.Get<uint8_t>();
    t.ipa = r.Str(r.Get<uint8_t>());
  }
  for (uint32_t i = 0, n = r.Get<uint32_t>(); i < n; ++i) {
    std::vector<int> o(r.Get<uint8_t>());
    for (int& c : o) c = r.Get<uint8_t>();
    g->onsets.insert(o);
  }
  std::map<std::string, std::vector<float>> t;
  for (uint32_t i = 0, n = r.Get<uint32_t>(); i < n; ++i) {
    const std::string name = r.Str(r.Get<uint8_t>());
    size_t count = 1;
    for (int k = 0, nd = r.Get<uint8_t>(); k < nd; ++k) count *= size_t(r.Get<int32_t>());
    auto& v = t[name];
    v.resize(count);
    if (r.p + count * 2 > r.end) throw std::runtime_error("guesser: truncated");
    for (size_t k = 0; k < count; ++k, r.p += 2) {
      hwy::float16_t h;
      std::memcpy(&h, r.p, 2);
      v[k] = hwy::F32FromF16(h);
    }
  }
  const int d = g->d;
  auto T = [&](const std::string& n) -> std::vector<float>& {
    auto it = t.find(n);
    if (it == t.end()) throw std::runtime_error("guesser: missing " + n);
    return it->second;
  };
  auto lin = [&](const std::string& p) {
    auto &w = T(p + ".weight"), &b = T(p + ".bias");
    return Pack(int(b.size()), int(w.size() / b.size()), w.data(), b.data());
  };
  auto cat = [&](const std::vector<std::string>& ps) {
    std::vector<float> w, b;
    for (auto& p : ps) {
      w.insert(w.end(), T(p + ".weight").begin(), T(p + ".weight").end());
      b.insert(b.end(), T(p + ".bias").begin(), T(p + ".bias").end());
    }
    return Pack(int(b.size()), d, w.data(), b.data());
  };
  auto norm = [&](const std::string& p) { return Guesser::Norm{T(p + ".weight"), T(p + ".bias")}; };
  const float scale = std::sqrt(float(d));
  g->src_emb = T("src_emb.weight"), g->tgt_emb = T("tgt_emb.weight");
  for (float& x : g->src_emb) x *= scale;
  for (float& x : g->tgt_emb) x *= scale;
  g->src_pos = T("src_pos.weight"), g->tgt_pos = T("tgt_pos.weight");
  for (int i = 0; i < ne + nd; ++i) {
    const bool is_dec = i >= ne;
    const std::string p = is_dec ? "dec." + std::to_string(i - ne) : "enc." + std::to_string(i);
    Guesser::Layer L;
    L.n1 = norm(p + ".n1"), L.n3 = norm(p + ".n3");
    L.qkv = cat({p + ".sa.q", p + ".sa.k", p + ".sa.v"}), L.o = lin(p + ".sa.o");
    L.f1 = lin(p + ".f1"), L.f2 = lin(p + ".f2");
    if (is_dec) L.n2 = norm(p + ".n2"), L.cq = lin(p + ".ca.q"), L.ckv = cat({p + ".ca.k", p + ".ca.v"}), L.co = lin(p + ".ca.o");
    (is_dec ? g->dec : g->enc).push_back(std::move(L));
  }
  g->enc_norm = norm("enc_norm"), g->dec_norm = norm("dec_norm");
  g->out = lin("out");
  return g.release();
}

std::string Guess(const Guesser& g, const std::string& word) {
  std::vector<int> src;
  if (g.tag >= 0) src.push_back(g.tag);
  for (size_t i = 0; i < word.size() && int(src.size()) < g.max_src; ++i) {
    uint8_t c = uint8_t(word[i]);
    if (c == 0xE2 && i + 2 < word.size() && uint8_t(word[i + 1]) == 0x80 && uint8_t(word[i + 2]) == 0x99) c = '\'', i += 2;
    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    if (g.char_id[c] >= 0) src.push_back(g.char_id[c]);
  }
  const int S = int(src.size()), d = g.d, nd = int(g.dec.size());
  if (S == (g.tag >= 0)) return "";
  thread_local std::vector<float> buf;
  const int M = std::max(S, g.max_tgt + 1);
  buf.resize(size_t(M) * (8 * d + g.ff) + size_t(nd) * (2 * S + 2 * (g.max_tgt + 1)) * d + M);
  float *x = buf.data(), *y = x + M * d, *h = y + M * d, *qkv = h + M * d, *ctx = qkv + 3 * M * d, *f = ctx + M * d,
        *cross = f + size_t(M) * g.ff, *cache = cross + size_t(nd) * 2 * S * d,
        *s = cache + size_t(nd) * 2 * (g.max_tgt + 1) * d;
  for (int i = 0; i < S; ++i)
    for (int c = 0; c < d; ++c) x[i * d + c] = g.src_emb[src[i] * d + c] + g.src_pos[i * d + c];
  for (auto& L : g.enc) {
    for (int i = 0; i < S; ++i) LayerNorm(L.n1, x + i * d, h + i * d, d);
    Lin(L.qkv, h, S, qkv);
    for (int i = 0; i < S; ++i) Attend(qkv + i * 3 * d, qkv + d, 3 * d, qkv + 2 * d, 3 * d, S, d, g.heads, ctx + i * d, s);
    Lin(L.o, ctx, S, y, x);
    for (int i = 0; i < S; ++i) LayerNorm(L.n3, y + i * d, h + i * d, d);
    Lin(L.f1, h, S, f, nullptr, true);
    Lin(L.f2, f, S, x, y);
  }
  for (int i = 0; i < S; ++i) LayerNorm(g.enc_norm, x + i * d, h + i * d, d);
  for (int l = 0; l < nd; ++l) Lin(g.dec[l].ckv, h, S, cross + size_t(l) * 2 * S * d);
  std::vector<int> toks;
  int tok = 1;
  for (int t = 0; t < g.max_tgt; ++t) {
    for (int c = 0; c < d; ++c) x[c] = g.tgt_emb[tok * d + c] + g.tgt_pos[t * d + c];
    for (int l = 0; l < nd; ++l) {
      const auto& L = g.dec[l];
      float* kv = cache + size_t(l) * 2 * (g.max_tgt + 1) * d;
      LayerNorm(L.n1, x, h, d);
      Lin(L.qkv, h, 1, qkv);
      std::memcpy(kv + size_t(t) * 2 * d, qkv + d, 2 * d * sizeof(float));
      Attend(qkv, kv, 2 * d, kv + d, 2 * d, t + 1, d, g.heads, ctx, s);
      Lin(L.o, ctx, 1, y, x);
      LayerNorm(L.n2, y, h, d);
      Lin(L.cq, h, 1, qkv);
      const float* kc = cross + size_t(l) * 2 * S * d;
      Attend(qkv, kc, 2 * d, kc + d, 2 * d, S, d, g.heads, ctx, s);
      Lin(L.co, ctx, 1, x, y);
      LayerNorm(L.n3, x, h, d);
      Lin(L.f1, h, 1, f, nullptr, true);
      Lin(L.f2, f, 1, y, x);
      std::swap(x, y);
    }
    LayerNorm(g.dec_norm, x, h, d);
    Lin(g.out, h, 1, y);
    tok = int(std::max_element(y, y + g.out.out) - y);
    if (g.tgt[tok].kind == 0) break;
    toks.push_back(tok);
  }
  std::vector<int> ph;  // phone ids, -1 primary, -2 secondary
  for (int k : toks) {
    if (const int st = g.tgt[k].stress) {
      size_t j = ph.size();
      while (j > 0 && ph[j - 1] >= 0 && g.tgt[ph[j - 1]].kind == 1) --j;
      size_t on = j;
      if (j > 0) {
        on = ph.size();
        for (size_t a = j; a < ph.size(); ++a)
          if (g.onsets.count(std::vector<int>(ph.begin() + a, ph.end()))) {
            on = a;
            break;
          }
      }
      ph.insert(ph.begin() + on, -st);
    }
    ph.push_back(k);
  }
  std::string out;
  for (int p : ph) out += p == -1 ? "ˈ" : p == -2 ? "ˌ" : g.tgt[p].ipa;
  return out;
}

}  // namespace kk::g2p
#endif
