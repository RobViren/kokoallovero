#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace kk::g2p {

struct Tagger {
  const uint32_t* slots;
  const uint32_t* entries;
  uint64_t mask;
  uint32_t n_entries;
  std::vector<std::string> tags;
  std::vector<uint32_t> owned;
};

Tagger* LoadTagger(const unsigned char* data, size_t size);
std::vector<std::string> Tag(const Tagger&, const std::vector<std::string>& tokens);

namespace {

constexpr uint32_t kMagic = 0x47544B4B, kVersion = 1;
constexpr uint8_t kSep = 0x1f;
enum : uint8_t { kBias, kWord, kSuf2, kSuf3, kSuf4, kPre1, kPre2, kShape, kPrev, kPrevSuf3, kPrevShape, kPrev2,
                 kNext, kNextSuf3, kNextShape, kNext2, kPrevWord, kWordNext, kP1 = 30, kP2, kP1P2, kP1W };

struct Hash {
  uint64_t h = 0xCBF29CE484222325ull;
  Hash& B(uint8_t x) { h = (h ^ x) * 0x100000001B3ull; return *this; }
  Hash& S(std::string_view s) { for (unsigned char c : s) B(c); return *this; }
};

std::string_view Suffix(std::string_view s, size_t n) { return s.substr(s.size() > n ? s.size() - n : 0); }

std::string Shape(std::string_view w) {
  std::string out;
  int last = -1, run = 0;
  for (unsigned char b : w) {
    int c = b >= 'A' && b <= 'Z' ? 'X' : b >= 'a' && b <= 'z' ? 'x' : b >= '0' && b <= '9' ? 'd' : b >= 128 ? 'u' : b;
    run = c == last ? run + 1 : 1;
    last = c;
    if (run <= 4) out.push_back(char(c));
  }
  return out;
}

std::string Lower(std::string_view w) {
  std::string out(w);
  for (char& c : out) if (c >= 'A' && c <= 'Z') c = char(c + 32);
  return out;
}

void Add(const Tagger& t, uint64_t h, int32_t* scores) {
  uint32_t fp = uint32_t(h >> 32);
  if (!fp) fp = 1;
  for (uint64_t s = h & t.mask;; s = (s + 1) & t.mask) {
    uint32_t key = t.slots[2 * s];
    if (!key) return;
    if (key != fp) continue;
    for (const uint32_t* e = t.entries + t.slots[2 * s + 1];; ++e) {
      scores[*e & 0xff] += int16_t(*e >> 16);
      if (*e & 0x100) return;
    }
  }
}

uint64_t F(uint8_t id, std::string_view v) { return Hash().B(id).B(kSep).S(v).h; }
uint64_t F(uint8_t id, std::string_view a, std::string_view b) { return Hash().B(id).B(kSep).S(a).B(kSep).S(b).h; }

}  // namespace

Tagger* LoadTagger(const unsigned char* data, size_t size) {
  uint32_t hdr[5];
  if (size < sizeof hdr) return nullptr;
  std::memcpy(hdr, data, sizeof hdr);
  auto [magic, version, ntags, cap, nent] = hdr;
  size_t need = sizeof hdr + 8ull * ntags + 8ull * cap + 4ull * nent;
  if (magic != kMagic || version != kVersion || ntags == 0 || ntags > 256 || (cap & (cap - 1)) || size < need)
    return nullptr;
  auto* t = new Tagger;
  const unsigned char* p = data + sizeof hdr;
  for (uint32_t i = 0; i < ntags; ++i, p += 8) t->tags.emplace_back(reinterpret_cast<const char*>(p), strnlen(reinterpret_cast<const char*>(p), 8));
  if (reinterpret_cast<uintptr_t>(p) % alignof(uint32_t)) {
    t->owned.resize(2ull * cap + nent);
    std::memcpy(t->owned.data(), p, 4 * t->owned.size());
    p = reinterpret_cast<const unsigned char*>(t->owned.data());
  }
  t->slots = reinterpret_cast<const uint32_t*>(p);
  t->entries = t->slots + 2ull * cap;
  t->mask = cap - 1;
  t->n_entries = nent;
  return t;
}

std::vector<std::string> Tag(const Tagger& t, const std::vector<std::string>& tokens) {
  const size_t n = tokens.size(), nt = t.tags.size();
  std::vector<std::string> lw(n + 4), sh(n + 4);
  lw[0] = "-S2-", lw[1] = "-S-", lw[n + 2] = "-E-", lw[n + 3] = "-E2-";
  for (size_t i = 0; i < 4; ++i) sh[i < 2 ? i : n + i] = Shape(lw[i < 2 ? i : n + i]);
  for (size_t i = 0; i < n; ++i) lw[i + 2] = Lower(tokens[i]), sh[i + 2] = Shape(tokens[i]);
  std::vector<std::string> out(n);
  std::string_view p1 = "-S-", p2 = "-S2-";
  std::vector<int32_t> scores(nt);
  for (size_t i = 0; i < n; ++i) {
    const size_t k = i + 2;
    std::string_view w = lw[k], p = lw[k - 1], nx = lw[k + 1];
    std::fill(scores.begin(), scores.end(), 0);
    const uint64_t hs[] = {
        F(kBias, ""), F(kWord, w), F(kSuf2, Suffix(w, 2)), F(kSuf3, Suffix(w, 3)), F(kSuf4, Suffix(w, 4)),
        F(kPre1, w.substr(0, 1)), F(kPre2, w.substr(0, 2)), F(kShape, sh[k]), F(kPrev, p),
        F(kPrevSuf3, Suffix(p, 3)), F(kPrevShape, sh[k - 1]), F(kPrev2, lw[k - 2]), F(kNext, nx),
        F(kNextSuf3, Suffix(nx, 3)), F(kNextShape, sh[k + 1]), F(kNext2, lw[k + 2]), F(kPrevWord, p, w),
        F(kWordNext, w, nx), F(kP1, p1), F(kP2, p2), F(kP1P2, p1, p2), F(kP1W, p1, w),
    };
    for (uint64_t h : hs) Add(t, h, scores.data());
    size_t best = 0;
    for (size_t c = 1; c < nt; ++c) if (scores[c] > scores[best]) best = c;
    out[i] = t.tags[best];
    p2 = p1, p1 = t.tags[best];
  }
  return out;
}

}  // namespace kk::g2p
