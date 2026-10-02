#include "src/g2p/g2p.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace kk::g2p {
namespace {

using sv = std::string_view;
using Opt = std::optional<std::string>;

// ---- UTF-8 and Python str helpers

int CpLen(unsigned char c) { return c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4; }

char32_t Decode(sv s, size_t& i) {
  const unsigned char c = s[i];
  const int n = CpLen(c);
  char32_t cp = n == 1 ? c : c & (0x3F >> (n - 1));
  for (int k = 1; k < n; ++k) cp = cp << 6 | (s[i + k] & 0x3F);
  i += n;
  return cp;
}

void Append(std::string& o, char32_t c) {
  if (c < 0x80) {
    o += char(c);
  } else if (c < 0x800) {
    o += char(0xC0 | c >> 6), o += char(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    o += char(0xE0 | c >> 12), o += char(0x80 | (c >> 6 & 0x3F)), o += char(0x80 | (c & 0x3F));
  } else {
    o += char(0xF0 | c >> 18), o += char(0x80 | (c >> 12 & 0x3F)), o += char(0x80 | (c >> 6 & 0x3F));
    o += char(0x80 | (c & 0x3F));
  }
}

std::u32string U32(sv s) {
  std::u32string o;
  for (size_t i = 0; i < s.size();) o += Decode(s, i);
  return o;
}

std::string U8(std::u32string_view s) {
  std::string o;
  for (char32_t c : s) Append(o, c);
  return o;
}

size_t Len(sv s) {
  size_t n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

char32_t Last(sv s) {
  if (s.empty()) return 0;
  size_t i = s.size() - 1;
  while (i && (s[i] & 0xC0) == 0x80) --i;
  return Decode(s, i);
}

// Invalid bytes become U+FFFD so every regex call can skip PCRE2's UTF check.
std::string Sanitize(sv s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    int n = c < 0x80 ? 1 : c >= 0xC2 && c < 0xE0 ? 2 : c >= 0xE0 && c < 0xF0 ? 3 : c >= 0xF0 && c < 0xF5 ? 4 : 0;
    bool ok = n && i + n <= s.size();
    for (int k = 1; ok && k < n; ++k) ok = (s[i + k] & 0xC0) == 0x80;
    if (ok && n > 1) {
      size_t j = i;
      const char32_t cp = Decode(s, j);
      ok = cp >= (n == 2 ? 0x80u : n == 3 ? 0x800u : 0x10000u) && cp <= 0x10FFFF && (cp < 0xD800 || cp > 0xDFFF);
    }
    if (ok) o.append(s.substr(i, n)), i += n;
    else o += "\xEF\xBF\xBD", ++i;
  }
  return o;
}

bool IsSpace(char32_t c) {
  return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

sv Strip(sv s) {
  size_t b = 0, e = s.size();
  while (b < e) {
    size_t j = b;
    if (!IsSpace(Decode(s, j))) break;
    b = j;
  }
  while (e > b) {
    size_t i = e - 1;
    while (i > b && (s[i] & 0xC0) == 0x80) --i;
    size_t j = i;
    if (!IsSpace(Decode(s, j))) break;
    e = i;
  }
  return s.substr(b, e - b);
}

sv StripChars(sv s, sv chars) {
  const size_t b = s.find_first_not_of(chars);
  if (b == sv::npos) return {};
  return s.substr(b, s.find_last_not_of(chars) - b + 1);
}

std::string Replace(std::string s, sv a, sv b) {
  for (size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) s.replace(p, a.size(), b);
  return s;
}

bool Ends(sv s, sv x) { return s.size() >= x.size() && s.substr(s.size() - x.size()) == x; }
bool Starts(sv s, sv x) { return s.substr(0, x.size()) == x; }

bool Has(sv set, char32_t c) {
  for (size_t i = 0; i < set.size();)
    if (Decode(set, i) == c) return true;
  return false;
}

std::string Lower(sv s) {
  std::string o;
  for (size_t i = 0; i < s.size();) {
    char32_t c = Decode(s, i);
    if ((c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7)) c += 32;
    else if (c == 0x152) c = 0x153;
    Append(o, c);
  }
  return o;
}

std::string Upper(sv s) {
  std::string o(s);
  for (char& c : o)
    if (c >= 'a' && c <= 'z') c -= 32;
  return o;
}

bool IsUpper(sv s) {
  bool up = false;
  for (char c : s) {
    if (c >= 'a' && c <= 'z') return false;
    up |= c >= 'A' && c <= 'Z';
  }
  return up;
}

std::vector<sv> Split(sv s, char d) {
  std::vector<sv> o;
  for (size_t p = 0;;) {
    const size_t q = s.find(d, p);
    o.push_back(s.substr(p, q == sv::npos ? sv::npos : q - p));
    if (q == sv::npos) return o;
    p = q + 1;
  }
}

std::string Join(const std::vector<std::string>& v, sv d) {
  std::string o;
  for (size_t i = 0; i < v.size(); ++i) o += (i ? std::string(d) : "") + v[i];
  return o;
}

// ---- PCRE2 in Python re semantics (UTF + UCP, LF newlines)

pcre2_compile_context* CompileCtx() {
  static pcre2_compile_context* c = [] {
    auto* c = pcre2_compile_context_create(nullptr);
    pcre2_set_newline(c, PCRE2_NEWLINE_LF);
    return c;
  }();
  return c;
}

struct Re {
  pcre2_code* code;
  explicit Re(sv pat, uint32_t opt = 0) {
    int err;
    PCRE2_SIZE off;
    code = pcre2_compile((PCRE2_SPTR)pat.data(), pat.size(), PCRE2_UTF | PCRE2_UCP | opt, &err, &off, CompileCtx());
    if (!code) {
      PCRE2_UCHAR buf[256];
      pcre2_get_error_message(err, buf, sizeof buf);
      throw std::runtime_error("regex at " + std::to_string(off) + ": " + (char*)buf + " in " + std::string(pat.substr(0, 120)));
    }
    pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);
  }
};

Re Full(sv pat) { return Re("\\A(?:" + std::string(pat) + ")\\z"); }

struct M {
  pcre2_match_data* md;
  PCRE2_SIZE* ov = nullptr;
  int n = 0;
  sv s;
  explicit M(const Re& r) : md(pcre2_match_data_create_from_pattern(r.code, nullptr)) {}
  M(const M&) = delete;
  ~M() { pcre2_match_data_free(md); }
  bool Find(const Re& r, sv subj, size_t from = 0) {
    s = subj;
    n = pcre2_match(r.code, (PCRE2_SPTR)subj.data(), subj.size(), from, PCRE2_NO_UTF_CHECK, md, nullptr);
    ov = pcre2_get_ovector_pointer(md);
    return n > 0;
  }
  bool Set(int i) const { return i < n && ov[2 * i] != PCRE2_UNSET; }
  size_t B(int i = 0) const { return ov[2 * i]; }
  size_t E(int i = 0) const { return ov[2 * i + 1]; }
  sv G(int i = 0) const { return Set(i) ? s.substr(B(i), E(i) - B(i)) : sv(); }
};

bool Search(const Re& r, sv s) { return M(r).Find(r, s); }

template <class F> std::string Sub(const Re& r, sv s, F&& f) {
  std::string o;
  M m(r);
  size_t last = 0;
  while (last <= s.size() && m.Find(r, s, last)) {
    o.append(s.substr(last, m.B() - last));
    o += f(m);
    last = m.E();
    if (m.E() == m.B()) {
      if (last >= s.size()) break;
      const int k = CpLen(s[last]);
      o.append(s.substr(last, k));
      last += k;
    }
  }
  if (last < s.size()) o.append(s.substr(last));
  return o;
}

std::string SubT(const Re& r, sv s, sv tmpl) {
  return Sub(r, s, [&](const M& m) {
    std::string o;
    for (size_t i = 0; i < tmpl.size(); ++i)
      if (tmpl[i] == '\\' && i + 1 < tmpl.size() && tmpl[i + 1] >= '1' && tmpl[i + 1] <= '9') o += m.G(tmpl[++i] - '0');
      else o += tmpl[i];
    return o;
  });
}

template <class F> void FindIter(const Re& r, sv s, F&& f) {
  M m(r);
  for (size_t p = 0; p <= s.size() && m.Find(r, s, p);) {
    f(m);
    p = m.E() > m.B() ? m.E() : m.E() + (m.E() < s.size() ? CpLen(s[m.E()]) : 1);
  }
}

// ---- rules.py

const char* const kOnes[] = {"",        "one",     "two",       "three",    "four",     "five",    "six",
                             "seven",   "eight",   "nine",      "ten",      "eleven",   "twelve",  "thirteen",
                             "fourteen", "fifteen", "sixteen",  "seventeen", "eighteen", "nineteen"};
const char* const kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

// Python ints are unbounded and every scale is a power of ten, so divmod is a split of the digit string.
std::string IntWords(sv s) {
  while (!s.empty() && s[0] == '0') s.remove_prefix(1);
  if (s.empty()) return "zero";
  if (s.size() <= 2) {
    const int n = std::stoi(std::string(s));
    if (n < 20) return kOnes[n];
    return std::string(kTens[n / 10]) + (n % 10 ? std::string(" ") + kOnes[n % 10] : "");
  }
  auto rest = [](sv r) {
    while (!r.empty() && r[0] == '0') r.remove_prefix(1);
    return r.empty() ? std::string() : " " + IntWords(r);
  };
  if (s.size() == 3) return std::string(kOnes[s[0] - '0']) + " hundred" + rest(s.substr(1));
  static const std::pair<size_t, const char*> scales[] = {{12, "trillion"}, {9, "billion"}, {6, "million"}, {3, "thousand"}};
  for (auto [k, name] : scales)
    if (s.size() > k) return IntWords(s.substr(0, s.size() - k)) + " " + name + rest(s.substr(s.size() - k));
  return {};
}

std::string IntWords(long n) { return IntWords(std::to_string(n)); }

std::vector<std::string> DigitWords(sv s) {
  std::vector<std::string> o;
  for (char c : s)
    if (c >= '0' && c <= '9') o.push_back(c == '0' ? "zero" : kOnes[c - '0']);
  return o;
}

std::string OrdinalWords(sv s) {
  const std::string w = IntWords(s);
  const size_t sp = w.rfind(' ');
  const std::string tail = sp == std::string::npos ? w : w.substr(sp + 1);
  static const std::pair<sv, sv> ord[] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
                                          {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
  std::string rep;
  for (auto [a, b] : ord)
    if (tail == a) rep = b;
  if (rep.empty()) rep = Ends(tail, "y") ? tail.substr(0, tail.size() - 1) + "ieth" : tail + "th";
  return w.substr(0, w.size() - tail.size()) + rep;
}

std::string YearWords(int n) {
  const int hi = n / 100, lo = n % 100;
  if (lo == 0) return IntWords(hi) + " hundred";
  if (lo < 10) return IntWords(hi) + " oh " + kOnes[lo];
  return IntWords(hi) + " " + IntWords(lo);
}

using Lookup = std::function<Opt(const std::string&)>;

bool EndsAny(sv s, std::initializer_list<sv> xs) {
  for (sv x : xs)
    if (Ends(s, x)) return true;
  return false;
}
const std::initializer_list<sv> kSibilant = {"tʃ", "dʒ", "s", "z", "ʃ", "ʒ"};

std::string PluralSuffix(const std::string& b) {
  if (EndsAny(b, kSibilant)) return b + "ɪz";
  return b + (EndsAny(b, {"p", "t", "k", "f", "θ"}) ? "s" : "z");
}

std::string JoinCompound(const std::vector<std::string>& ipas, size_t head) {
  std::string o;
  for (size_t k = 0; k < ipas.size(); ++k) o += k == head ? ipas[k] : Replace(ipas[k], "ˈ", "ˌ");
  return o;
}

bool Parts(const std::string& word, const Lookup& lookup, size_t count, std::vector<sv>& parts, std::vector<std::string>& ipas) {
  if (word.find('-') == std::string::npos) return false;
  parts = Split(word, '-');
  for (sv p : parts)
    if (p.empty()) return false;
  if (count && parts.size() != count) return false;
  for (sv p : parts) {
    Opt i = lookup(std::string(p));
    if (!i) return false;
    ipas.push_back(*i);
  }
  return true;
}

Opt CompoundNumeral(const std::string& w, const Lookup& l) {
  static const std::unordered_set<sv> num = [] {
    std::unordered_set<sv> s = {"and", "hundred", "thousand", "million", "billion"};
    for (int i = 1; i < 20; ++i) s.insert(kOnes[i]);
    for (int i = 2; i < 10; ++i) s.insert(kTens[i]);
    return s;
  }();
  std::vector<sv> p;
  std::vector<std::string> i;
  if (!Parts(w, l, 0, p, i)) return {};
  for (sv x : p)
    if (!num.count(x)) return {};
  return JoinCompound(i, i.size() - 1);
}

Opt ParticipialCompound(const std::string& w, const Lookup& l) {
  std::vector<sv> p;
  std::vector<std::string> i;
  if (Parts(w, l, 2, p, i) && (Ends(p[1], "ing") || Ends(p[1], "ed"))) return JoinCompound(i, 1);
  return {};
}

Opt PrefixedCompound(const std::string& w, const Lookup& l) {
  static const std::unordered_set<sv> pre = {"anti", "non", "self", "post", "pre", "co", "auto", "bio", "neuro",
      "multi", "micro", "sub", "over", "under", "intra", "hypo", "hyper", "ultra", "pseudo", "immuno", "cardio",
      "radio", "electro", "photo", "thermo", "semi"};
  std::vector<sv> p;
  std::vector<std::string> i;
  if (Parts(w, l, 2, p, i) && pre.count(p[0])) return JoinCompound(i, 1);
  return {};
}

std::string Drop(const std::string& w, size_t n) { return w.substr(0, w.size() - n); }

Opt SuffixInflection(const std::string& w, const Lookup& l) {
  Opt base;
  if (Ends(w, "'s")) {
    base = l(Drop(w, 2));
  } else if (Ends(w, "es") && Len(w) > 2) {
    Opt drop_es = Len(w) >= 5 ? l(Drop(w, 2)) : Opt();
    if (drop_es && !drop_es->empty() && EndsAny(*drop_es, kSibilant)) {
      base = drop_es;
    } else {
      base = l(Drop(w, 1));
      if (!base && Ends(Drop(w, 2), "i")) base = l(Drop(w, 3) + "y");
    }
  } else if (Ends(w, "s") && Len(w) > 1) {
    base = l(Drop(w, 1));
  } else {
    return {};
  }
  return base && !base->empty() ? PluralSuffix(*base) : Opt();
}

Opt Base(const std::string& w, sv suffix, const Lookup& l) {
  if (!Ends(w, suffix)) return {};
  const std::string stem = Drop(w, suffix.size());
  std::vector<std::string> c = {stem};
  if (Has("eaiou", suffix[0])) c.push_back(stem + "e");
  if (Ends(stem, "i") && Len(stem) > 1) c.push_back(Drop(stem, 1) + "y");
  const size_t n = stem.size();
  if (Len(stem) > 2 && stem[n - 1] == stem[n - 2] && Has("bdgklmnprtvz", stem[n - 1])) c.push_back(Drop(stem, 1));
  for (auto& s : c)
    if (Opt i = l(s)) return i;
  return {};
}

template <class F> auto SuffixRule(sv suffix, F add) {
  return [=](const std::string& w, const Lookup& l) -> Opt {
    Opt b = Base(w, suffix, l);
    return b && !b->empty() ? Opt(add(*b)) : Opt();
  };
}

Opt PastEd(const std::string& w, const Lookup& l) {
  Opt b = Base(w, "ed", l);
  if (!b) return {};
  if (EndsAny(*b, {"t", "d"})) return *b + "ɪd";
  return *b + (EndsAny(*b, {"p", "t", "k", "f", "θ", "s", "ʃ", "tʃ"}) ? "t" : "d");
}

Opt NounNess(const std::string& w, const Lookup& l) {
  if (w == "business" || w == "businesses" || w == "wilderness" || w == "wildernesses") return {};
  Opt b = Base(w, "ness", l);
  return b && !b->empty() ? Opt(*b + "nəs") : Opt();
}

Opt NegativeUn(const std::string& w, const Lookup& l) {
  if (Starts(w, "un") && Len(w) >= 4 && w != "unit" && w != "units" && w != "union" && w != "unions") {
    Opt b = l(w.substr(2));
    return b && !b->empty() ? Opt("ʌn" + *b) : Opt();
  }
  return {};
}

Opt BritishOur(const std::string& w, const Lookup& l) {
  if (w != "our" && Ends(w, "our") && Len(w) > 3 && w != "tour" && w != "tours" && w != "amour" && w != "amours")
    return l(Drop(w, 3) + "or");
  return {};
}

Opt ApplyRules(const std::string& w, const Lookup& l) {
  static const std::function<Opt(const std::string&, const Lookup&)> rules[] = {
      CompoundNumeral, ParticipialCompound, PrefixedCompound, SuffixInflection,
      SuffixRule("ly", [](const std::string& b) { return Ends(b, "l") ? b + "i" : b + "li"; }),
      SuffixRule("er", [](const std::string& b) { return b + "ɚ"; }),
      SuffixRule("est", [](const std::string& b) { return b + "ɪst"; }),
      SuffixRule("ing", [](const std::string& b) { return b + "ɪŋ"; }),
      PastEd, NounNess,
      SuffixRule("less", [](const std::string& b) { return b + "ləs"; }),
      NegativeUn, BritishOur};
  for (auto& r : rules)
    if (Opt i = r(w, l)) return i;
  return {};
}

}  // namespace

// ---- lexicon.bin

struct Lexicon {
  std::vector<sv> keys, vals, skeys, svals;
  std::unordered_set<char32_t> vocab;
  std::unordered_map<uint64_t, char32_t> nfc_pair;
  std::unordered_map<char32_t, char32_t> nfc_single;
  std::unique_ptr<Re> prefix, suffix, infix, url;
  // Tokenizer._special_matcher: special cases containing affixes, as their affix-split token texts.
  std::unordered_map<std::string, std::vector<std::pair<std::vector<std::string>, sv>>> matcher;
};

namespace {

struct Row {
  char status;
  bool import;
  sv ipa, def, opts, conds;
  Opt Cond(sv c) const {
    if (conds.empty() || c.empty()) return {};
    for (sv kv : Split(conds, '\x1e')) {
      const size_t p = kv.find('\x1d');
      if (kv.substr(0, p) == c && p + 1 < kv.size()) return std::string(kv.substr(p + 1));
    }
    return {};
  }
};

std::optional<sv> Find(const std::vector<sv>& keys, const std::vector<sv>& vals, sv k) {
  auto it = std::lower_bound(keys.begin(), keys.end(), k);
  if (it == keys.end() || *it != k) return {};
  return vals[it - keys.begin()];
}

std::optional<Row> GetRow(const Lexicon& L, sv word) {
  auto v = Find(L.keys, L.vals, word);
  if (!v) return {};
  auto f = Split(v->substr(2), '\x1f');
  return Row{(*v)[0], (*v)[1] == '1', f[1], f[2], f[3], f[4]};
}

// ---- kokoro_map.py

struct Unmappable {};

constexpr sv kVowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻᵊ";
constexpr sv kPunct = ",.!?;:…";

std::string Restress(sv ps) {
  std::string o;
  char32_t pending = 0;
  for (size_t i = 0; i < ps.size();) {
    const char32_t c = Decode(ps, i);
    if (c == U'ˈ' || c == U'ˌ') {
      pending = c == U'ˈ' || pending != U'ˈ' ? c : pending;
    } else if (Has(kVowels, c) && pending) {
      Append(o, pending), Append(o, c);
      pending = 0;
    } else {
      Append(o, c);
    }
  }
  return o;
}

std::string ToK(const Lexicon& L, sv ipa, bool unstressed = false) {
  static const std::pair<sv, sv> seq[] = {{"ɝ", "ɜɹ"}, {"ɚ", "əɹ"}, {"eɪ", "A"}, {"aɪ", "I"}, {"oʊ", "O"}, {"aʊ", "W"},
      {"ɔɪ", "Y"}, {"dʒ", "ʤ"}, {"tʃ", "ʧ"}, {"g", "ɡ"}, {"r", "ɹ"}, {"ɾ", "T"}, {"ʔ", "t"}, {"ː", ""}, {"e", "A"},
      {"o", "O"}, {"a", "æ"}, {"iɹ", "ɪɹ"}};
  std::string s = Replace(std::string(ipa), " ", "");
  for (auto [a, b] : seq) s = Replace(std::move(s), a, b);
  s = unstressed ? Replace(Replace(std::move(s), "ˈ", ""), "ˌ", "") : Restress(s);
  for (size_t i = 0; i < s.size();) {
    const char32_t c = Decode(s, i);
    if (c != ' ' && !L.vocab.count(c)) throw Unmappable{};
  }
  return s;
}

// ---- normalize.py

const std::pair<sv, sv> kExt[] = {{"py", "pie"}, {"md", "MD"}, {"txt", "text"}, {"jsonl", "JSON L"}, {"sh", "SH"},
    {"db", "DB"}, {"tsv", "TSV"}, {"csv", "CSV"}, {"js", "JS"}, {"ts", "TS"}, {"rs", "RS"}, {"wav", "wave"},
    {"pt", "PT"}, {"onnx", "onyx"}, {"yml", "YAML"}, {"yaml", "YAML"}, {"toml", "tommel"}, {"cfg", "config"},
    {"so", "SO"}, {"m4a", "M4A"}, {"log", "log"}, {"jsx", "JSX"}, {"tsx", "TSX"}, {"mjs", "MJS"}, {"cjs", "CJS"},
    {"json", "JSON"}, {"html", "HTML"}, {"htm", "HTML"}, {"css", "CSS"}, {"scss", "SCSS"}, {"svg", "SVG"},
    {"png", "PNG"}, {"jpg", "jay peg"}, {"jpeg", "jay peg"}, {"gif", "gif"}, {"webp", "web P"}, {"pdf", "PDF"},
    {"mp3", "MP3"}, {"mp4", "MP4"}, {"webm", "web M"}, {"opus", "opus"}, {"flac", "flack"}, {"ogg", "ogg"},
    {"kt", "KT"}, {"kts", "KTS"}, {"java", "java"}, {"go", "go"}, {"c", "C"}, {"h", "H"}, {"cpp", "CPP"},
    {"hpp", "HPP"}, {"cu", "CU"}, {"rb", "RB"}, {"lua", "lua"}, {"sql", "sequel"}, {"xml", "XML"}, {"ini", "INI"},
    {"env", "env"}, {"lock", "lock"}, {"zsh", "ZSH"}, {"bash", "bash"}, {"ps1", "PS1"}, {"bat", "bat"},
    {"exe", "exe"}, {"dll", "DLL"}, {"apk", "APK"}, {"zip", "zip"}, {"gz", "GZ"}, {"tar", "tar"}, {"gguf", "GGUF"},
    {"bin", "bin"}, {"safetensors", "safe tensors"}, {"ckpt", "checkpoint"}, {"npy", "NPY"}, {"npz", "NPZ"},
    {"ipynb", "notebook"}, {"gradle", "gradle"}, {"vue", "view"}, {"svelte", "svelte"}, {"swift", "swift"}};

std::optional<sv> Ext(sv e) {
  for (auto [k, v] : kExt)
    if (k == e) return v;
  return {};
}

std::string PathWords(sv m) {
  std::string s(m);
  const std::string lead = Starts(s, "~/") ? "home slash " : Starts(s, "/") ? "slash " : "";
  const size_t f = s.find_first_not_of("~/");
  s = f == std::string::npos ? "" : s.substr(f);
  while (!s.empty() && s.back() == '/') s.pop_back();
  static const Re ext(R"(\.([A-Za-z0-9]{1,11})\b)"), slash(R"(\s*/\s*)");
  s = Sub(ext, s, [](const M& e) {
    auto x = Ext(Lower(e.G(1)));
    return " dot " + std::string(x ? *x : e.G(1));
  });
  return " " + lead + Replace(SubT(slash, s, " slash "), "_", " ") + " ";
}

std::string Markdown(sv t) {
  static const Re fence("```.*?(```|$)", PCRE2_DOTALL), link(R"(!?\[([^\]]+)\]\([^)]*\))"),
      head(R"((?m)^\s{0,3}#{1,6}\s+)"), bullet(R"((?m)^\s*(?:[-*+•]|\d+[.)])\s+)"),
      rule(R"((?m)^\s*\|?[\s:|-]+\|[\s:|-]*$)"), bold(R"((\*\*|__)(.+?)\1)"),
      ital(R"((?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*]))");
  std::string s = SubT(fence, t, " ");
  s = SubT(link, s, "\\1");
  s = SubT(head, s, "");
  s = SubT(bullet, s, "");
  s = SubT(rule, s, "");
  s = SubT(bold, s, "\\2");
  s = SubT(ital, s, "\\1");
  return Replace(Replace(s, "`", ""), "**", "");
}

// Divergence from ag2p: 2000-2099 read as years only after a year cue; leading zeros read digit by digit.
bool YearCue(sv before) {
  static const std::string months = "January|February|March|April|May|June|July|August|September|October|November|December";
  static const Re cue(R"re((?:\b(?i:in|since|by|until|till|from|during|year|circa|before|after|through|around|early|late|mid))re"
                      R"re(|\b(?i:summer|winter|spring|fall|autumn|class|end|start|beginning|middle)\s+of)re"
                      R"re(|\b(?:)re" + months + R"re()\b(?:\s+\w+)?,?|\b(?:1[1-9]|20)\d\d\s+(?:to|and|or|through))\s+\z)re");
  return Search(cue, before.substr(before.size() > 80 ? before.size() - 80 : 0));
}

std::string Number(const M& m) {
  static const Re comma(R"(\d,\d)");
  std::string s = Replace(std::string(m.G()), ",", "");
  if (const size_t dot = s.find('.'); dot != std::string::npos) {
    std::string o = IntWords(sv(s).substr(0, dot)) + " point";
    return o + " " + Join(DigitWords(sv(s).substr(dot + 1)), " ");
  }
  if (s.size() > 1 && s[0] == '0' && m.G().find(',') == sv::npos) {
    std::vector<std::string> w = DigitWords(s);
    for (auto& x : w) x = x == "zero" ? "oh" : x;
    return Join(w, " ");
  }
  sv d = s;
  while (d.size() > 1 && d[0] == '0') d.remove_prefix(1);
  const long n = d.size() <= 4 ? std::stol(std::string(d)) : 0;
  if (n >= 1100 && n <= 1999 && s.size() == 4 && !Search(comma, m.G())) return YearWords(int(n));
  if (n >= 2000 && n <= 2099 && s.size() == 4 && YearCue(m.s.substr(0, m.B()))) return n < 2010 ? IntWords(n) : YearWords(int(n));
  return IntWords(s);
}

std::string Decade(const M& m) {
  const std::string d(m.G(1));
  std::string w = d.size() == 2 ? IntWords(d) : d < "2000" ? YearWords(std::stoi(d)) : d < "2010" ? IntWords(d) : YearWords(std::stoi(d));
  return Ends(w, "y") ? Drop(w, 1) + "ies" : w + "s";
}

bool KeepDot(const M& m) {
  static const Re next(R"(\s*\z|\s+[A-Z])");
  M n(next);
  return Ends(m.G(), ".") && n.Find(next, m.s, m.E()) && n.B() == m.E();
}

int Roman(sv s) {
  std::vector<int> v;
  for (char c : s) {
    switch (c | 32) {
      case 'i': v.push_back(1); break;
      case 'v': v.push_back(5); break;
      case 'x': v.push_back(10); break;
      case 'l': v.push_back(50); break;
      case 'c': v.push_back(100); break;
      case 'd': v.push_back(500); break;
      case 'm': v.push_back(1000); break;
    }
  }
  int sum = 0;
  for (size_t i = 0; i < v.size(); ++i) sum += i + 1 < v.size() && v[i] < v[i + 1] ? -v[i] : v[i];
  return sum;
}

std::string Numbers(std::string t) {
  static const std::string roman = "M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})";
  static const Re roman_ctx(R"(\b((?i:chapter|book|part|volume|section|act|scene|canto))\s+(?:(?=[MDCLXVI]+\b)()" + roman +
                            R"()|(?=[ivx]+\b)(x{0,3}(?:ix|iv|v?i{0,3})))(?<=[MDCLXVIivx])\b(?<!\bI\b(?=\s+[a-z])))"),
      dollars(R"(\$\s*(\d[\d,]*)(?:\.(\d\d))?\b)"), dotted(R"(\bv?(\d+(?:\.\d+){2,})\b)"),
      version(R"(\bv(\d+(?:\.\d+)?)\b)"), decade(R"((?<![\w.,'])'?([1-9]0|1[1-9]\d0|20\d0)'?s\b)"),
      ampm(R"((?<=\d)\s*([apAP])\.?[mM]\b\.?)"), clock(R"(\b(\d{1,2}):(\d\d)(?!\d))"),
      tilde(R"(~\s*(?=\d))"), range(R"((?<![\w.])(\d+)\s*[-—–]\s*(\d+)(?![\w.-]))"),
      units(R"((?<![A-Za-z\d.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:(k|K|M|B|x|×)|\s?)"
            R"((KB|kB|MB|GB|TB|PB|ms|ns|µs|μs|GHz|MHz|kHz|Hz|Gbps|Mbps|kbps|fps|km|kg|cm|mm))\b)"),
      ordinal(R"(\b(\d+)(st|nd|rd|th)\b)"), plus(R"((\d)\+)"),
      number(R"((?<![A-Za-z\d])(?>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![A-Za-z\d]))");
  t = Sub(roman_ctx, t, [](const M& m) {
    return std::string(m.G(1)) + " " + std::to_string(Roman(!m.G(2).empty() ? m.G(2) : m.G(3)));
  });
  t = Sub(dollars, t, [](const M& m) {
    return std::string(m.G(1)) + " dollars" + (m.Set(2) ? " and " + std::string(m.G(2)) + " cents" : "");
  });
  t = Sub(dotted, t, [](const M& m) {
    std::vector<std::string> w;
    for (sv p : Split(m.G(1), '.')) w.push_back(IntWords(p));
    return Join(w, " point ");
  });
  t = SubT(version, t, "version \\1");
  t = Sub(decade, t, Decade);
  t = Sub(ampm, t, [](const M& m) { return " " + Upper(m.G(1)) + " M" + (KeepDot(m) ? "." : ""); });
  t = Sub(clock, t, [](const M& m) {
    sv mm = m.G(2);
    return IntWords(m.G(1)) + " " + (mm == "00" ? "" : mm[0] == '0' ? "oh " + IntWords(mm) : IntWords(mm));
  });
  t = SubT(tilde, t, "about ");
  t = SubT(range, t, "\\1 to \\2");
  t = Sub(units, t, [](const M& m) {
    static const std::pair<sv, sv> u[] = {{"k", "thousand"}, {"K", "thousand"}, {"M", "million"}, {"B", "billion"},
        {"x", "times"}, {"×", "times"}, {"KB", "kilobytes"}, {"kB", "kilobytes"}, {"MB", "megabytes"},
        {"GB", "gigabytes"}, {"TB", "terabytes"}, {"PB", "petabytes"}, {"ms", "milliseconds"}, {"ns", "nanoseconds"},
        {"µs", "microseconds"}, {"μs", "microseconds"}, {"GHz", "gigahertz"}, {"MHz", "megahertz"},
        {"kHz", "kilohertz"}, {"Hz", "hertz"}, {"Gbps", "gigabits per second"}, {"Mbps", "megabits per second"},
        {"kbps", "kilobits per second"}, {"fps", "frames per second"}, {"km", "kilometers"}, {"kg", "kilograms"},
        {"cm", "centimeters"}, {"mm", "millimeters"}};
    const sv unit = m.Set(2) ? m.G(2) : m.G(3);
    for (auto [k, v] : u)
      if (k == unit) {
        std::string name(v);
        const size_t e = std::min(name.find(" per"), name.size());
        if (m.G(1) == "1" && !m.Set(2) && name[e - 1] == 's') name.erase(e - 1, 1);
        return std::string(m.G(1)) + " " + name;
      }
    return std::string(m.G());
  });
  t = Sub(ordinal, t, [](const M& m) { return OrdinalWords(m.G(1)); });
  t = SubT(plus, t, "\\1 plus");
  return Sub(number, t, Number);
}

std::string Nfc(const Lexicon& L, sv s) {
  bool ascii = true;
  for (unsigned char c : s) ascii &= c < 0x80;
  if (ascii) return std::string(s);
  std::u32string o;
  for (size_t i = 0; i < s.size();) {
    char32_t c = Decode(s, i);
    if (auto it = L.nfc_single.find(c); it != L.nfc_single.end()) c = it->second;
    if (!o.empty())
      if (auto it = L.nfc_pair.find(uint64_t(o.back()) << 32 | c); it != L.nfc_pair.end()) {
        o.back() = it->second;
        continue;
      }
    o += c;
  }
  return U8(o);
}

std::string NormalizeImpl(const Lexicon& L, sv text) {
  std::string t = Nfc(L, text);
  t = Replace(Replace(Replace(Replace(t, "“", "\""), "”", "\""), "‘", "'"), "’", "'");
  t = Markdown(t);
  static const std::pair<sv, sv> abbrev[] = {{"e.g.", "for example"}, {"e.g", "for example"}, {"i.e.", "that is"},
      {"i.e", "that is"}, {"etc.", "et cetera"}, {"vs.", "versus"}, {"vs", "versus"}, {"approx.", "approximately"}};
  static const std::vector<std::unique_ptr<Re>> abbrev_re = [] {
    std::vector<std::unique_ptr<Re>> v;
    for (auto [k, _] : abbrev) v.push_back(std::make_unique<Re>(R"((?<!\w))" + Replace(std::string(k), ".", "\\.") + R"((?!\w))"));
    return v;
  }();
  static const Re title(R"(\b(Dr|Mr|Mrs|Ms|Prof|Mt)\.(?=\s+[A-Z]))"), saint(R"(\bSt\.)"),
      street(R"([a-z0-9,]\s+[A-Z][a-z]+\s+\z)"), name_next(R"(\s+[A-Z])"), suffix(R"(\b(Jr|Sr)\.)"),
      etc(R"((?<!\w)etc\.(?=\s*\z|\s+[A-Z]))"), month(R"(\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s+\d))");
  t = Sub(title, t, [](const M& m) {
    static const std::pair<sv, sv> w[] = {{"Dr", "Doctor"}, {"Mr", "Mister"}, {"Mrs", "Missus"}, {"Ms", "Miz"},
                                          {"Prof", "Professor"}, {"Mt", "Mount"}};
    for (auto [k, v] : w)
      if (k == m.G(1)) return std::string(v);
    return std::string(m.G());
  });
  t = Sub(saint, t, [](const M& m) {
    const sv before = m.s.substr(0, m.B());
    if (Search(street, before.substr(before.size() > 40 ? before.size() - 40 : 0)))
      return std::string("Street") + (KeepDot(m) ? "." : "");
    M n(name_next);
    return n.Find(name_next, m.s, m.E()) && n.B() == m.E() ? std::string("Saint") : std::string(m.G());
  });
  t = Sub(suffix, t, [](const M& m) { return std::string(m.G(1) == "Jr" ? "Junior" : "Senior") + (KeepDot(m) ? "." : ""); });
  t = SubT(etc, t, "et cetera.");
  t = Sub(month, t, [](const M& m) {
    static const std::pair<sv, sv> w[] = {{"Jan", "January"}, {"Feb", "February"}, {"Mar", "March"}, {"Apr", "April"},
        {"Jun", "June"}, {"Jul", "July"}, {"Aug", "August"}, {"Sep", "September"}, {"Sept", "September"},
        {"Oct", "October"}, {"Nov", "November"}, {"Dec", "December"}};
    for (auto [k, v] : w)
      if (k == m.G(1)) return std::string(v);
    return std::string(m.G());
  });
  for (size_t i = 0; i < std::size(abbrev); ++i) t = SubT(*abbrev_re[i], t, abbrev[i].second);
  static const Re url(R"((?:https?://|www\.)[^\s)>\]]+?(?=[.,;:!?)]*(?:\s|$)))"), url_head(R"(^(https?://)?(www\.)?)"),
      path(R"((?<![\w/])(?:~/|/|\w[\w.-]*/)[\w./~-]*\w/?|\b\w[\w-]*\.[A-Za-z][A-Za-z0-9]{0,4}\b(?=[\s,;:!?)]|\.(?:\s|$)|$))"),
      plusplus(R"(\b([A-Za-z]{2,})\+\+)"), cpp(R"(\bC\+\+)"), dots(R"(\b(?:[A-Z]\.){2,}(?![A-Za-z]))"),
      acr_num(R"(\b([A-Z]{2,})-(\d+)\b)"), dot(R"((?<=[A-Za-z])\.(?=[A-Za-z]))"), slash(R"((?<=\w)/(?=\w))"),
      under(R"((?<=\w)_(?=\w))"), eq(R"((?<=\w)=(?=\w))"), dash(R"(-{2,}|[—–]|\s-\s)"), ell(R"(\s*\.{3}|…)"),
      ws(R"([ \t]+)"), dotline(".*");
  t = Sub(url, t, [](const M& m) {
    std::string s = SubT(url_head, m.G(), "");
    while (!s.empty() && s.back() == '/') s.pop_back();
    M d(dotline);
    d.Find(dotline, s);
    return PathWords(d.G());
  });
  t = Sub(path, t, [](const M& m) {
    sv s = m.G();
    const size_t dp = s.rfind('.');
    const std::string ext = Lower(dp == sv::npos ? s : s.substr(dp + 1));
    if (s.find('/') != sv::npos || Ext(ext) || s.find('_') != sv::npos) return PathWords(s);
    return std::string(s);
  });
  t = SubT(plusplus, t, "\\1 plus plus");
  t = SubT(cpp, t, "C plus plus");
  t = Sub(dots, t, [](const M& m) { return Replace(std::string(m.G()), ".", ""); });
  t = SubT(acr_num, t, "\\1 \\2");
  t = Numbers(std::move(t));
  t = SubT(dot, t, " dot ");
  t = SubT(slash, t, " slash ");
  t = SubT(under, t, " ");
  t = SubT(eq, t, " equals ");
  static const std::pair<sv, sv> symbols[] = {{"→", " to "}, {"←", " from "}, {"↔", " and "}, {"≈", " about "},
      {"≤", " at most "}, {"≥", " at least "}, {"&", " and "}, {"=", " equals "}, {"×", " times "}, {"@", " at "},
      {"#", " number "}, {"•", ", "}, {"|", ", "}, {"+", " plus "}, {"%", " percent "}, {"°", " degrees "},
      {"§", " section "}};
  for (auto [k, v] : symbols) t = Replace(std::move(t), k, v);
  t = SubT(dash, t, ", ");
  t = SubT(ell, t, "...");
  return std::string(Strip(SubT(ws, t, " ")));
}

// ---- spaCy tokenizer (Tokenizer.__call__: _tokenize_affixes, then _apply_special_cases)

void TokSpan(const Lexicon& L, sv full, size_t b, size_t e, std::vector<std::pair<size_t, size_t>>& out, bool specials = true) {
  auto special = [&](size_t s, size_t t) {
    return specials ? Find(L.skeys, L.svals, full.substr(s, t - s)) : std::optional<sv>();
  };
  auto emit_special = [&](size_t s, sv pieces) {
    for (sv p : Split(pieces, '\x1f')) out.emplace_back(s, s + p.size()), s += p.size();
  };
  if (auto sp = special(b, e)) return emit_special(b, *sp);
  std::vector<std::pair<size_t, size_t>> pre, suf;
  M m(*L.prefix), ms(*L.suffix);
  size_t last = SIZE_MAX;
  while (e > b && e - b != last) {
    if (special(b, e)) break;
    last = e - b;
    const size_t pre_len = m.Find(*L.prefix, full.substr(b, e - b)) ? m.E() - m.B() : 0;
    if (pre_len && b + pre_len < e && special(b + pre_len, e)) {
      pre.emplace_back(b, b + pre_len), b += pre_len;
      break;
    }
    const size_t suf_len = ms.Find(*L.suffix, full.substr(b + pre_len, e - b - pre_len)) ? ms.E() - ms.B() : 0;
    if (suf_len && e - suf_len > b && special(b, e - suf_len)) {
      suf.emplace_back(e - suf_len, e), e -= suf_len;
      break;
    }
    if (pre_len && suf_len && pre_len + suf_len <= e - b) {
      pre.emplace_back(b, b + pre_len), suf.emplace_back(e - suf_len, e);
      b += pre_len, e -= suf_len;
    } else if (pre_len) {
      pre.emplace_back(b, b + pre_len), b += pre_len;
    } else if (suf_len) {
      suf.emplace_back(e - suf_len, e), e -= suf_len;
    }
  }
  out.insert(out.end(), pre.begin(), pre.end());
  if (e > b) {
    sv s = full.substr(b, e - b);
    if (auto sp = special(b, e)) {
      emit_special(b, *sp);
    } else if (M u(*L.url); u.Find(*L.url, s)) {
      out.emplace_back(b, e);
    } else {
      size_t start = 0;
      bool any = false;
      FindIter(*L.infix, s, [&](const M& x) {
        any = true;
        if (x.B() == 0) return;
        if (x.B() != start) out.emplace_back(b + start, b + x.B());
        if (x.B() != x.E()) out.emplace_back(b + x.B(), b + x.E());
        start = x.E();
      });
      if (!any) out.emplace_back(b, e);
      else if (start < s.size()) out.emplace_back(b + start, e);
    }
  }
  out.insert(out.end(), suf.rbegin(), suf.rend());
}

void TokenizeSpans(const Lexicon& L, sv text, std::vector<std::pair<size_t, size_t>>& out, bool specials = false) {
  size_t start = 0;
  bool in_ws = true;
  for (size_t i = 0; i <= text.size();) {
    size_t j = i;
    const bool ws = i == text.size() || IsSpace(Decode(text, j));
    if (ws && !in_ws) TokSpan(L, text, start, i, out, specials);
    if (!ws && in_ws) start = i;
    in_ws = ws;
    if (i == text.size()) break;
    i = j;
  }
}

std::vector<std::pair<size_t, size_t>> TokenizeImpl(const Lexicon& L, sv text) {
  std::vector<std::pair<size_t, size_t>> out;
  TokenizeSpans(L, text, out, true);
  struct Span {
    size_t s, n;
    sv key;
  };
  std::vector<Span> spans;
  for (size_t i = 0; i < out.size(); ++i) {
    auto it = L.matcher.find(std::string(text.substr(out[i].first, out[i].second - out[i].first)));
    if (it == L.matcher.end()) continue;
    for (auto& [toks, key] : it->second) {
      bool ok = i + toks.size() <= out.size();
      for (size_t k = 1; ok && k < toks.size(); ++k)
        ok = text.substr(out[i + k].first, out[i + k].second - out[i + k].first) == toks[k];
      if (ok) spans.push_back({i, toks.size(), key});
    }
  }
  if (spans.empty()) return out;
  std::stable_sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.n != b.n ? a.n > b.n : a.s < b.s; });
  std::vector<char> taken(out.size());
  std::vector<const Span*> keep(out.size());
  for (auto& sp : spans) {
    if (std::any_of(taken.begin() + sp.s, taken.begin() + sp.s + sp.n, [](char c) { return c; })) continue;
    std::fill(taken.begin() + sp.s, taken.begin() + sp.s + sp.n, 1);
    keep[sp.s] = &sp;
  }
  std::vector<std::pair<size_t, size_t>> res;
  for (size_t i = 0; i < out.size();) {
    const Span* sp = keep[i];
    const size_t b = out[i].first;
    if (sp && text.substr(b, out[i + sp->n - 1].second - b) == sp->key) {
      size_t s = b;
      for (sv p : Split(*Find(L.skeys, L.svals, sp->key), '\x1f')) res.emplace_back(s, s + p.size()), s += p.size();
      i += sp->n;
    } else {
      res.push_back(out[i++]);
    }
  }
  return res;
}

// ---- g2p.py

// Latin-1 and Œœ letters to ASCII, for the accent-stripped lexicon retry (a divergence from ag2p).
std::string Fold(sv s) {
  static const char* const latin1[] = {"A", "A", "A", "A", "A", "A", "AE", "C", "E", "E", "E", "E", "I", "I", "I", "I",
      "D", "N", "O", "O", "O", "O", "O", nullptr, "O", "U", "U", "U", "U", "Y", "TH", "ss", "a", "a", "a", "a", "a", "a",
      "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", "d", "n", "o", "o", "o", "o", "o", nullptr, "o", "u", "u", "u",
      "u", "y", "th", "y"};
  std::string o;
  for (size_t i = 0; i < s.size();) {
    const size_t b = i;
    const char32_t c = Decode(s, i);
    if (c >= 0xC0 && c <= 0xFF && latin1[c - 0xC0]) o += latin1[c - 0xC0];
    else if (c == 0x152 || c == 0x153) o += c == 0x152 ? "OE" : "oe";
    else o.append(s.substr(b, i - b));
  }
  return o;
}

constexpr sv kIpaChars = "əɚɝɪʊʌɛæɑɔɹʃʒθðŋˈˌɾʔ";
const char* const kLetters[] = {"eɪ", "bi", "si", "di", "i", "ɛf", "dʒi", "eɪtʃ", "aɪ", "dʒeɪ", "keɪ", "ɛl", "ɛm",
                                "ɛn", "oʊ", "pi", "kju", "ɑɹ", "ɛs", "ti", "ju", "vi", "dʌbəlju", "ɛks", "waɪ", "zi"};

sv TagClass(sv tag) {
  static const std::pair<sv, sv> m[] = {{"NN", "noun"}, {"NNS", "noun"}, {"NNP", "noun"}, {"NNPS", "noun"},
      {"VB", "verb"}, {"VBP", "verb"}, {"VBZ", "verb"}, {"VBG", "verb"}, {"VBD", "past"}, {"VBN", "past"},
      {"JJ", "adj"}, {"JJR", "adj"}, {"JJS", "adj"}, {"RB", "adv"}, {"RBR", "adv"}, {"RBS", "adv"}, {"IN", "prep"}};
  for (auto [k, v] : m)
    if (k == tag) return v;
  return {};
}

sv TagFallback(sv klass) { return klass == "past" ? "verb" : klass == "adv" ? "adj" : sv(); }

bool In(sv x, std::initializer_list<sv> set) { return std::find(set.begin(), set.end(), x) != set.end(); }

struct Part {
  std::string text;
  Opt ipa;
  std::string ps, src;
};

struct Hit {
  std::string ipa, src;
  bool unstressed = false;
};

struct Slime {
  const G2P& g;
  const Lexicon& L;

  std::string K(sv ipa, bool unstressed = false) const { return ToK(L, ipa, unstressed); }
  std::optional<Row> Get(sv w) const { return GetRow(L, w); }

  Lookup RuleLookup(sv tag) const {
    const sv klass = TagClass(tag);
    return [this, klass](const std::string& w) -> Opt {
      auto r = Get(w);
      if (!r) return {};
      if (r->status == 'r') return std::string(r->ipa);
      if (r->status == 'c' && !klass.empty()) {
        if (Opt v = r->Cond(klass)) return v;
        return r->Cond(TagFallback(klass));
      }
      return {};
    };
  }

  std::optional<Hit> Lex(const std::string& key, sv tag) const {
    auto r = Get(key);
    if (!r) return {};
    static const std::pair<sv, sv> strong[] = {{"us", "ʌs"}, {"them", "ðɛm"}, {"her", "hɝ"}, {"your", "jɔɹ"},
                                               {"am", "æm"}, {"as", "æz"}, {"than", "ðæn"}};
    if (r->status == 'r') {
      for (auto [k, v] : strong)
        if (k == key) return Hit{std::string(v), "lex-strong", true};
      return Hit{std::string(r->ipa), r->import ? "lex-import" : "lex"};
    }
    if (r->status == 'w') {
      Opt v = r->Cond("strong");
      return Hit{v ? *v : std::string(Split(r->opts, '\x1e')[0]), "lex-strong", true};
    }
    const std::string ts = tag.empty() ? "None" : std::string(tag);
    if (r->status == 'c') {
      const sv klass = TagClass(tag);
      if (!klass.empty())
        if (Opt v = r->Cond(klass)) return Hit{*v, "variant+" + ts};
      const sv fb = In(tag, {"NN", "NNS", "NNP", "NNPS", "CD", "FW"}) ? sv("noun") : TagFallback(klass);
      if (!fb.empty())
        if (Opt v = r->Cond(fb)) return Hit{*v, "variant-" + std::string(fb) + "+" + ts};
      return Hit{std::string(r->def), "ambiguous-default"};
    }
    if (Opt d = ApplyRules(key, RuleLookup(tag)); d && !d->empty()) {
      const std::string dd = Replace(*d, "ˌ", "");
      for (sv o : Split(r->opts, '\x1e'))
        if (Replace(std::string(o), "ˌ", "") == dd) return Hit{*d, "rule+" + ts};
    }
    return Hit{std::string(r->def), "ambiguous-default"};
  }

  std::string LexKey(sv w) const {
    static const Re a = Full("(?=.*[A-Z])[A-Z0-9]{2,}s?"), b = Full("[B-HJ-Z]");
    return Search(a, w) || Search(b, w) ? std::string(w) : Lower(w);
  }

  Part Spell(sv word) const {
    const bool plural = Ends(word, "s") && !IsUpper(word);
    sv letters = word;
    if (plural)
      while (!letters.empty() && letters.back() == 's') letters.remove_suffix(1);
    std::string up;
    for (char c : letters.empty() ? word : letters)
      if (c >= 'A' && c <= 'Z') up += c;
    std::string ipa;
    for (size_t i = 0; i < up.size(); ++i) ipa += (i + 1 < up.size() ? "ˌ" : "ˈ") + std::string(kLetters[up[i] - 'A']);
    if (plural && !ipa.empty()) ipa = PluralSuffix(ipa);
    return {std::string(word), ipa, K(ipa), "norm-spell"};
  }

  void DigitIpa(sv s, bool between, std::vector<std::string>& out) const {
    std::vector<std::string> words;
    if (between) words = {"to"};
    else if (Len(s) < 5)
      for (const std::string n = IntWords(s); sv w : Split(n, ' ')) words.emplace_back(w);
    else words = DigitWords(s);
    for (auto& w : words)
      if (auto h = Lex(w, "")) out.push_back(w == "to" ? Replace(h->ipa, "ˈ", "") : h->ipa);
  }

  Part Alnum(sv word) const {
    static const Re r("[A-Za-z]+|\\d+");
    const std::string up = Upper(word);
    std::vector<std::string> out;
    FindIter(r, up, [&](const M& m) {
      sv s = m.G();
      if (s[0] >= '0' && s[0] <= '9') {
        DigitIpa(s, m.B() > 0 && m.E() < word.size() && s == "2", out);
      } else if (std::string i = *Spell(s).ipa; !i.empty()) {
        out.push_back(i);
      }
    });
    std::vector<std::string> ps;
    for (auto& p : out) ps.push_back(K(p));
    return {std::string(word), Join(out, " "), Join(ps, " "), "norm-spell"};
  }

  std::pair<std::string, std::string> Model(const std::string& word) const {
    std::string ipa;
    if (g.oracle) {
      auto it = g.oracle->model.find(word);
      if (it != g.oracle->model.end()) ipa = it->second;
      else ++g.oracle->misaligned;
    } else if (g.guesser) {
      ipa = Guess(*g.guesser, word);
    }
    if (!ipa.empty()) try {
        return {ipa, K(ipa)};
      } catch (const Unmappable&) {
      }
    static const Re alnum("[A-Za-z0-9]");
    const std::string f = Fold(word);
    Part p = Search(alnum, f) ? Alnum(f) : Spell("X");
    return {*p.ipa, p.ps};
  }

  std::vector<Part> SplitParts(sv word, sv tag) const {
    std::vector<sv> parts;
    for (sv p : Split(word, '-'))
      if (!p.empty()) parts.push_back(p);
    std::vector<Part> o;
    for (size_t i = 0; i < parts.size(); ++i)
      for (auto& x : Resolve(std::string(parts[i]), i == parts.size() - 1 ? tag : sv("NN"))) o.push_back(std::move(x));
    return o;
  }

  std::vector<Part> Resolve(const std::string& word, sv tag) const {
    for (size_t i = 0; i < word.size();)
      if (Has(kIpaChars, Decode(word, i))) {
        std::string f;
        for (size_t j = 0; j < word.size();) {
          const char32_t c = Decode(word, j);
          if (L.vocab.count(c) || c == U'ˈ' || c == U'ˌ') Append(f, c);
        }
        return {{word, word, K(f), "norm"}};
      }
    const std::string key = LexKey(word), lw = Lower(word);
    auto hit = Lex(key, tag);
    static const Re keep = Full("[A-Z]{2,3}s?|[A-Z]+[0-9][A-Za-z0-9]*|[^AEIOUY]+s?");
    if (!hit && key != lw && !Search(keep, word)) hit = Lex(lw, tag);
    if (hit) return {{word, hit->ipa, K(hit->ipa, hit->unstressed), hit->src}};
    static const Re acr_s = Full("[A-Z]{2,}s");
    const bool poss = Ends(lw, "'s") && Len(word) > 2;
    if (poss || Search(acr_s, word)) {
      auto parts = Resolve(word.substr(0, word.size() - (poss ? 2 : 1)), tag);
      if (parts.empty()) return parts;
      Part last = std::move(parts.back());
      parts.pop_back();
      if (last.ipa) {
        const std::string ipa = PluralSuffix(*last.ipa);
        parts.push_back({word, ipa, K(ipa), last.src});
      } else {
        const char32_t c = Last(last.ps);
        parts.push_back({word, {}, last.ps + (!c || Has("szʃʒʧʤ", c) ? "ɪz" : "z"), last.src});
      }
      return parts;
    }
    static const Re camel("[a-z][A-Z]|[A-Z]{2}[a-z]"),
        camel_parts(R"([A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+(?:'[a-z]+)?|[A-Z]+s?(?![a-z])|\d+)");
    if (Search(camel, word) && word.find('-') == std::string::npos) {
      std::vector<std::string> parts;
      FindIter(camel_parts, word, [&](const M& m) { parts.emplace_back(m.G()); });
      if (parts.size() > 1) {
        std::vector<Part> o;
        for (auto& p : parts)
          for (auto& x : Resolve(p, tag)) o.push_back(std::move(x));
        return o;
      }
    }
    static const Re alnum = Full("[A-Z]+[0-9]+[A-Za-z0-9]*|[a-z]{1,3}[0-9]+[a-z0-9]{0,3}"), acr = Full("[A-Z]{2,}s?"),
                    vowel("[AEIOU]");
    if (Search(alnum, word)) return {Alnum(word)};
    if (Search(acr, word)) {
      sv base = word;
      while (!base.empty() && base.back() == 's') base.remove_suffix(1);
      if (Len(base) <= 3 || !Search(vowel, word)) return {Spell(word)};
    }
    const bool hyphen = StripChars(word, "-").find('-') != sv::npos;
    if (hyphen) {
      bool known = true;
      for (sv p : Split(word, '-'))
        if (!p.empty() && !Get(LexKey(p)) && !Get(Lower(p))) known = false;
      if (known) return SplitParts(word, tag);
    }
    if (Opt ipa = ApplyRules(key, RuleLookup(tag))) return {{word, *ipa, K(*ipa), "rule"}};
    if (hyphen) {
      static const std::pair<sv, sv> prefixes[] = {{"re", "ˌɹi"}, {"un", "ˌʌn"}, {"de", "ˌdi"}, {"pre", "ˌpɹi"},
                                                   {"co", "ˌkoʊ"}, {"non", "ˌnɑn"}, {"sub", "ˌsʌb"}};
      const size_t d = word.find('-');
      const std::string head = word.substr(0, d), rest = word.substr(d + 1);
      for (auto [k, v] : prefixes)
        if (k == Lower(head) && !rest.empty()) {
          std::vector<Part> o = {{head, std::string(v), K(v), "rule"}};
          for (auto& x : Resolve(rest, tag)) o.push_back(std::move(x));
          return o;
        }
      return SplitParts(word, tag);
    }
    static const Re digit("[0-9]"), runs("[0-9]+|[^0-9]+");
    if (Search(digit, word)) {
      std::vector<Part> o;
      FindIter(runs, word, [&](const M& m) {
        if (m.G()[0] >= '0' && m.G()[0] <= '9') {
          std::vector<std::string> ipas, ps;
          DigitIpa(m.G(), false, ipas);
          for (auto& i : ipas) ps.push_back(K(i));
          if (!ipas.empty()) o.push_back({std::string(m.G()), Join(ipas, " "), Join(ps, " "), "norm-num"});
        } else if (const std::string r(StripChars(m.G(), "-'")); !r.empty()) {
          for (auto& x : Resolve(r, tag)) o.push_back(std::move(x));
        }
      });
      return o;
    }
    if (const std::string f = Fold(word); f != word) {
      const std::string fk = LexKey(f), fl = Lower(f);
      auto h = Lex(fk, tag);
      if (!h && fk != fl) h = Lex(fl, tag);
      if (h) return {{word, h->ipa, K(h->ipa, h->unstressed), h->src + "+fold"}};
      if (Opt ipa = ApplyRules(fl, RuleLookup(tag))) return {{word, *ipa, K(*ipa), "rule+fold"}};
    }
    auto [ipa, ps] = Model(word);
    return {{word, ipa, ps, "model-fallback"}};
  }

  std::vector<Tok> Tokens(sv line, std::string& text, TraceLine* tr) const {
    text = NormalizeImpl(L, line);
    static const Re token_re(R"re([A-Za-z]+[0-9]+[A-Za-z0-9]*|[A-Za-zÀ-ÖØ-öø-ÿŒœÆæəɚɝɪʊʌɛæɑɔɹʃʒθðŋˈˌɾʔ'-]+|[0-9]+|\.\.\.|[,.!?;:()"])re");
    struct Tagged {
      size_t b, e;
      std::string tag;
    };
    std::vector<Tagged> tagged;
    if (g.oracle) {
      Oracle& o = *g.oracle;
      if (o.next < o.lines.size()) {
        size_t pos = 0;
        for (auto& [tok, tag] : o.lines[o.next]) {
          const size_t f = text.find(tok, pos);
          if (f == std::string::npos) {
            ++o.misaligned;
            continue;
          }
          tagged.push_back({f, f + tok.size(), tag});
          pos = f + tok.size();
          if (tr) tr->spacy.push_back(tok);
        }
      } else {
        ++o.misaligned;
      }
      ++o.next;
    } else {
      std::vector<std::string> words;
      for (auto [b, e] : TokenizeImpl(L, text)) {
        tagged.push_back({b, e, ""});
        words.emplace_back(sv(text).substr(b, e - b));
      }
      if (g.tagger) {
        auto tags = Tag(*g.tagger, words);
        for (size_t i = 0; i < tags.size() && i < tagged.size(); ++i) tagged[i].tag = tags[i];
      }
      if (tr) tr->spacy = words;
    }
    std::vector<Tok> out;
    size_t j = 0;
    FindIter(token_re, text, [&](const M& m) {
      const size_t s = m.B(), e = m.E();
      const sv w = m.G();
      if (w.find_first_not_of("'-,.!?;:()\"") == sv::npos) {
        const std::string p = w == "(" || w == ")" ? "," : w == "\"" ? "" : w == "..." ? "…" : std::string(w);
        if (!p.empty()) out.push_back({std::string(w), "", "", p, "punct"});
        return;
      }
      const std::string w2(StripChars(w, "-'"));
      if (w2.empty()) return;
      while (j < tagged.size() && tagged[j].e <= s) ++j;
      const std::string tag = j < tagged.size() && tagged[j].b < e ? tagged[j].tag : "";
      std::vector<Part> parts;
      try {
        parts = Resolve(w2, tag);
      } catch (const Unmappable&) {
        auto [ipa, ps] = Model(w2);
        parts = {{w2, ipa, ps, "model-fallback"}};
      }
      static const std::initializer_list<sv> function_tags = {"CC", "DT", "IN", "PRP", "PRP$", "TO", "MD", "WDT", "WP", "WP$", "EX", "PDT"};
      for (auto& p : parts) {
        std::string ps = p.ps;
        if (In(tag, function_tags) && parts.size() == 1) {
          int vowels = 0;
          for (size_t i = 0; i < ps.size();) vowels += Has(kVowels, Decode(ps, i));
          if (vowels == 1) ps = Replace(Replace(ps, "ˈ", ""), "ˌ", "");
        }
        out.push_back({p.text, tag, p.ipa.value_or(""), ps, p.src});
      }
    });
    return out;
  }

  std::string Line(sv line) const {
    std::string text;
    TraceLine tl;
    TraceLine* tr = g.trace ? &tl : nullptr;
    auto toks = Tokens(line, text, tr);
    std::string ps;
    for (auto& t : toks) {
      if (t.source == "punct") {
        const char32_t c = Last(ps);
        if (!ps.empty() && !Has(kPunct, c)) ps += t.ps;
        else if (!ps.empty() && c == ',' && t.ps != ",") ps = ps.substr(0, ps.size() - 1) + t.ps;
      } else if (!t.ps.empty()) {
        ps += (ps.empty() ? "" : " ") + t.ps;
      }
    }
    if (tr) {
      tl.norm = text;
      tl.toks = std::move(toks);
      g.trace->push_back(std::move(tl));
    }
    return std::string(StripChars(ps, " ,;:"));
  }
};

using U = std::u32string;

U Trim(std::u32string_view s) {
  const size_t b = s.find_first_not_of(U' ');
  return b == U::npos ? U() : U(s.substr(b, s.find_last_not_of(U' ') - b + 1));
}

bool Speakable(const U& s) {
  for (char32_t c : s)
    if (c != U' ' && !Has(kPunct, c)) return true;
  return false;
}

// Cuts s into pieces of at most max: last clause mark past max/4, else last space, else a hard cut.
void Pieces(U s, size_t max, std::vector<U>& out) {
  while (s.size() > max) {
    size_t cut = 0;
    for (size_t i = max; i-- > max / 4 && !cut;)
      if (Has(",;:…", s[i])) cut = i + 1;
    for (size_t i = max + 1; i-- > 1 && !cut;)
      if (s[i] == U' ') cut = i;
    if (!cut) cut = max;
    if (U h = Trim(std::u32string_view(s).substr(0, cut)); Speakable(h)) out.push_back(std::move(h));
    s = Trim(std::u32string_view(s).substr(cut));
  }
  if (Speakable(s)) out.push_back(std::move(s));
}

}  // namespace

std::vector<std::string> Chunk(const std::vector<std::string>& lines, const ChunkOpts& o) {
  const size_t max = std::max<size_t>(o.max, 1), target = std::min(o.target, max);
  std::vector<U> sents;
  for (auto& line : lines) {
    U s = Trim(U32(line));
    if (s.empty()) continue;
    if (!Has(kPunct, s.back())) s += U'.';
    for (size_t b = 0, i = 0; i < s.size(); ++i)
      if (i + 1 == s.size() || (Has(".!?", s[i]) && s[i + 1] == U' ')) {
        if (U t = Trim(std::u32string_view(s).substr(b, i + 1 - b)); !t.empty()) sents.push_back(std::move(t));
        b = i + 1;
      }
  }
  std::vector<std::string> out;
  U cur;
  auto flush = [&] {
    if (!cur.empty()) out.push_back(U8(cur)), cur.clear();
  };
  for (size_t k = 0; k < sents.size(); ++k) {
    std::vector<U> atoms;
    U& s = sents[k];
    auto cut_first = [&](size_t from, sv marks) {
      for (size_t i = from; i + 1 < s.size() && i < max; ++i)
        if (Has(marks, s[i])) {
          if (U h = Trim(std::u32string_view(s).substr(0, i + 1)); Speakable(h)) atoms.push_back(std::move(h));
          s = Trim(std::u32string_view(s).substr(i + 1));
          return true;
        }
      return false;
    };
    if (k == 0 && s.size() > o.first_max && !cut_first(std::max<size_t>(o.first_min, 1) - 1, ",;:…"))
      cut_first((o.first_min + o.first_max) / 2, " ");
    Pieces(std::move(s), max, atoms);
    for (auto& a : atoms) {
      if (!cur.empty() && cur.size() + 1 + a.size() > target) flush();
      cur = cur.empty() ? std::move(a) : cur + U' ' + a;
      if (out.empty()) flush();
    }
  }
  flush();
  return out;
}

std::vector<std::string> PhonemizeLines(const G2P& g, const std::string& text) {
  static const Re nl("\n+");
  const std::string t = Sanitize(text);
  const std::string body(Strip(t));
  std::vector<std::string> lines;
  Slime s{g, *g.lex};
  size_t p = 0;
  M m(nl);
  for (bool more = true; more;) {
    size_t e = body.size(), np = body.size();
    if (m.Find(nl, body, p)) e = m.B(), np = m.E();
    else more = false;
    const sv line = sv(body).substr(p, e - p);
    p = np;
    if (Strip(line).empty()) continue;
    std::string ps;
    try {
      ps = s.Line(line);
    } catch (...) {
    }
    if (!ps.empty()) lines.push_back(std::move(ps));
  }
  return lines;
}

std::vector<std::string> Phonemize(const G2P& g, const std::string& text, const ChunkOpts& o) {
  return Chunk(PhonemizeLines(g, text), o);
}


G2P LoadG2P(const unsigned char* data, size_t size, const Tagger* tagger, const Guesser* guesser) {
  auto* L = new Lexicon;
  if (size < 12 || std::memcmp(data, "KKG2P\x01\0\0", 8)) throw std::runtime_error("lexicon.bin: bad magic");
  size_t p = 8;
  auto u32 = [&] {
    if (p + 4 > size) throw std::runtime_error("lexicon.bin: truncated");
    uint32_t v;
    std::memcpy(&v, data + p, 4);
    p += 4;
    return v;
  };
  auto str = [&] {
    const uint32_t n = u32();
    if (p + n > size) throw std::runtime_error("lexicon.bin: truncated");
    sv s((const char*)data + p, n);
    p += n;
    return s;
  };
  std::unordered_map<std::string, std::vector<sv>> sec;
  for (uint32_t ns = u32(); ns--;) {
    const std::string name(str());
    auto& v = sec[name];
    v.resize(u32());
    for (auto& s : v) s = str();
  }
  L->keys = std::move(sec["lex.k"]), L->vals = std::move(sec["lex.v"]);
  L->skeys = std::move(sec["spacy.k"]), L->svals = std::move(sec["spacy.v"]);
  for (sv s : sec["vocab"]) L->vocab.insert(U32(s)[0]);
  for (sv s : sec["nfc.pair"]) {
    auto u = U32(s);
    L->nfc_pair[uint64_t(u[0]) << 32 | u[1]] = u[2];
  }
  for (sv s : sec["nfc.single"]) {
    auto u = U32(s);
    L->nfc_single[u[0]] = u[1];
  }
  auto& pat = sec["spacy.pat"];
  if (pat.size() != 4) throw std::runtime_error("lexicon.bin: missing spaCy patterns");
  L->prefix = std::make_unique<Re>(pat[0]), L->suffix = std::make_unique<Re>(pat[1]);
  L->infix = std::make_unique<Re>(pat[2]), L->url = std::make_unique<Re>(pat[3]);
  for (sv k : L->skeys) {
    bool affix = k.find(' ') != sv::npos || M(*L->prefix).Find(*L->prefix, k) || M(*L->suffix).Find(*L->suffix, k);
    FindIter(*L->infix, k, [&](const M&) { affix = true; });
    if (!affix) continue;
    std::vector<std::pair<size_t, size_t>> spans;
    TokenizeSpans(*L, k, spans);
    std::vector<std::string> toks;
    for (auto [b, e] : spans) toks.emplace_back(k.substr(b, e - b));
    if (!toks.empty()) L->matcher[toks[0]].emplace_back(std::move(toks), k);
  }
  G2P g;
  g.lex = L, g.tagger = tagger, g.guesser = guesser;
  return g;
}

std::string Normalize(const G2P& g, const std::string& text) { return NormalizeImpl(*g.lex, Sanitize(text)); }

std::vector<std::pair<size_t, size_t>> Tokenize(const G2P& g, const std::string& text) {
  return TokenizeImpl(*g.lex, text);
}

std::string ToKokoro(const G2P& g, const std::string& ipa, bool unstressed) { return ToK(*g.lex, ipa, unstressed); }

}  // namespace kk::g2p
