#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "src/g2p/g2p.h"

namespace {

using namespace kk::g2p;

std::vector<unsigned char> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(f), {}};
}

std::string Unescape(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\' || i + 1 == s.size()) {
      o += s[i];
      continue;
    }
    const char c = s[++i];
    o += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;
  }
  return o;
}

std::string Escape(const std::string& s) {
  std::string o;
  for (char c : s) o += c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\t' ? "\\t" : c == '\r' ? "\\r" : std::string(1, c);
  return o;
}

std::vector<std::string> Fields(const std::string& line, char d) {
  std::vector<std::string> o;
  std::stringstream ss(line);
  for (std::string f; std::getline(ss, f, d);) o.push_back(f);
  if (!line.empty() && line.back() == d) o.emplace_back();
  return o;
}

struct Record {
  std::string id, text;
  Oracle oracle;
};

// Parity harness records: "#\tid\ttext", then per normalized line "S\ttok\x1ftag\t...", "M\tword\tipa".
std::vector<Record> ReadRecords(const std::string& path) {
  std::ifstream f(path);
  std::vector<Record> rs;
  for (std::string line; std::getline(f, line);) {
    auto fs = Fields(line, '\t');
    if (fs.empty()) continue;
    if (fs[0] == "#") {
      rs.push_back({fs[1], Unescape(fs.size() > 2 ? fs[2] : "")});
    } else if (fs[0] == "S") {
      auto& l = rs.back().oracle.lines.emplace_back();
      for (size_t i = 1; i < fs.size(); ++i) {
        auto tt = Fields(fs[i], '\x1f');
        l.emplace_back(Unescape(tt[0]), tt.size() > 1 ? tt[1] : "");
      }
    } else if (fs[0] == "M") {
      rs.back().oracle.model[Unescape(fs[1])] = fs.size() > 2 ? Unescape(fs[2]) : "";
    }
  }
  return rs;
}

}  // namespace

int main(int argc, char** argv) {
  std::string lex = "weights/g2p/lexicon.bin", tagger_path, guesser_path, records;
  bool oracle = false, norm = false, chunk_mode = false;
  ChunkOpts co;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "-l") lex = next();
    else if (a == "--tagger") tagger_path = next();
    else if (a == "--guesser") guesser_path = next();
    else if (a == "--records") records = next();
    else if (a == "--oracle") oracle = true;
    else if (a == "--norm") norm = true;
    else if (a == "--chunks") chunk_mode = true;
    else if (a == "--max") co.max = std::stoul(next());
    else if (a == "--target") co.target = std::stoul(next());
    else if (a == "--first-max") co.first_max = std::stoul(next());
    else if (a == "--first-min") co.first_min = std::stoul(next());
    else {
      std::fprintf(stderr,
                   "usage: kkg2p [-l lexicon.bin] [--tagger f] [--guesser f] [--records f [--oracle]] [--norm] < lines\n"
                   "       kkg2p --chunks [--max n] [--target n] [--first-max n] [--first-min n] < text  (len<TAB>chunk)\n");
      return 2;
    }
  }
  const auto lex_bytes = ReadFile(lex);
  std::vector<unsigned char> tagger_bytes, guesser_bytes;
  const Tagger* tagger = nullptr;
  const Guesser* guesser = nullptr;
  if (!tagger_path.empty()) tagger_bytes = ReadFile(tagger_path), tagger = LoadTagger(tagger_bytes.data(), tagger_bytes.size());
  if (!guesser_path.empty())
    guesser_bytes = ReadFile(guesser_path), guesser = LoadGuesser(guesser_bytes.data(), guesser_bytes.size());
  auto t0 = std::chrono::steady_clock::now();
  G2P g = LoadG2P(lex_bytes.data(), lex_bytes.size(), tagger, guesser);
  const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::vector<double> ms;
  auto timed = [&](const std::string& text) {
    auto t = std::chrono::steady_clock::now();
    auto lines = PhonemizeLines(g, text);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
    return lines;
  };
  auto cps = [](const std::string& s) { return std::count_if(s.begin(), s.end(), [](char c) { return (c & 0xC0) != 0x80; }); };
  if (chunk_mode) {
    const std::string text{std::istreambuf_iterator<char>(std::cin), {}};
    for (auto& c : Chunk(timed(text), co)) std::cout << cps(c) << '\t' << c << '\n';
  } else if (records.empty()) {
    for (std::string line; std::getline(std::cin, line);) {
      if (norm) std::cout << Normalize(g, line) << '\t';
      auto chunks = Chunk(timed(line), co);
      for (size_t i = 0; i < chunks.size(); ++i) std::cout << (i ? "\t" : "") << chunks[i];
      std::cout << '\n';
    }
  } else {
    int misaligned = 0;
    for (auto& r : ReadRecords(records)) {
      std::vector<TraceLine> trace;
      g.trace = &trace;
      g.oracle = oracle ? &r.oracle : nullptr;
      std::vector<std::string> lines, chunks;
      std::string err;
      try {
        lines = timed(r.text);
        chunks = Chunk(lines, co);
      } catch (const std::exception& e) {
        err = e.what();
      }
      misaligned += r.oracle.misaligned;
      std::cout << "#\t" << r.id << '\n';
      for (auto& tl : trace) {
        std::cout << "N\t" << Escape(tl.norm) << "\nK";
        for (auto& t : tl.spacy) std::cout << '\t' << Escape(t);
        std::cout << '\n';
        for (auto& t : tl.toks)
          std::cout << "W\t" << Escape(t.text) << '\t' << t.tag << '\t' << Escape(t.ipa) << '\t' << Escape(t.ps) << '\t'
                    << t.source << '\n';
      }
      if (!err.empty()) std::cout << "E\t" << Escape(err) << '\n';
      std::cout << 'P';
      for (auto& l : lines) std::cout << '\t' << Escape(l);
      std::cout << "\nC";
      for (auto& c : chunks) std::cout << '\t' << Escape(c);
      std::cout << '\n';
    }
    if (oracle) std::fprintf(stderr, "oracle misaligned: %d\n", misaligned);
  }
  if (!ms.empty()) {
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double x : s) sum += x;
    std::fprintf(stderr, "kkg2p: load %.1f ms, %zu texts, mean %.3f ms, p50 %.3f, p99 %.3f, max %.3f\n", load_ms, s.size(),
                 sum / s.size(), s[s.size() / 2], s[s.size() * 99 / 100], s.back());
  }
}
