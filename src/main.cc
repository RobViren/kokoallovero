#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include "src/kk.h"
#include "src/g2p/g2p.h"

namespace {

std::vector<char32_t> Utf8(const std::string& s) {
  std::vector<char32_t> out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    const int n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    char32_t cp = n == 1 ? c : c & (0x3F >> (n - 1));
    for (int k = 1; k < n; ++k) cp = cp << 6 | (s[i + k] & 0x3F);
    out.push_back(cp);
    i += n;
  }
  return out;
}

void WriteWav(const std::string& path, const std::vector<float>& a) {
  std::vector<int16_t> pcm(a.size());
  for (size_t i = 0; i < a.size(); ++i) pcm[i] = int16_t(std::clamp(a[i], -1.f, 1.f) * 32767);
  const uint32_t data = pcm.size() * 2, riff = 36 + data, fmt = 16, rate = 24000, bps = rate * 2;
  const uint16_t pcm_fmt = 1, ch = 1, align = 2, bits = 16;
  FILE* f = std::fopen(path.c_str(), "wb");
  std::fwrite("RIFF", 1, 4, f), std::fwrite(&riff, 4, 1, f), std::fwrite("WAVEfmt ", 1, 8, f);
  std::fwrite(&fmt, 4, 1, f), std::fwrite(&pcm_fmt, 2, 1, f), std::fwrite(&ch, 2, 1, f);
  std::fwrite(&rate, 4, 1, f), std::fwrite(&bps, 4, 1, f), std::fwrite(&align, 2, 1, f), std::fwrite(&bits, 2, 1, f);
  std::fwrite("data", 1, 4, f), std::fwrite(&data, 4, 1, f), std::fwrite(pcm.data(), 2, pcm.size(), f);
  std::fclose(f);
}

std::vector<unsigned char> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

kk::g2p::G2P LoadG2P(bool embedded, const std::string& dir) {
  static std::vector<unsigned char> files[3];
  auto get = [&](int i, const char* name) -> kk::Asset {
    if (embedded) return kk::EmbeddedAsset(name);
    files[i] = ReadBytes(dir + "/" + name);
    return {files[i].data(), files[i].size()};
  };
  const kk::Asset lex = get(0, "g2p/lexicon.bin"), tag = get(1, "g2p/tagger.bin"), guess = get(2, "g2p/guesser.bin");
  return kk::g2p::LoadG2P(lex.data, lex.size, kk::g2p::LoadTagger(tag.data, tag.size),
                          kk::g2p::LoadGuesser(guess.data, guess.size));
}

void WritePcm(const std::vector<float>& a) {
  std::vector<int16_t> pcm(a.size());
  for (size_t i = 0; i < a.size(); ++i) pcm[i] = int16_t(std::clamp(a[i], -1.f, 1.f) * 32767);
  std::fwrite(pcm.data(), 2, pcm.size(), stdout);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = "weights", voice = "af_heart", phonemes, text, out = "kk.wav", dump, noise_dir;
  int threads = 4, reps = 1;
  bool dir_given = false, has_text = false;
  float speed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "-w") dir = v, dir_given = true;
    else if (a == "-v") voice = v;
    else if (a == "-p") phonemes = v;
    else if (a == "-o") out = v;
    else if (a == "-t") threads = std::stoi(v);
    else if (a == "-r") reps = std::stoi(v);
    else if (a == "-s") speed = std::stof(v);
    else if (a == "--dump") dump = v;
    else if (a == "--noise") noise_dir = v;
    else if (a == "-i") text = v, has_text = true;
  }
  if (phonemes.empty() && !has_text) {
    std::ifstream f(dir + "/ref/phonemes.txt");
    std::getline(f, phonemes);
  }

  const auto t0 = std::chrono::steady_clock::now();
  const bool embedded = !dir_given && kk::EmbeddedAsset("kokoro.bin").size;
  auto floats = [](const char* n) { return reinterpret_cast<const float*>(kk::EmbeddedAsset(n).data); };
  kk::Model* model = embedded ? kk::LoadModel(kk::Weights(floats("kokoro.bin"), kk::EmbeddedAsset("kokoro.tsv").str()))
                              : kk::LoadModel(dir);
  const kk::Weights voices = embedded ? kk::Weights(floats("voices.bin"), kk::EmbeddedAsset("voices.tsv").str())
                                      : kk::Weights(dir + "/voices");
  std::unordered_map<char32_t, int> vocab;
  std::stringstream vf(embedded ? kk::EmbeddedAsset("vocab.tsv").str() : std::string());
  if (!embedded) vf << std::ifstream(dir + "/vocab.tsv").rdbuf();
  for (uint32_t cp, id; vf >> cp >> id;) vocab[cp] = id;
  std::fprintf(stderr, "%-14s %8.1f\n", "load",
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

  kk::Pool pool(threads);
  const bool stream = out == "-";
  std::vector<float> audio;
  bool spoke = false;
  const std::vector<float> gap(size_t(0.15 * 24000), 0.f);
  auto speak = [&](const std::string& ps) {
    const auto cps = Utf8(ps);
    if (cps.empty()) return;
    std::vector<int> ids{0};
    for (char32_t c : cps)
      if (auto it = vocab.find(c); it != vocab.end()) ids.push_back(it->second);
    ids.push_back(0);
    const float* style = voices(voice) + (std::min<size_t>(cps.size(), 510) - 1) * 256;
    std::vector<float> chunk;
    for (int r = 0; r < reps; ++r) {
      kk::Noise noise;
      if (!noise_dir.empty()) {
        noise.initial_phase = kk::ReadNpy(noise_dir + "/initial_phase.npy");
        noise.randn = kk::ReadNpy(noise_dir + "/noise.npy");
      }
      kk::Trace trace{r + 1 == reps ? dump : "", has_text};
      chunk = kk::Synthesize(*model, pool, ids, style, speed, noise, trace);
      double ms = 0;
      for (auto& [name, lap] : trace.laps) ms += lap;
      std::fprintf(stderr, "%-14s %8.1f  phonemes %zu audio %.2fs rtf %.3f\n", "synth", ms, cps.size(),
                   chunk.size() / 24000.0, ms / 1e3 / (chunk.size() / 24000.0));
    }
    if (stream) {
      if (spoke) WritePcm(gap);
      WritePcm(chunk);
      spoke = true;
    } else {
      if (!audio.empty()) audio.insert(audio.end(), gap.begin(), gap.end());
      audio.insert(audio.end(), chunk.begin(), chunk.end());
    }
  };

  if (!has_text) {
    speak(phonemes);
  } else {
    const kk::g2p::G2P g2p = LoadG2P(embedded, dir);
    auto say = [&](const std::string& line) {
      const auto t = std::chrono::steady_clock::now();
      const auto chunks = kk::g2p::Phonemize(g2p, line);
      std::fprintf(stderr, "%-14s %8.2f  chunks %zu\n", "g2p",
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(),
                   chunks.size());
      for (const auto& ps : chunks) speak(ps);
    };
    if (text == "-")
      for (std::string line; std::getline(std::cin, line);) say(line);
    else
      say(text);
  }
  if (!stream) WriteWav(out, audio);
}
