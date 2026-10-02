#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Port of ag2p's speaking path (Slime.phonemize): text in, Kokoro phoneme chunks out. The lexicon stores ag2p IPA;
// kokoro_map runs last, as in Slime.
namespace kk::g2p {

struct Tagger;   // opaque, implemented in src/g2p/tagger.cc by the tagger agent
struct Guesser;  // opaque, implemented in src/g2p/guesser.cc by the guesser agent
Tagger* LoadTagger(const unsigned char* data, size_t size);
std::vector<std::string> Tag(const Tagger&, const std::vector<std::string>& tokens);  // spaCy en_core_web_sm tag_
Guesser* LoadGuesser(const unsigned char* data, size_t size);
std::string Guess(const Guesser&, const std::string& word);  // ag2p IPA, empty if no answer

struct Lexicon;

// Parity hooks. An oracle replaces the tokenizer+tagger (spaCy tokens and tags per normalized line, in call order)
// and the guesser (word -> IPA, empty for no answer).
struct Oracle {
  std::vector<std::vector<std::pair<std::string, std::string>>> lines;
  std::unordered_map<std::string, std::string> model;
  size_t next = 0;
  int misaligned = 0;
};
struct Tok {
  std::string text, tag, ipa, ps, source;
};
struct TraceLine {
  std::string norm;
  std::vector<std::string> spacy;  // tokens the tagger saw
  std::vector<Tok> toks;
};

struct G2P {
  const Lexicon* lex = nullptr;
  const Tagger* tagger = nullptr;
  const Guesser* guesser = nullptr;
  Oracle* oracle = nullptr;
  std::vector<TraceLine>* trace = nullptr;
};

// Chunks never exceed max Kokoro phonemes and are never empty. The first chunk is the first sentence alone (cut at its
// first clause mark past first_min if longer than first_max); later sentences pack greedily up to target. A sentence over
// max splits at its last clause mark (,;:…) past max/4, else its last space, else a hard cut. Newlines end sentences.
struct ChunkOpts {
  size_t max = 510, target = 400, first_max = 150, first_min = 60;
};

G2P LoadG2P(const unsigned char* lexicon, size_t size, const Tagger* tagger = nullptr, const Guesser* guesser = nullptr);
std::vector<std::string> Phonemize(const G2P&, const std::string& text, const ChunkOpts& = {});
// Phonemes per non-empty input line, unchunked: what ag2p's Slime.phonemize_line returns.
std::vector<std::string> PhonemizeLines(const G2P&, const std::string& text);
std::vector<std::string> Chunk(const std::vector<std::string>& lines, const ChunkOpts& = {});
std::string Normalize(const G2P&, const std::string& text);
// spaCy en_core_web_sm tokenizer: [begin, end) byte spans of the non-space tokens.
std::vector<std::pair<size_t, size_t>> Tokenize(const G2P&, const std::string& text);
std::string ToKokoro(const G2P&, const std::string& ipa, bool unstressed = false);

}  // namespace kk::g2p
