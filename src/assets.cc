#include "src/kk.h"

namespace kk {
namespace {
alignas(64) const unsigned char kKokoroBin[] = {
#embed "kokoro.bin"
};
const unsigned char kKokoroTsv[] = {
#embed "kokoro.tsv"
};
alignas(64) const unsigned char kVoicesBin[] = {
#embed "voices.bin"
};
const unsigned char kVoicesTsv[] = {
#embed "voices.tsv"
};
const unsigned char kVocabTsv[] = {
#embed "vocab.tsv"
};
alignas(64) const unsigned char kLexicon[] = {
#embed "g2p/lexicon.bin"
};
alignas(64) const unsigned char kTagger[] = {
#embed "g2p/tagger.bin"
};
alignas(64) const unsigned char kGuesser[] = {
#embed "g2p/guesser.bin"
};
}  // namespace

Asset EmbeddedAsset(const std::string& name) {
  static const std::unordered_map<std::string, Asset> table = {
      {"kokoro.bin", {kKokoroBin, sizeof kKokoroBin}}, {"kokoro.tsv", {kKokoroTsv, sizeof kKokoroTsv}},
      {"voices.bin", {kVoicesBin, sizeof kVoicesBin}}, {"voices.tsv", {kVoicesTsv, sizeof kVoicesTsv}},
      {"vocab.tsv", {kVocabTsv, sizeof kVocabTsv}},
      {"g2p/lexicon.bin", {kLexicon, sizeof kLexicon}}, {"g2p/tagger.bin", {kTagger, sizeof kTagger}},
      {"g2p/guesser.bin", {kGuesser, sizeof kGuesser}},
  };
  const auto it = table.find(name);
  return it == table.end() ? Asset{nullptr, 0} : it->second;
}
}  // namespace kk
