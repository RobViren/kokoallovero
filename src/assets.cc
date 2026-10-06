#include "src/kk.h"

#define KK_INCBIN(sym, file)                                                              \
  __asm__(".section .rodata\n.balign 64\n" #sym ":\n.incbin \"" KK_ASSET_DIR "/" file "\"\n" #sym "_end:\n.previous\n"); \
  extern "C" const unsigned char sym[], sym##_end[];
KK_INCBIN(kk_asset_kokoro_bin, "kokoro.bin")
KK_INCBIN(kk_asset_kokoro_tsv, "kokoro.tsv")
KK_INCBIN(kk_asset_voices_bin, "voices.bin")
KK_INCBIN(kk_asset_voices_tsv, "voices.tsv")
KK_INCBIN(kk_asset_vocab_tsv, "vocab.tsv")
KK_INCBIN(kk_asset_lexicon, "g2p/lexicon.bin")
KK_INCBIN(kk_asset_tagger, "g2p/tagger.bin")
KK_INCBIN(kk_asset_guesser, "g2p/guesser.bin")

namespace kk {
Asset EmbeddedAsset(const std::string& name) {
#define KK_ASSET(sym) {sym, size_t(sym##_end - sym)}
  static const std::unordered_map<std::string, Asset> table = {
      {"kokoro.bin", KK_ASSET(kk_asset_kokoro_bin)}, {"kokoro.tsv", KK_ASSET(kk_asset_kokoro_tsv)},
      {"voices.bin", KK_ASSET(kk_asset_voices_bin)}, {"voices.tsv", KK_ASSET(kk_asset_voices_tsv)},
      {"vocab.tsv", KK_ASSET(kk_asset_vocab_tsv)},
      {"g2p/lexicon.bin", KK_ASSET(kk_asset_lexicon)}, {"g2p/tagger.bin", KK_ASSET(kk_asset_tagger)},
      {"g2p/guesser.bin", KK_ASSET(kk_asset_guesser)},
  };
  const auto it = table.find(name);
  return it == table.end() ? Asset{nullptr, 0} : it->second;
}
}  // namespace kk
