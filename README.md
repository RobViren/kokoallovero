# kokoallovero

## Background
I have looked all over for a standalone, fast, and reasonable TTS engine for some projects I am working on. CrispASR, audio.cpp and others provided interfaces for my TTS of choice, but I found two things that I wanted to chase down. G2P is the current rabbit hole I find myself in. I have a whole side project trying to agentically build out a corpus, modeling for ambiguous IPA, and text normalization. Some of what was rolled into this project came from that. Not quite ready to share. But contained in this application is a greatly expanded **English Only** (sorry) G2P which has been informed by my exploration into IPA. The G2P is quite reasonable, text normalization could use improvement but works for my primary use case of creating a voice interface for Claude Code. Bundling in the G2P and having it, in my opinion, out perform the quality of espeak for American English was my goal. The setup for kokoro as deployed in python is a little ham-fisted and I wanted something I could just copy around without much effort.

The second item I wanted to dig into was why ggml/gguf performance was lagging behind python on the CPU which was counter intuitive. Given the smallness of the kokoro model it did not make sense to me that it would benefit greatly from quantization. So I wanted to explore what a SIMD focused kokoro implementation could yield. My buddy Claude and I dug into the implementations and pulled in Highway for SIMD so someone else could benefit from the effort if it panned out. My actual deployment target for this work is a potato speed server that sits in my dresser drawer with an i7-7700 so I was really only going to use acceleration that was relevant to that. After several iterations the CPU on both targets were essentially maxed out on what could be done without taking more drastic means of optimization. Performance came out ahead of any full model kokoro I have seen for CPU, but I am sure given enough model time and enough wrangling more could be found. Good enough for my use cases.

I'll keep exploring IPA modeling and my Claude Code voice interface and see when I feel like sharing. I keep most of my code local until I am at least reasonably happy about it so that is why some of the bin files pulled in do not have sources/training data attached. The rest of this is largely LLM drafted for full disclosure. I obligate myself to be clear about what is and is not AI generated.

## What is it
A standalone C++ inference engine for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text to speech on CPU. Text in, 24 kHz audio out, from one binary with no Python, no espeak and no runtime downloads. SIMD is [Google Highway](https://github.com/google/highway) with runtime dispatch. Sister project to [parakeetogo](https://github.com/RobViren/parakeetogo), which does the same for Parakeet speech to text.

It runs about 2x faster than PyTorch on the same CPU. Synthesis only, 4 threads, one 6.6 s sentence, voice `af_heart`:

| CPU | kk | torch | speedup |
| --- | --- | --- | --- |
| i7-14700F | 470 ms (RTF 0.071) | 916 ms | 1.95x |
| i7-7700 | 902 ms (RTF 0.137) | 1770 ms | 1.96x |

`bench/torch_synth.py` is the torch side of that comparison; `kk -p "$(cat phonemes.txt)" -r 7` is ours.

## Build

Needs CMake 3.24+ and GCC (Clang builds but its conv microkernel is about 0.6x the speed). Highway and PCRE2 are fetched at configure time.

```
cmake -B build -G Ninja
cmake --build build
```

Options:
- `-DKK_EMBED=ON` embeds every asset from `KK_ASSET_DIR` (default `weights/`), giving a single 350 MB executable that needs no files.
- `-DKK_AVX512=ON` lets Highway dispatch to its AVX-512 targets. The kernels are tuned and tested on AVX2 only, so this is off by default.
- `-DKK_STATIC=ON` links fully static, for a binary that runs on older glibc.

## ARM

The same source builds for aarch64 with NEON (Highway's SVE targets are disabled). Under qemu the audio matches the x86 render to within the model's own float noise floor. Cross-compiling from x86 (Debian: `g++-aarch64-linux-gnu`):

```
cmake -B build-arm64 -G Ninja -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DKK_STATIC=ON -DKK_EMBED=ON
```

ARM speed is untuned: the Winograd configuration and tile widths were swept on Intel (`KK_WINO`, `KK_WCOLS` override them).

## Weights

Configuring downloads the assets into `weights/` from [rodbiren/kokoro_parakeet_aio_models](https://huggingface.co/rodbiren/kokoro_parakeet_aio_models) if they are missing (`-DKK_FETCH_ASSETS=OFF` to skip, `-DKK_ASSET_URL=` to point elsewhere). Nothing is downloaded at runtime.

How they were made:
- **Model and voices** (`kokoro.{bin,tsv}`, `voices.{bin,tsv}`, `vocab.tsv`): `uv run tools/export_weights.py [voice ...]` downloads `kokoro-v1_0.pth` and the voice packs from hexgrad/Kokoro-82M and writes flat f32 blobs with weight norm folded, which the binary mmaps. It reproduces the hosted files byte for byte. With no arguments it exports `af_heart`, `am_michael` and `bf_emma`.
- **Lexicon** (`g2p/lexicon.bin`): baked from a private pronunciation database, so it ships as a binary only.
- **Tagger** (`g2p/tagger.bin`): `tools/tag_corpus.py` tags a text corpus with spaCy, and `tools/train_tagger.py` distils that into an averaged perceptron. The corpus used is private.
- **Guesser** (`g2p/guesser.bin`): `tools/g2p/train.py` trains the character-to-phoneme transformer on word and phone TSVs (format in `tools/g2p/data.py`), and `tools/g2p/export.py` writes the f16 binary. The training lexicon is private.

## Use

```
build/kk -i "The kettle is boiling now." -o out.wav
cat book.txt | build/kk -i - -o - | aplay -f S16_LE -r 24000 -c 1
```

- `-i text` synthesizes text; `-i -` reads lines from stdin.
- `-p phonemes` takes a Kokoro phoneme string directly and skips G2P.
- `-o file.wav` writes a wav; `-o -` streams raw s16le 24 kHz mono to stdout chunk by chunk.
- `-v voice` (default `af_heart`), `-s speed`, `-t threads` (default 4), `-w dir` (weights directory, default `weights`), `-r n` (repeat, for timing).
- `build/kkg2p` is the G2P alone: text lines on stdin, phoneme lines on stdout.

Per-stage timings go to stderr.

## How it works

- `src/kokoro.cc`, `src/ops.cc`, `src/gemm.cc`: the model. The iSTFTNet vocoder dominates the cost, so its resblock convs are direct convolutions on time tiles that stay in L2, using 1D Winograd, with AdaIN, snake and the residual fused into the same pass. ConvTranspose is polyphase. Threads split over time tiles on a pool pinned to physical cores.
- `src/g2p/`: English text to Kokoro phonemes. A lexicon with part-of-speech variants, an averaged-perceptron tagger to pick homograph readings, and a small transformer that guesses unknown words. Long text is chunked at sentence and clause boundaries, never over Kokoro's 510 phoneme limit.
- `ref/kokoro.py`: a compact PyTorch reimplementation of Kokoro that loads the published checkpoint unchanged. It is the spec the C++ was ported from and the baseline it is measured against.
- `tools/`: how the assets were made (weight export, tagger and guesser training) and the parity checks. `export_weights.py`, `dump_ref.py`, `parity.py` and `bench/torch_synth.py` run as they are. The G2P training scripts need data you supply.

The vocoder is chaotic at the float ulp level, so parity is measured as SNR against a floor: torch compared with itself after nudging the style vector by one ulp (`tools/dump_ref.py`, `tools/parity.py`).

## License

Apache 2.0, the same as Kokoro-82M, which this derives from. Kokoro builds on StyleTTS2 and iSTFTNet.
