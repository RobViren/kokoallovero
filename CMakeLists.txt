cmake_minimum_required(VERSION 3.24)
# Clang spills an accumulator on every iteration of the conv microkernel (about 0.6x the throughput), so default to GCC.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX} AND NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_CXX_COMPILER g++)
endif()
project(kokoallovero C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The kernels are tuned and tested on AVX2 only, and current clang rejects Highway's AVX3 target attributes.
option(KK_AVX512 "Let Highway dispatch to its AVX-512 targets" OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # The kernels keep vectors in arrays, which SVE's sizeless types do not allow. NEON only.
  add_compile_definitions("HWY_DISABLED_TARGETS=(HWY_SVE|HWY_SVE2|HWY_SVE_256|HWY_SVE2_128)")
elseif(NOT KK_AVX512)
  add_compile_definitions("HWY_DISABLED_TARGETS=(HWY_AVX3|HWY_AVX3_DL|HWY_AVX3_ZEN4|HWY_AVX3_SPR|HWY_AVX10_2)")
endif()
add_compile_options(-Wno-ignored-attributes)

include(FetchContent)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
FetchContent_Declare(highway
  URL https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON)

# PCRE2 gives the G2P the Python-re semantics it needs (lookbehind, Unicode \w \b). The release tarball, unlike the
# tag archive, includes the sljit sources the JIT needs.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(pcre2
  URL https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.49/pcre2-10.49.tar.bz2
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(highway pcre2)

option(KK_EMBED "Embed weights, voices and G2P assets from KK_ASSET_DIR into the binary" OFF)
option(KK_STATIC "Link fully static" OFF)
if(KK_STATIC)
  add_link_options(-static)
endif()
set(KK_ASSET_DIR ${CMAKE_SOURCE_DIR}/weights CACHE PATH "")

# Assets missing from KK_ASSET_DIR are downloaded at configure time. Nothing is fetched at runtime.
option(KK_FETCH_ASSETS "Download missing assets into KK_ASSET_DIR when configuring" ON)
set(KK_ASSET_URL https://huggingface.co/rodbiren/kokoro_parakeet_aio_models/resolve/766eb98b496915a437f38b1cec8e9458879cb2f3/kokoro CACHE STRING "")
if(KK_FETCH_ASSETS)
  foreach(f kokoro.bin kokoro.tsv voices.bin voices.tsv vocab.tsv g2p/lexicon.bin g2p/tagger.bin g2p/guesser.bin)
    if(NOT EXISTS ${KK_ASSET_DIR}/${f})
      message(STATUS "Downloading ${f}")
      file(DOWNLOAD ${KK_ASSET_URL}/${f} ${KK_ASSET_DIR}/${f}.part STATUS status)
      list(GET status 0 code)
      if(code)
        file(REMOVE ${KK_ASSET_DIR}/${f}.part)
        message(WARNING "Could not download ${f}: ${status}")
      else()
        file(RENAME ${KK_ASSET_DIR}/${f}.part ${KK_ASSET_DIR}/${f})
      endif()
    endif()
  endforeach()
endif()
if(KK_EMBED)
  set(KK_ASSETS src/assets.cc)
  set_source_files_properties(src/assets.cc PROPERTIES COMPILE_OPTIONS "--embed-dir=${KK_ASSET_DIR};-O0"
    OBJECT_DEPENDS "${KK_ASSET_DIR}/kokoro.bin;${KK_ASSET_DIR}/voices.bin;${KK_ASSET_DIR}/g2p/lexicon.bin;${KK_ASSET_DIR}/g2p/tagger.bin;${KK_ASSET_DIR}/g2p/guesser.bin")
else()
  set(KK_ASSETS src/noassets.cc)
endif()

add_library(kkg2p_lib STATIC src/g2p/g2p.cc src/g2p/tagger.cc src/g2p/guesser.cc)
target_include_directories(kkg2p_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(kkg2p_lib PUBLIC pcre2-8-static hwy)

add_executable(kk src/main.cc src/kokoro.cc src/ops.cc src/gemm.cc ${KK_ASSETS})
target_link_libraries(kk PRIVATE kkg2p_lib)

# G2P alone: text lines in, phoneme lines out.
add_executable(kkg2p src/g2p/kkg2p.cc)
target_link_libraries(kkg2p PRIVATE kkg2p_lib)
