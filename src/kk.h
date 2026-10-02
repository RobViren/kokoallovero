#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hwy/aligned_allocator.h"

namespace kk {

constexpr int kM = 4;
constexpr int kHalo = 64;

inline size_t RoundUp(size_t a, size_t m) { return (a + m - 1) / m * m; }
size_t TimeStep();

// Persistent workers pinned one per physical core, fastest cores first, within the process affinity mask. Hybrid
// CPUs need this: the static work split makes the slowest core set the pace.
struct Pool {
  int n;
  explicit Pool(int n);
  ~Pool();
  template <class F> void Run(F&& f) {
    call_ = [](void* c, int i) { (*static_cast<F*>(c))(i); };
    ctx_ = &f;
    pending_.store(n - 1);
    gen_.fetch_add(1);
    gen_.notify_all();
    f(0);
    for (int p; (p = pending_.load()) != 0;) pending_.wait(p);
  }
  template <class F> void For(int count, F&& f) {
    Run([&](int i) {
      for (int j = count * i / n, e = count * (i + 1) / n; j < e; ++j) f(j);
    });
  }

 private:
  void Work(int i, int cpu);
  void (*call_)(void*, int) = nullptr;
  void* ctx_ = nullptr;
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> pending_{0};
  std::atomic<bool> stop_{false};
  std::vector<std::thread> workers_;
};

// Channel-major [C][T], time contiguous, kHalo zeros around every row. Rows are padded to kM and the time axis to
// the conv time step; everything past T stays zero.
struct Act {
  int C = 0, T = 0;
  size_t stride = 0;
  hwy::AlignedFreeUniquePtr<float[]> buf;
  Act() = default;
  Act(int C, int T);
  float* operator[](int c) { return buf.get() + c * stride + kHalo; }
  const float* operator[](int c) const { return buf.get() + c * stride + kHalo; }
};

struct Conv {
  hwy::AlignedFreeUniquePtr<float[]> w;  // [Cout/kM][Cin][K][kM]
  std::vector<float> b;
  int cin = 0, cout = 0, K = 0, dil = 1;
  int wm = 0;                  // Winograd F(wm, K) outputs per tile when enabled
  std::vector<float> AT, BT;   // [wm][A], [A][A]
  std::vector<Conv> U;         // A transformed 1x1 convs
};

void EnableWinograd(Conv& c, int m);

Conv PackConv(int cout, int cin, int K, int dil, const float* b, const std::function<float(int, int, int)>& w);
void ConvRaw(Pool& pool, const Conv& c, const float* x, size_t xs, float* y, size_t ys, size_t T);
void Apply(Pool& pool, const Conv& c, const Act& x, Act& y);
struct Moments {
  std::vector<double> sum, sq;
};

// y = conv(snake(a*x + b)) (+ y if residual), with alpha per input channel; m gets the moments of the new y.
void ConvSnake(Pool& pool, const Conv& c, const Act& x, const float* a, const float* b, const float* alpha, Act& y,
               bool residual, Moments& m);
void Linear(Pool* pool, const float* x, int n, int in, const float* w, const float* b, float* y, int out);

// Linear weights packed at load into column panels [out/NR][in][NR] for a register-blocked GEMM. Each *Part call does
// slice `part` of `parts`, so a caller can run a whole sequence of them inside one Pool::Run.
struct Gemm {
  hwy::AlignedFreeUniquePtr<float[]> w, b;
  int in = 0, out = 0;
};

Gemm PackGemm(int out, int in, const float* w, const float* b);  // w is torch Linear layout [out][in]
void MatmulPart(const Gemm& g, const float* x, int n, float* y, int part, int parts, const float* res = nullptr,
                bool gelu = false);
void Matmul(Pool& pool, const Gemm& g, const float* x, int n, float* y);
void AttentionPart(const float* qkv, int n, int D, int heads, float* ctx, int part, int parts);
void LayerNormPart(float* x, int n, int C, const float* g, const float* b, float eps, int part, int parts);

struct Weights {
  struct Tensor {
    const float* p;
    std::vector<int> shape;
  };
  std::unordered_map<std::string, Tensor> t;
  explicit Weights(const std::string& stem);
  Weights(const float* base, const std::string& index);
  const float* operator()(const std::string& name) const { return t.at(name).p; }
  const std::vector<int>& Shape(const std::string& name) const { return t.at(name).shape; }
};

struct Trace {
  std::string dump_dir;
  bool quiet = false;
  std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
  std::vector<std::pair<std::string, double>> laps;
  void Lap(const char* stage);
  void Dump(const std::string& name, const float* p, std::vector<int> shape);
  void Dump(const std::string& name, const Act& a);
};

struct Noise {
  std::vector<float> initial_phase, randn;  // [9], [L][9]; empty means draw from the rng
  unsigned seed = 0;
};

struct Model;
Model* LoadModel(const std::string& dir);
Model* LoadModel(Weights w);

struct Asset {
  const unsigned char* data;
  size_t size;
  std::string str() const { return {reinterpret_cast<const char*>(data), size}; }
};
Asset EmbeddedAsset(const std::string& name);  // size 0 when not embedded
std::vector<float> Synthesize(Model& m, Pool& pool, const std::vector<int>& ids, const float* style, float speed,
                              Noise& noise, Trace& trace);

void WriteNpy(const std::string& path, const float* p, const std::vector<int>& shape);
std::vector<float> ReadNpy(const std::string& path);

}  // namespace kk
