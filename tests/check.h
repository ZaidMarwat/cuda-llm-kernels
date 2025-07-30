// A 40-line test harness, so the repo needs no test framework dependency.
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace check {

struct Test { std::string name; std::function<void()> fn; };
inline std::vector<Test>& registry() { static std::vector<Test> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, f}); } };

#define TEST(name) static void name(); static check::Reg reg_##name(#name, name); static void name()
#define EXPECT(cond) do { if (!(cond)) { std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++check::failures(); } } while (0)

// Max |a-b| / (atol + rtol*|b|) <= 1 elementwise. Returns worst ratio for reporting.
inline double close_ratio(const float* a, const float* b, size_t n, double atol, double rtol) {
  double worst = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::isnan(a[i]) != std::isnan(b[i])) return INFINITY;
    double r = std::fabs(double(a[i]) - b[i]) / (atol + rtol * std::fabs(double(b[i])));
    if (r > worst) worst = r;
  }
  return worst;
}

inline std::vector<float> randn(size_t n, unsigned seed, float scale = 1.0f) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

inline int run_all() {
  for (auto& t : registry()) {
    int before = failures();
    t.fn();
    std::printf("[%s] %s\n", failures() == before ? "  OK  " : " FAIL ", t.name.c_str());
  }
  std::printf("%zu tests, %d failures\n", registry().size(), failures());
  return failures() == 0 ? 0 : 1;
}

}  // namespace check
