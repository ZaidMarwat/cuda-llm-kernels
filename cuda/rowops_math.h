// The per-lane math of the warp softmax and LayerNorm kernels, as plain functions usable
// from both device code and host code. Keeping them here lets tests/test_rowops_math.cpp
// check the algorithms (online softmax, Welford + parallel merge, the 32-lane butterfly)
// on a CPU with no GPU. It checks the math, not the kernels themselves.
#pragma once
#include <cmath>

#include "hd.h"

namespace ck {

// Running softmax state: m = max seen so far, d = sum of exp(x - m) over what was seen.
// Adding one element costs one exp: whichever of (old max, new x) is smaller gets rescaled.
CK_HD void online_push(float& m, float& d, float x) {
  if (x > m) {
    d = d * expf(m - x) + 1.0f;  // m = -inf on the first element gives d*0 + 1
    m = x;
  } else if (x != -INFINITY) {  // masked logits (-inf) add nothing; also avoids -inf - -inf
    d += expf(x - m);
  }
}

// Merge two softmax states (from two lanes): rescale each sum to the larger max.
CK_HD void online_merge(float& m, float& d, float m2, float d2) {
  const float mn = fmaxf(m, m2);
  if (mn == -INFINITY) return;  // both empty; avoids exp(-inf - -inf) = NaN
  d = d * expf(m - mn) + d2 * expf(m2 - mn);
  m = mn;
}

// Welford state for n values: mean and M2 = sum of squared deviations from the mean.
// Welford rather than sum / sum-of-squares: E[x^2] - E[x]^2 cancels catastrophically when the
// mean is large relative to the spread.
CK_HD void welford_push(float& n, float& mean, float& m2, float x) {
  n += 1.0f;
  const float delta = x - mean;
  mean += delta / n;
  m2 += delta * (x - mean);
}

// Chan, Golub & LeVeque (1979) parallel merge. n is a float so it can go through a shuffle.
CK_HD void welford_merge(float& n, float& mean, float& m2, float nb, float meanb, float m2b) {
  const float nn = n + nb;
  if (nn == 0.0f) return;
  const float delta = meanb - mean;
  mean += delta * nb / nn;
  m2 += m2b + delta * delta * n * nb / nn;
  n = nn;
}

}  // namespace ck
