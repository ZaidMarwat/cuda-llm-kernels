// Online-softmax bookkeeping for decode attention, shared by the CUDA kernel and the CPU
// emulation test (tests/test_attention_math.cpp).
//
// A partial attention result over some set of positions is (m, l, acc[D]):
//   m   = max score seen,  l = sum_t exp(s_t - m),  acc = sum_t exp(s_t - m) * v_t
// The final output is acc / l. Two partial results over disjoint position sets combine
// exactly by rescaling both to the larger max. This is what lets the KV cache be split
// across thread blocks (split-KV, as in FlashDecoding) and merged afterwards.
#pragma once
#include <cmath>

#include "hd.h"

namespace ck {

// Add one position with score s. The caller then updates acc[i] = acc[i]*scale + p*v[i].
CK_HD void attn_push(float& m, float& l, float s, float& scale, float& p) {
  const float mn = fmaxf(m, s);
  scale = (m == -INFINITY) ? 0.0f : expf(m - mn);  // first position: nothing to rescale
  p = expf(s - mn);
  l = l * scale + p;
  m = mn;
}

// Merge partial (m2, l2, acc2) into (m, l, acc). The caller then updates
// acc[i] = acc[i]*s1 + acc2[i]*s2. Either side may be empty (m = -inf, l = 0).
CK_HD void attn_merge(float& m, float& l, float m2, float l2, float& s1, float& s2) {
  const float mn = fmaxf(m, m2);
  if (mn == -INFINITY) { s1 = s2 = 0.0f; return; }
  s1 = (m == -INFINITY) ? 0.0f : expf(m - mn);
  s2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - mn);
  l = l * s1 + l2 * s2;
  m = mn;
}

}  // namespace ck
