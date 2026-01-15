#pragma once
#include <vector>

namespace fp {

template <typename T> class LSSMatcher {
public:
  double computeScore(const std::vector<T> &setA, const std::vector<T> &setB) {
    if (setA.empty() || setB.empty())
      return 0.0;

    double sum_max_sim = 0.0;
    for (const auto &a : setA) {
      double best = 0.0;
      for (const auto &b : setB) {
        best = std::max(best, T::compare(a, b));
      }
      sum_max_sim += best;
    }
    return sum_max_sim / std::max(setA.size(), setB.size());
  }
};

} // namespace fp
