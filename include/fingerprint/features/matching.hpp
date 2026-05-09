#pragma once
#include "fingerprint/core/types.hpp"
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

  std::vector<MatchedPair> matchPairs(const std::vector<T> &setA,
                                      const std::vector<T> &setB) {
    std::vector<MatchedPair> result;
    for (int i = 0; i < (int)setA.size(); i++) {
      double best = -1.0;
      int best_j = 0;
      for (int j = 0; j < (int)setB.size(); j++) {
        double sim = T::compare(setA[i], setB[j]);
        if (sim > best) {
          best = sim;
          best_j = j;
        }
      }
      if (best > 0.0)
        result.push_back({i, best_j});
    }
    return result;
  }
};

} // namespace fp
