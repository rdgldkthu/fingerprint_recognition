#pragma once
#include "fingerprint/core/types.hpp"
#include <vector>

namespace fp {

struct AlignmentMatcherParams {
    float spatial_tolerance = 15.0f;
    float angle_tolerance   = 0.35f;
};

struct MatchedPair {
    int idx_a;
    int idx_b;
};

class AlignmentMatcher {
public:
    explicit AlignmentMatcher(AlignmentMatcherParams params = {});

    double computeScore(const std::vector<Minutia>& A, const std::vector<Minutia>& B);
    std::vector<MatchedPair> findBestAlignment(const std::vector<Minutia>& A,
                                               const std::vector<Minutia>& B);

private:
    AlignmentMatcherParams params_;

    std::vector<MatchedPair> matchUnderAlignment(const std::vector<Minutia>& A,
                                                  const std::vector<Minutia>& B,
                                                  float dtheta, float tx, float ty);
};

} // namespace fp
