#include "fingerprint/features/alignment_matcher.hpp"
#include "fingerprint/core/utility.hpp"
#include <algorithm>
#include <cmath>

namespace fp {

AlignmentMatcher::AlignmentMatcher(AlignmentMatcherParams params) : params_(params) {}

double AlignmentMatcher::computeScore(const std::vector<Minutia>& A,
                                       const std::vector<Minutia>& B) {
    if (A.empty() || B.empty()) return 0.0;
    auto matches = findBestAlignment(A, B);
    return static_cast<double>(matches.size()) / std::max(A.size(), B.size());
}

std::vector<MatchedPair> AlignmentMatcher::findBestAlignment(const std::vector<Minutia>& A,
                                                               const std::vector<Minutia>& B) {
    if (A.empty() || B.empty()) return {};

    std::vector<MatchedPair> best;

    for (int i = 0; i < (int)A.size(); ++i) {
        for (int j = 0; j < (int)B.size(); ++j) {
            float base_dtheta = B[j].theta - A[i].theta;
            for (int k = 0; k < 2; ++k) {
                float dtheta = normalizeAngle(base_dtheta + k * static_cast<float>(CV_PI));
                float cos_dt = std::cos(dtheta);
                float sin_dt = std::sin(dtheta);
                float rot_ai_x = A[i].x * cos_dt - A[i].y * sin_dt;
                float rot_ai_y = A[i].x * sin_dt + A[i].y * cos_dt;
                float tx = B[j].x - rot_ai_x;
                float ty = B[j].y - rot_ai_y;

                auto matches = matchUnderAlignment(A, B, dtheta, tx, ty);
                if (matches.size() > best.size())
                    best = std::move(matches);
            }
        }
    }
    return best;
}

std::vector<MatchedPair> AlignmentMatcher::matchUnderAlignment(
    const std::vector<Minutia>& A, const std::vector<Minutia>& B,
    float dtheta, float tx, float ty)
{
    float cos_dt = std::cos(dtheta);
    float sin_dt = std::sin(dtheta);

    struct Candidate { float dist; int ia, ib; };
    std::vector<Candidate> candidates;

    for (int i = 0; i < (int)A.size(); ++i) {
        float ta_x = A[i].x * cos_dt - A[i].y * sin_dt + tx;
        float ta_y = A[i].x * sin_dt + A[i].y * cos_dt + ty;
        float ta_theta = normalizeAngle(A[i].theta + dtheta);

        for (int j = 0; j < (int)B.size(); ++j) {
            if (A[i].type != B[j].type) continue;
            float ddx = ta_x - B[j].x;
            float ddy = ta_y - B[j].y;
            float dist = std::sqrt(ddx * ddx + ddy * ddy);
            if (dist > params_.spatial_tolerance) continue;
            if (angleDiff(ta_theta, B[j].theta) > params_.angle_tolerance) continue;
            candidates.push_back({dist, i, j});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

    std::vector<bool> used_a(A.size(), false);
    std::vector<bool> used_b(B.size(), false);
    std::vector<MatchedPair> matches;

    for (const auto& c : candidates) {
        if (!used_a[c.ia] && !used_b[c.ib]) {
            used_a[c.ia] = true;
            used_b[c.ib] = true;
            matches.push_back({c.ia, c.ib});
        }
    }
    return matches;
}

} // namespace fp
