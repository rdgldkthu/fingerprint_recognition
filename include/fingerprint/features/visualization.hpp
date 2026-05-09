#pragma once
#include "fingerprint/core/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace fp {

cv::Mat visualizeMatching(const cv::Mat& img1, const cv::Mat& img2,
                           const std::vector<Minutia>& m1,
                           const std::vector<Minutia>& m2,
                           const std::vector<MatchedPair>& matches);

} // namespace fp
