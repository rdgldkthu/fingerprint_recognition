#pragma once
#include "fingerprint/core/types.hpp"
#include <opencv2/core.hpp>

namespace fp {

std::vector<Minutia> detectMinutiae(const cv::Mat &skeleton,
                                    const cv::Mat &orientation,
                                    const cv::Mat &mask,
                                    float angle_tolerance = CV_PI / 6,
                                    float border_dist_min = 8.0f,
                                    int image_margin = 10);
cv::Mat visualizeMinutiae(const cv::Mat &skeleton,
                          const std::vector<Minutia> &minutiae, int radius = 3,
                          int arrow_len = 12);

} // namespace fp
