#pragma once
#include "fingerprint/core/types.hpp"
#include <opencv2/core.hpp>

namespace fp {

std::vector<Minutia> detectMinutiae(const cv::Mat &skeleton,
                                    const cv::Mat &orientation,
                                    float angle_tolerance = CV_PI / 6);

void pruneByMaskDistance(std::vector<Minutia> &minutiae, const cv::Mat &mask,
                         float min_dist);
void pruneByImageBorder(std::vector<Minutia> &minutiae, int width, int height,
                        int margin);

cv::Mat visualizeMinutiae(const cv::Mat &skeleton,
                          const std::vector<Minutia> &minutiae, int radius = 3,
                          int arrow_len = 12);

} // namespace fp
