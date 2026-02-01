#pragma once
#include <opencv2/core.hpp>

namespace fp {

cv::Mat extractFingerprintROI(const cv::Mat &gray_img, cv::Size ksize);
void removeThinkRidgesFromROI(cv::Mat &roi, const cv::Mat enhanced_img, float max_half_width);

} // namespace fp
