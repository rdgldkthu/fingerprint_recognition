#pragma once
#include <opencv2/core.hpp>

namespace fp {

cv::Mat extractFingerprintROI(const cv::Mat &img, cv::Size ksize);

} // namespace fp
