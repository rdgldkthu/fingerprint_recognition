#pragma once
#include <opencv2/core.hpp>

namespace fp {

void skeletonizeRidges(const cv::Mat &src, cv::Mat &dst);

} // namespace fp
