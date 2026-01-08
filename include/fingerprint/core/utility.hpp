#pragma once
#include <opencv2/core.hpp>

namespace fp {

void normalizeImage(const cv::Mat &src, cv::Mat &dst, double dmean,
                    double dstd);

float angleDiff(float a, float b);

} // namespace fp