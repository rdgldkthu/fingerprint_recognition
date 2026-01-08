#pragma once
#include <opencv2/core.hpp>

namespace fp {

cv::Mat estimateRidgeFrequency(const cv::Mat &img,
                               const cv::Mat &orientation_img, int block_size);

} // namespace fp
