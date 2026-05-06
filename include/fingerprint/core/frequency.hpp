#pragma once
#include <opencv2/core.hpp>

namespace fp {

cv::Mat estimateRidgeFrequency(const cv::Mat &img,
                               const cv::Mat &orientation_img, int block_size,
                               float min_period = 3.0f, float max_period = 25.0f,
                               int interp_ksize = 7, float interp_sigma = 3.0f);

} // namespace fp
