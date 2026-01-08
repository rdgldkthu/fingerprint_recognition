#pragma once
#include <opencv2/core.hpp>

namespace fp {

void applyGaborFilter(const cv::Mat &src, cv::Mat &dst,
                      const cv::Mat &orientation_img,
                      const cv::Mat &frequency_img, const cv::Mat &region_mask,
                      float kx, float ky, int filter_size);

} // namespace fp
