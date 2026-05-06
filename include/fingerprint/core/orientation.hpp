#pragma once
#include <opencv2/core.hpp>

namespace fp {

cv::Mat estimateRidgeOrientation(const cv::Mat &img, int block_size,
                                 int smooth_ksize = 5,
                                 float smooth_sigma = 3.0f);
void showRidgeOrientation(const cv::Mat &bg_img, const cv::Mat &orientation_img,
                          int block_size, const char *winname = "Orientation");

} // namespace fp
