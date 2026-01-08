#pragma once
#include <opencv2/core.hpp>

namespace fp {

cv::Mat estimateRidgeOrientation(const cv::Mat &img, int block_size);
void showRidgeOrientation(const cv::Mat &bg_img, const cv::Mat &orientation_img,
                          int block_size, const char *winname = "Orientation");

} // namespace fp
