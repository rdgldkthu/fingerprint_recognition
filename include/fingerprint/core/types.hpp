#pragma once
#include <opencv2/core.hpp>

namespace fp {

enum class MinutiaType { ENDING, BIFURCATION };

struct Minutia {
  int y;
  int x;
  float theta; // radian, ridge direction
  MinutiaType type;
};

struct EnhancementResult {
  cv::Mat enhanced_img;    // enhanced binary image
  cv::Mat orientation_img; // CV_32F, radian, block-wise
  cv::Mat frequency_img;   // CV_32F
  cv::Mat mask;            // CV_8U
};

} // namespace fp
