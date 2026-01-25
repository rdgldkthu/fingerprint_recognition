#pragma once
#include <opencv2/core.hpp>

namespace fp {

inline float angleDiff(float a, float b) {
  float d = std::fabs(a - b);
  return std::min(d, float(CV_PI) - d);
}

} // namespace fp