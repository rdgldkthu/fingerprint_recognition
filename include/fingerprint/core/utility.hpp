#pragma once
#include <opencv2/core.hpp>

namespace fp {

static const int dx[8] = {0, -1, -1, -1, 0, 1, 1, 1};
static const int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};

inline float normalizeAngle(float theta) {
  while (theta >= CV_PI) theta -= 2 * CV_PI;
  while (theta < -CV_PI) theta += 2 * CV_PI;
  return theta;
}

inline float angleDiff(float a, float b) {
  float d = std::fabs(a - b);
  return std::min(d, float(CV_PI) - d);
}

} // namespace fp
