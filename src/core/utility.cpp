#include "fingerprint/core/utility.hpp"

namespace fp {

void normalizeImage(const cv::Mat &src, cv::Mat &dst, double dmean,
                    double dstd) {
  CV_Assert(!src.empty());

  cv::Mat src_F;
  src.convertTo(src_F, CV_32FC1);

  cv::Scalar m, s;
  cv::meanStdDev(src_F, m, s);

  double mean = m[0];
  double stddev = s[0];

  double scale = stddev == 0 ? 0 : dstd / stddev;

  dst = (src_F - mean) * scale + dmean;
}

float angleDiff(float a, float b) {
  float d = std::fabs(a - b);
  return std::min(d, float(CV_PI) - d);
}

} // namespace fp
