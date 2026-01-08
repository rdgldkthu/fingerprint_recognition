#pragma once
#include "fingerprint/core/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace fp {

class Detector {
public:
  Detector() = default;

  std::vector<Minutia> detect(const cv::Mat &enhanced_img,
                              const cv::Mat &orientation,
                              const cv::Mat &mask) const;
};

} // namespace fp
