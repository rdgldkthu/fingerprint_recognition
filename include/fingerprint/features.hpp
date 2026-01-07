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

private:
  void thin(const cv::Mat &src, cv::Mat &dst) const;
  std::vector<Minutia> detectMinutiae(const cv::Mat &skeleton,
                                      const cv::Mat &orientation) const;
  cv::Mat visualizeMinutiae(const cv::Mat &skeleton,
                            const std::vector<Minutia> &minutiae,
                            int radius = 3, int arrow_len = 12) const;
};

} // namespace fp
