#pragma once
#include "fingerprint/core/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace fp {

struct DetectorParams {
  int spur_max_len = 9;
  int island_min_size = 30;
  int lake_max_area = 150;
  float angle_tolerance = CV_PI / 6;
  float border_dist_min = 8.0f;
  int image_margin = 10;
};

class Detector {
public:
  Detector() = default;
  explicit Detector(DetectorParams params);

  std::vector<Minutia> detect(const cv::Mat &enhanced_img,
                              const cv::Mat &orientation,
                              const cv::Mat &mask) const;

private:
  DetectorParams params_;
};

} // namespace fp
