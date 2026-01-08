#pragma once
#include "fingerprint/core/types.hpp"
#include "fingerprint/enhancement/params.hpp"
#include <opencv2/core.hpp>

namespace fp {

class Enhancer {
public:
  explicit Enhancer(const EnhancerParams &params = {}) : params_(params) {}

  EnhancementResult enhance(const cv::Mat &img) const;

private:
  EnhancerParams params_;
};

} // namespace fp
