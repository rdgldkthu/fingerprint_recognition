#pragma once
#include <opencv2/core.hpp>

namespace fp {

struct EnhancerParams {
  // Normalization
  double target_mean = 100.0;
  double target_std = 100.0;

  // Orientation Image Estimation
  int ori_block_size = 10;

  // Frequency Image Estimation
  int freq_block_size = 16;

  // Region Mask Extraction
  cv::Size roi_ksize{31, 61};

  // Recoverability Check
  double recoverable_threshold = 40.0;

  // Gabor Filtering
  float kx = 4.0f;
  float ky = 4.0f;
  int gabor_filter_size = 11;
};

} // namespace fp
