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
  float max_half_width = 4.0f;

  // Recoverability Check
  double recoverable_threshold = 40.0;

  // Gabor Filtering
  float kx = 4.0f;
  float ky = 4.0f;
  int gabor_filter_size = 11;
  int gabor_angle_step_deg = 3;

  // Frequency Estimation
  float freq_min_period = 3.0f;
  float freq_max_period = 25.0f;
  int freq_interp_kernel_size = 7;
  float freq_interp_sigma = 3.0f;

  // Orientation Smoothing
  int ori_smooth_ksize = 5;
  float ori_smooth_sigma = 3.0f;
};

} // namespace fp
