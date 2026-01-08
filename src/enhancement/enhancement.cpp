#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/core/frequency.hpp"
#include "fingerprint/core/gabor.hpp"
#include "fingerprint/core/orientation.hpp"
#include "fingerprint/core/roi.hpp"
#include "fingerprint/core/utility.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fp {

EnhancementResult Enhancer::enhance(const cv::Mat &img) const {
  if (img.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    return EnhancementResult();
  }

  cv::Mat gray_img;
  if (img.channels() == 3)
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
  else
    gray_img = img.clone();

  // Normalization
  cv::Mat normalized_img;
  normalizeImage(gray_img, normalized_img, params_.target_mean,
                 params_.target_std);

  // Orientation Image Estimation
  cv::Mat orientation_img =
      estimateRidgeOrientation(normalized_img, params_.ori_block_size);

  // Frequency Image Estimation
  cv::Mat frequency_img = estimateRidgeFrequency(
      normalized_img, orientation_img, params_.freq_block_size);

  // Region Mask Extraction
  cv::Mat region_mask = extractFingerprintROI(gray_img, params_.roi_ksize);

  // Recoverability Check
  double RECOVERABLE_THRESHOLD = params_.recoverable_threshold;
  double recoverable_ratio = cv::sum(region_mask / 255)[0] /
                             (region_mask.rows * region_mask.cols) * 100.0;
  if (recoverable_ratio < RECOVERABLE_THRESHOLD) {
    std::cout << "Image Rejected: Recoverable region is " << recoverable_ratio
              << "% of the image, which is below threshold("
              << RECOVERABLE_THRESHOLD << "%)" << std::endl;
    return EnhancementResult();
  }

  // Gabor Filtering
  cv::Mat enhanced_img;
  applyGaborFilter(normalized_img, enhanced_img, orientation_img, frequency_img,
                   region_mask, params_.kx, params_.ky,
                   params_.gabor_filter_size);

// Visulization
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Original", cv::WINDOW_NORMAL);
  cv::moveWindow("Original", 50, 50);
  cv::imshow("Original", gray_img);

  cv::Mat norm_vis_img;
  cv::convertScaleAbs(normalized_img, norm_vis_img);
  cv::namedWindow("Normalized", cv::WINDOW_NORMAL);
  cv::moveWindow("Normalized", 450, 50);
  cv::imshow("Normalized", norm_vis_img);

  showRidgeOrientation(gray_img, orientation_img, params_.ori_block_size);
  cv::namedWindow("Orientation Image", cv::WINDOW_NORMAL);
  cv::resizeWindow("Orientation Image", gray_img.size());
  cv::moveWindow("Orientation Image", 850, 410);
  cv::imshow("Orientation Image", orientation_img);

  cv::namedWindow("Frequency Image", cv::WINDOW_NORMAL);
  cv::resizeWindow("Frequency Image", gray_img.size());
  cv::moveWindow("Frequency Image", 1150, 410);
  cv::imshow("Frequency Image", frequency_img);

  cv::namedWindow("Region Mask", cv::WINDOW_NORMAL);
  cv::moveWindow("Region Mask", 50, 410);
  cv::imshow("Region Mask", region_mask);
  std::cout << "Recoverable region: " << recoverable_ratio << '%' << std::endl;

  cv::namedWindow("Enhanced Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Enhanced Image", 450, 410);
  cv::imshow("Enhanced Image", enhanced_img);
  cv::waitKey(0);
#endif

  enhanced_img.convertTo(enhanced_img, CV_8UC1);
  return {enhanced_img, orientation_img, frequency_img, region_mask};
}

} // namespace fp
