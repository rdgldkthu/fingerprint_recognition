#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/core/frequency.hpp"
#include "fingerprint/core/gabor.hpp"
#include "fingerprint/core/orientation.hpp"
#include "fingerprint/core/roi.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

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

}

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

  // ROI Extraction
  cv::Mat roi = extractFingerprintROI(gray_img, params_.roi_ksize);

  // Recoverability Check
  double RECOVERABILITY_THRESHOLD = params_.recoverable_threshold;
  double recoverability =
      cv::sum(roi / 255)[0] / (roi.rows * roi.cols) * 100.0;
  if (recoverability < RECOVERABILITY_THRESHOLD) {
    std::cout << "Image Rejected: Recoverable region is " << recoverability
              << "% of the image, which is below threshold("
              << RECOVERABILITY_THRESHOLD << "%)" << std::endl;
    return EnhancementResult();
  }

  // Gabor Filtering
  cv::Mat enhanced_img;
  applyGaborFilter(normalized_img, enhanced_img, orientation_img, frequency_img,
                   roi, params_.kx, params_.ky,
                   params_.gabor_filter_size);
  enhanced_img.convertTo(enhanced_img, CV_8UC1);

  // Update ROI
  removeThinkRidgesFromROI(roi, enhanced_img, params_.max_half_width);

  enhanced_img.setTo(255, ~roi);

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

  cv::namedWindow("ROI", cv::WINDOW_NORMAL);
  cv::moveWindow("ROI", 50, 410);
  cv::imshow("ROI", roi);
  std::cout << "Recoverable region: " << recoverability << '%' << std::endl;

  cv::namedWindow("Enhanced Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Enhanced Image", 450, 410);
  cv::imshow("Enhanced Image", enhanced_img);
  cv::waitKey(0);
#endif

  return {enhanced_img, orientation_img, frequency_img, roi};
}

} // namespace fp
