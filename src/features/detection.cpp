#include "fingerprint/features/detection.hpp"
#include "fingerprint/core/minutiae.hpp"
#include "fingerprint/core/pruning.hpp"
#include "fingerprint/core/skeleton.hpp"
#include "fingerprint/core/types.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fp {

Detector::Detector(DetectorParams params) : params_(params) {}

std::vector<Minutia> Detector::detect(const cv::Mat &enhanced_img,
                                      const cv::Mat &orientation,
                                      const cv::Mat &mask) const {
  std::vector<Minutia> minutiae;
  if (enhanced_img.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    return minutiae;
  }

  cv::Mat skeleton;
  skeletonizeRidges(enhanced_img, skeleton);
  pruneIslands(skeleton, params_.island_min_size);
  pruneSpurs(skeleton, params_.spur_max_len);
  // pruneLakes(skeleton, params_.lake_area_thresh);
  // pruneHBreaks(skeleton);

  minutiae = detectMinutiae(skeleton, orientation, params_.angle_tolerance);
  pruneByMaskDistance(minutiae, mask, params_.border_dist_min);
  pruneByImageBorder(minutiae, skeleton.cols, skeleton.rows, params_.image_margin);

#ifdef FP_DEBUG_VIS
  cv::namedWindow("Skeleton Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Skeleton Image", 50, 750);
  cv::imshow("Skeleton Image", skeleton);

  cv::Mat vis = visualizeMinutiae(skeleton, minutiae);
  cv::namedWindow("Visualize Minutiae", cv::WINDOW_NORMAL);
  cv::moveWindow("Visualize Minutiae", 450, 750);
  cv::imshow("Visualize Minutiae", vis);
  cv::waitKey(0);
#endif

  return minutiae;
}

} // namespace fp
