#include "fingerprint/features/detection.hpp"
#include "fingerprint/core/minutiae.hpp"
#include "fingerprint/core/skeleton.hpp"
#include "fingerprint/core/types.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fp {

std::vector<Minutia> Detector::detect(const cv::Mat &enhanced_img,
                                      const cv::Mat &orientation,
                                      const cv::Mat &mask) const {
  std::vector<Minutia> minutiae;
  if (enhanced_img.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    return minutiae;
  }

  cv::Mat thinned_img;
  skeletonizeRidges(enhanced_img, thinned_img);

  minutiae = detectMinutiae(thinned_img, orientation, mask);

#ifdef FP_DEBUG_VIS
  cv::Mat vis = visualizeMinutiae(thinned_img, minutiae);
  cv::namedWindow("Visualize Minutiae", cv::WINDOW_NORMAL);
  cv::moveWindow("Visualize Minutiae", 450, 750);
  cv::imshow("Visualize Minutiae", vis);
  cv::waitKey(0);
#endif

  return minutiae;
}

} // namespace fp
