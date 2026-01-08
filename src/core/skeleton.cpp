#include "fingerprint/core/skeleton.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace fp {

void skeletonizeRidges(const cv::Mat &src, cv::Mat &dst) {
  CV_Assert(!src.empty());

  cv::Mat binary_img;
  cv::threshold(src, binary_img, 127, 255, cv::THRESH_BINARY);

  cv::Mat inverted_img;
  cv::bitwise_not(binary_img, inverted_img);

  cv::Mat thinned_img;
  cv::ximgproc::thinning(inverted_img, thinned_img,
                         cv::ximgproc::THINNING_ZHANGSUEN);

  cv::Mat reinverted_img;
  cv::bitwise_not(thinned_img, reinverted_img);
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Thinned Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Thinned Image", 50, 750);
  cv::imshow("Thinned Image", reinverted_img);
#endif

  dst = reinverted_img;
  return;
}

} // namespace fp
