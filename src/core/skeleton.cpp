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

  cv::rectangle(thinned_img, cv::Point(0, 0),
                cv::Point(src.cols - 1, src.rows - 1), 0, 1);

  cv::bitwise_not(thinned_img, dst);

  return;
}

} // namespace fp
