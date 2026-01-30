#include "fingerprint/core/skeleton.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace {

void pruneIslands(cv::Mat &skel, int min_size) {
  CV_Assert(skel.type() == CV_8UC1);

  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(skel, labels, stats, centroids,
                                                 8, CV_32S);

  for (int i = 1; i < nLabels; ++i) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);

    if (area < min_size) {
      skel.setTo(0, labels == i);
    }
  }
}

}

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

  pruneIslands(thinned_img, 10);

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
