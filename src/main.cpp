#include <iostream>
#include <opencv2/opencv.hpp>

#include "fingerprint/enhancement.hpp"
#include "fingerprint/feature_detection.hpp"
#include "fingerprint/matching.hpp"

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

int main() {
  std::string img_path1 = std::string(DATA_DIR) + "/raw/101_1.tif";
  cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
  fp::Enhancer enhancer;
  cv::Mat e1 = enhancer.enhance(img1);
  /*
  {
    std::string img_path1 = std::string(DATA_DIR) + "/raw/101_1.tif";
    std::string img_path2 = std::string(DATA_DIR) + "/raw/101_2.tif";

    cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img_path2, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
      std::cerr << "Failed to load images: " << img_path1 << " or " << img_path2
                << std::endl;
      return 1;
    }

    fp::Enhancer enhancer;
    fp::FeatureDetector detector;
    fp::Matcher matcher;

    cv::Mat e1 = enhancer.enhance(img1);
    cv::Mat e2 = enhancer.enhance(img2);

    auto m1 = detector.detectMinutiae(e1);
    auto m2 = detector.detectMinutiae(e2);

    auto res = matcher.match(m1, m2);

    std::cout << "Match score: " << res.score << std::endl;
  }
  */
  return 0;
}
