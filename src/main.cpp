#include "fingerprint/enhancement.hpp"
#include "fingerprint/features.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

int main() {
  std::string img_path = std::string(DATA_DIR) + "/raw/101_1.tif";
  cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  fp::Enhancer enhancer;
  auto enhanced = enhancer.enhance(img);
  fp::Detector detector;
  auto minutiae = detector.detect(enhanced.enhanced_img,
                                  enhanced.orientation_img, enhanced.mask);
  std::cout << minutiae.size() << "minutiae detected" << std::endl;
  return 0;
}
