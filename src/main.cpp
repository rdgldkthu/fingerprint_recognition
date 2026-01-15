#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/features/detection.hpp"
#include "fingerprint/features/matching.hpp"
#include "fingerprint/features/mcc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

int main(int argc, char **argv) {
  // Read Fingerprint Images
  std::string img_path1 = std::string(DATA_DIR) + "/raw/101_1.tif";
  std::string img_path2 = std::string(DATA_DIR) + "/raw/101_2.tif";
  cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(img_path2, cv::IMREAD_GRAYSCALE);

  // Initialization
  fp::Enhancer enhancer;
  fp::Detector detector;
  fp::MCCExtractor descriptor;
  fp::LSSMatcher<fp::Cylinder> matcher;

  // Fingerprint Enhancement
  auto enhanced1 = enhancer.enhance(img1);
  auto enhanced2 = enhancer.enhance(img2);

  // Minutiae Detection
  auto minutiae1 = detector.detect(enhanced1.enhanced_img,
                                   enhanced1.orientation_img, enhanced1.mask);
  auto minutiae2 = detector.detect(enhanced2.enhanced_img,
                                   enhanced2.orientation_img, enhanced2.mask);
  std::cout << "Minutiae detected in image 1: " << minutiae1.size()
            << std::endl;
  std::cout << "Minutiae detected in image 2: " << minutiae2.size()
            << std::endl;

  // Descriptor Extraction
  auto descriptors1 = descriptor.extract(minutiae1);
  auto descriptors2 = descriptor.extract(minutiae2);

  // Descriptor Matching
  auto match_score = matcher.computeScore(descriptors1, descriptors2);
  std::cout << match_score << std::endl;

  return 0;
}
