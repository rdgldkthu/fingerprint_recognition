#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/features/alignment_matcher.hpp"
#include "fingerprint/features/detection.hpp"
#include "fingerprint/features/visualization.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

int main(int argc, char **argv) {
  // Read Fingerprint Images
  std::string img_path1 = std::string(DATA_DIR) + "/raw/104_3.tif";
  std::string img_path2 = std::string(DATA_DIR) + "/raw/104_7.tif";
  cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(img_path2, cv::IMREAD_GRAYSCALE);

  if (!img1.empty())
    std::cout << img_path1 << " read successfully" << std::endl;
  if (!img2.empty())
    std::cout << img_path2 << " read successfully" << std::endl;

  // Initialization
  fp::Enhancer enhancer;
  fp::Detector detector;
  fp::AlignmentMatcher matcher;

  // Fingerprint Enhancement
  auto enhanced1 = enhancer.enhance(img1);
  auto enhanced2 = enhancer.enhance(img2);

  // Minutiae Detection
  auto minutiae1 = detector.detect(enhanced1.enhanced_img,
                                   enhanced1.orientation_img, enhanced1.mask);
  auto minutiae2 = detector.detect(enhanced2.enhanced_img,
                                   enhanced2.orientation_img, enhanced2.mask);
  std::cout << "Minutiae detected in image 1: " << minutiae1.size() << std::endl;
  std::cout << "Minutiae detected in image 2: " << minutiae2.size() << std::endl;

  // Matching
  auto matches = matcher.findBestAlignment(minutiae1, minutiae2);
  double score = matcher.computeScore(minutiae1, minutiae2);
  std::cout << "Match score: " << score << std::endl;

  // Visualization
  cv::Mat vis = fp::visualizeMatching(enhanced1.enhanced_img, enhanced2.enhanced_img,
                                      minutiae1, minutiae2, matches);
  cv::imshow("Matching Result", vis);
  cv::waitKey(0);

  return 0;
}
