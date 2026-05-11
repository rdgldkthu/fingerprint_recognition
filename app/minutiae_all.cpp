#include "fingerprint/core/minutiae.hpp"
#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/features/detection.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

namespace fs = std::filesystem;

int main() {
  fs::path raw_dir = fs::path(DATA_DIR) / "raw";
  fs::path out_dir = fs::path(DATA_DIR) / "minutiae_vis";

  fs::create_directories(out_dir);

  fp::Enhancer enhancer;
  fp::Detector detector;
  int ok = 0, failed = 0;

  for (const auto &entry : fs::directory_iterator(raw_dir)) {
    if (entry.path().extension() != ".tif")
      continue;

    cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "Failed to load: " << entry.path() << "\n";
      ++failed;
      continue;
    }

    auto enh = enhancer.enhance(img);
    auto minutiae = detector.detect(enh.enhanced_img, enh.orientation_img, enh.mask);

    cv::Mat raw_bgr;
    cv::cvtColor(img, raw_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat side_by_side;
    cv::hconcat(raw_bgr, fp::visualizeMinutiae(enh.enhanced_img, minutiae), side_by_side);

    fs::path out = out_dir / entry.path().stem();
    out += ".png";
    cv::imwrite(out.string(), side_by_side);
    std::cout << entry.path().filename().string()
              << "  →  " << minutiae.size() << " minutiae\n";
    ++ok;
  }

  std::cout << "\nDone: " << ok << " processed, " << failed << " failed.\n";
  return failed > 0 ? 1 : 0;
}
