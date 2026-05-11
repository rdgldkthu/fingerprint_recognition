#include "fingerprint/enhancement/enhancement.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

namespace fs = std::filesystem;

int main() {
  fs::path raw_dir     = fs::path(DATA_DIR) / "raw";
  fs::path compare_dir = fs::path(DATA_DIR) / "compare";

  fs::create_directories(compare_dir);

  fp::Enhancer enhancer;
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

    auto result = enhancer.enhance(img);

    cv::Mat raw_bgr, enhanced_bgr;
    cv::cvtColor(img, raw_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(result.enhanced_img, enhanced_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat side_by_side;
    cv::hconcat(raw_bgr, enhanced_bgr, side_by_side);

    fs::path out = compare_dir / entry.path().stem();
    out += ".png";
    cv::imwrite(out.string(), side_by_side);
    std::cout << "Compared: " << entry.path().filename().string() << "\n";
    ++ok;
  }

  std::cout << "\nDone: " << ok << " compared, " << failed << " failed.\n";
  return failed > 0 ? 1 : 0;
}
