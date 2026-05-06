#include "fingerprint/enhancement/enhancement.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

namespace fs = std::filesystem;

int main() {
  fs::path raw_dir      = fs::path(DATA_DIR) / "raw";
  fs::path enhanced_dir = fs::path(DATA_DIR) / "enhanced";

  fs::create_directories(enhanced_dir);

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
    fs::path out = enhanced_dir / entry.path().filename();
    cv::imwrite(out.string(), result.enhanced_img);
    std::cout << "Enhanced: " << entry.path().filename().string() << "\n";
    ++ok;
  }

  std::cout << "\nDone: " << ok << " enhanced, " << failed << " failed.\n";
  return failed > 0 ? 1 : 0;
}
