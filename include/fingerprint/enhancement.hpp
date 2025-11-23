#pragma once
#include <opencv2/core.hpp>

namespace fp {

class Enhancer {
public:
  Enhancer(int block_size = 16, int kernel_size = 5)
      : block_size(block_size), kernel_size(kernel_size) {}

  cv::Mat enhance(const cv::Mat &input) const;

private:
  const int block_size;
  const int kernel_size;

  cv::Mat estimateOrientation(const cv::Mat &img) const;
  cv::Mat estimateFrequency(const cv::Mat &img,
                            const cv::Mat &orientation_img) const;
  cv::Mat generateRegionMask(const cv::Mat &img) const;
  void normalize(const cv::Mat &src, cv::Mat &dst, double dmean = 100,
                 double dstd0 = 100) const;
  void applyGabor(const cv::Mat &src, cv::Mat &dst,
                  const cv::Mat &orientation_img,
                  const cv::Mat &frequency_img) const;
  void showOrientation(const cv::Mat &bg_img, const cv::Mat &orientation_img,
                       const char *winname = "Orientation") const;

  // Static Member Functions
  static void calculateGradients(const cv::Mat &img, cv::Mat &grad_x,
                                 cv::Mat &grad_y);
  static void convertSinCos(const cv::Mat &img, cv::Mat &sin_img,
                            cv::Mat &cos_img);
};

} // namespace fp
