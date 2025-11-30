#pragma once
#include <opencv2/core.hpp>

namespace fp {

class Enhancer {
public:
  Enhancer(int block_size = 16, int kernel_size = 5)
      : block_size(block_size), kernel_size(kernel_size) {}

  cv::Mat enhance(const cv::Mat &input) const;

private:
  // === Preprocessing ===
  void normalize(const cv::Mat &src, cv::Mat &dst, double dmean = 100,
                 double dstd0 = 100) const;

  // === Orientation ===
  cv::Mat estimateOrientation(const cv::Mat &img) const;
  void computeGradients(const cv::Mat &img, cv::Mat &grad_x,
                        cv::Mat &grad_y) const;
  void convertSinCos(const cv::Mat &img, cv::Mat &sin_img,
                     cv::Mat &cos_img) const;

  // === Frequency ===
  cv::Mat estimateFrequency(const cv::Mat &img,
                            const cv::Mat &orientation_img) const;
  float computeBlockFrequency(const cv::Mat &img, float ori, int cy,
                              int cx) const;
  float estimatePeriod(const std::vector<float> &x_sig) const;

  // === Region Mask ===
  cv::Mat generateRegionMask(const cv::Mat &img) const;

  // === Enhancement ===
  void applyGabor(const cv::Mat &src, cv::Mat &dst,
                  const cv::Mat &orientation_img, const cv::Mat &frequency_img,
                  const cv::Mat &region_mask, float kx = 4.0, float ky = 4.0,
                  int filter_size = 11) const;
  cv::Mat createGaborFilter(float frequency, float orientation, float kx,
                            float ky, int filter_size) const;
  void buildGaborFilterBank(const std::vector<float> &unique_freqs,
                            std::vector<std::vector<cv::Mat>> &bank, float kx,
                            float ky, int filter_size) const;
  void applyGaborFilterBank(const cv::Mat &src, cv::Mat &dst,
                            const cv::Mat &ori_img_resized,
                            const cv::Mat &freq_img_resized,
                            const std::unordered_map<int, int> &freq_index_map,
                            const std::vector<std::vector<cv::Mat>> &bank,
                            int filter_size) const;

  // === Debug/Visualization ===
  void showOrientation(const cv::Mat &bg_img, const cv::Mat &orientation_img,
                       const char *winname = "Orientation") const;

  const int block_size;
  const int kernel_size;
  const int window_length = 2 * block_size;
};

} // namespace fp
