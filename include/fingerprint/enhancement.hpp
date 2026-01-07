#pragma once
#include <opencv2/core.hpp>

namespace fp {

struct EnhancementResult {
  cv::Mat enhanced_img;     // enhanced binary image
  cv::Mat orientation_img;  // CV_32F, radian, block-wise
  cv::Mat frequency_img;    // CV_32F
  cv::Mat mask;             // CV_8U
};

class Enhancer {
public:
  Enhancer() = default;

  EnhancementResult enhance(const cv::Mat &img) const;

private:
  // === Preprocessing ===
  void normalize(const cv::Mat &src, cv::Mat &dst, double dmean = 100,
                 double dstd0 = 100) const;

  // === Orientation ===
  cv::Mat estimateRidgeOrientation(const cv::Mat &img, int block_size = 16) const;
  void convert2SinCosImg(const cv::Mat &img, cv::Mat &sin_img,
                     cv::Mat &cos_img) const;

  // === Frequency ===
  cv::Mat estimateRidgeFrequency(const cv::Mat &img, const cv::Mat &orientation_img,
                            int block_size = 16) const;
  float computeBlockFrequency(const cv::Mat &img, float ori, int cy, int cx,
                              int block_size) const;
  float estimatePeriodFromXSignature(const std::vector<float> &x_sig) const;

  // === Region Mask ===
  cv::Mat generateRegionMask(const cv::Mat &img) const;

  // === Enhancement ===
  void applyGabor(const cv::Mat &src, cv::Mat &dst,
                  const cv::Mat &orientation_img, const cv::Mat &frequency_img,
                  const cv::Mat &region_mask, float kx = 4.0, float ky = 4.0,
                  int filter_size = 11) const;
  cv::Mat createGaborFilter(float freq, float ori, float kx,
                            float ky, int filter_size) const;
  void buildGaborFilterBank(const std::vector<float> &unique_freqs,
                            std::vector<std::vector<cv::Mat>> &bank, float kx,
                            float ky, int filter_size) const;
  void applyGaborFilterBank(const cv::Mat &src, cv::Mat &dst,
                            const cv::Mat &ori_img_resized,
                            const cv::Mat &freq_img_resized,
                            const cv::Mat &region_mask,
                            const std::unordered_map<int, int> &freq_index_map,
                            const std::vector<std::vector<cv::Mat>> &bank,
                            int filter_size) const;

  // === Debug/Visualization ===
  void showOrientation(const cv::Mat &bg_img, const cv::Mat &orientation_img,
                       int block_size = 16,
                       const char *winname = "Orientation") const;
};

} // namespace fp
