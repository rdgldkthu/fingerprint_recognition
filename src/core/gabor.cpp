#include "fingerprint/core/gabor.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <set>

namespace {

cv::Mat createGaborFilter(float freq, float ori, float kx, float ky,
                          int filter_size) {
  cv::Mat gabor_filter(filter_size, filter_size, CV_32FC1);

  float sigma_x = kx / freq;
  float sigma_y = ky / freq;

  float cos_phi = std::cos(ori);
  float sin_phi = std::sin(ori);

  int half_size = filter_size / 2;
  for (int y = -half_size; y <= half_size; ++y) {
    for (int x = -half_size; x <= half_size; ++x) {
      // Rotate coordinates
      float x_phi = x * cos_phi + y * sin_phi;
      float y_phi = -x * sin_phi + y * cos_phi;

      // Gabor formula
      float exponent = -0.5f * ((x_phi * x_phi) / (sigma_x * sigma_x) +
                                (y_phi * y_phi) / (sigma_y * sigma_y));
      float cosine = std::cos(2.0f * CV_PI * freq * x_phi);

      gabor_filter.at<float>(y + half_size, x + half_size) =
          std::exp(exponent) * cosine;
    }
  }
  return gabor_filter;
}

void buildGaborFilterBank(const std::vector<float> &unique_freqs,
                          std::vector<std::vector<cv::Mat>> &bank, float kx,
                          float ky, int filter_size) {
  int angle_increment = 3;
  int angle_steps = 180 / angle_increment;

  bank.resize(unique_freqs.size(), std::vector<cv::Mat>(angle_steps));

  for (int fi = 0; fi < unique_freqs.size(); ++fi) {
    float freq = unique_freqs[fi];
    for (int oi = 0; oi < angle_steps; ++oi) {
      float ori = (oi * angle_increment) * CV_PI / 180.f;
      bank[fi][oi] = createGaborFilter(freq, ori, kx, ky, filter_size);
    }
  }
}

void applyGaborFilterBank(const cv::Mat &src, cv::Mat &dst,
                          const cv::Mat &ori_img_resized,
                          const cv::Mat &freq_img_resized,
                          const cv::Mat &region_mask,
                          const std::unordered_map<int, int> &freq_index_map,
                          const std::vector<std::vector<cv::Mat>> &bank,
                          int filter_size) {
  int rows = src.rows;
  int cols = src.cols;

  int angle_increment = 3;
  int angle_steps = 180 / angle_increment;

  dst = cv::Mat::zeros(rows, cols, CV_32FC1);

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (region_mask.at<uchar>(y, x) == 0) {
        dst.at<float>(y, x) = 255.f;
        continue;
      }

      float ori = ori_img_resized.at<float>(y, x);
      float freq = freq_img_resized.at<float>(y, x);

      if (freq <= 0.f)
        continue;

      int freq_key = static_cast<int>(std::round(freq * 100));
      auto it = freq_index_map.find(freq_key);
      if (it == freq_index_map.end())
        continue;

      int fi = it->second;
      int oi =
          static_cast<int>(std::round(ori * 180 / CV_PI / angle_increment)) %
          angle_steps;
      if (oi < 0)
        oi += angle_steps;

      const cv::Mat &gabor_filter = bank[fi][oi];

      // Apply Gabor filter
      float sum = 0.f;
      for (int fy = -filter_size / 2; fy <= filter_size / 2; ++fy) {
        if (y + fy < 0 || y + fy >= rows)
          continue;
        const float *src_ptr = src.ptr<float>(y + fy);
        const float *filter_ptr = gabor_filter.ptr<float>(fy + filter_size / 2);
        for (int fx = -filter_size / 2; fx <= filter_size / 2; ++fx) {
          if (x + fx < 0 || x + fx >= cols)
            continue;
          sum += src_ptr[x + fx] * filter_ptr[fx + filter_size / 2];
        }
      }

      dst.at<float>(y, x) = sum;
    }
  }
}
} // namespace

namespace fp {

void applyGaborFilter(const cv::Mat &src, cv::Mat &dst,
                      const cv::Mat &orientation_img,
                      const cv::Mat &frequency_img, const cv::Mat &region_mask,
                      float kx, float ky, int filter_size) {
  CV_Assert(!src.empty());
  CV_Assert(!orientation_img.empty());
  CV_Assert(!frequency_img.empty());
  CV_Assert(!region_mask.empty());
  CV_Assert(src.type() == CV_32FC1);
  CV_Assert(orientation_img.type() == CV_32FC1);
  CV_Assert(frequency_img.type() == CV_32FC1);
  CV_Assert(region_mask.type() == CV_8UC1);

  // Resize orientation and frequency images to match source image size
  cv::Mat ori_resized, freq_resized;
  cv::resize(orientation_img, ori_resized, src.size(), 0, 0, cv::INTER_NEAREST);
  cv::resize(frequency_img, freq_resized, src.size(), 0, 0, cv::INTER_NEAREST);

  // Extract unique frequencies
  std::set<float> freq_set;
  for (int y = 0; y < freq_resized.rows; ++y) {
    const float *freq_ptr = freq_resized.ptr<float>(y);
    for (int x = 0; x < freq_resized.cols; ++x) {
      float f = std::round(freq_ptr[x] * 100) / 100.f;
      if (f > 0.f)
        freq_set.insert(f);
    }
  }
  std::vector<float> unique_freqs(freq_set.begin(), freq_set.end());

  // Map frequency to index
  std::unordered_map<int, int> freq_index_map;
  for (int i = 0; i < unique_freqs.size(); ++i)
    freq_index_map[static_cast<int>(unique_freqs[i] * 100)] = i;

  // Build Gabor filter bank
  std::vector<std::vector<cv::Mat>> bank;
  buildGaborFilterBank(unique_freqs, bank, kx, ky, filter_size);

  // Apply Gabor filter bank
  applyGaborFilterBank(src, dst, ori_resized, freq_resized, region_mask,
                       freq_index_map, bank, filter_size);
}

} // namespace fp
