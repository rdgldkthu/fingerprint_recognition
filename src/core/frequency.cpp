#include "fingerprint/core/frequency.hpp"
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>

namespace {

float estimatePeriodFromXSignature(const std::vector<float> &x_sig) {
  // Mean centering
  std::vector<float> sig = x_sig;
  float mean = 0.f;
  for (float v : sig)
    mean += v;
  mean /= sig.size();
  for (float &v : sig)
    v -= mean;

  // Find zero-crossings
  std::vector<int> zero_crossings;
  for (size_t i = 1; i < sig.size(); ++i) {
    if (sig[i - 1] * sig[i] < 0) {
      zero_crossings.push_back(static_cast<int>(i));
    }
  }

  // Compute average period
  if (zero_crossings.size() < 2)
    return -1.f;

  float total_period = 0.f;
  int count = 0;
  for (size_t i = 1; i < zero_crossings.size(); ++i) {
    int period = zero_crossings[i] - zero_crossings[i - 1];
    total_period += period;
    count++;
  }

  float period = total_period / count * 2.f;

  return period;
}

float computeBlockFrequency(const cv::Mat &img, float ori, int cy, int cx,
                            int block_size) {
  // Sample the gray values along the orthogonal direction of the ridge
  // orientation
  const int window_length = 2 * block_size;
  std::vector<float> x_sig(window_length, 0.f);

  float sin_ori = std::sin(ori);
  float cos_ori = std::cos(ori);

  for (int k = 0; k < window_length; ++k) {
    float sum = 0.f;
    float dk = (k - window_length / 2.f);

    for (int t = 0; t < block_size; ++t) {
      float dt = (t - block_size / 2.f);

      float u = cy + dk * sin_ori - dt * cos_ori;
      float v = cx + dk * cos_ori + dt * sin_ori;

      if (u < 0 || u >= img.rows || v < 0 || v >= img.cols)
        continue;

      sum += img.at<float>(u, v);
    }

    x_sig[k] = sum / block_size;
  }

  // Estimate the period from the x-signature
  float T = estimatePeriodFromXSignature(x_sig);
  if (T < 3.0f || T > 25.0f)
    return -1.f;

  // Compute frequency
  float freq = 1.0f / T;

  return freq;
}

} // namespace

namespace fp {

cv::Mat estimateRidgeFrequency(const cv::Mat &img,
                               const cv::Mat &orientation_img, int block_size) {
  CV_Assert(!img.empty());
  CV_Assert(!orientation_img.empty());
  CV_Assert(img.type() == CV_32FC1);
  CV_Assert(orientation_img.type() == CV_32FC1);

  const int rows = img.rows / block_size;
  const int cols = img.cols / block_size;

  cv::Mat frequency_img(rows, cols, CV_32FC1, cv::Scalar(-1));

  for (int by = 0; by < rows; ++by) {
    for (int bx = 0; bx < cols; ++bx) {
      float ori = orientation_img.at<float>(by, bx);

      // Center of the block
      float cx = bx * block_size + block_size * 0.5f;
      float cy = by * block_size + block_size * 0.5f;

      float freq = computeBlockFrequency(img, ori, cy, cx, block_size);

      frequency_img.at<float>(by, bx) = freq;
    }
  }

  // Interpolation to fill the missing frequency values
  cv::Mat kernel = cv::getGaussianKernel(7, 3, CV_32F);
  cv::Mat kernel2D = kernel * kernel.t();

  for (int by = 0; by < rows; ++by) {
    for (int bx = 0; bx < cols; ++bx) {
      if (frequency_img.at<float>(by, bx) == -1.f) {
        float weighted_sum = 0.f;
        float weight_sum = 0;

        for (int dy = -3; dy <= 3; ++dy) {
          for (int dx = -3; dx <= 3; ++dx) {
            int ny = by + dy;
            int nx = bx + dx;
            if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
              float f = frequency_img.at<float>(ny, nx);
              if (f > 0.f) {
                float w = kernel2D.at<float>(dy + 3, dx + 3);
                weighted_sum += f * w;
                weight_sum += w;
              }
            }
          }
        }

        frequency_img.at<float>(by, bx) = weighted_sum / weight_sum;
      }
    }
  }

  // Perform smoothing to the frequency image
  cv::GaussianBlur(frequency_img, frequency_img, cv::Size(7, 7), 0);

  return frequency_img;
}

} // namespace fp
