#include "fingerprint/enhancement.hpp"
#include <cmath>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fp {

cv::Mat Enhancer::enhance(const cv::Mat &input) const {
  if (input.empty())
    return cv::Mat();

  cv::Mat gray_img;
  if (input.channels() == 3)
    cv::cvtColor(input, gray_img, cv::COLOR_BGR2GRAY);
  else
    gray_img = input.clone();

  // Normalization
  cv::Mat normalized_img;
  normalize(gray_img, normalized_img);

  // Orientation Image Estimation
  cv::Mat orientation_img = estimateOrientation(normalized_img);

  // Frequency Image Estimation
  cv::Mat frequency_img = estimateFrequency(normalized_img, orientation_img);

  // Recoverability Check
  cv::Mat region_mask = generateRegionMask(gray_img);
  double RECOVERABLE_THRESHOLD = 40.0;
  double recoverable_ratio =
      cv::sum(region_mask / 255)[0] / (region_mask.rows * region_mask.cols) * 100.0;

  if (recoverable_ratio < RECOVERABLE_THRESHOLD) {
    std::cout << "Image Rejected: Recoverable region is " <<
    recoverable_ratio
              << "% of the image, which is less than threshold: "
              << RECOVERABLE_THRESHOLD << '%' << std::endl;
    return cv::Mat();
  }

  // Gabor Filtering
  cv::Mat enhanced_img;
  applyGabor(normalized_img, enhanced_img, orientation_img, frequency_img);

#ifdef FP_DEBUG_VIS // Visualize Process
  {
    cv::imshow("Original", gray_img);
    cv::waitKey(0);

    cv::Mat norm_vis;
    cv::convertScaleAbs(normalized_img, norm_vis);
    cv::imshow("Normalized", norm_vis);
    cv::waitKey(0);

    showOrientation(gray_img, orientation_img);
    cv::imshow("Orientation Image", orientation_img);
    cv::waitKey(0);

    cv::imshow("Frequency Image", frequency_img);
    cv::waitKey(0);

    cv::imshow("Region Mask", region_mask);
    cv::waitKey(0);
    std::cout << "Recoverable region: " << recoverable_ratio << '%'
              << std::endl;

    // cv::imshow("Enhanced", enhanced);
    // cv::waitKey(0);
  }
#endif

  return enhanced_img;
}

// === Preprocessing ===
void Enhancer::normalize(const cv::Mat &src, cv::Mat &dst, double dmean,
                         double dstd) const {
  CV_Assert(!src.empty());

  cv::Mat srcF;
  src.convertTo(srcF, CV_32FC1);

  cv::Scalar m, s;
  cv::meanStdDev(srcF, m, s);

  double mean = m[0];
  double stddev = s[0];

  double scale = stddev == 0 ? 0 : dstd / stddev;

  dst = (srcF - mean) * scale + dmean;
}

// === Orientation ===
cv::Mat Enhancer::estimateOrientation(const cv::Mat &img) const {
  CV_Assert(!img.empty());
  CV_Assert(img.type() == CV_32FC1);

  // Compute gradients in the x and y directions of image
  cv::Mat grad_x, grad_y;
  computeGradients(img, grad_x, grad_y);

  const int img_rows = img.rows;
  const int img_cols = img.cols;
  const int ori_rows = img_rows / block_size;
  const int ori_cols = img_cols / block_size;

  cv::Mat orientation_img(ori_rows, ori_cols, CV_32FC1);

  // Estimate ridge orientation block-wise
  for (int by = 0; by < ori_rows; by++) {
    for (int bx = 0; bx < ori_cols; bx++) {
      float Vy = 0.f;
      float Vx = 0.f;

      for (int y = by * block_size; y < (by + 1) * block_size; y++) {
        const float *grad_y_ptr = grad_y.ptr<float>(y);
        const float *grad_x_ptr = grad_x.ptr<float>(y);

        for (int x = bx * block_size; x < (bx + 1) * block_size; x++) {
          Vy += 2 * grad_x_ptr[x] * grad_y_ptr[x];
          Vx += grad_x_ptr[x] * grad_x_ptr[x] - grad_y_ptr[x] * grad_y_ptr[x];
        }
      }

      float theta = 0.5f * atan2(Vy, Vx);
      orientation_img.at<float>(by, bx) = theta;
    }
  }

  // Perform low-pass filtering
  cv::Mat smooth_orientation_img(ori_rows, ori_cols, CV_32FC1);
  cv::Mat phi_y(ori_rows, ori_cols, CV_32FC1);
  cv::Mat phi_x(ori_rows, ori_cols, CV_32FC1);

  convertSinCos(2 * orientation_img, phi_y, phi_x);

  cv::GaussianBlur(phi_y, phi_y, cv::Size(5, 5), 3);
  cv::GaussianBlur(phi_x, phi_x, cv::Size(5, 5), 3);

  for (int y = 0; y < ori_rows; y++) {
    const float *phi_x_ptr = phi_x.ptr<float>(y);
    const float *phi_y_ptr = phi_y.ptr<float>(y);

    for (int x = 0; x < ori_cols; x++) {
      float theta_s = 0.5f * atan2(phi_y_ptr[x], phi_x_ptr[x]);
      smooth_orientation_img.at<float>(y,x) = theta_s;
    }
  }

  return smooth_orientation_img;
}

void Enhancer::computeGradients(const cv::Mat &img, cv::Mat &grad_x,
                                cv::Mat &grad_y) const {
  CV_Assert(!img.empty());

  cv::Scharr(img, grad_x, CV_32FC1, 1, 0);
  cv::Scharr(img, grad_y, CV_32FC1, 0, 1);
}

void Enhancer::convertSinCos(const cv::Mat &img, cv::Mat &sin_img,
                             cv::Mat &cos_img) const {
  CV_Assert(!img.empty());

  sin_img.create(img.rows, img.cols, CV_32FC1);
  cos_img.create(img.rows, img.cols, CV_32FC1);

  for (int y = 0; y < img.rows; ++y) {
    const float *img_ptr = img.ptr<float>(y);
    float *sin_ptr = sin_img.ptr<float>(y);
    float *cos_ptr = cos_img.ptr<float>(y);
    for (int x = 0; x < img.cols; ++x) {
      sin_ptr[x] = std::sin(img_ptr[x]);
      cos_ptr[x] = std::cos(img_ptr[x]);
    }
  }
}

// === Frequency ===
cv::Mat Enhancer::estimateFrequency(const cv::Mat &img,
                                    const cv::Mat &orientation_img) const {
  CV_Assert(!img.empty());
  CV_Assert(!orientation_img.empty());
  CV_Assert(img.type() == CV_32FC1);
  CV_Assert(orientation_img.type() == CV_32FC1);

  const int rows = orientation_img.rows;
  const int cols = orientation_img.cols;

  cv::Mat frequency_img(rows, cols, CV_32FC1, cv::Scalar(-1));

  for (int by = 0; by < rows; ++by) {
    for (int bx = 0; bx < cols; ++bx) {
      float ori = orientation_img.at<float>(by, bx);

      // Center of the block
      float cx = bx * block_size + block_size * 0.5f;
      float cy = by * block_size + block_size * 0.5f;

      float freq = computeBlockFrequency(img, ori, cy, cx);

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

  //Perform smoothing to the frequency image
  cv::GaussianBlur(frequency_img, frequency_img, cv::Size(7, 7), 0);

  return frequency_img;
}

float Enhancer::computeBlockFrequency(const cv::Mat &img, float ori, int cy,
                                      int cx) const {
  CV_Assert(!img.empty());
  CV_Assert(img.type() == CV_32FC1);

#ifdef FP_DEBUG_VIS
  cv::Mat debug;
  img.convertTo(debug, CV_8UC1, 255.0);
  cv::cvtColor(debug, debug, cv::COLOR_GRAY2BGR);
#endif

  // Sample the gray values along the orthogonal direction of the ridge orientation
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

#ifdef FP_DEBUG_VIS
      cv::circle(debug, cv::Point(v, u), 1, cv::Scalar(0, 0, 255), -1);
#endif
    }

    x_sig[k] = sum / block_size;
  }

#ifdef FP_DEBUG_VIS
  cv::imshow("Sampling Debug", debug);
  cv::waitKey(0);
#endif

  // Estimate the period from the x-signature
  float T = estimatePeriod(x_sig);
  if (T < 3.0f || T > 25.0f)
    return -1.f;

  // Compute frequency
  float freq = 1.0f / T;

  return freq;
}

float Enhancer::estimatePeriod(const std::vector<float> &x_sig) const {
  // Mean centering
  std::vector<float> sig = x_sig;
  float mean = 0.f;
  for (float v : sig) mean += v;
  mean /= sig.size();
  for (float &v : sig) v -= mean;

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

// === Region Mask ===
cv::Mat Enhancer::generateRegionMask(const cv::Mat &gray_img) const {
  CV_Assert(!gray_img.empty());
  CV_Assert(gray_img.type() == CV_8UC1);

  cv::Mat blur;
  cv::GaussianBlur(gray_img, blur, cv::Size(7, 7), 0);

  cv::Mat bin;
  cv::threshold(blur, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  cv::Size ksize = cv::Size(31, 61);
  cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, ksize);

  cv::Mat region_mask;
  cv::morphologyEx(~bin, region_mask, cv::MORPH_CLOSE, se);
  cv::morphologyEx(region_mask, region_mask, cv::MORPH_OPEN, se);

  return region_mask;
}

// === Enhancement ===
void Enhancer::applyGabor(const cv::Mat &src, cv::Mat &dst,
                          const cv::Mat &orientation_img,
                          const cv::Mat &frequency_img) const {
  CV_Assert(!src.empty());
  CV_Assert(!orientation_img.empty());
  CV_Assert(!frequency_img.empty());
  CV_Assert(src.type() == CV_32FC1);
  CV_Assert(orientation_img.type() == CV_32FC1);
  CV_Assert(frequency_img.type() == CV_32FC1);
  // TODO 4: Implement Gabor filtering
}

// === Debug/Visualization ===
void Enhancer::showOrientation(const cv::Mat &bg_img,
                               const cv::Mat &orientation_img,
                               const char *winname) const {
  CV_Assert(!bg_img.empty());
  CV_Assert(!orientation_img.empty());
  CV_Assert(orientation_img.type() == CV_32FC1);

  cv::Mat vis = bg_img.clone();
  cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

  const int ori_rows = orientation_img.rows;
  const int ori_cols = orientation_img.cols;

  const float len = block_size * 0.8f;

  for (int r = 0; r < ori_rows; ++r) {
    for (int c = 0; c < ori_cols; ++c) {
      float angle = orientation_img.at<float>(r, c) + CV_PI * 0.5f;

      if (std::isnan(angle))
        continue;

      // center of the block
      float cx = c * block_size + block_size * 0.5f;
      float cy = r * block_size + block_size * 0.5f;

      float dx = std::cos(angle) * (len * 0.5f);
      float dy = std::sin(angle) * (len * 0.5f);

      cv::Point p1(cv::saturate_cast<int>(std::round(cx + dx)),
                   cv::saturate_cast<int>(std::round(cy + dy)));
      cv::Point p2(cv::saturate_cast<int>(std::round(cx - dx)),
                   cv::saturate_cast<int>(std::round(cy - dy)));

      // draw line
      cv::line(vis, p1, p2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }
  }

  cv::imshow(winname, vis);
  cv::waitKey(1);
}

} // namespace fp
