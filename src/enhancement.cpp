#include "fingerprint/enhancement.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace fp {

void Enhancer::enhance(const cv::Mat &src, cv::Mat &dst) const {
  if (src.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    dst = cv::Mat();
    return;
  }

  cv::Mat gray_img;
  if (src.channels() == 3)
    cv::cvtColor(src, gray_img, cv::COLOR_BGR2GRAY);
  else
    gray_img = src.clone();
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Original", cv::WINDOW_NORMAL);
  cv::moveWindow("Original", 50, 50);
  cv::imshow("Original", gray_img);
#endif

  // Normalization
  cv::Mat normalized_img;
  normalize(gray_img, normalized_img);
#ifdef FP_DEBUG_VIS
  cv::Mat norm_vis_img;
  cv::convertScaleAbs(normalized_img, norm_vis_img);
  cv::namedWindow("Normalized", cv::WINDOW_NORMAL);
  cv::moveWindow("Normalized", 450, 50);
  cv::imshow("Normalized", norm_vis_img);
#endif

  // Orientation Image Estimation
  cv::Mat orientation_img = estimateRidgeOrientation(normalized_img, 10);
#ifdef FP_DEBUG_VIS
  showOrientation(gray_img, orientation_img, 10);
  cv::namedWindow("Orientation Image", cv::WINDOW_NORMAL);
  cv::resizeWindow("Orientation Image", gray_img.size());
  cv::moveWindow("Orientation Image", 850, 410);
  cv::imshow("Orientation Image", orientation_img);
#endif

  // Frequency Image Estimation
  cv::Mat frequency_img = estimateRidgeFrequency(normalized_img, orientation_img);
  // cv::Mat frequency_img = cv::Mat::ones(orientation_img.rows, orientation_img.cols, CV_32FC1) * 0.11f;
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Frequency Image", cv::WINDOW_NORMAL);
  cv::resizeWindow("Frequency Image", gray_img.size());
  cv::moveWindow("Frequency Image", 1150, 410);
  cv::imshow("Frequency Image", frequency_img);
#endif

  // Recoverability Check
  cv::Mat region_mask = generateRegionMask(gray_img);
  double RECOVERABLE_THRESHOLD = 40.0;
  double recoverable_ratio = cv::sum(region_mask / 255)[0] /
                             (region_mask.rows * region_mask.cols) * 100.0;
  if (recoverable_ratio < RECOVERABLE_THRESHOLD) {
    std::cout << "Image Rejected: Recoverable region is " << recoverable_ratio
              << "% of the image, which is less than threshold("
              << RECOVERABLE_THRESHOLD << "%)" << std::endl;
    dst = cv::Mat();
    return;
  }
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Region Mask", cv::WINDOW_NORMAL);
  cv::moveWindow("Region Mask", 50, 410);
  cv::imshow("Region Mask", region_mask);
  std::cout << "Recoverable region: " << recoverable_ratio << '%' << std::endl;
#endif

  // Gabor Filtering
  cv::Mat enhanced_img;
  applyGabor(normalized_img, enhanced_img, orientation_img, frequency_img,
             region_mask);
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Enhanced Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Enhanced Image", 450, 410);
  cv::imshow("Enhanced Image", enhanced_img);
  cv::waitKey(0);
#endif

  enhanced_img.convertTo(dst, CV_8UC1);
  return;
}

void Enhancer::thin(const cv::Mat &src, cv::Mat &dst) const {
  if (src.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    dst = cv::Mat();
    return;
  }

  cv::Mat binary_img;
  cv::threshold(src, binary_img, 127, 255, cv::THRESH_BINARY);

  cv::Mat inverted_img;
  cv::bitwise_not(binary_img, inverted_img);

  cv::Mat thinned_img;
  cv::ximgproc::thinning(inverted_img, thinned_img, cv::ximgproc::THINNING_ZHANGSUEN);

  cv::Mat reinverted_img;
  cv::bitwise_not(thinned_img, reinverted_img);
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Thinned Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Thinned Image", 450, 750);
  cv::imshow("Thinned Image", reinverted_img);
  cv::waitKey(0);
#endif

  dst = reinverted_img;
  return;
}

// === Preprocessing ===
void Enhancer::normalize(const cv::Mat &src, cv::Mat &dst, double dmean,
                         double dstd) const {
  CV_Assert(!src.empty());

  cv::Mat src_F;
  src.convertTo(src_F, CV_32FC1);

  cv::Scalar m, s;
  cv::meanStdDev(src_F, m, s);

  double mean = m[0];
  double stddev = s[0];

  double scale = stddev == 0 ? 0 : dstd / stddev;

  dst = (src_F - mean) * scale + dmean;
}

// === Orientation ===
cv::Mat Enhancer::estimateRidgeOrientation(const cv::Mat &img,
                                      int block_size) const {
  CV_Assert(!img.empty());
  CV_Assert(img.type() == CV_32FC1);

  // Compute gradients in the x and y directions of image
  cv::Mat grad_x, grad_y;
  cv::Scharr(img, grad_x, CV_32FC1, 1, 0);
  cv::Scharr(img, grad_y, CV_32FC1, 0, 1);

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

  convert2SinCosImg(2 * orientation_img, phi_y, phi_x);

  cv::GaussianBlur(phi_y, phi_y, cv::Size(5, 5), 3);
  cv::GaussianBlur(phi_x, phi_x, cv::Size(5, 5), 3);

  for (int y = 0; y < ori_rows; y++) {
    const float *phi_x_ptr = phi_x.ptr<float>(y);
    const float *phi_y_ptr = phi_y.ptr<float>(y);

    for (int x = 0; x < ori_cols; x++) {
      float theta_s = 0.5f * atan2(phi_y_ptr[x], phi_x_ptr[x]);
      smooth_orientation_img.at<float>(y, x) = theta_s;
    }
  }

  return smooth_orientation_img;
}

void Enhancer::convert2SinCosImg(const cv::Mat &img, cv::Mat &sin_img,
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
cv::Mat Enhancer::estimateRidgeFrequency(const cv::Mat &img,
                                    const cv::Mat &orientation_img,
                                    int block_size) const {
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

float Enhancer::computeBlockFrequency(const cv::Mat &img, float ori, int cy,
                                      int cx, int block_size) const {
#ifdef FP_DEBUG_VIS_FREQ
  cv::Mat debug;
  img.convertTo(debug, CV_8UC1, 255.0);
  cv::cvtColor(debug, debug, cv::COLOR_GRAY2BGR);
#endif

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

#ifdef FP_DEBUG_VIS_FREQ
      cv::circle(debug, cv::Point(v, u), 1, cv::Scalar(0, 0, 255), -1);
#endif
    }

    x_sig[k] = sum / block_size;
  }

#ifdef FP_DEBUG_VIS_FREQ
  cv::imshow("Sampling Debug", debug);
  cv::waitKey(0);
#endif

  // Estimate the period from the x-signature
  float T = estimatePeriodFromXSignature(x_sig);
  if (T < 3.0f || T > 25.0f)
    return -1.f;

  // Compute frequency
  float freq = 1.0f / T;

  return freq;
}

float Enhancer::estimatePeriodFromXSignature(const std::vector<float> &x_sig) const {
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

// === Region Mask ===
cv::Mat Enhancer::generateRegionMask(const cv::Mat &gray_img) const {
  CV_Assert(!gray_img.empty());
  CV_Assert(gray_img.type() == CV_8UC1);

  cv::Mat norm_img;
  cv::normalize(gray_img, norm_img, 0, 255, cv::NORM_MINMAX);

  cv::Mat blur;
  cv::GaussianBlur(norm_img, blur, cv::Size(7, 7), 0);

  cv::Mat bin;
  cv::threshold(blur, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  cv::Size ksize(31, 61);
  cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, ksize);

  cv::Mat region_mask;
  cv::morphologyEx(~bin, region_mask, cv::MORPH_CLOSE, se);
  cv::morphologyEx(region_mask, region_mask, cv::MORPH_OPEN, se);

  return region_mask;
}

// === Enhancement ===
void Enhancer::applyGabor(const cv::Mat &src, cv::Mat &dst,
                          const cv::Mat &orientation_img,
                          const cv::Mat &frequency_img,
                          const cv::Mat &region_mask, float kx, float ky,
                          int filter_size) const {
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

  std::cout << ori_resized.at<float>(95, 296) << std::endl;
  std::cout << freq_resized.at<float>(95, 296) << std::endl;

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

cv::Mat Enhancer::createGaborFilter(float freq, float ori, float kx, float ky,
                                    int filter_size) const {
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

void Enhancer::buildGaborFilterBank(const std::vector<float> &unique_freqs,
                                    std::vector<std::vector<cv::Mat>> &bank,
                                    float kx, float ky, int filter_size) const {
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

void Enhancer::applyGaborFilterBank(
    const cv::Mat &src, cv::Mat &dst, const cv::Mat &ori_img_resized,
    const cv::Mat &freq_img_resized, const cv::Mat &region_mask,
    const std::unordered_map<int, int> &freq_index_map,
    const std::vector<std::vector<cv::Mat>> &bank, int filter_size) const {
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

// === Debug/Visualization ===
void Enhancer::showOrientation(const cv::Mat &bg_img,
                               const cv::Mat &orientation_img, int block_size,
                               const char *winname) const {
  CV_Assert(!bg_img.empty());
  CV_Assert(!orientation_img.empty());
  CV_Assert(orientation_img.type() == CV_32FC1);

  cv::Mat vis_img = bg_img.clone();
  cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);

  cv::namedWindow(winname, cv::WINDOW_NORMAL);
  cv::moveWindow(winname, 850, 50);

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
      cv::line(vis_img, p1, p2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }
  }

  cv::imshow(winname, vis_img);
  cv::waitKey(1);
}

} // namespace fp
