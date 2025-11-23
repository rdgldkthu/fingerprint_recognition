#include "fingerprint/enhancement.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fp {

cv::Mat Enhancer::enhance(const cv::Mat &input) const {
  if (input.empty())
    return cv::Mat();

  cv::Mat gray;
  if (input.channels() == 3)
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  else
    gray = input.clone();

#ifdef FP_DEBUG_VIS
  cv::imshow("Original", gray);
  cv::waitKey(0);
#endif

  // Normalization
  cv::Mat normalized_img;
  normalize(gray, normalized_img);

#ifdef FP_DEBUG_VIS
  cv::Mat norm_vis;
  cv::convertScaleAbs(normalized_img, norm_vis);
  cv::imshow("Normalized", norm_vis);
  cv::waitKey(0);
#endif

  // Orientation Image Estimation
  cv::Mat orientation_img = estimateOrientation(gray);
#ifdef FP_DEBUG_VIS
  showOrientation(gray, orientation_img);
  cv::waitKey(0);
#endif

  return normalized_img;

  //   // Frequency Image Estimation
  //   cv::Mat frequency_img = estimateFrequency(normalized_img,
  //   orientation_img);

  // #ifdef FP_DEBUG_VIS
  //   cv::imshow("Frequency Image", frequency_img);
  //   cv::waitKey(0);
  // #endif

  //   // Region Mask Generation
  //   cv::Mat region_mask;
  //   generateRegionMask(normalized_img, region_mask);

  // #ifdef FP_DEBUG_VIS
  //   cv::imshow("Region Mask", region_mask);
  //   cv::waitKey(0);
  // #endif

  //   // Recoverability Check
  //   double RECOVERABLE_THRESHOLD = 40.0;
  //   double recoverable_ratio =
  //       cv::sum(region_mask)[0] / (region_mask.rows * region_mask.cols);
  //   if (recoverable_ratio < RECOVERABLE_THRESHOLD) {
  //     std::cout << "Image Rejected: Recoverable region is " <<
  //     recoverable_ratio
  //               << "% of the image, which is less than threshold: "
  //               << RECOVERABLE_THRESHOLD << std::endl;
  //     return cv::Mat();
  //   }

  //   // Filtering
  //   cv::Mat filtered_img;
  //   applyGabor(normalized_img, filtered_img, orientation_img, frequency_img);

  //   cv::Mat enhanced = filtered_img;

  // #ifdef FP_DEBUG_VIS
  //   cv::imshow("Enhanced", enhanced);
  //   cv::waitKey(0);
  // #endif

  //   return enhanced;
}

cv::Mat Enhancer::estimateOrientation(const cv::Mat &img) const {
  if (img.empty())
    return cv::Mat();

  cv::Mat imgF;
  img.convertTo(imgF, CV_32FC1);

  // Compute gradients in the x and y directions of image
  cv::Mat grad_x, grad_y;
  calculateGradients(imgF, grad_x, grad_y);

  const int h = imgF.rows;
  const int w = imgF.cols;
  const int ori_h = h / block_size;
  const int ori_w = w / block_size;

  cv::Mat orientation_img(ori_h, ori_w, CV_32FC1);

  // Estimate ridge orientation block-wise
  for (int by = 0; by < ori_h; by++) {
    for (int bx = 0; bx < ori_w; bx++) {
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
  cv::Mat smooth_orientation_img(ori_h, ori_w, CV_32FC1);
  cv::Mat phi_y(ori_h, ori_w, CV_32FC1);
  cv::Mat phi_x(ori_h, ori_w, CV_32FC1);

  convertSinCos(2 * orientation_img, phi_y, phi_x);

  cv::GaussianBlur(phi_y, phi_y, cv::Size(5, 5), 3);
  cv::GaussianBlur(phi_x, phi_x, cv::Size(5, 5), 3);

  for (int y = 0; y < ori_h; y++) {
    const float *phi_x_ptr = phi_x.ptr<float>(y);
    const float *phi_y_ptr = phi_y.ptr<float>(y);
    float *ori_ptr = smooth_orientation_img.ptr<float>(y);

    for (int x = 0; x < ori_w; x++) {
      float theta_s = 0.5f * atan2(phi_y_ptr[x], phi_x_ptr[x]);
      ori_ptr[x] = theta_s;
    }
  }

  return smooth_orientation_img;
}

cv::Mat Enhancer::estimateFrequency(const cv::Mat &img,
                                    const cv::Mat &orientation_img) const {
  if (img.empty())
    return cv::Mat();

  cv::Mat srcF;
  img.convertTo(srcF, CV_32FC1);

  return cv::Mat();
}

cv::Mat Enhancer::generateRegionMask(const cv::Mat &img) const {
  if (img.empty())
    return cv::Mat();

  cv::Mat imgF;
  img.convertTo(imgF, CV_32FC1);

  return cv::Mat();
}

void Enhancer::normalize(const cv::Mat &src, cv::Mat &dst, double dmean,
                         double dstd) const {
  if (src.empty())
    return;

  cv::Mat srcF;
  src.convertTo(srcF, CV_32FC1);

  cv::Scalar m, s;
  cv::meanStdDev(srcF, m, s);

  double mean = m[0];
  double stddev = s[0];

  double scale = stddev == 0 ? 0 : dstd / stddev;

  dst = (srcF - mean) * scale + dmean;
}

void Enhancer::applyGabor(const cv::Mat &src, cv::Mat &dst,
                          const cv::Mat &orientation_img,
                          const cv::Mat &frequency_img) const {
  if (src.empty() || orientation_img.empty() || frequency_img.empty())
    return;
}

void Enhancer::showOrientation(const cv::Mat &bg_img,
                               const cv::Mat &orientation_img,
                               const char *winname) const {
  if (bg_img.empty() || orientation_img.empty())
    return;

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

// Static Member Functions

void Enhancer::calculateGradients(const cv::Mat &img, cv::Mat &grad_x,
                                  cv::Mat &grad_y) {
  if (img.empty())
    return;

  cv::Scharr(img, grad_x, CV_32FC1, 1, 0);
  cv::Scharr(img, grad_y, CV_32FC1, 0, 1);
}

void Enhancer::convertSinCos(const cv::Mat &img, cv::Mat &sin_img,
                             cv::Mat &cos_img) {
  if (img.empty())
    return;

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

} // namespace fp
