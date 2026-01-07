#include "fingerprint/features.hpp"
#include "fingerprint/core/types.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

/// Internal Utility Functions
namespace {

inline float angleDiff(float a, float b) {
  float d = std::fabs(a - b);
  return std::min(d, float(CV_PI) - d);
}

int crossingNumber(const cv::Mat &win) {
  static const int dx[8] = {0, -1, -1, -1, 0, 1, 1, 1};
  static const int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};

  int cn = 0;
  for (int i = 0; i < 8; ++i) {
    int p = win.at<uchar>(dy[i] + 1, dx[i] + 1);
    int pn = win.at<uchar>(dy[(i + 1) % 8] + 1, dx[(i + 1) % 8] + 1);
    cn += std::abs(p - pn);
  }
  return cn / 2;
}

float computeEndingDirection(const cv::Mat &win) {
  static const int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  static const int dy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

  for (int k = 0; k < 8; ++k) {
    if (win.at<uchar>(dy[k] + 1, dx[k] + 1) == 1)
      return std::atan2(dy[k], dx[k]);
  }
  return 0.f;
}

std::vector<float> computeBifurcationDirections(const cv::Mat &win) {
  static const int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  static const int dy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

  std::vector<float> dirs;
  for (int k = 0; k < 8; ++k) {
    if (win.at<uchar>(dy[k] + 1, dx[k] + 1) == 1)
      dirs.push_back(std::atan2(dy[k], dx[k]));
  }
  return dirs;
}

bool pruneEnding(float theta_m, float theta_f) {
  return angleDiff(theta_m, theta_f) < CV_PI / 6; // 30 deg
}

bool pruneBifurcation(const std::vector<float> &dirs, float theta_f) {
  int valid = 0;
  for (float d : dirs) {
    if (angleDiff(d, theta_f) < CV_PI / 6)
      valid++;
  }
  return valid >= 2;
}

cv::Mat computeMaskDistance(const cv::Mat &mask) {
  CV_Assert(mask.type() == CV_8UC1);

  cv::Mat dist;
  cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

  return dist;
}

void pruneByMaskDistance(std::vector<fp::Minutia> &minutiae,
                         const cv::Mat &mask_dist, float min_dist = 8.0f) {
  std::vector<fp::Minutia> pruned;
  pruned.reserve(minutiae.size());

  for (const auto &m : minutiae) {

    if (m.type == fp::MinutiaType::ENDING) {

      float d = mask_dist.at<float>(m.y, m.x);
      if (d < min_dist)
        continue;
    }

    pruned.push_back(m);
  }

  minutiae.swap(pruned);
}

} // namespace

namespace fp {

/// Feature Detection
std::vector<Minutia> Detector::detect(const cv::Mat &enhanced_img,
                                      const cv::Mat &orientation,
                                      const cv::Mat &mask) const {
  std::vector<Minutia> minutiae;
  if (enhanced_img.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    return minutiae;
  }

  cv::Mat thinned_img;
  thin(enhanced_img, thinned_img);

  minutiae = detectMinutiae(thinned_img, orientation);

  cv::Mat mask_dist = computeMaskDistance(mask);

  pruneByMaskDistance(minutiae, mask_dist);

#ifdef FP_DEBUG_VIS
  cv::Mat vis = visualizeMinutiae(thinned_img, minutiae);
  cv::namedWindow("Visualize Minutiae", cv::WINDOW_NORMAL);
  cv::moveWindow("Visualize Minutiae", 450, 750);
  cv::imshow("Visualize Minutiae", vis);
  cv::waitKey(0);
#endif

  return minutiae;
}

void Detector::thin(const cv::Mat &src, cv::Mat &dst) const {
  CV_Assert(!src.empty());

  cv::Mat binary_img;
  cv::threshold(src, binary_img, 127, 255, cv::THRESH_BINARY);

  cv::Mat inverted_img;
  cv::bitwise_not(binary_img, inverted_img);

  cv::Mat thinned_img;
  cv::ximgproc::thinning(inverted_img, thinned_img,
                         cv::ximgproc::THINNING_ZHANGSUEN);

  cv::Mat reinverted_img;
  cv::bitwise_not(thinned_img, reinverted_img);
#ifdef FP_DEBUG_VIS
  cv::namedWindow("Thinned Image", cv::WINDOW_NORMAL);
  cv::moveWindow("Thinned Image", 50, 750);
  cv::imshow("Thinned Image", reinverted_img);
#endif

  dst = reinverted_img;
  return;
}

// 0=ridge, 255=background
// CV_32F, radian
std::vector<Minutia>
Detector::detectMinutiae(const cv::Mat &skeleton,
                         const cv::Mat &orientation) const {
  CV_Assert(skeleton.type() == CV_8UC1);
  CV_Assert(orientation.type() == CV_32F);

  cv::Mat ori_resized;
  cv::resize(orientation, ori_resized, skeleton.size(), 0, 0,
             cv::INTER_NEAREST);

  std::vector<Minutia> minutiae;

  for (int y = 1; y < skeleton.rows - 1; ++y) {
    for (int x = 1; x < skeleton.cols - 1; ++x) {

      if (skeleton.at<uchar>(y, x) != 0)
        continue;

      // local window (ridge=1)
      cv::Mat win(3, 3, CV_8UC1);
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          win.at<uchar>(dy + 1, dx + 1) =
              skeleton.at<uchar>(y + dy, x + dx) == 0 ? 1 : 0;

      int cn = crossingNumber(win);
      float theta_f = ori_resized.at<float>(y, x) - CV_PI / 2;

      if (cn == 1) {
        float theta_m = computeEndingDirection(win);
        if (!pruneEnding(theta_m, theta_f))
          continue;

        minutiae.push_back({y, x, theta_f, MinutiaType::ENDING});
      } else if (cn == 3) {
        auto dirs = computeBifurcationDirections(win);
        if (!pruneBifurcation(dirs, theta_f))
          continue;

        minutiae.push_back({y, x, theta_f, MinutiaType::BIFURCATION});
      }
    }
  }
  return minutiae;
}

cv::Mat Detector::visualizeMinutiae(const cv::Mat &enhanced,
                                    const std::vector<Minutia> &minutiae,
                                    int radius, int arrow_len) const {
  CV_Assert(enhanced.type() == CV_8UC1);

  cv::Mat vis;
  cv::cvtColor(enhanced, vis, cv::COLOR_GRAY2BGR);

  for (const auto &m : minutiae) {
    cv::Point p(m.x, m.y);

    cv::Scalar color;
    if (m.type == MinutiaType::ENDING)
      color = cv::Scalar(0, 0, 255); // red
    else
      color = cv::Scalar(255, 0, 0); // blue

    // draw point
    cv::circle(vis, p, radius, color, -1);

    // draw orientation arrow
    cv::Point q(static_cast<int>(p.x + arrow_len * std::cos(m.theta)),
                static_cast<int>(p.y + arrow_len * std::sin(m.theta)));

    cv::arrowedLine(vis, p, q, color, 1, cv::LINE_AA, 0, 0.3);
  }

  return vis;
}

} // namespace fp
