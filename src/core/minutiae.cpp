#include "fingerprint/core/minutiae.hpp"
#include "fingerprint/core/utility.hpp"
#include <opencv2/imgproc.hpp>

namespace {

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
  return fp::angleDiff(theta_m, theta_f) < CV_PI / 6; // 30 deg
}

bool pruneBifurcation(const std::vector<float> &dirs, float theta_f) {
  int valid = 0;
  for (float d : dirs) {
    if (fp::angleDiff(d, theta_f) < CV_PI / 6)
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

inline bool isNearImageBorder(int x, int y, int width, int height, int margin) {
  return (x < margin || y < margin || x >= width - margin ||
          y >= height - margin);
}

void pruneByImageBorder(std::vector<fp::Minutia> &minutiae, int width, int height,
                        int margin) {
  std::vector<fp::Minutia> pruned;
  pruned.reserve(minutiae.size());

  for (const auto &m : minutiae) {

    if (isNearImageBorder(m.x, m.y, width, height, margin)) continue;

    pruned.push_back(m);
  }

  minutiae.swap(pruned);
}

} // namespace

namespace fp {

// 0=ridge, 255=background
// CV_32F, radian
std::vector<Minutia> detectMinutiae(const cv::Mat &skeleton,
                                    const cv::Mat &orientation,
                                    const cv::Mat &mask) {
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

  cv::Mat mask_dist = computeMaskDistance(mask);

  pruneByMaskDistance(minutiae, mask_dist);

  pruneByImageBorder(minutiae, mask.cols, mask.rows, 10);

  return minutiae;
}

cv::Mat visualizeMinutiae(const cv::Mat &enhanced,
                          const std::vector<Minutia> &minutiae, int radius,
                          int arrow_len) {
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
