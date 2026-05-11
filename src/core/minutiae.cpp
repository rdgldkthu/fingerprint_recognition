#include "fingerprint/core/minutiae.hpp"
#include "fingerprint/core/utility.hpp"
#include <limits>
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


std::vector<float> traceBranchDirections(const cv::Mat &skel, int cx, int cy,
                                         int steps = 3) {
  static const int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  static const int dy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

  std::vector<float> dirs;
  for (int k = 0; k < 8; ++k) {
    int nx = cx + dx[k], ny = cy + dy[k];
    if (nx < 0 || ny < 0 || nx >= skel.cols || ny >= skel.rows) continue;
    if (skel.at<uchar>(ny, nx) != 0) continue;

    // Walk up to `steps` more pixels along this branch.
    int px = cx, py = cy, x = nx, y = ny;
    for (int s = 0; s < steps; ++s) {
      int next_x = -1, next_y = -1;
      bool ambiguous = false;
      for (int ddy = -1; ddy <= 1 && !ambiguous; ++ddy) {
        for (int ddx = -1; ddx <= 1 && !ambiguous; ++ddx) {
          if (ddx == 0 && ddy == 0) continue;
          int qx = x + ddx, qy = y + ddy;
          if (qx == px && qy == py) continue;
          if (qx < 0 || qy < 0 || qx >= skel.cols || qy >= skel.rows) continue;
          if (skel.at<uchar>(qy, qx) == 0) {
            if (next_x == -1) { next_x = qx; next_y = qy; }
            else ambiguous = true;
          }
        }
      }
      if (ambiguous || next_x == -1) break;
      px = x; py = y; x = next_x; y = next_y;
    }
    dirs.push_back(std::atan2(y - cy, x - cx));
  }
  return dirs;
}

bool pruneEnding(float theta_m, float theta_f, float angle_tolerance) {
  return fp::angleDiff(theta_m, theta_f) < angle_tolerance;
}

float computeBifurcationTheta(const std::vector<float> &dirs, float theta_f) {
  if (dirs.size() < 2) return 0.f;

  // Stem = branch most aligned with the ridge axis (180° symmetry via angleDiff).
  // Using theta_f is more reliable than pure geometric isolation: it handles
  // symmetric configs and extra neighbors from thick skeletons without ambiguity.
  int stem = 0;
  float min_diff = std::numeric_limits<float>::max();
  for (int i = 0; i < (int)dirs.size(); ++i) {
    float diff = fp::angleDiff(dirs[i], theta_f);
    if (diff < min_diff) { min_diff = diff; stem = i; }
  }

  // Fork bisector: unit-vector average of all non-stem branches.
  float bx = 0.f, by = 0.f;
  for (int i = 0; i < (int)dirs.size(); ++i) {
    if (i == stem) continue;
    bx += std::cos(dirs[i]);
    by += std::sin(dirs[i]);
  }
  // Degenerate: fork branches exactly cancel — point away from stem instead.
  if (bx * bx + by * by < 1e-6f) {
    float s = dirs[stem];
    return s > 0 ? s - (float)CV_PI : s + (float)CV_PI;
  }
  return std::atan2(by, bx);
}

cv::Mat computeMaskDistance(const cv::Mat &mask) {
  CV_Assert(mask.type() == CV_8UC1);

  cv::Mat dist;
  cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

  return dist;
}

void pruneByMaskDistance(std::vector<fp::Minutia> &minutiae,
                         const cv::Mat &mask_dist, float min_dist) {
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
                                    const cv::Mat &mask,
                                    float angle_tolerance,
                                    float border_dist_min,
                                    int image_margin) {
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
        auto dirs = traceBranchDirections(skeleton, x, y);
        if (dirs.empty()) continue;
        float theta_m = dirs[0];
        if (!pruneEnding(theta_m, theta_f, angle_tolerance))
          continue;

        minutiae.push_back({y, x, theta_m, MinutiaType::ENDING});
      } else if (cn == 3) {
        auto dirs = traceBranchDirections(skeleton, x, y);
        float theta_b = computeBifurcationTheta(dirs, theta_f);
        minutiae.push_back({y, x, theta_b, MinutiaType::BIFURCATION});
      }
    }
  }

  cv::Mat mask_dist = computeMaskDistance(mask);

  pruneByMaskDistance(minutiae, mask_dist, border_dist_min);

  pruneByImageBorder(minutiae, mask.cols, mask.rows, image_margin);

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
