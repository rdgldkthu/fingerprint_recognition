#include "fingerprint/core/pruning.hpp"
#include <opencv2/imgproc.hpp>

namespace fp {

void pruneIslands(cv::Mat &skeleton, int min_size) {
  CV_Assert(skeleton.type() == CV_8UC1);

  cv::Mat inv;
  cv::bitwise_not(skeleton, inv);

  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(inv, labels, stats, centroids,
                                                 8, CV_32S);
  for (int i = 1; i < nLabels; ++i) {
    if (stats.at<int>(i, cv::CC_STAT_AREA) < min_size)
      inv.setTo(0, labels == i);
  }

  cv::bitwise_not(inv, skeleton);
}

void pruneSpurs(cv::Mat &skeleton, int max_len) {
  CV_Assert(skeleton.type() == CV_8UC1);

  cv::Mat inv;
  cv::bitwise_not(skeleton, inv);

  for (int y = 1; y < inv.rows - 1; ++y) {
    for (int x = 1; x < inv.cols - 1; ++x) {
      if (inv.at<uchar>(y, x) != 255) continue;

      // Count 8-connected white neighbors
      int degree = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (dy == 0 && dx == 0) continue;
          if (inv.at<uchar>(y + dy, x + dx) == 255) ++degree;
        }
      if (degree != 1) continue; // only endpoints

      std::vector<cv::Point> path;
      cv::Point prev(-1, -1), curr(x, y);
      bool is_spur = true;

      while ((int)path.size() <= max_len) {
        path.push_back(curr);

        int deg = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            cv::Point nb(curr.x + dx, curr.y + dy);
            if (nb.x < 0 || nb.y < 0 || nb.x >= inv.cols || nb.y >= inv.rows) continue;
            if (inv.at<uchar>(nb.y, nb.x) == 255) ++deg;
          }

        if (curr != cv::Point(x, y) && deg >= 3) break; // reached bifurcation

        cv::Point next(-1, -1);
        for (int dy = -1; dy <= 1 && next.x == -1; ++dy)
          for (int dx = -1; dx <= 1 && next.x == -1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            cv::Point nb(curr.x + dx, curr.y + dy);
            if (nb == prev) continue;
            if (nb.x < 0 || nb.y < 0 || nb.x >= inv.cols || nb.y >= inv.rows) continue;
            if (inv.at<uchar>(nb.y, nb.x) == 255) next = nb;
          }

        if (next.x == -1) break; // dead end without bifurcation

        prev = curr;
        curr = next;

        if ((int)path.size() > max_len) { is_spur = false; break; }
      }

      if (is_spur && (int)path.size() <= max_len)
        for (const auto &p : path)
          inv.at<uchar>(p.y, p.x) = 0;
    }
  }

  cv::bitwise_not(inv, skeleton);
}

void pruneLakes(cv::Mat &skeleton, int area_thresh) {
  CV_Assert(skeleton.type() == CV_8UC1);

  const int H = skeleton.rows, W = skeleton.cols;

  cv::Mat labels, stats, centroids;
  // 4-connectivity on the skeleton (ridge=0); background holes are non-zero components
  int numCC = cv::connectedComponentsWithStats(skeleton, labels, stats, centroids,
                                               4, CV_32S);

  for (int label = 1; label < numCC; ++label) {
    int area = stats.at<int>(label, cv::CC_STAT_AREA);
    int x    = stats.at<int>(label, cv::CC_STAT_LEFT);
    int y    = stats.at<int>(label, cv::CC_STAT_TOP);
    int w    = stats.at<int>(label, cv::CC_STAT_WIDTH);
    int h    = stats.at<int>(label, cv::CC_STAT_HEIGHT);

    if ((x == 0) || (y == 0) || (x + w >= W) || (y + h >= H)) continue;
    if (area > area_thresh) continue;

    // Erase ridge pixels immediately surrounding the hole
    for (int r = y - 1; r <= y + h; ++r)
      for (int c = x - 1; c <= x + w; ++c) {
        if (r < 0 || r >= H || c < 0 || c >= W) continue;
        if (labels.at<int>(r, c) != label && skeleton.at<uchar>(r, c) == 0)
          skeleton.at<uchar>(r, c) = 255;
      }
  }
}

void pruneHBreaks(cv::Mat &skeleton) {
  CV_Assert(skeleton.type() == CV_8UC1);

  static const int dx8[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
  static const int dy8[8] = {-1, -1, -1, 0, 1, 1, 1, 0};

  auto ridgeDegree = [&](int x, int y) {
    int deg = 0;
    for (int k = 0; k < 8; ++k) {
      int nx = x + dx8[k], ny = y + dy8[k];
      if (nx >= 0 && ny >= 0 && nx < skeleton.cols && ny < skeleton.rows)
        if (skeleton.at<uchar>(ny, nx) == 0) ++deg;
    }
    return deg;
  };

  cv::Mat result = skeleton.clone();

  for (int y = 1; y < skeleton.rows - 1; ++y) {
    for (int x = 1; x < skeleton.cols - 1; ++x) {
      if (skeleton.at<uchar>(y, x) != 0) continue;

      cv::Point n1(-1, -1), n2(-1, -1);
      int cnt = 0;
      for (int k = 0; k < 8 && cnt <= 2; ++k) {
        int nx = x + dx8[k], ny = y + dy8[k];
        if (skeleton.at<uchar>(ny, nx) == 0) {
          if (cnt == 0) n1 = {nx, ny};
          else if (cnt == 1) n2 = {nx, ny};
          ++cnt;
        }
      }
      if (cnt != 2) continue;

      // Both neighbors must have ≥2 other ridge connections (degree ≥ 3)
      if (ridgeDegree(n1.x, n1.y) < 3) continue;
      if (ridgeDegree(n2.x, n2.y) < 3) continue;

      // Neighbors must not be 8-adjacent (otherwise this is a valid bend, not a bridge)
      if (std::abs(n1.x - n2.x) <= 1 && std::abs(n1.y - n2.y) <= 1) continue;

      result.at<uchar>(y, x) = 255;
    }
  }

  skeleton = result;
}

} // namespace fp
