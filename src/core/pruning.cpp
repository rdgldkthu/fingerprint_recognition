#include "fingerprint/core/pruning.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>

namespace {

inline int countWhiteNeighbors(const cv::Mat &img, int y, int x) {
  int cnt = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dy == 0 && dx == 0) continue;
      int ny = y + dy;
      int nx = x + dx;
      if (ny < 0 || ny >= img.rows || nx < 0 || nx >= img.cols) continue;
      if (img.at<uchar>(ny, nx) == 255) cnt++;
    }
  }
  return cnt;
}

} // namespace

namespace fp {

void pruneSpurs(cv::Mat &skeleton, int max_len) {
  CV_Assert(skeleton.type() == CV_8UC1);

  cv::Mat inv;
  cv::bitwise_not(skeleton, inv);

  for (int y = 1; y < inv.rows - 1; ++y) {
    for (int x = 1; x < inv.cols - 1; ++x) {

      if (inv.at<uchar>(y, x) != 255) continue;
      if (countWhiteNeighbors(inv, y, x) != 1) continue; // endpoint

      std::vector<cv::Point> path;
      cv::Point prev(-1, -1);
      cv::Point curr(x, y);

      bool valid_spur = true;

      while (true) {
        path.push_back(curr);

        // Total degree of current pixel
        int degree = countWhiteNeighbors(inv, curr.y, curr.x);

        // Bifurcation
        if (curr != cv::Point(x, y) && degree >= 3) break;

        cv::Point next(-1, -1);
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            cv::Point np(curr.x + dx, curr.y + dy);
            if (np.x < 0 || np.y < 0 || np.x >= inv.cols || np.y >= inv.rows) continue;
            if (inv.at<uchar>(np.y, np.x) == 255 && np != prev) {
              next = np;
              break;
            }
          }
          if (next.x != -1) break;
        }

        if (next.x == -1) break;

        prev = curr;
        curr = next;

        if ((int)path.size() > max_len) {
          valid_spur = false;
          break;
        }
      }

      if (valid_spur && path.size() <= max_len) {
        for (const auto &p : path) {
          inv.at<uchar>(p.y, p.x) = 0;
        }
      }
    }
  }
  cv::bitwise_not(inv, skeleton);
}

void pruneIslands(cv::Mat &skeleton, int min_size) {
  CV_Assert(skeleton.type() == CV_8UC1);

  cv::Mat inv;
  cv::bitwise_not(skeleton, inv);

  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(inv, labels, stats, centroids,
                                                 8, CV_32S);

  for (int i = 1; i < nLabels; ++i) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);

    if (area < min_size) {
      inv.setTo(0, labels == i);
    }
  }

  cv::bitwise_not(inv, skeleton);
}

void pruneLakes(cv::Mat &skeleton, int areaThresh) {
  CV_Assert(skeleton.type() == CV_8UC1);

  const int H = skeleton.rows;
  const int W = skeleton.cols;

  cv::Mat labels, stats, centroids;
  int numCC = cv::connectedComponentsWithStats(skeleton, labels, stats, centroids,
                                               4, CV_32S);

  for (int label = 1; label < numCC; ++label) // label 0 = outer background
  {
    int area = stats.at<int>(label, cv::CC_STAT_AREA);
    int x = stats.at<int>(label, cv::CC_STAT_LEFT);
    int y = stats.at<int>(label, cv::CC_STAT_TOP);
    int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);

    // Skip if touches border
    bool touchesBorder = (x == 0) || (y == 0) || (x + w >= W) || (y + h >= H);

    if (touchesBorder) continue;

    // Skip if area is big enough
    if (area > areaThresh) continue;

    // Remove ridges that form the lake
    for (int r = y - 1; r <= y + h; ++r) {
      for (int c = x - 1; c <= x + w; ++c) {
        if (r < 0 || r >= H || c < 0 || c >= W) continue;

        if (labels.at<int>(r, c) == label) {
          // Skip if within the lake
          continue;
        }

        // Remove ridges near the lake boundary
        if (skeleton.at<uchar>(r, c) == 0) {
          skeleton.at<uchar>(r, c) = 255;
        }
      }
    }
  }
}

} // namespace fp
