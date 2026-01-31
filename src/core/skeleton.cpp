#include "fingerprint/core/skeleton.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace {

void pruneIslands(cv::Mat &skel, int min_size) {
  CV_Assert(skel.type() == CV_8UC1);

  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(skel, labels, stats, centroids,
                                                 8, CV_32S);

  for (int i = 1; i < nLabels; ++i) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);

    if (area < min_size) {
      skel.setTo(0, labels == i);
    }
  }
}

inline int countWhiteNeighbors(const cv::Mat &img, int y, int x) {
  int cnt = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dy == 0 && dx == 0)
        continue;
      int ny = y + dy;
      int nx = x + dx;
      if (ny < 0 || ny >= img.rows || nx < 0 || nx >= img.cols)
        continue;
      if (img.at<uchar>(ny, nx) == 255)
        cnt++;
    }
  }
  return cnt;
}

void pruneSpurs(cv::Mat &skel, int max_len) {
  CV_Assert(skel.type() == CV_8UC1);

  for (int y = 1; y < skel.rows - 1; ++y) {
    for (int x = 1; x < skel.cols - 1; ++x) {

      if (skel.at<uchar>(y, x) != 255) continue;
      if (countWhiteNeighbors(skel, y, x) != 1) continue; // endpoint

      std::vector<cv::Point> path;
      cv::Point prev(-1, -1);
      cv::Point curr(x, y);

      bool valid_spur = true;

      while (true) {
        path.push_back(curr);

        // Total degree of current pixel
        int degree = countWhiteNeighbors(skel, curr.y, curr.x);

        // Bifurcation
        if (curr != cv::Point(x, y) && degree >= 3) break;

        cv::Point next(-1, -1);
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            cv::Point np(curr.x + dx, curr.y + dy);
            if (np.x < 0 || np.y < 0 || np.x >= skel.cols || np.y >= skel.rows) continue;
            if (skel.at<uchar>(np.y, np.x) == 255 && np != prev) {
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
          skel.at<uchar>(p.y, p.x) = 0;
        }
      }
    }
  }
}

} // namespace

namespace fp {

void skeletonizeRidges(const cv::Mat &src, cv::Mat &dst) {
  CV_Assert(!src.empty());

  cv::Mat binary_img;
  cv::threshold(src, binary_img, 127, 255, cv::THRESH_BINARY);

  cv::Mat inverted_img;
  cv::bitwise_not(binary_img, inverted_img);

  cv::Mat thinned_img;
  cv::ximgproc::thinning(inverted_img, thinned_img,
                         cv::ximgproc::THINNING_ZHANGSUEN);

  cv::rectangle(thinned_img, cv::Point(0, 0),
                cv::Point(src.cols - 1, src.rows - 1), 0, 1);

  pruneIslands(thinned_img, 30);
  pruneSpurs(thinned_img, 9);

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

} // namespace fp
