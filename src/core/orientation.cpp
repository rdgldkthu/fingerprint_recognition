#include "fingerprint/core/orientation.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

void convertImg2SinCosImg(const cv::Mat &img, cv::Mat &sin_img,
                          cv::Mat &cos_img) {
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

} // namespace

namespace fp {

cv::Mat estimateRidgeOrientation(const cv::Mat &img, int block_size) {
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

  convertImg2SinCosImg(2 * orientation_img, phi_y, phi_x);

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

void showRidgeOrientation(const cv::Mat &bg_img, const cv::Mat &orientation_img,
                          int block_size, const char *winname) {
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
