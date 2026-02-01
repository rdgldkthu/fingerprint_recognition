#include "fingerprint/core/roi.hpp"
#include <opencv2/imgproc.hpp>

namespace fp {

cv::Mat extractFingerprintROI(const cv::Mat &gray_img, cv::Size ksize) {
  CV_Assert(!gray_img.empty());
  CV_Assert(gray_img.type() == CV_8UC1);

  cv::Mat norm_img;
  cv::normalize(gray_img, norm_img, 0, 255, cv::NORM_MINMAX);

  cv::Mat blur;
  cv::GaussianBlur(norm_img, blur, cv::Size(7, 7), 0);

  cv::Mat bin;
  cv::threshold(blur, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, ksize);

  cv::Mat region_mask;
  cv::morphologyEx(~bin, region_mask, cv::MORPH_CLOSE, se);
  cv::morphologyEx(region_mask, region_mask, cv::MORPH_OPEN, se);

  return region_mask;
}

void removeThinkRidgesFromROI(cv::Mat &roi, const cv::Mat enhanced_img,
                              float max_half_width) {
  CV_Assert(!enhanced_img.empty());
  CV_Assert(enhanced_img.type() == CV_8UC1);

  // ridge -> white
  cv::Mat ridge_bin = enhanced_img < 128;
  ridge_bin.convertTo(ridge_bin, CV_8UC1, 255);

  // distance inside ridge
  cv::Mat dist;
  cv::distanceTransform(ridge_bin, dist, cv::DIST_L2, 3);

  // too-thick ridge cores
  cv::Mat thick_mask = dist > max_half_width;
  thick_mask.convertTo(thick_mask, CV_8UC1, 255);

  // blob core expansion
  cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
  cv::morphologyEx(thick_mask, thick_mask, cv::MORPH_DILATE, se);

  roi.setTo(0, thick_mask);
}

} // namespace fp
