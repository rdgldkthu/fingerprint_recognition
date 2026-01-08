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

} // namespace fp
