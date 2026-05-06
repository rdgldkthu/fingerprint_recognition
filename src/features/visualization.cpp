#include "fingerprint/features/visualization.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace fp {

namespace {

void drawMinutia(cv::Mat& canvas, const Minutia& m, int x_offset) {
    const cv::Point center(m.x + x_offset, m.y);
    const int radius = 4;
    const int arrow_len = 8;

    const cv::Scalar color = (m.type == MinutiaType::ENDING)
        ? cv::Scalar(0, 0, 220)
        : cv::Scalar(220, 0, 0);

    cv::circle(canvas, center, radius, color, 1, cv::LINE_AA);

    const cv::Point tip(
        center.x + static_cast<int>(arrow_len * std::cos(m.theta)),
        center.y - static_cast<int>(arrow_len * std::sin(m.theta))
    );
    cv::arrowedLine(canvas, center, tip, color, 1, cv::LINE_AA, 0, 0.3);
}

cv::Mat toBGR(const cv::Mat& img) {
    cv::Mat out;
    if (img.channels() == 1)
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else
        out = img.clone();
    return out;
}

} // namespace

cv::Mat visualizeMatching(const cv::Mat& img1, const cv::Mat& img2,
                           const std::vector<Minutia>& m1,
                           const std::vector<Minutia>& m2,
                           const std::vector<MatchedPair>& matches) {
    const int h = std::max(img1.rows, img2.rows);
    const int w = img1.cols + img2.cols;

    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat left  = toBGR(img1);
    cv::Mat right = toBGR(img2);

    left.copyTo(canvas(cv::Rect(0, 0, img1.cols, img1.rows)));
    right.copyTo(canvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    for (const auto& m : m1)
        drawMinutia(canvas, m, 0);
    for (const auto& m : m2)
        drawMinutia(canvas, m, img1.cols);

    for (const auto& pair : matches) {
        const cv::Point p1(m1[pair.idx_a].x, m1[pair.idx_a].y);
        const cv::Point p2(m2[pair.idx_b].x + img1.cols, m2[pair.idx_b].y);
        cv::line(canvas, p1, p2, cv::Scalar(0, 220, 0), 1, cv::LINE_AA);
    }

    return canvas;
}

} // namespace fp
