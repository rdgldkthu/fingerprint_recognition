#include "fingerprint/core/minutiae.hpp"
#include "fingerprint/core/pruning.hpp"
#include "fingerprint/core/skeleton.hpp"
#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/features/alignment_matcher.hpp"
#include "fingerprint/features/detection.hpp"
#include "fingerprint/features/matching.hpp"
#include "fingerprint/features/mcc.hpp"
#include "fingerprint/features/visualization.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

namespace {

cv::Mat renderRidgeOrientation(const cv::Mat &bg_img,
                                const cv::Mat &orientation_img,
                                int block_size) {
    cv::Mat vis;
    cv::cvtColor(bg_img, vis, cv::COLOR_GRAY2BGR);

    const float len = block_size * 0.8f;
    for (int r = 0; r < orientation_img.rows; ++r) {
        for (int c = 0; c < orientation_img.cols; ++c) {
            float angle = orientation_img.at<float>(r, c) + CV_PI * 0.5f;
            if (std::isnan(angle))
                continue;
            float cx = c * block_size + block_size * 0.5f;
            float cy = r * block_size + block_size * 0.5f;
            float dx = std::cos(angle) * (len * 0.5f);
            float dy = std::sin(angle) * (len * 0.5f);
            cv::Point p1(cv::saturate_cast<int>(std::round(cx + dx)),
                         cv::saturate_cast<int>(std::round(cy + dy)));
            cv::Point p2(cv::saturate_cast<int>(std::round(cx - dx)),
                         cv::saturate_cast<int>(std::round(cy - dy)));
            cv::line(vis, p1, p2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
    }
    return vis;
}

cv::Mat colorizeFrequency(const cv::Mat &frequency, const cv::Mat &mask) {
    cv::Mat vis;
    double mn, mx;
    cv::minMaxLoc(frequency, &mn, &mx, nullptr, nullptr, mask);
    frequency.convertTo(vis, CV_8U, 255.0 / (mx - mn + 1e-6),
                        -mn * 255.0 / (mx - mn + 1e-6));
    cv::Mat colored;
    cv::applyColorMap(vis, colored, cv::COLORMAP_JET);
    // Zero out background
    for (int r = 0; r < mask.rows; r++)
        for (int c = 0; c < mask.cols; c++)
            if (!mask.at<uchar>(r, c))
                colored.at<cv::Vec3b>(r, c) = {0, 0, 0};
    return colored;
}

cv::Mat makeEnhancementPanel(const cv::Mat &raw,
                              const fp::EnhancementResult &res) {
    int h = raw.rows, w = raw.cols;

    cv::Mat rawBGR, enhBGR;
    cv::cvtColor(raw, rawBGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(res.enhanced_img, enhBGR, cv::COLOR_GRAY2BGR);

    int block_size = raw.rows / res.orientation_img.rows;
    cv::Mat oriColor = renderRidgeOrientation(raw, res.orientation_img, block_size);

    cv::Mat freqFull, maskFull;
    cv::resize(res.frequency_img, freqFull, {w, h}, 0, 0, cv::INTER_NEAREST);
    cv::resize(res.mask, maskFull, {w, h}, 0, 0, cv::INTER_NEAREST);
    cv::Mat freqColor = colorizeFrequency(freqFull, maskFull);

    cv::Mat panel(h, w * 4, CV_8UC3);
    rawBGR.copyTo(panel(cv::Rect(0,     0, w, h)));
    oriColor.copyTo(panel(cv::Rect(w,   0, w, h)));
    freqColor.copyTo(panel(cv::Rect(w*2, 0, w, h)));
    enhBGR.copyTo(panel(cv::Rect(w*3,  0, w, h)));

    // Thin white dividers
    for (int col : {w, w*2, w*3})
        cv::line(panel, {col, 0}, {col, h - 1}, {255, 255, 255}, 1);

    // Labels
    auto label = [&](const std::string &txt, int x) {
        int baseline = 0;
        cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::rectangle(panel, {x, 0}, {x + sz.width + 8, sz.height + baseline + 6},
                      {0, 0, 0}, cv::FILLED);
        cv::putText(panel, txt, {x + 4, sz.height + 3}, cv::FONT_HERSHEY_SIMPLEX,
                    0.45, {255, 255, 255}, 1, cv::LINE_AA);
    };
    label("Raw",         0);
    label("Orientation", w);
    label("Frequency",   w * 2);
    label("Enhanced",    w * 3);

    return panel;
}

cv::Mat makeMinutiaePanel(const fp::EnhancementResult &res) {
    int h = res.enhanced_img.rows, w = res.enhanced_img.cols;

    cv::Mat enhBGR;
    cv::cvtColor(res.enhanced_img, enhBGR, cv::COLOR_GRAY2BGR);

    cv::Mat skeleton_raw;
    fp::skeletonizeRidges(res.enhanced_img, skeleton_raw);

    cv::Mat skeleton_pruned = skeleton_raw.clone();
    fp::pruneIslands(skeleton_pruned);
    fp::pruneSpurs(skeleton_pruned);
    // fp::pruneLakes(skeleton_pruned);
    // fp::pruneHBreaks(skeleton_pruned);

    auto minutiae = fp::detectMinutiae(skeleton_pruned, res.orientation_img);
    fp::pruneByMaskDistance(minutiae, res.mask, 8.0f);
    fp::pruneByImageBorder(minutiae, skeleton_pruned.cols, skeleton_pruned.rows, 10);
    cv::Mat minutiaeVis = fp::visualizeMinutiae(skeleton_pruned, minutiae);

    cv::Mat skelRawBGR, skelPrunedBGR;
    cv::cvtColor(skeleton_raw, skelRawBGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(skeleton_pruned, skelPrunedBGR, cv::COLOR_GRAY2BGR);

    cv::Mat panel(h, w * 4, CV_8UC3);
    enhBGR.copyTo(panel(cv::Rect(0,     0, w, h)));
    skelRawBGR.copyTo(panel(cv::Rect(w,  0, w, h)));
    skelPrunedBGR.copyTo(panel(cv::Rect(w*2, 0, w, h)));
    minutiaeVis.copyTo(panel(cv::Rect(w*3, 0, w, h)));

    for (int col : {w, w*2, w*3})
        cv::line(panel, {col, 0}, {col, h - 1}, {255, 255, 255}, 1);

    auto label = [&](const std::string &txt, int x) {
        int baseline = 0;
        cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::rectangle(panel, {x, 0}, {x + sz.width + 8, sz.height + baseline + 6},
                      {0, 0, 0}, cv::FILLED);
        cv::putText(panel, txt, {x + 4, sz.height + 3}, cv::FONT_HERSHEY_SIMPLEX,
                    0.45, {255, 255, 255}, 1, cv::LINE_AA);
    };
    label("Enhanced",  0);
    label("Skeleton",  w);
    label("Pruned",    w * 2);
    label("Minutiae",  w * 3);

    return panel;
}

} // namespace

int main() {
    std::string path1 = std::string(DATA_DIR) + "/raw/101_1.tif";
    std::string path2 = std::string(DATA_DIR) + "/raw/101_2.tif";

    cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load images\n";
        return 1;
    }

    fp::Enhancer enhancer;
    fp::Detector detector;
    fp::AlignmentMatcher alignment_matcher;
    fp::MCCExtractor mcc_extractor;
    fp::LSSMatcher<fp::Cylinder> lss_matcher;

    auto res1 = enhancer.enhance(img1);
    auto res2 = enhancer.enhance(img2);

    auto min1 = detector.detect(res1.enhanced_img, res1.orientation_img, res1.mask);
    auto min2 = detector.detect(res2.enhanced_img, res2.orientation_img, res2.mask);

    auto desc1 = mcc_extractor.extract(min1);
    auto desc2 = mcc_extractor.extract(min2);

    auto align_pairs = alignment_matcher.findBestAlignment(min1, min2);
    auto mcc_pairs   = lss_matcher.matchPairs(desc1, desc2);

    std::filesystem::create_directories("vis");

    cv::imwrite("vis/enhancement.png",    makeEnhancementPanel(img1, res1));
    cv::imwrite("vis/minutiae.png",       makeMinutiaePanel(res1));
    cv::imwrite("vis/match_alignment.png",
                fp::visualizeMatching(res1.enhanced_img, res2.enhanced_img,
                                      min1, min2, align_pairs));
    cv::imwrite("vis/match_mcc.png",
                fp::visualizeMatching(res1.enhanced_img, res2.enhanced_img,
                                      min1, min2, mcc_pairs));

    std::cout << "Wrote vis/enhancement.png\n"
              << "Wrote vis/minutiae.png\n"
              << "Wrote vis/match_alignment.png\n"
              << "Wrote vis/match_mcc.png\n";
    return 0;
}
