#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/features/alignment_matcher.hpp"
#include "fingerprint/features/detection.hpp"
#include "fingerprint/features/visualization.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

#define LOG(msg) (std::cerr << msg << std::endl)

int main() {
    const std::string path1 = std::string(DATA_DIR) + "/raw/104_3.tif";
    const std::string path2 = std::string(DATA_DIR) + "/raw/104_7.tif";

    LOG("Loading images...");
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        LOG("Failed to load images from " << DATA_DIR << "/raw/");
        return 1;
    }
    LOG("img1: " << img1.cols << "x" << img1.rows
        << "  img2: " << img2.cols << "x" << img2.rows);

    try {
        fp::Enhancer enhancer;
        fp::Detector detector;
        fp::AlignmentMatcher matcher;

        LOG("Enhancing img1...");
        auto res1 = enhancer.enhance(img1);
        LOG("enhanced_img1 type=" << res1.enhanced_img.type()
            << " size=" << res1.enhanced_img.cols << "x" << res1.enhanced_img.rows
            << " empty=" << res1.enhanced_img.empty());

        LOG("Enhancing img2...");
        auto res2 = enhancer.enhance(img2);

        LOG("Detecting minutiae...");
        auto m1 = detector.detect(res1.enhanced_img, res1.orientation_img, res1.mask);
        auto m2 = detector.detect(res2.enhanced_img, res2.orientation_img, res2.mask);
        LOG("Minutiae: img1=" << m1.size() << "  img2=" << m2.size());

        LOG("Aligning...");
        auto pairs = matcher.findBestAlignment(m1, m2);
        double score = matcher.computeScore(m1, m2);
        LOG("Pairs=" << pairs.size() << "  score=" << score);

        LOG("Building visualization...");
        cv::Mat vis = fp::visualizeMatching(res1.enhanced_img, res2.enhanced_img,
                                            m1, m2, pairs);
        LOG("vis: " << vis.cols << "x" << vis.rows
            << " type=" << vis.type() << " empty=" << vis.empty());

        const std::string out_path = "vis_result.png";
        LOG("Writing to " << out_path << "...");
        if (!cv::imwrite(out_path, vis))
            LOG("ERROR: imwrite returned false");
        else
            LOG("Saved: " << out_path);

    } catch (const cv::Exception& e) {
        LOG("cv::Exception: " << e.what());
        return 1;
    } catch (const std::exception& e) {
        LOG("std::exception: " << e.what());
        return 1;
    }

    return 0;
}
