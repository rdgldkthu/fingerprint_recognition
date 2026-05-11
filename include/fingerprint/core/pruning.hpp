#pragma once
#include <opencv2/core.hpp>

namespace fp {

void pruneIslands(cv::Mat &skeleton, int min_size = 30);
void pruneSpurs(cv::Mat &skeleton, int max_len = 9);
void pruneLakes(cv::Mat &skeleton, int area_thresh = 150);
void pruneHBreaks(cv::Mat &skeleton);

} // namespace fp
