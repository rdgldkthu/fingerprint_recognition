#pragma once
#include <opencv2/core.hpp>

namespace fp {

void pruneSpurs(cv::Mat &skeleton, int max_len);
void pruneIslands(cv::Mat &skeleton, int min_size);
void pruneLakes(cv::Mat &skeleton, int areaThresh = 150);

} // namespace fp
