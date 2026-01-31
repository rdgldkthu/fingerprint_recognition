#include "fingerprint/enhancement/enhancement.hpp"
#include "fingerprint/features/detection.hpp"
#include "fingerprint/features/matching.hpp"
#include "fingerprint/features/mcc.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifndef DATA_DIR
#define DATA_DIR "."
#endif

using namespace std;

double matchFingerprint(const cv::Mat &img1, const cv::Mat &img2) {
  fp::Enhancer enhancer;
  fp::Detector detector;
  fp::MCCExtractor descriptor;
  fp::LSSMatcher<fp::Cylinder> matcher;

  auto enhanced1 = enhancer.enhance(img1);
  auto enhanced2 = enhancer.enhance(img2);

  auto minutiae1 = detector.detect(enhanced1.enhanced_img,
                                   enhanced1.orientation_img, enhanced1.mask);
  auto minutiae2 = detector.detect(enhanced2.enhanced_img,
                                   enhanced2.orientation_img, enhanced2.mask);

  auto descriptors1 = descriptor.extract(minutiae1);
  auto descriptors2 = descriptor.extract(minutiae2);

  auto match_score = matcher.computeScore(descriptors1, descriptors2);
  return match_score;
}

struct ScoreLabel {
  double score;
  int label; // 1 = genuine, 0 = impostor
};

double computeEER(const vector<ScoreLabel> &data) {
  vector<ScoreLabel> sorted = data;
  sort(sorted.begin(), sorted.end(),
       [](auto &a, auto &b) { return a.score > b.score; });

  int P = 0, N = 0;
  for (auto &s : sorted)
    (s.label == 1) ? P++ : N++;

  int TP = 0, FP = 0;
  double eer = 1.0;
  double minDiff = 1.0;

  for (auto &s : sorted) {
    if (s.label == 1) TP++;
    else FP++;

    double FAR = FP / (double)N;
    double FRR = 1.0 - TP / (double)P;
    double diff = fabs(FAR - FRR);

    if (diff < minDiff) {
      minDiff = diff;
      eer = (FAR + FRR) * 0.5;
    }
  }
  return eer;
}

int main() {
  vector<vector<cv::Mat>> fingerprints(101);

  for (int id = 1; id <= 10; id++) {
    for (int k = 1; k <= 8; k++) {
      char name[64];
      sprintf(name, "1%02d_%d.tif", id, k);
      string path = std::string(DATA_DIR) + "/raw/" + name;

      cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        cerr << "Failed to load: " << path << endl;
        return -1;
      }
      fingerprints[id].push_back(img);
    }
  }

  vector<ScoreLabel> scores;

  // Genuine pairs
  for (int id = 1; id <= 10; id++) {
    for (int i = 0; i < 8; i++) {
      for (int j = i + 1; j < 8; j++) {
        double s = matchFingerprint(fingerprints[id][i], fingerprints[id][j]);
        scores.push_back({s, 1});
      }
    }
  }

  // Imposter pairs
  for (int i = 1; i <= 10; i++) {
    for (int j = i + 1; j <= 10; j++) {
      double s = matchFingerprint(fingerprints[i][0], fingerprints[j][0]);
      scores.push_back({s, 0});
    }
  }

  // Result
  double eer = computeEER(scores);
  cout << "Total pairs: " << scores.size() << endl;
  cout << "EER: " << eer * 100.0 << " %" << endl;

  return 0;
}
