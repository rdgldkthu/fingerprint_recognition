#include "fingerprint/features/alignment_matcher.hpp"
#include <cmath>
#include <iostream>

using namespace fp;

static int fails = 0;

static void check(bool cond, const char* msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; ++fails; }
    else        { std::cout << "PASS: " << msg << "\n"; }
}

int main() {
    AlignmentMatcher matcher;

    // 1. Empty inputs
    check(matcher.computeScore({}, {}) == 0.0,                                          "empty A and B -> 0.0");
    check(matcher.computeScore({{100, 100, 0.f, MinutiaType::ENDING}}, {}) == 0.0,     "empty B -> 0.0");
    check(matcher.computeScore({}, {{100, 100, 0.f, MinutiaType::ENDING}}) == 0.0,     "empty A -> 0.0");

    std::vector<Minutia> A = {
        {100, 100, 0.5f,  MinutiaType::ENDING},
        {150, 200, 1.0f,  MinutiaType::BIFURCATION},
        { 50,  80, -0.3f, MinutiaType::ENDING},
    };

    // 2. Identity: A vs A
    check(std::abs(matcher.computeScore(A, A) - 1.0) < 1e-9, "identity -> 1.0");

    // 3. findBestAlignment returns all 3 pairs for identity
    auto pairs = matcher.findBestAlignment(A, A);
    check(pairs.size() == 3, "identity alignment -> 3 matched pairs");

    // 4. Pure translation: B = A shifted by (dx=7, dy=-4), same angles
    std::vector<Minutia> B;
    for (const auto& m : A)
        B.push_back({m.y - 4, m.x + 7, m.theta, m.type});
    check(std::abs(matcher.computeScore(A, B) - 1.0) < 1e-9, "pure translation -> 1.0");

    // 5. Subset: B contains only 2 of A's minutiae -> score = 2/3
    std::vector<Minutia> A_sub = {A[0], A[2]};
    check(std::abs(matcher.computeScore(A, A_sub) - 2.0 / 3.0) < 1e-9, "subset -> 2/3");

    // 6. Reversed order: same minutiae in B but shuffled
    std::vector<Minutia> A_rev = {A[2], A[0], A[1]};
    check(std::abs(matcher.computeScore(A, A_rev) - 1.0) < 1e-9, "reversed order -> 1.0");

    // 7. Single minutia matched against itself
    std::vector<Minutia> single = {{200, 300, 0.8f, MinutiaType::BIFURCATION}};
    check(std::abs(matcher.computeScore(single, single) - 1.0) < 1e-9, "single minutia -> 1.0");

    if (fails == 0) std::cout << "\nAll tests passed.\n";
    return fails;
}
