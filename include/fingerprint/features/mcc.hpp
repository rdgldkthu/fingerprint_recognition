#pragma once
#include "fingerprint/features/description.hpp"
#include <bitset>

namespace MCC_PARAMS {
const int NS = 8;
const int ND = 6;
const int NC = NS * NS * ND;
const float R = 70.0f;
const float SIGMA_S = 8.0f;
const float SIGMA_D = 0.5f;
} // namespace MCC_PARAMS

namespace fp {

class Cylinder : public Descriptor {
public:
  std::bitset<MCC_PARAMS::NC> bit_vector;
  std::bitset<MCC_PARAMS::NC> mask;

  static double compare(const Cylinder &a, const Cylinder &b);
};

class MCCExtractor : public DescriptorExtractor<Cylinder> {
public:
  std::vector<Cylinder> extract(const std::vector<Minutia> &minutiae) override;

private:
  Cylinder computeSingleCylinder(const Minutia &center,
                                 const std::vector<Minutia> &all);
  float getContribution(const Minutia &m, float cell_x, float cell_y,
                        float cell_theta);
};

} // namespace fp
