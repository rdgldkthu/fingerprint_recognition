#include "fingerprint/features/mcc.hpp"
#include <cmath>

namespace fp {

double Cylinder::compare(const Cylinder &a, const Cylinder &b) {
  auto valid_mask = a.mask & b.mask;
  if (valid_mask.none())
    return 0.0;
  auto diff = (a.bit_vector ^ b.bit_vector) & valid_mask;
  return 1.0 - (static_cast<double>(diff.count()) / valid_mask.count());
}

std::vector<Cylinder>
MCCExtractor::extract(const std::vector<Minutia> &minutiae) {
  std::vector<Cylinder> cylinders;
  cylinders.reserve(minutiae.size());
  for (const auto &center : minutiae) {
    cylinders.push_back(computeSingleCylinder(center, minutiae));
  }
  return cylinders;
}

Cylinder MCCExtractor::computeSingleCylinder(const Minutia &center,
                                             const std::vector<Minutia> &all) {
  Cylinder cyl;
  float cos_c = std::cos(center.theta);
  float sin_c = std::sin(center.theta);

  for (int k = 0; k < MCC_PARAMS::ND; ++k) {
    float cell_theta = (2.0f * M_PI * k) / MCC_PARAMS::ND;
    for (int i = 0; i < MCC_PARAMS::NS; ++i) {
      for (int j = 0; j < MCC_PARAMS::NS; ++j) {
        // Cell local coordinates
        float cx = (i - (MCC_PARAMS::NS - 1) / 2.0f) *
                   (2.0f * MCC_PARAMS::R / MCC_PARAMS::NS);
        float cy = (j - (MCC_PARAMS::NS - 1) / 2.0f) *
                   (2.0f * MCC_PARAMS::R / MCC_PARAMS::NS);

        // Rotate the cell coordinates around the center
        float rx = center.x + (cx * cos_c - cy * sin_c);
        float ry = center.y + (cx * sin_c + cy * cos_c);

        float total_contribution = 0.0f;
        for (const auto &m : all) {
          if (&m == &center)
            continue;
          total_contribution +=
              getContribution(m, rx, ry, center.theta + cell_theta);
        }

        // Sigmoid binarization
        int bit_idx =
            k * (MCC_PARAMS::NS * MCC_PARAMS::NS) + i * MCC_PARAMS::NS + j;
        if (1.0f / (1.0f + std::exp(-10.0f * (total_contribution - 0.5f))) >
            0.5f) {
          cyl.bit_vector.set(bit_idx);
        }
        cyl.mask.set(bit_idx);
      }
    }
  }
  return cyl;
}

float MCCExtractor::getContribution(const Minutia &m, float cell_x,
                                    float cell_y, float cell_theta) {
  float dx = cell_x - m.x;
  float dy = cell_y - m.y;
  float d2 = dx * dx + dy * dy;

  // Spatial contribution
  float c_s =
      std::exp(-d2 / (2.0f * MCC_PARAMS::SIGMA_S * MCC_PARAMS::SIGMA_S));

  // Directional contribution
  float d_theta = std::abs(cell_theta - m.theta);
  if (d_theta > M_PI)
    d_theta = 2.0f * M_PI - d_theta;
  float c_d = std::exp(-(d_theta * d_theta) /
                       (2.0f * MCC_PARAMS::SIGMA_D * MCC_PARAMS::SIGMA_D));

  return c_s * c_d;
}

} // namespace fp
