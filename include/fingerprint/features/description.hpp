#pragma once
#include "fingerprint/core/types.hpp"
#include <vector>

namespace fp {

class Descriptor {
public:
  virtual ~Descriptor() = default;
};

template <typename T> class DescriptorExtractor {
public:
  virtual ~DescriptorExtractor() = default;
  virtual std::vector<T> extract(const std::vector<Minutia> &minutiae) = 0;
};

} // namespace fp
