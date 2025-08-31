//
// Created by stefan on 8/31/25.
//

#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

float GetMean(const std::vector<float>& values)
{
  const float sum = std::accumulate(values.begin(), values.end(), 0.0);
  const float mean = sum / values.size();
  return mean;
}

float GetStd(const std::vector<float>& values)
{
  const float sum = std::accumulate(values.begin(), values.end(), 0.0);
  float mean = sum / values.size();

  std::vector<float> diff(values.size());
  std::ranges::transform(
      values,
      diff.begin(),
      std::bind(std::minus<float>(), std::placeholders::_1, mean));

  const float sq_sum
      = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

  const float std = std::sqrt(sq_sum / values.size());
  return std;
}
