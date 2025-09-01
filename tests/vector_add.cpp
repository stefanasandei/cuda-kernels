//
// Created by stefan on 8/31/25.
//

#include <gtest/gtest.h>

#include <common/common.h>
#include <kernels.h>

inline void initVectors(std::vector<int>& a, std::vector<int>& b, int size);
inline void computeExpected(
    std::vector<int>& expected,
    const std::vector<int>& a,
    const std::vector<int>& b,
    int size);

// ------------ Tests ------------

TEST(VectorAddTest, Correctness)
{
  constexpr int size = 1000;
  std::vector<int> a(size), b(size), c(size), expected(size);

  initVectors(a, b, size);
  computeExpected(expected, a, b, size);

  vectorAddHost(a.data(), b.data(), c.data(), size);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(c[i], expected[i]);
  }
}

TEST(VectorAddTest, MiniBenchmark)
{
  constexpr int size = 1000000;
  auto benchmarkPayload = BenchmarkPayload(2, 10);

  std::vector<int> a(size), b(size), c(size);
  initVectors(a, b, size);

  vectorAddHost(a.data(), b.data(), c.data(), size, benchmarkPayload);

  const auto mean = GetMean(*benchmarkPayload.OutTimes);
  const auto std = GetStd(*benchmarkPayload.OutTimes);
  std::print("Time: {:.3f} ± {:.3f} ms\n", mean, std);
  // std::print("Throughput: {} Mops/s\n", (size / (mean / 1000.0f)) / 1e6);

  EXPECT_EQ(c[0], a[0] + b[0]);
}

// ------------ Helper Functions ------------

inline void initVectors(std::vector<int>& a, std::vector<int>& b, int size)
{
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), size);
}

inline void computeExpected(
    std::vector<int>& expected,
    const std::vector<int>& a,
    const std::vector<int>& b,
    const int size)
{
  for (int i = 0; i < size; ++i) {
    expected[i] = a[i] + b[i];
  }
}