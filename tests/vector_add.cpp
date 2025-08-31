//
// Created by stefan on 8/31/25.
//

#include <numeric>

#include <gtest/gtest.h>

#include <common/common.h>
#include <vector_add/vector_add.h>

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

TEST(VectorAddTest, EndToEnd)
{
  constexpr int size = 1000000;
  std::vector<int> a(size), b(size), c(size);

  initVectors(a, b, size);

  const CudaTimer timer;
  timer.start();
  vectorAddHost(a.data(), b.data(), c.data(), size);
  const float time = timer.stop();

  std::cout << "End-to-end time: " << time << " ms" << std::endl;
  std::cout << "Throughput: " << (size / (time / 1000.0f)) / 1e6 << " Mops/s"
            << std::endl;

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