project(tests)

file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE GTest::gtest_main cuda_kernels common)
target_include_directories(tests PRIVATE ../lib ../src)
