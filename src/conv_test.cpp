#include <gtest/gtest.h>
#include "conv.hpp"

// Tests for the estimate_exec_time function
TEST(EstimateExecTimeTest, WhenMatrixSmallerThanKernel) {
    // If M < K, the result should depend only on the CPU
    int result = estimate_exec_time(2, 10, 3, 100, 200, 50, 5);
    EXPECT_GT(result, 0); // The result must be positive
    
    // Check that changing num_offloaded_rows does not affect the result
    int result2 = estimate_exec_time(2, 10, 3, 100, 200, 50, 0);
    EXPECT_EQ(result, result2);
}

TEST(EstimateExecTimeTest, WhenMatrixEqualToKernel) {
    // When M = K the result should be valid
    int result = estimate_exec_time(3, 10, 3, 100, 200, 50, 3);
    EXPECT_GT(result, 0);
}

TEST(EstimateExecTimeTest, IncreasedGpuPerformanceReducesTime) {
    // As GPU performance increases, time should decrease
    int time1 = estimate_exec_time(100, 100, 3, 500, 1000, 200, 50);
    int time2 = estimate_exec_time(100, 100, 3, 500, 2000, 200, 50);
    EXPECT_GE(time1, time2);
}

TEST(EstimateExecTimeTest, IncreasedBandwidthReducesTime) {
    // As throughput increases, time should decrease
    int time1 = estimate_exec_time(100, 100, 3, 500, 1000, 100, 50);
    int time2 = estimate_exec_time(100, 100, 3, 500, 1000, 200, 50);
    EXPECT_GE(time1, time2);
}

TEST(EstimateExecTimeTest, SpecificParameterSet) {
    // Check with specific parameters
    int time_est = estimate_exec_time(100, 100, 3, 4, 10, 2, 3);
    EXPECT_GT(time_est, 0);
}

// Tests for get_recommended_number_offloaded_rows function
TEST(GetRecommendedNumberOffloadedRowsTest, WhenMatrixSmallerThanKernel) {
    // If M < K, should return 0
    int result = get_recommended_number_offloaded_rows(2, 10, 3, 100, 200, 50);
    EXPECT_EQ(result, 0);
}

TEST(GetRecommendedNumberOffloadedRowsTest, WhenMatrixEqualToKernel) {
    // For M = K the result depends on CPU/GPU performance and bandwidth
    int result = get_recommended_number_offloaded_rows(3, 10, 3, 100, 200, 50);
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 3);
}

TEST(GetRecommendedNumberOffloadedRowsTest, WhenGpuMuchFasterThanCpu) {
    // If the GPU is much faster than the CPU, most of the work should be done on the GPU
    int M = 20, N = 20, K = 3;
    int cpu_ops = 100, gpu_ops = 10000, bandwidth = 10000;
    int result = get_recommended_number_offloaded_rows(M, N, K, cpu_ops, gpu_ops, bandwidth);
    
    // We expect at least half of the rows to be offloaded to the GPU
    EXPECT_GE(result, M / 2);
}

TEST(GetRecommendedNumberOffloadedRowsTest, WhenCpuMuchFasterThanGpu) {
    // If the CPU is much faster than the GPU, most of the work should be done on the CPU
    int M = 20, N = 20, K = 3;
    int cpu_ops = 10000, gpu_ops = 100, bandwidth = 100;
    int result = get_recommended_number_offloaded_rows(M, N, K, cpu_ops, gpu_ops, bandwidth);
    
    // Expect less than half of the rows to be offloaded to the GPU
    EXPECT_LE(result, M / 2);
}

TEST(GetRecommendedNumberOffloadedRowsTest, WhenBandwidthIsVeryLow) {
    // At very low throughput, offloading to GPU is not profitable
    int M = 20, N = 20, K = 3;
    int cpu_ops = 100, gpu_ops = 1000, bandwidth = 1;
    int result = get_recommended_number_offloaded_rows(M, N, K, cpu_ops, gpu_ops, bandwidth);
    
    // Expect minimal or 0 rows on GPU
    EXPECT_LE(result, M / 4);
}

// Tests for get_recommended_number_offloaded_rows_optimized function
TEST(GetRecommendedNumberOffloadedRowsOptimizedTest, WhenMatrixSmallerThanKernel) {
    // If M < K, should return 0
    int result = get_recommended_number_offloaded_rows_optimized(2, 10, 3, 100, 200, 50);
    EXPECT_EQ(result, 0);
}

TEST(GetRecommendedNumberOffloadedRowsOptimizedTest, MatchesFullSearchForSmallMatrix) {
    // For small matrices the results should match the exhaustive search
    int M = 10, N = 10, K = 3;
    int cpu_ops = 100, gpu_ops = 200, bandwidth = 50;
    
    int full_result = get_recommended_number_offloaded_rows(M, N, K, cpu_ops, gpu_ops, bandwidth);
    int opt_result = get_recommended_number_offloaded_rows_optimized(M, N, K, cpu_ops, gpu_ops, bandwidth);
    
    EXPECT_EQ(full_result, opt_result);
}

TEST(GetRecommendedNumberOffloadedRowsOptimizedTest, OptimizedAlgorithmForLargeMatrix) {
    // Test for large matrices
    int M = 1000, N = 1000, K = 5;
    int cpu_ops = 500, gpu_ops = 2000, bandwidth = 100;
    
    int result = get_recommended_number_offloaded_rows_optimized(M, N, K, cpu_ops, gpu_ops, bandwidth);
    
    // The result must be within acceptable limits
    EXPECT_GE(result, 0);
    EXPECT_LE(result, M);
    
    // Check that the found solution really minimizes the time
    int time_at_result = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, result);
    
    // Check neighboring points
    int sample_points[] = {0, result/4, result/2, result*3/4, result-1, result, result+1, 
                          std::min(result*2, M), std::min(result*4, M), M};
    
    for (int point : sample_points) {
        if (point >= 0 && point <= M && point != result) {
            int time_at_point = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, point);
            EXPECT_GE(time_at_point, time_at_result) << "Найдено лучшее время при " << point << " строках чем при " << result;
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
