#include "conv.hpp"

int estimate_exec_time(int M, int N, int K, int cpu_ops, int gpu_ops, int bandwidth, int num_offloaded_rows) {
    const int data_size = sizeof(float);
    // Each fold position requires K×K multiplications and K×K - 1 additions, so the number of operations per position = 2×K×K - 1
    int ops_per_position = 2 * K * K - 1;

    int output_height = M - K + 1;
    int output_width = N - K + 1;
    long long total_output_positions = static_cast<long long>(output_height) * output_width;

    int gpu_output_rows = std::max(0, std::min(num_offloaded_rows - K + 1, output_height));
    int cpu_output_rows = output_height - gpu_output_rows;
    long long gpu_positions = static_cast<long long>(gpu_output_rows) * output_width;
    long long cpu_positions = total_output_positions - gpu_positions;

    long long cpu_total_ops = cpu_positions * ops_per_position;
    long long gpu_total_ops = gpu_positions * ops_per_position;

    double cpu_time = static_cast<double>(cpu_total_ops) / cpu_ops;
    double gpu_compute_time = static_cast<double>(gpu_total_ops) / gpu_ops;

    int rows_to_transfer = std::min(gpu_output_rows + K - 1, M);
    int gpu_input_data_size = rows_to_transfer * N * data_size;
    int gpu_output_data_size = gpu_positions * data_size;

    double transfer_to_gpu_time = static_cast<double>(gpu_input_data_size) / bandwidth;
    double transfer_from_gpu_time = static_cast<double>(gpu_output_data_size) / bandwidth;
    double gpu_total_time = transfer_to_gpu_time + gpu_compute_time + transfer_from_gpu_time;

    double total_time = std::max(cpu_time, gpu_total_time);
    return static_cast<int>(std::ceil(total_time));
}

int get_recommended_number_offloaded_rows(int M, int N, int K, int cpu_ops, int gpu_ops, int bandwidth) {
    // If the matrix is ​​smaller than the convolution kernel, GPU computation is not possible
    if (M < K) {
        return 0;
    }

    int best_rows = 0;
    int min_time = std::numeric_limits<int>::max();

    // We go through all possible distribution options
    // from 0 to M lines per GPU
    for (int rows = 0; rows <= M; ++rows) {
        int time = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, rows);

        if (time < min_time) {
            min_time = time;
            best_rows = rows;
        }
    }

    return best_rows;
}

int get_recommended_number_offloaded_rows_optimized(int M, int N, int K, int cpu_ops, int gpu_ops, int bandwidth) {
    if (M < K) {
        return 0;
    }

    // For small matrices we use a complete enumeration
    if (M < 100) {
        return get_recommended_number_offloaded_rows(M, N, K, cpu_ops, gpu_ops, bandwidth);
    }

    // For large matrices we use binary search
    // We assume that the time function has one minimum

    int left = 0;
    int right = M;

    while (right - left > 2) {
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;

        int time1 = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, mid1);
        int time2 = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, mid2);

        if (time1 <= time2) {
            right = mid2;
        }
        else {
            left = mid1;
        }
    }

    // Accurate testing on a narrow range
    int best_rows = left;
    int min_time = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, left);

    for (int rows = left + 1; rows <= right; ++rows) {
        int time = estimate_exec_time(M, N, K, cpu_ops, gpu_ops, bandwidth, rows);
        if (time < min_time) {
            min_time = time;
            best_rows = rows;
        }
    }

    return best_rows;
}
