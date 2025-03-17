#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cassert>

int estimate_exec_time(int M, int N, int K, int cpu_ops, int gpu_ops, int bandwidth, int num_offloaded_rows);

int get_recommended_number_offloaded_rows(int M, int N, int K, int cpu_ops, int gpu_ops, int bandwidth);

int get_recommended_number_offloaded_rows_optimized(int M, int N, int K, int cpu_ops, int gpu_ops, int bandwidth);