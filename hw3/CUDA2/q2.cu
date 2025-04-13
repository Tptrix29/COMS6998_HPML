#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 1000000  // Array size (adjustable)

// Kernel function for array addition
__global__ void add_arrays(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    // Each thread processes multiple elements
    for (int i = 0; i < (size - 1) / totalThreads + 1; i++) {
        int index = i * totalThreads + idx;
        if (index < size) {
            c[index] = a[index] + b[index];
        }
    }
}

// Utility to check CUDA errors
void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Runs one scenario
void run_scenario(int blocks, int threads_per_block, int size, int K) {
    std::cout << "Running with " << blocks << " blocks and "
              << threads_per_block << " threads per block, K = " << K << std::endl;

    size_t bytes = size * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = 1;
        h_b[i] = 1;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    check_cuda_error(cudaMalloc(&d_a, bytes), "cudaMalloc d_a failed");
    check_cuda_error(cudaMalloc(&d_b, bytes), "cudaMalloc d_b failed");
    check_cuda_error(cudaMalloc(&d_c, bytes), "cudaMalloc d_c failed");

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Warm up the GPU
    add_arrays<<<blocks, threads_per_block>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), "Kernel launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "Kernel execution failed");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start event
    cudaEventRecord(start);

    // Launch kernel
    add_arrays<<<blocks, threads_per_block>>>(d_a, d_b, d_c, size);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "Execution time: " << duration << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host (optional for profiling, but keep for correctness)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    // Verify correctness with a few samples
    std::cout << "Sample results: c[0]=" << h_c[0] << ", c[size/2]=" << h_c[size/2] 
              << ", c[size-1]=" << h_c[size-1] << std::endl;
    // Check sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += h_c[i];
    }
    std::cout << "Sum of c: " << sum << std::endl;

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " K" << std::endl;
        std::cerr << "K is the size of arrays in millions" << std::endl;
        return 1;
    }

    int K = atoi(argv[1]);
    int array_size = K * N;

    // Scenario 1: 1 block, 1 thread
    run_scenario(1, 1, array_size, K);

    // Scenario 2: 1 block, 256 threads
    run_scenario(1, 256, array_size, K);

    // Scenario 3: Multiple blocks with 256 threads per block
    int threads = 256; // 256 threads
    int blocks = (array_size + threads - 1) / threads;
    run_scenario(blocks, threads, array_size, K);

    return 0;
}
