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

// Runs one scenario with unified memory
void run_scenario(int blocks, int threads_per_block, int size, int K) {
    std::cout << "Running with " << blocks << " blocks and "
              << threads_per_block << " threads per block, K = " << K << std::endl;

    size_t bytes = size * sizeof(float);

    // Allocate unified memory
    float *a, *b, *c;
    check_cuda_error(cudaMallocManaged(&a, bytes), "cudaMallocManaged a failed");
    check_cuda_error(cudaMallocManaged(&b, bytes), "cudaMallocManaged b failed");
    check_cuda_error(cudaMallocManaged(&c, bytes), "cudaMallocManaged c failed");

    // Initialize arrays directly (no need for separate host arrays)
    for (int i = 0; i < size; i++) {
        a[i] = 1;
        b[i] = 1;
    }

    // Warm up the GPU
    add_arrays<<<blocks, threads_per_block>>>(a, b, c, size);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), "Kernel launch failed");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start event
    cudaEventRecord(start);

    // Launch kernel
    add_arrays<<<blocks, threads_per_block>>>(a, b, c, size);

    // Stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "Execution time: " << duration << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host (optional for profiling, but keep for correctness)
    // Verify correctness with a few samples
    std::cout << "Sample results: c[0]=" << c[0] << ", c[size/2]=" << c[size/2] 
              << ", c[size-1]=" << c[size-1] << std::endl;
    // Check sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += c[i];
    }
    std::cout << "Sum of c: " << sum << std::endl;

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
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
    int threads = 256;
    int blocks = (array_size + threads - 1) / threads;
    run_scenario(blocks, threads, array_size, K);

    return 0;
}