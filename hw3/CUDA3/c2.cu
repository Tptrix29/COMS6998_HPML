#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

#define H 1024
#define W 1024
#define C 3
#define P 1
#define FW 3
#define FH 3
#define K 64

#define TILE_WIDTH 16

// Kernel function to perform 2D convolution without tiling
__global__ void conv2d(double* input, double* filter, double* output) {
    // For same convolution with P = 1, the shared memory tile size must include a 1-pixel border.
    const int tile_rows = TILE_WIDTH + 2 * P; // 16 + 2 = 18
    const int tile_cols = TILE_WIDTH + 2 * P; // 18

    // Allocate shared memory: one tile for each channel.
    __shared__ double sharedInput[C][tile_rows][tile_cols];

    // Compute the output pixel indices this thread is responsible for.
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z;  // Each block in z-dimension handles one filter (kernel) index.

    // The top-left corner (in the global input) corresponding to the tile, without padding.
    int in_row_start = blockIdx.y * blockDim.y;
    int in_col_start = blockIdx.x * blockDim.x;

    // Each thread loads one or more elements of the shared tile.
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int numElements = tile_rows * tile_cols;

    // Load the tile (with halo) into shared memory.
    for (int idx = threadId; idx < numElements; idx += numThreads) {
        int i = idx / tile_cols;  // Row index in the shared tile.
        int j = idx % tile_cols;  // Column index in the shared tile.
        // Adjust index in global in put for padding offset
        int global_row = in_row_start - P + i;
        int global_col = in_col_start - P + j;
        for (int c = 0; c < C; c++) {
            if (global_row >= 0 && global_row < H && global_col >= 0 && global_col < W) {
                sharedInput[c][i][j] = input[c * H * W + global_row * W + global_col];
            } else {
                sharedInput[c][i][j] = 0.0;
            }
        }
    }

    __syncthreads();

    double sum = 0.0;
    // Compute convolution if the output pixel is within the bounds.
    if (out_row < H && out_col < W) {
        // Iterate over channels and the filter window.
        for (int c = 0; c < C; c++) {
            for (int fh = 0; fh < FH; fh++) {
                for (int fw = 0; fw < FW; fw++) {
                    // Use threadIdx plus the filter offset to index into shared memory.
                    double in_val = sharedInput[c][threadIdx.y + fh][threadIdx.x + fw];
                    // Filter is stored as [K][C][FH][FW].
                    double filter_val = filter[k * (C * FH * FW) +
                                               c * (FH * FW) +
                                               fh * FW +
                                               fw];
                    sum += in_val * filter_val;
                }
            }
        }
        output[k * (H * W) + out_row * W + out_col] = sum;
    }
}


void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char* argv[]) {
    double *h_input, *h_output, *h_filter;
    double *d_input, *d_output, *d_filter;
    size_t input_size = H * W * C * sizeof(double);
    size_t output_size = H * W * K * sizeof(double);
    size_t filter_size = K * C * FH * FW * sizeof(double);

    // Allocate host memory
    h_input = (double*)malloc(input_size);
    h_output = (double*)malloc(output_size);
    h_filter = (double*)malloc(filter_size);
    if (!h_input || !h_output || !h_filter) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return -1;
    }

    // Initialize input and filter arrays
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                h_input[c * H * W + h * W + w] = c * (h + w);
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int fh = 0; fh < FH; ++fh) {
                for (int fw = 0; fw < FW; ++fw) {
                    h_filter[k * C * FH * FW + c * FH * FW + fh * FW + fw] = (c + k) * (fh + fw);
                }
            }
        }
    }
    // Allocate device memory
    check_cuda_error(cudaMalloc((void**)&d_input, input_size), "Failed to allocate device input memory");
    check_cuda_error(cudaMalloc((void**)&d_output, output_size), "Failed to allocate device output memory");
    check_cuda_error(cudaMalloc((void**)&d_filter, filter_size), "Failed to allocate device filter memory");
    // Copy input and filter arrays from host to device
    check_cuda_error(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Failed to copy input to device");
    check_cuda_error(cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice), "Failed to copy filter to device");
    
    // Define grid and block sizes
    dim3 block(16, 16, 1);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, K);

    // Warm up
    conv2d<<<grid, block>>>(d_input, d_filter, d_output);
    check_cuda_error(cudaGetLastError(), "Kernel launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "Kernel execution failed");
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start event
    cudaEventRecord(start);

    // Launch kernel
    conv2d<<<grid, block>>>(d_input, d_filter, d_output);
    check_cuda_error(cudaGetLastError(), "Kernel launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "Kernel execution failed");

    // Stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "Execution time: " << std::fixed << std::setprecision(3) << duration << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy output array from device to host
    check_cuda_error(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Failed to copy output to host");
    // Checksum
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                checksum += h_output[k * H * W + h * W + w];
            }
        }
    }
    std::cout << "Checksum: " << std::fixed << std::setprecision(0) << checksum << std::endl;
    // Free device memory
    check_cuda_error(cudaFree(d_input), "Failed to free device input memory");
    check_cuda_error(cudaFree(d_output), "Failed to free device output memory");
    check_cuda_error(cudaFree(d_filter), "Failed to free device filter memory");
    // Free host memory
    free(h_input);
    free(h_output);
    free(h_filter);
    return 0;
}


