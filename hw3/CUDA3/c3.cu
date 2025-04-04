#include <iostream>
#include <iomanip>
#include <chrono>
#include <cudnn.h>

#define H 1024
#define W 1024
#define C 3
#define P 1
#define FW 3
#define FH 3
#define K 64

void checkCUDNN(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << msg << ": " << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCUDA(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    double *h_input, *h_output, *h_filter;
    double *d_input, *d_output, *d_filter;
    size_t input_size = (H + 2 * P) * (W + 2 * P) * C * sizeof(double);
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
    checkCUDA(cudaMalloc(&d_input, input_size), "cudaMalloc input");
    checkCUDA(cudaMalloc(&d_output, output_size), "cudaMalloc output");
    checkCUDA(cudaMalloc(&d_filter, filter_size), "cudaMalloc filter");

    // Copy input and filter arrays from host to device
    checkCUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Memcpy input");
    checkCUDA(cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice), "Memcpy filter");

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn), "Create cuDNN handle");

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_desc), "Create input tensor descriptor");
    checkCUDNN(cudnnCreateTensorDescriptor(&output_desc), "Create output tensor descriptor");
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc), "Create filter descriptor");
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc), "Create convolution descriptor");

    checkCUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE,
                                          1, C, H, W),
               "Set input descriptor");

    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc,
                                          CUDNN_DATA_DOUBLE,
                                          CUDNN_TENSOR_NCHW,
                                          K, C, FH, FW),
               "Set filter descriptor");

    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                               P, P,  // padding height, width
                                               1, 1,  // stride height, width
                                               1, 1,  // dilation height, width
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_DOUBLE),
               "Set convolution descriptor");

    // Set the output descriptor
    int n, c, h, w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w),
               "Get output dimensions");

    checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE,
                                          n, c, h, w),
               "Set output descriptor");

    // Algorithm selection
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    // Get workspace size
    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_desc,
                                                       filter_desc,
                                                       conv_desc,
                                                       output_desc,
                                                       algo,
                                                       &workspace_size),
               "Get workspace size");

    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        checkCUDA(cudaMalloc(&d_workspace, workspace_size), "Malloc workspace");
    }

    // Perform the convolution
    // Note: The alpha and beta values are used for scaling the output
    const double alpha = 1.0, beta = 0.0;

    // Warm up
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_desc,
                                       d_input,
                                       filter_desc,
                                       d_filter,
                                       conv_desc,
                                       algo,
                                       d_workspace,
                                       workspace_size,
                                       &beta,
                                       output_desc,
                                       d_output),
               "Warm up");
    // Synchronize to ensure the warm-up is complete
    checkCUDA(cudaDeviceSynchronize(), "Warm up synchronize");

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform the convolution
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_desc,
                                       d_input,
                                       filter_desc,
                                       d_filter,
                                       conv_desc,
                                       algo,
                                       d_workspace,
                                       workspace_size,
                                       &beta,
                                       output_desc,
                                       d_output),
               "Convolution forward");

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "Execution time: " << std::fixed << std::setprecision(3) << duration << " ms" << std::endl;

    // Clean cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCUDA(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Memcpy output");

    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                checksum += h_output[k * H * W + h * W + w];
            }
        }
    }
    std::cout << "Checksum: " << std::fixed << std::setprecision(0) << checksum << std::endl;


    if (workspace_size > 0) cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    free(h_input);
    free(h_output);
    free(h_filter);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
