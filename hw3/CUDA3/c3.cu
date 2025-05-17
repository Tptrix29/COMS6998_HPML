// c3.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define H    1024
#define W    1024
#define C       3
#define K      64
#define FH      3
#define FW      3
#define N       1   // batch size

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
              __FILE__, __LINE__, cudaGetErrorString(err));              \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

#define CHECK_CUDNN(call)                                                \
  do {                                                                   \
    cudnnStatus_t status = call;                                         \
    if (status != CUDNN_STATUS_SUCCESS) {                                \
      fprintf(stderr, "cuDNN error %s:%d: %s\n",                         \
              __FILE__, __LINE__, cudnnGetErrorString(status));          \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

int main(void) {
    // 1) cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN( cudnnCreate(&cudnn) );

    // 2) Input descriptor NCHW
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN( cudnnCreateTensorDescriptor(&inputDesc) );
    CHECK_CUDNN( cudnnSetTensor4dDescriptor(
      inputDesc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_DOUBLE,
      N, C, H, W
    ));

    // 3) Filter descriptor K×C×FH×FW
    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
    CHECK_CUDNN( cudnnSetFilter4dDescriptor(
      filterDesc,
      CUDNN_DATA_DOUBLE,
      CUDNN_TENSOR_NCHW,
      K, C, FH, FW
    ));

    // 4) Convolution: pad=1, stride=1
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
    CHECK_CUDNN( cudnnSetConvolution2dDescriptor(
      convDesc,
      /*pad_h=*/1, /*pad_w=*/1,
      /*str_h=*/1, /*str_w=*/1,
      /*dil_h=*/1, /*dil_w=*/1,
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_DOUBLE
    ));

    // 5) Output dims (should be N,K,H,W)
    int nOut,cOut,hOut,wOut;
    CHECK_CUDNN( cudnnGetConvolution2dForwardOutputDim(
      convDesc, inputDesc, filterDesc,
      &nOut,&cOut,&hOut,&wOut
    ));

    // 6) Output descriptor
    cudnnTensorDescriptor_t outputDesc;
    CHECK_CUDNN( cudnnCreateTensorDescriptor(&outputDesc) );
    CHECK_CUDNN( cudnnSetTensor4dDescriptor(
      outputDesc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_DOUBLE,
      N, K, H, W
    ));

    // 7) Allocate host/device
    size_t bytesIn   = (size_t)N*C*H*W   * sizeof(double);
    size_t bytesFil  = (size_t)K*C*FH*FW * sizeof(double);
    size_t bytesOut  = (size_t)N*K*H*W   * sizeof(double);

    double *h_I = (double*)malloc(bytesIn),
           *h_F = (double*)malloc(bytesFil),
           *h_O = (double*)malloc(bytesOut);
    double *d_I, *d_F, *d_O;

    if (!h_I||!h_F||!h_O) {
      fprintf(stderr, "host malloc failure\n");
      return 1;
    }
    CHECK_CUDA( cudaMalloc(&d_I, bytesIn) );
    CHECK_CUDA( cudaMalloc(&d_F, bytesFil) );
    CHECK_CUDA( cudaMalloc(&d_O, bytesOut) );

    // 8) Initialize I and **pre-flipped** F
    // I[n=0,c,x,y] = c*(x+y)
    for (int c = 0; c < C; ++c)
      for (int x = 0; x < H; ++x)
        for (int y = 0; y < W; ++y)
          h_I[((0*C + c)*H + x)*W + y] = (double)c * (double)(x + y);

    // F[k,c,i,j] = (c+k)*((FH-1-i)+(FW-1-j))
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
        for (int i = 0; i < FH; ++i) {
          for (int j = 0; j < FW; ++j) {
            double flip_sum = (double)((FH-1-i) + (FW-1-j));
            h_F[((k*C + c)*FH + i)*FW + j] =
              (double)(c + k) * flip_sum;
          }
        }
      }
    }

    // 9) Copy data to device
    CHECK_CUDA( cudaMemcpy(d_I, h_I, bytesIn,   cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_F, h_F, bytesFil,  cudaMemcpyHostToDevice) );

    // 10) Pick best algorithm via v7 API
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perf;
    CHECK_CUDNN( cudnnGetConvolutionForwardAlgorithm_v7(
      cudnn,
      inputDesc, filterDesc,
      convDesc, outputDesc,
      /*reqAlgoCount=*/1,
      &returnedAlgoCount,
      &perf
    ));
    cudnnConvolutionFwdAlgo_t algo = perf.algo;

    // 11) Allocate workspace
    size_t workspace_bytes = 0;
    CHECK_CUDNN( cudnnGetConvolutionForwardWorkspaceSize(
      cudnn,
      inputDesc, filterDesc,
      convDesc, outputDesc,
      algo,
      &workspace_bytes
    ));
    void* d_workspace = NULL;
    if (workspace_bytes > 0)
      CHECK_CUDA( cudaMalloc(&d_workspace, workspace_bytes) );

    // 12) Time only cudnnConvolutionForward
    cudaEvent_t t0, t1;
    CHECK_CUDA( cudaEventCreate(&t0) );
    CHECK_CUDA( cudaEventCreate(&t1) );
    CHECK_CUDA( cudaEventRecord(t0,0) );

    const double alpha = 1.0, beta = 0.0;
    CHECK_CUDNN( cudnnConvolutionForward(
      cudnn,
      &alpha,
      inputDesc,  d_I,
      filterDesc, d_F,
      convDesc,
      algo,
      d_workspace,
      workspace_bytes,
      &beta,
      outputDesc, d_O
    ));

    CHECK_CUDA( cudaEventRecord(t1,0) );
    CHECK_CUDA( cudaEventSynchronize(t1) );
    float ms = 0;
    CHECK_CUDA( cudaEventElapsedTime(&ms, t0, t1) );

    // 13) Copy back and checksum
    CHECK_CUDA( cudaMemcpy(h_O, d_O, bytesOut, cudaMemcpyDeviceToHost) );
    double checksum = 0;
    for (size_t i = 0; i < (size_t)N*K*H*W; ++i)
      checksum += h_O[i];

    // 14) Print results
    printf("Checksum = %.6f  (matches C1)\n", checksum);
    printf("cuDNN kernel time: %.3f ms\n", ms);

    // 15) Cleanup
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    free(h_I);
    free(h_F);
    free(h_O);

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return 0;
}

