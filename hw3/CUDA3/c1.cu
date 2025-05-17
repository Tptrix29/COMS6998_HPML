// c1_timed.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define H   1024
#define W   1024
#define C     3
#define FH    3
#define FW    3
#define K    64
#define P     1   // padding

// CUDA kernel: each (k, x, y) thread computes one output pixel O[k, x, y]
__global__ void conv_kernel(
    const double *I0,  // padded input: [C][H+2P][W+2P]
    const double *F,   // filters:      [K][C][FH][FW]
          double *O)   // output:       [K][H][W]
{
    int k = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= H || y >= W) return;

    double sum = 0.0;
    int stride_Ic = (H + 2*P) * (W + 2*P);
    int stride_Ix = (W + 2*P);
    int stride_Fk = C * FH * FW;
    int stride_Fc = FH * FW;

    for (int c = 0; c < C; ++c) {
        for (int j = 0; j < FH; ++j) {
            for (int i = 0; i < FW; ++i) {
                int fi = FW - 1 - i, fj = FH - 1 - j;
                int xi = x + i, yj = y + j;
                double in_val = I0[c * stride_Ic + xi * stride_Ix + yj];
                double f_val  = F[k * stride_Fk + c * stride_Fc + fj * FW + fi];
                sum += in_val * f_val;
            }
        }
    }
    O[k * (H*W) + x * W + y] = sum;
}

int main(void) {
    size_t bytes_I   = (size_t)C * H * W          * sizeof(double);
    size_t bytes_I0  = (size_t)C * (H+2*P) * (W+2*P) * sizeof(double);
    size_t bytes_F   = (size_t)K * C * FH * FW    * sizeof(double);
    size_t bytes_O   = (size_t)K * H * W          * sizeof(double);

    // Host allocations (with casts)
    double *h_I  = (double*)malloc(bytes_I);
    double *h_I0 = (double*)calloc((size_t)C*(H+2*P)*(W+2*P), sizeof(double));
    double *h_F  = (double*)malloc(bytes_F);
    double *h_O  = (double*)malloc(bytes_O);
    if (!h_I||!h_I0||!h_F||!h_O) {
        fprintf(stderr,"host alloc failed\n");
        return 1;
    }

    // Init I[c,x,y] = c*(x+y)
    for (int c = 0; c < C; ++c)
    for (int x = 0; x < H; ++x)
    for (int y = 0; y < W; ++y)
        h_I[c*H*W + x*W + y] = (double)c * (double)(x + y);

    // Pad I -> I0
    for (int c = 0; c < C; ++c)
    for (int x = 0; x < H; ++x)
    for (int y = 0; y < W; ++y) {
        int xo = x + P, yo = y + P;
        h_I0[c*(H+2*P)*(W+2*P) + xo*(W+2*P) + yo] =
            h_I[c*H*W + x*W + y];
    }

    // Init F[k,c,i,j] = (c+k)*(i+j)
    for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c)
    for (int i = 0; i < FH; ++i)
    for (int j = 0; j < FW; ++j)
        h_F[(k*C + c)*FH*FW + i*FW + j] =
            (double)(c + k) * (double)(i + j);

    // Device allocations
    double *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, bytes_I0);
    cudaMalloc(&d_F,  bytes_F);
    cudaMalloc(&d_O,  bytes_O);

    // Copy to device
    cudaMemcpy(d_I0, h_I0, bytes_I0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,  h_F,  bytes_F,  cudaMemcpyHostToDevice);

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel & time it
    dim3 threads(16,16);
    dim3 blocks((H+15)/16, (W+15)/16, K);
    cudaEventRecord(start, 0);
    conv_kernel<<<blocks, threads>>>(d_I0, d_F, d_O);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy back
    cudaMemcpy(h_O, d_O, bytes_O, cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)K*H*W; ++i)
        checksum += h_O[i];

    // Print results
    printf("Checksum = %.6f\n", checksum);
    printf("Kernel time   = %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);
    free(h_I);
    free(h_I0);
    free(h_F);
    free(h_O);

    return 0;
}

