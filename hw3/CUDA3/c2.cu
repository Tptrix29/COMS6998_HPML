// c2.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define H    1024
#define W    1024
#define C       3
#define FH      3
#define FW      3
#define K      64
#define P       1   // padding = (FH-1)/2 = 1

// tile size in output space
#define TILE_X 16   // along H (rows)
#define TILE_Y 16   // along W (cols)

// shared‐mem patch dims = tile + (kernel−1)
#define SMEM_H (TILE_X + FH - 1)   // 16 + 3 - 1 = 18
#define SMEM_W (TILE_Y + FW - 1)   // 18

// straightforward flipped‐filter 2D conv, tiled + shared‐mem
__global__ void conv_tiled(
    const double * __restrict__ I0,  // [C][H+2P][W+2P]
    const double * __restrict__ F,   // [K][C][FH][FW]
          double *             O)     // [K][H][W]
{
    // which output‐filter slice?
    int k = blockIdx.z;

    // block origin in the padded input
    int base_x = blockIdx.x * TILE_X;
    int base_y = blockIdx.y * TILE_Y;

    // thread’s local coords
    int tx = threadIdx.x;   // 0..TILE_X-1
    int ty = threadIdx.y;   // 0..TILE_Y-1

    // shared memory: one patch per channel
    __shared__ double sI[C][SMEM_H][SMEM_W];

    // load the SMEM tile (each thread strides to cover the whole SMEM_H×SMEM_W)
    for (int c = 0; c < C; ++c) {
        for (int r = tx; r < SMEM_H; r += blockDim.x) {
            for (int s = ty; s < SMEM_W; s += blockDim.y) {
                int xi = base_x + r;           // padded‐input row
                int yj = base_y + s;           // padded‐input col
                double v = 0.0;
                // bounds‐check against padded dims:
                if (xi < (H + 2*P) && yj < (W + 2*P)) {
                    v = I0[
                        c * (H+2*P)*(W+2*P)
                      + xi * (W+2*P)
                      + yj
                    ];
                }
                sI[c][r][s] = v;
            }
        }
    }

    __syncthreads();

    // now compute one output pixel per thread
    int x = base_x + tx;  // global output row
    int y = base_y + ty;  // global output col

    if (x < H && y < W) {
        double sum = 0.0;
        // 3×3 flipped filter
        for (int c = 0; c < C; ++c) {
            for (int j = 0; j < FH; ++j) {
                for (int i = 0; i < FW; ++i) {
                    double in_val = sI[c][tx + j][ty + i];
                    double f_val = F[
                        k*C*FH*FW
                      + c*FH*FW
                      + (FH-1-j)*FW
                      + (FW-1-i)
                    ];
                    sum += in_val * f_val;
                }
            }
        }
        O[k*H*W + x*W + y] = sum;
    }
}

int main(void) {
    // byte‐sizes
    size_t bytes_I0  = (size_t)C * (H+2*P) * (W+2*P) * sizeof(double);
    size_t bytes_F   = (size_t)K * C * FH * FW       * sizeof(double);
    size_t bytes_O   = (size_t)K * H * W             * sizeof(double);

    // allocate + init host
    double *h_I0 = (double*)calloc((size_t)C*(H+2*P)*(W+2*P), sizeof(double));
    double *h_F  = (double*)malloc(bytes_F);
    double *h_O  = (double*)malloc(bytes_O);
    if (!h_I0 || !h_F || !h_O) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    // fill I0: pad a “c*(x+y)” input
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                int xo = x + P, yo = y + P;
                h_I0[
                  c*(H+2*P)*(W+2*P)
                + xo*(W+2*P)
                + yo
                ] = (double)c * (double)(x + y);
            }
        }
    }
    // fill F[k,c,i,j] = (c+k)*(i+j)
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    h_F[(k*C + c)*FH*FW + i*FW + j] =
                      (double)(c + k) * (double)(i + j);
                }
            }
        }
    }

    // device buffers
    double *d_I0 = NULL, *d_F = NULL, *d_O = NULL;
    cudaMalloc(&d_I0, bytes_I0);
    cudaMalloc(&d_F,  bytes_F);
    cudaMalloc(&d_O,  bytes_O);

    // copy in
    cudaMemcpy(d_I0, h_I0, bytes_I0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,  h_F,  bytes_F,  cudaMemcpyHostToDevice);

    // set up grid / block
    dim3 threads(TILE_X, TILE_Y);
    dim3 blocks((H+TILE_X-1)/TILE_X,
                (W+TILE_Y-1)/TILE_Y,
                K);

    // timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // launch + time only the kernel
    cudaEventRecord(start, 0);
    conv_tiled<<<blocks, threads>>>(d_I0, d_F, d_O);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // copy back & checksum
    cudaMemcpy(h_O, d_O, bytes_O, cudaMemcpyDeviceToHost);
    double checksum = 0;
    for (size_t i = 0; i < (size_t)K*H*W; ++i) checksum += h_O[i];

    // report
    printf("Checksum = %.6f  (should match C1)\n", checksum);
    printf("Kernel time: %.3f ms\n", ms);

    // cleanup
    free(h_I0);
    free(h_F);
    free(h_O);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

