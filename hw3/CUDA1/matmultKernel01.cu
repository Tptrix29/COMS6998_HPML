#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Kernel where each thread computes a 2x2 tile (four values) of C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

  // Pointers for the submatrices.
  float *Asub, *Bsub, *Csub;

  // Thread indices in the block.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each thread block now computes a submatrix of size:
  // (2 * BLOCK_SIZE) x (2 * BLOCK_SIZE)
  // Adjust the Csub pointer accordingly.
  Csub = &C.elements[C.stride * (2 * BLOCK_SIZE * block_row) + (2 * BLOCK_SIZE * block_col)];

  // Each thread computes a 2x2 output tile.
  float Cvalue00 = 0, Cvalue01 = 0, Cvalue10 = 0, Cvalue11 = 0;

  // Loop over all tiles in the K dimension.
  // (A.width is the inner dimension size, which remains tiled by BLOCK_SIZE.)
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // For matrix A, the submatrix has 2*BLOCK_SIZE rows and BLOCK_SIZE columns.
    Asub = &A.elements[A.stride * (2 * BLOCK_SIZE * block_row) + (BLOCK_SIZE * m)];
    // For matrix B, the submatrix has BLOCK_SIZE rows and 2*BLOCK_SIZE columns.
    Bsub = &B.elements[B.stride * (BLOCK_SIZE * m) + (2 * BLOCK_SIZE * block_col)];

    // Declare shared memory for the tiles.
    __shared__ float shared_A[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][2 * BLOCK_SIZE];

    // Each thread loads two elements of A (its corresponding two rows).
    shared_A[2 * thread_row][thread_col]         = Asub[(2 * thread_row) * A.stride + thread_col];
    shared_A[2 * thread_row + 1][thread_col]       = Asub[(2 * thread_row + 1) * A.stride + thread_col];

    // Each thread loads two elements of B (its corresponding two columns).
    shared_B[thread_row][2 * thread_col]           = Bsub[thread_row * B.stride + (2 * thread_col)];
    shared_B[thread_row][2 * thread_col + 1]         = Bsub[thread_row * B.stride + (2 * thread_col + 1)];

    // Synchronize so that the tiles are fully loaded.
    __syncthreads();

    // Compute the 2x2 partial block product using the shared memory.
    #pragma unroll
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      // Load the two A values for the two rows of the tile.
      float a0 = shared_A[2 * thread_row][e];
      float a1 = shared_A[2 * thread_row + 1][e];
      // Load the two B values for the two columns of the tile.
      float b0 = shared_B[e][2 * thread_col];
      float b1 = shared_B[e][2 * thread_col + 1];

      // Accumulate partial results for the 2x2 output tile.
      Cvalue00 += a0 * b0;
      Cvalue01 += a0 * b1;
      Cvalue10 += a1 * b0;
      Cvalue11 += a1 * b1;
    }

    // Synchronize before loading the next tile.
    __syncthreads();
  }

  // Write the computed 2x2 tile from registers to the global memory.
  int C_row = 2 * thread_row;
  int C_col = 2 * thread_col;
  Csub[C_row * C.stride + C_col]         = Cvalue00;
  Csub[C_row * C.stride + C_col + 1]       = Cvalue01;
  Csub[(C_row + 1) * C.stride + C_col]     = Cvalue10;
  Csub[(C_row + 1) * C.stride + C_col + 1]   = Cvalue11;
}