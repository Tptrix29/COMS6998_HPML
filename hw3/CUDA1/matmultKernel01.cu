// Matrix multiplication kernel with shared memory
//
#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

  int row = block_row * BLOCK_SIZE + thread_row;
  int col = block_col * BLOCK_SIZE + thread_col;
  float Cvalue = 0;
  // Loop over all sub matrices in block_row of A and block_col of B
  for (int m = 0;  m < ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m){
    if (row < A.height && m * BLOCK_SIZE + thread_col < A.width)
      shared_A[thread_row][thread_col] = A.elements[row * A.stride + m * BLOCK_SIZE + thread_col];
    else
      shared_A[thread_row][thread_col] = 0.0f;
    if (col < B.width && m * BLOCK_SIZE + thread_row < B.height)
      shared_B[thread_row][thread_col] = B.elements[(m * BLOCK_SIZE + thread_row) * B.stride + col];
    else
      shared_B[thread_row][thread_col] = 0.0f;

    // Synchronize to ensure all elements are read
    __syncthreads();
    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
    #pragma unroll
    // This unroll is not necessary, but it helps the compiler
    // to optimize the code. It is not a good idea to unroll
    for(int e=0; e<BLOCK_SIZE; ++e)
      Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];
    __syncthreads();
  }
  if (row < C.height && col < C.width)
    C.elements[row * C.stride + col] = Cvalue;
}

