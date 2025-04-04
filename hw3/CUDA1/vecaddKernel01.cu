__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int totalThreads = blockDim.x * gridDim.x;

    for (int i = 0; i < N; i++){
        int index = i * totalThreads + tid;
        C[index] = A[index] + B[index];
    }

}