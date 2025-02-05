#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl_cblas.h>


float bdp(long N, float *pA, float *pB) {
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R;
}

int main(int argc, char *argv[]) {
    long N = atol(argv[1]);
    int nloop = atoi(argv[2]);
    float A[N], B[N];

    struct timespec start, end;
    double elapsed_time, total_time = 0.0;
    double avg_time, bandwidth, throughput;

    // Initialize A and B
    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }
    
    // Execute dp function
    for (int i = 0; i < nloop; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        bdp(N, A, B);
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        if (i / 2 == 0) {
            total_time += elapsed_time;
        }
    }

    // Calculate average time, bandwidth, and throughput
    avg_time = total_time / (nloop / 2);
    bandwidth = 2 * N * sizeof(float) / avg_time / 1073741824;
    throughput = 2 * N / avg_time;

    printf("N: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n", N, avg_time, bandwidth, throughput);

    return 0;
}