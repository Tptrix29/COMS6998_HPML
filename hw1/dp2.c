#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    return R;
}


int main(int argc, char *argv[]) {
    long N = atol(argv[1]);
    int nloop = atoi(argv[2]);
    float *A = (float *)malloc(N * sizeof(float));
    float *B = (float *)malloc(N * sizeof(float));

    struct timespec start, end;
    double elapsed_time, total_time = 0.0;
    double total_bandwidth = 0.0, total_throughput = 0.0;
    // double total_bandwidth_inv = 0.0, total_throughput_inv = 0.0;
    double avg_time, bandwidth, throughput;
    volatile float result;

    // Initialize A and B
    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }
    
    // Execute dp function
    for (int i = 0; i < nloop; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        result = dpunroll(N, A, B);
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        if (i >= nloop / 2) {
            total_time += elapsed_time;
            total_bandwidth += 2 * N * sizeof(float) / elapsed_time;
            total_throughput += 2 * N / elapsed_time;
            // total_bandwidth_inv += elapsed_time * 1e9 / (2 * N * sizeof(float));
            // total_throughput_inv += elapsed_time / (2 * N);
        }
    }

    // Calculate average time, bandwidth, and throughput
    avg_time = total_time / (nloop / 2);
    bandwidth = total_bandwidth / (nloop / 2) / 1073741824;
    throughput = total_throughput / (nloop / 2);

    // bandwidth = nloop / total_bandwidth_inv;
    // throughput = nloop / total_throughput_inv;

    printf("Result: %.2f\n", result);
    printf("N: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n", N, avg_time, bandwidth, throughput);
    printf("N: %ld <T>: %.2e sec B: %.3e GB/sec F: %.3e FLOP/sec\n", N, avg_time, bandwidth, throughput);

    free(A);
    free(B);

    return 0;
}