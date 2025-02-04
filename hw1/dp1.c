float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j = 0; j < N; j++) {
        R += pA[j] * pB[j];
    }
    return R;
}