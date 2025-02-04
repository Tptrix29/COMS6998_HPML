float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    return R;
}