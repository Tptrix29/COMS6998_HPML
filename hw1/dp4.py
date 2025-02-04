import numpy as np

# for a simple loop
def dp(N,A,B):
    A = np.ones(N,dtype=np.float32)
    B = np.ones(N,dtype=np.float32)
    R = 0.0
    for j in range(0,N):
        R += A[j]*B[j]
    return R