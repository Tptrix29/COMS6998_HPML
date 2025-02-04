import numpy as np

def dp(N,A,B):
    A = np.ones(N,dtype=np.float32)
    B = np.ones(N,dtype=np.float32)
    return np.dot(A, B)