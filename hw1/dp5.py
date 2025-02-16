import time
import argparse
import numpy as np

# for a simple loop
def dp(N, A, B):
    return np.dot(A, B)

def argparser():
    parser = argparse.ArgumentParser(description='Vector Dot Product')
    parser.add_argument('-N', type=int, default=1000, help='Size of the vector')
    parser.add_argument('-R', type=int, default=1, help='Number of repetitions')
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    N = args.N
    R = args.R
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    total_time = 0.0
    total_bandwidth, total_throughput = 0.0, 0.0
    for i in range(R):
        start = time.time()
        result = dp(N, A, B)    
        end = time.time()
        if i >= R/2:
            total_time += end - start
            total_bandwidth += 2 * N * 4 / (end - start)
            total_throughput += 2 * N / (end - start)
    avg_time = total_time / (R/2)
    bandwidth = total_bandwidth / (R/2) / 1073741824
    throughput = total_throughput / (R/2)

    print(f"N: {N} <T>: {avg_time: .6f} sec B: {bandwidth: .3f} GB/sec F: {throughput: .3f} FLOP/sec")
    print(f"R: {R} <T>: {avg_time: .2e} sec B: {bandwidth: .3e} GB/sec F: {throughput: .3e} FLOP/sec")