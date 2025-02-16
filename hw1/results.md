## Q1
Explain the rationale and expected consequence of only using the second half of the measurements for the computation of the mean execution time. Moreover, explain what type of mean is appropriate for the calculations, and why. 

**Solution:**

***Rationale***: 
Using only the second half of the measurements to compute the mean execution time primarily aims to capture the system's performance after it has reached a stable state, avoiding initial transient effects such as warming up or setup anomalies. This approach helps to focus on a more consistent and representative dataset, providing a clearer picture of the system's typical operational efficiency and minimizing the impact of any initial outliers or irregularities.

***Expected consequence***:
It is beneficial to reduce the impact of outliers on the mean. Using only the latter half of the measurements likely leads to a mean that better represents the system's performance under a typical load. If initial workload are abnormally high given the warm-up influence, excluding them helps in focusing on typical performance.

**Arithmetic mean** is appropriate for the calculations of mean execution time because each trail with same input size contribute equally to the final result.



## Q2

Draw a roofline model based on a peak performance of 200 GFLOPS and memory bandwidth of 30 GB/s. Add a vertical line for the arithmetic intensity. Plot points for the 10 measurements for the average results for each microbenchmark. The roofline model must be ”plotted” using matplotlib or an equivalent package.

Based on your plotted measurements, explain clearly whether the computations are compute or memory bound, and why. Discuss the underlying reasons for why these computations differ or don’t across each microbenchmark.

Lastly, identify any microbenchmarks that underperform relative to the roofline, and explain the algorithmic bottlenecks responsible for this performance gap.

<img src="img/roofline.png" alt="roofline" width="800"/>

**Solution:** 

The computation is memory-bound, given all benchmark experiments located at the left side of red line. When arithmetic intensity (FLOP/Byte) is low, computations spend more time waiting for data from memory rather than performing actual floating-point operations.

***Reason:*** The reason why the computations are differ across each microbenchmark is that the algorithm and programming language used in each benchmark is different. 

***Underperform benchmarks***: `dp1.c` and `dp5.py`

***Algorithmic bottleneck***: 

- `dp1.c`: The algorithm use the brute-force approach to compute the dot product, which is not efficient for the memory access pattern. Because when call a function for each element in the vector, the memory access pattern is not coalesced, leading to poor memory utilization.
- `dp5.py`: The algorithm use the brute-force approach to compute the dot product, which is not efficient for the memory access pattern. Given the Python's overhead, the performance is even worse.



## Q3

Using the N = 300000000 simple loop as the baseline, explain the the difference in performance for the 5 measurements in the C and Python variants. Explain why this occurs by considering the underlying algorithms used.

Performance metrics of the 10 measurements in the C and Python variants.
<img src="img/metric-panel.png" alt="metric-panel" width="800" style="align: center"/>

**Solution:**
The performance comparison: 
$$
{dp4} < {dp1} < {dp2} < {dp5} < {dp3}
$$

***Explanation***: 

- `dp3.c` is written in C with the usage of `MKL` library, which optimizes parallel processing for linear algebra operations based on the hardware architecture.
- `dp5.py` is written in `numpy` library, which is optimized for vectorized operations and parallel processing.
- `dp1.c` and `dp2.c` are written in C without using any optimization libraries, but `dp2.c` use unrolled loop to improve the performance.
- `dp4.py` is written in Python without using any optimization libraries, so the performance is the worst.



## Q4

Check the result of the dot product computations against the analytically calculated result. Explain your findings, and why the results occur. (Hint: Floating point operations are not exact.)

**Solution:**
***Analytically calculated result***: (All elements in each vector are 1)
$$
A \cdot B = \sum_{i=0}^{N-1} 1 = N
$$

***Execution result***:

- When N = 1000000, all the results are 1000000.
- When N = 300000000, some execution results are not exactly 300000000.

***Explanation***:
When doing element-wise operation on floating point numbers, the result is not exactly the expected result due to the precision limitation of floating point representation.



