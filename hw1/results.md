## Q1
Explain the rationale and expected consequence of only using the second half of the measurements for the computation of the mean execution time. Moreover, explain what type of mean is appropriate for the calculations, and why. 

**Solution:**
1. Rationale: 
    Using only the second half of the measurements to compute the mean execution time primarily aims to capture the system's performance after it has reached a stable state, avoiding initial transient effects such as warming up or setup anomalies. This approach helps to focus on a more consistent and representative dataset, providing a clearer picture of the system's typical operational efficiency and minimizing the impact of any initial outliers or irregularities.
2. Expected consequence:
    It is beneficial to reduce the impact of outliers on the mean. Using only the latter half of the measurements likely leads to a mean that better represents the system's performance under a typical load. If initial workload are abnormally high given the warm-up influence, excluding them helps in focusing on typical performance.
3. **Arithmetic mean** is appropriate for the calculations because each trail with same input size contribute equally to the final result.


## Q2
Draw a roofline model based on a peak performance of 200 GFLOPS and memory bandwidth of 30 GB/s. Add a vertical line for the arithmetic intensity. Plot points for the 10 measurements for the average results for each microbenchmark. The roofline model must be ”plotted” using matplotlib or an equivalent package.

Based on your plotted measurements, explain clearly whether the computations are compute or memory bound, and why. Discuss the underlying reasons for why these computations differ or don’t across each microbenchmark.

Lastly, identify any microbenchmarks that underperform relative to the roofline, and explain the algorithmic bottlenecks responsible for this performance gap.

**Solution:**

## Q3
Using the N = 300000000 simple loop as the baseline, explain the the difference in performance for the 5 measurements in the C and Python variants. Explain why this occurs by considering the underlying algorithms used.

**Solution:**

## Q4
Check the result of the dot product computations against the analytically calculated result. Explain your findings, and why the results occur. (Hint: Floating point operations are not exact.)

**Solution:**