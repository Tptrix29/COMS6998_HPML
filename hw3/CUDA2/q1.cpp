#include <iostream>
#include <cstdlib>
#include <chrono>

#define N 1000000 // 1 million elements

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " K" << std::endl;
        std::cerr << "K is the size of arrays in millions" << std::endl;
        return 1;
    }
    
    // Parse K from command-line
    int K = std::atoi(argv[1]);
    size_t size = K * N; // K million elements
    
    std::cout << "Adding arrays with " << K << " million elements each..." << std::endl;
    
    // Allocate memory for arrays
    float* a = (float*)malloc(size * sizeof(float));
    float* b = (float*)malloc(size * sizeof(float));
    float* c = (float*)malloc(size * sizeof(float));
    
    if (!a || !b || !c) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }
    
    // Initialize arrays with some values
    for (size_t i = 0; i < size; ++i) {
        a[i] = 1;
        b[i] = 1;
    }
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Add arrays element-wise
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Print execution time
    std::cout << "Execution time: " << duration << " ms" << std::endl;
    
    // Verify correctness with a few samples
    std::cout << "Sample results: c[0]=" << c[0] << ", c[size/2]=" << c[size/2] 
              << ", c[size-1]=" << c[size-1] << std::endl;
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}