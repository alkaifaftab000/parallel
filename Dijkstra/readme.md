# Parallel Histogram Sort Implementation
## Parallel Computing Assignment

### 1. Program Description
This program implements a parallel version of histogram sorting using OpenMP. The implementation compares sequential and parallel approaches to histogram-based sorting, measuring performance differences and verifying correctness.

### 2. Source Code
```cpp
vector<int> histogram_sort_par(const vector<int>& input, int min_val, int max_val) {
    const int range = max_val - min_val + 1;
    vector<int> histogram(range, 0);
    vector<int> output(input.size());

    // Parallel histogram construction
    #pragma omp parallel for
    for (int i = 0; i < input.size(); ++i) {
        #pragma omp atomic
        histogram[input[i] - min_val]++;
    }

    // Sequential prefix sum
    for (int i = 1; i < range; ++i) {
        histogram[i] += histogram[i - 1];
    }

    // Parallel placement
    #pragma omp parallel for
    for (int i = input.size() - 1; i >= 0; --i) {
        int pos;
        #pragma omp atomic capture
        { pos = histogram[input[i] - min_val]; histogram[input[i] - min_val]--; }
        output[pos - 1] = input[i];
    }

    return output;
}
```

### 3. Implementation Details
- **Algorithm Structure**: Three-phase histogram-based sorting algorithm
- **Parallelization Strategy**:
  1. Parallel histogram construction with atomic updates
  2. Sequential prefix sum calculation 
  3. Parallel element placement with atomic capture operations
- **Thread Management**: Uses OpenMP for parallel region control
- **Synchronization**: Employs atomic operations to prevent race conditions

### 4. Sample Output
```
PARALLEL HISTOGRAM SORT
=======================
Enter number of elements to sort: 10000000
Enter number of threads to use (0 for auto): 4
Generating 10000000 random numbers (0 to 1000)...

Sequential histogram sort...
Time: 194 ms
Verified: Yes

Parallel histogram sort (4 threads)...
Time: 327 ms
Verified: Yes

Performance comparison:
Speedup: 0.59x
```

### 5. Performance Analysis
- The parallel implementation operates correctly but shows lower performance than sequential
- Possible performance bottlenecks:
  - Atomic operation overhead exceeds parallelization benefits
  - Sequential prefix sum calculation creates a synchronization point
  - Memory access patterns may cause cache contention
- Performance could potentially be improved by:
  - Implementing parallel prefix sum calculation
  - Optimizing memory access patterns
  - Tuning thread count based on problem size and hardware

### 6. Conclusions
This implementation demonstrates the challenges of effective parallelization. Not all algorithms benefit equally from parallel execution, especially when synchronization overhead outweighs computational parallelism. The results show that careful algorithm design and performance analysis are essential when developing parallel solutions.

---
*Submitted by: Alkaif Ansari*  
*Date: April 25, 2025*