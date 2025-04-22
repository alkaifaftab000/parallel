# Histogram Sorting using OpenMP
## Parallel Computing Assignment

### 1. Program Description
This program implements both sequential and parallel histogram sorting using OpenMP. The algorithm builds a histogram of input values, calculates prefix sums, and uses them for sorting. It compares the performance between sequential and parallel approaches, demonstrating the impact of parallelization on sorting operations.

### 2. Source Code
```cpp
// Parallel histogram sort implementation using OpenMP
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
- **Sequential Sort**: Traditional histogram-based sorting approach
- **Parallel Sort**: OpenMP implementation with three main phases:
  1. Parallel histogram construction with atomic updates
  2. Sequential prefix sum calculation
  3. Parallel element placement with atomic capture
- **Performance Metrics**: Measures execution time and calculates speedup
- **Verification**: Ensures sorted output correctness

### 4. Sample Output
```
PARALLEL HISTOGRAM SORT
=======================

Enter number of elements to sort: 100000000
Enter number of threads to use (0 for auto): 4

Generating 100000000 random numbers (0 to 1000)...

Sequential histogram sort...
Time: 1783 ms
Verified: Yes

Parallel histogram sort (4 threads)...
Time: 3501 ms
Verified: Yes

Performance comparison:
Speedup: 0.51x
```

### 5. Performance Analysis
- Parallel implementation shows interesting performance characteristics:
  - Atomic operations ensure thread safety but add overhead
  - Sequential prefix sum creates a bottleneck
  - Large data sets demonstrate the impact of parallel processing
- The current implementation shows lower performance than sequential version due to:
  - Atomic operation overhead
  - Memory access patterns
  - Data size and distribution effects

### 6. Conclusions
While the parallel implementation successfully maintains sorting accuracy, the current version shows that not all algorithms benefit directly from parallelization. The atomic operations and memory access patterns create overhead that outweighs the benefits of parallel processing in this specific implementation.

---
*Date: April 22, 2025*