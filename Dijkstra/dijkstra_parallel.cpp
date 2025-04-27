#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Generate random data
vector<int> generate_data(size_t size, int min_val, int max_val) {
    vector<int> data(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(min_val, max_val);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// Sequential histogram sort
vector<int> histogram_sort_seq(const vector<int>& input, int min_val, int max_val) {
    const int range = max_val - min_val + 1;
    vector<int> histogram(range, 0);
    vector<int> output(input.size());

    // Build histogram
    for (int num : input) {
        histogram[num - min_val]++;
    }

    // Calculate cumulative sum
    for (int i = 1; i < range; ++i) {
        histogram[i] += histogram[i - 1];
    }

    // Place elements in sorted order
    for (int i = input.size() - 1; i >= 0; --i) {
        output[--histogram[input[i] - min_val]] = input[i];
    }

    return output;
}

// Parallel histogram sort
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

// Verify sorting
bool is_sorted(const vector<int>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i - 1] > data[i]) return false;
    }
    return true;
}

int main() {
    // User configuration
    size_t data_size;
    int num_threads;
    const int min_val = 0;
    const int max_val = 1000;

    cout << "PARALLEL HISTOGRAM SORT\n";
    cout << "=======================\n\n";
    
    // Get user input
    cout << "Enter number of elements to sort: ";
    cin >> data_size;
    cout << "Enter number of threads to use (0 for auto): ";
    cin >> num_threads;

    // Validate input
    if (data_size <= 0) {
        cerr << "Error: Number of elements must be positive!\n";
        return 1;
    }

    // Set threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    // Generate data
    cout << "\nGenerating " << data_size << " random numbers (" 
         << min_val << " to " << max_val << ")...\n";
    auto data = generate_data(data_size, min_val, max_val);
    vector<int> seq_result, par_result;

    // Sequential sort
    cout << "\nSequential histogram sort...\n";
    auto start_seq = high_resolution_clock::now();
    seq_result = histogram_sort_seq(data, min_val, max_val);
    auto end_seq = high_resolution_clock::now();
    auto seq_time = duration_cast<milliseconds>(end_seq - start_seq).count();
    cout << "Time: " << seq_time << " ms\n";
    cout << "Verified: " << (is_sorted(seq_result) ? "Yes" : "No") << "\n";

    // Parallel sort
    cout << "\nParallel histogram sort (" << num_threads << " threads)...\n";
    auto start_par = high_resolution_clock::now();
    par_result = histogram_sort_par(data, min_val, max_val);
    auto end_par = high_resolution_clock::now();
    auto par_time = duration_cast<milliseconds>(end_par - start_par).count();
    cout << "Time: " << par_time << " ms\n";
    cout << "Verified: " << (is_sorted(par_result) ? "Yes" : "No") << "\n";

    // Performance comparison
    cout << "\nPerformance comparison:\n";
    if (par_time > 0) {
        cout << "Speedup: " << fixed << setprecision(2) 
             << (double)seq_time/par_time << "x\n";
    } else {
        cout << "Speedup: Too fast to measure (parallel time < 1ms)\n";
    }

    return 0;
}