#include <iostream>
#include <omp.h>

using namespace std;

// Example 1: Basic parallel region
void example1() {
    cout << "\n--- Example 1: Basic parallel region ---\n";
    
    // Set number of threads to 2 for this example
    omp_set_num_threads(2);
    
    #pragma omp parallel
    {
        cout << "A ";
        cout << "race ";
        cout << "car ";
    } // End of parallel region
    
    cout << "\n";
}

// Example 2: Single directive
void example2() {
    cout << "\n--- Example 2: Single directive ---\n";
    
    // Set number of threads to 2 for this example
    omp_set_num_threads(2);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "A ";
            cout << "race ";
            cout << "car ";
        }
    } // End of parallel region
    
    cout << "\n";
}

// Example 3: Tasks without taskwait
void example3() {
    cout << "\n--- Example 3: Tasks without taskwait ---\n";
    
    // Set number of threads to 2 for this example
    omp_set_num_threads(2);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "A ";
            #pragma omp task
            {cout << "race ";}
            #pragma omp task
            {cout << "car ";}
            cout << "is fun to watch ";
        }
    } // End of parallel region
    
    cout << "\n";
}

// Example 4: Tasks with taskwait
void example4() {
    cout << "\n--- Example 4: Tasks with taskwait ---\n";
    
    // Set number of threads to 2 for this example
    omp_set_num_threads(2);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "A ";
            #pragma omp task
            {cout << "car ";}
            #pragma omp task
            {cout << "race ";}
            #pragma omp taskwait
            cout << "is fun to watch ";
        }
    } // End of parallel region
    
    cout << "\n";
}

int main() {
    cout << "Running OpenMP examples with 2 threads\n";
    
    // Display current OpenMP version
    #ifdef _OPENMP
        cout << "OpenMP Version: " << _OPENMP << "\n";
    #else
        cout << "OpenMP not supported\n";
    #endif
    
    // Run all examples
    example1();
    example2();
    example3();
    example4();
    
    return 0;
}