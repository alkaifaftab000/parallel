# Parallel Dijkstra's Algorithm Implementation

## Parallel Computing Assignment

This program implements a parallel version of Dijkstra's shortest path algorithm using OpenMP. The implementation compares sequential and parallel approaches to finding shortest paths in a graph, measuring performance differences and verifying correctness.

## Source Code

```cpp
// Parallel Dijkstra's algorithm
vector<int> dijkstra_par(const Graph& graph, int src) {
    vector<int> dist(graph.V, numeric_limits<int>::max());
    dist[src] = 0;
    
    // Array to track processed vertices
    vector<bool> processed(graph.V, false);
    
    // Main loop for Dijkstra
    for (int count = 0; count < graph.V - 1; ++count) {
        // Find vertex with minimum distance
        int u = -1;
        int min_dist = numeric_limits<int>::max();
        
        #pragma omp parallel
        {
            int local_u = -1;
            int local_min = numeric_limits<int>::max();
            
            // Each thread finds its local minimum
            #pragma omp for nowait
            for (int v = 0; v < graph.V; ++v) {
                if (!processed[v] && dist[v] < local_min) {
                    local_min = dist[v];
                    local_u = v;
                }
            }
            
            // Update global minimum
            #pragma omp critical
            {
                if (local_u != -1 && local_min < min_dist) {
                    min_dist = local_min;
                    u = local_u;
                }
            }
        }
        
        if (u == -1) continue; // No reachable vertex found
        
        processed[u] = true;
        
        // Parallel relaxation step
        #pragma omp parallel for
        for (int i = 0; i < graph.adj[u].size(); ++i) {
            int v = graph.adj[u][i].dest;
            int weight = graph.adj[u][i].weight;
            
            #pragma omp critical
            {
                if (!processed[v] && dist[u] != numeric_limits<int>::max() && 
                    dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                }
            }
        }
    }
    
    return dist;
}
```

## Sample Output

```
PARALLEL DIJKSTRA'S ALGORITHM
=============================
Enter number of vertices: 1000
Enter edge density (edges per vertex): 10
Enter source vertex (0 to 999): 9
Enter number of threads to use (0 for auto): 4

Generating random graph with 1000 vertices and ~10 edges per vertex...

Sequential Dijkstra's algorithm...
Time: 5 ms

Parallel Dijkstra's algorithm (4 threads)...
Time: 130 ms

Verified: Yes

Performance comparison:
Speedup: 0.04x

Sample shortest path distances from vertex 9:
Vertex  Distance
----------------
0       69
1       84
2       117
3       98
4       77
5       71
6       81
7       84
8       72
9       0
... and 990 more
```
