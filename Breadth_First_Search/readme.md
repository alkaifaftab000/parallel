# Parallel BFS Traversal Implementation

## Parallel Computing Assignment

This program implements a parallel version of Breadth-First Search (BFS) traversal using OpenMP. The implementation compares sequential and parallel approaches to graph traversal, measuring performance differences and verifying correctness.

## Source Code

```cpp
// Parallel BFS traversal
vector<int> bfs_par(const Graph& graph, int start) {
    vector<bool> visited(graph.V, false);
    vector<int> traversal_order;
    
    // Mark the source vertex as visited
    visited[start] = true;
    
    // Use a queue to keep track of frontier vertices
    vector<int> current_frontier = {start};
    
    // Process frontier levels in parallel
    while (!current_frontier.empty()) {
        // Add current frontier to traversal order
        traversal_order.insert(traversal_order.end(), current_frontier.begin(), current_frontier.end());
        
        // Create new frontier
        vector<int> next_frontier;
        
        // Process current frontier in parallel to discover next frontier
        #pragma omp parallel
        {
            // Local frontier to avoid contention
            vector<int> local_frontier;
            
            // Process current frontier vertices in parallel
            #pragma omp for nowait
            for (int u : current_frontier) {
                // Examine all neighbors of u
                for (int v : graph.adj[u]) {
                    bool already_visited = false;
                    
                    // Atomic check and update of visited status
                    #pragma omp critical
                    {
                        if (!visited[v]) {
                            visited[v] = true;
                            already_visited = false;
                        } else {
                            already_visited = true;
                        }
                    }
                    
                    // If newly visited, add to local frontier
                    if (!already_visited) {
                        local_frontier.push_back(v);
                    }
                }
            }
            
            // Merge local frontier into global next frontier
            #pragma omp critical
            {
                next_frontier.insert(next_frontier.end(), local_frontier.begin(), local_frontier.end());
            }
        }
        
        // Update current frontier for next iteration
        current_frontier = next_frontier;
    }
    
    return traversal_order;
}
```

## Sample Output

```
PARALLEL BFS TRAVERSAL
=====================

Enter number of vertices: 10000
Enter edge density (edges per vertex): 5
Enter start vertex (0 to 9999): 0
Enter number of threads to use (0 for auto): 4

Generating random graph with 10000 vertices and ~5 edges per vertex...

Sequential BFS traversal...
Time: 38 ms
Nodes visited: 10000

Parallel BFS traversal (4 threads)...
Time: 356 ms
Nodes visited: 10000

Verified: Yes

Performance comparison:
Speedup: 0.11x

Sample of sequential BFS traversal from vertex 0:
0 6281 1663 8111 5765 5307 9143 7211 2325 9561 ... and 9990 more

Sample of parallel BFS traversal from vertex 0:
0 6281 1663 8111 5765 5307 9143 7211 2325 9561 ... and 9990 more
```