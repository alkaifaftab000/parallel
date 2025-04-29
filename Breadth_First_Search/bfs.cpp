#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <chrono>
#include <random>
#include <iomanip>
#include <atomic>

using namespace std;
using namespace std::chrono;

// Structure to represent a graph
class Graph {
public:
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

    Graph(int vertices) : V(vertices) {
        adj.resize(vertices);
    }

    // Add edge to graph
    void addEdge(int src, int dest) {
        adj[src].push_back(dest);
    }
};

// Generate random graph
Graph generate_graph(int vertices, int edge_density) {
    Graph graph(vertices);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dest_dist(0, vertices - 1);
    
    // Each vertex will have approximately edge_density outgoing edges
    #pragma omp parallel for
    for (int i = 0; i < vertices; ++i) {
        for (int j = 0; j < edge_density; ++j) {
            int dest = dest_dist(gen);
            
            // Avoid self-loops
            if (dest == i) {
                if (dest < vertices - 1) dest++;
                else dest--;
            }
            
            #pragma omp critical
            {
                graph.addEdge(i, dest);
            }
        }
    }
    
    return graph;
}

// Sequential BFS traversal
vector<int> bfs_seq(const Graph& graph, int start) {
    vector<bool> visited(graph.V, false);
    vector<int> traversal_order;
    queue<int> q;
    
    // Mark the source vertex as visited and enqueue it
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        // Dequeue a vertex from queue
        int u = q.front();
        q.pop();
        traversal_order.push_back(u);
        
        // Get all adjacent vertices of the dequeued vertex
        // If an adjacent vertex has not been visited, mark it visited and enqueue it
        for (int v : graph.adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    
    return traversal_order;
}

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

// Verify BFS results (Check if all nodes are visited in both traversals)
bool verify_results(const vector<int>& seq_result, const vector<int>& par_result, int vertex_count) {
    if (seq_result.size() != par_result.size()) return false;
    
    // Check if both traversals visit the same nodes (order may differ)
    vector<bool> seq_visited(vertex_count, false);
    vector<bool> par_visited(vertex_count, false);
    
    for (int v : seq_result) seq_visited[v] = true;
    for (int v : par_result) par_visited[v] = true;
    
    for (int i = 0; i < vertex_count; ++i) {
        if (seq_visited[i] != par_visited[i]) return false;
    }
    
    return true;
}

int main() {
    // User configuration
    int vertices;
    int edge_density;
    int start_vertex;
    int num_threads;

    cout << "PARALLEL BFS TRAVERSAL\n";
    cout << "=====================\n\n";
    
    // Get user input
    cout << "Enter number of vertices: ";
    cin >> vertices;
    cout << "Enter edge density (edges per vertex): ";
    cin >> edge_density;
    cout << "Enter start vertex (0 to " << vertices - 1 << "): ";
    cin >> start_vertex;
    cout << "Enter number of threads to use (0 for auto): ";
    cin >> num_threads;

    // Validate input
    if (vertices <= 0) {
        cerr << "Error: Number of vertices must be positive!\n";
        return 1;
    }
    if (edge_density <= 0) {
        cerr << "Error: Edge density must be positive!\n";
        return 1;
    }
    if (start_vertex < 0 || start_vertex >= vertices) {
        cerr << "Error: Start vertex must be between 0 and " << vertices - 1 << "!\n";
        return 1;
    }

    // Set threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    // Generate graph
    cout << "\nGenerating random graph with " << vertices << " vertices and ~" 
         << edge_density << " edges per vertex...\n";
    auto graph = generate_graph(vertices, edge_density);
    vector<int> seq_result, par_result;

    // Sequential BFS
    cout << "\nSequential BFS traversal...\n";
    auto start_seq = high_resolution_clock::now();
    seq_result = bfs_seq(graph, start_vertex);
    auto end_seq = high_resolution_clock::now();
    auto seq_time = duration_cast<milliseconds>(end_seq - start_seq).count();
    cout << "Time: " << seq_time << " ms\n";
    cout << "Nodes visited: " << seq_result.size() << "\n";

    // Parallel BFS
    cout << "\nParallel BFS traversal (" << num_threads << " threads)...\n";
    auto start_par = high_resolution_clock::now();
    par_result = bfs_par(graph, start_vertex);
    auto end_par = high_resolution_clock::now();
    auto par_time = duration_cast<milliseconds>(end_par - start_par).count();
    cout << "Time: " << par_time << " ms\n";
    cout << "Nodes visited: " << par_result.size() << "\n";

    // Verify results
    cout << "\nVerified: " << (verify_results(seq_result, par_result, vertices) ? "Yes" : "No") << "\n";

    // Performance comparison
    cout << "\nPerformance comparison:\n";
    if (par_time > 0) {
        cout << "Speedup: " << fixed << setprecision(2) 
             << (double)seq_time/par_time << "x\n";
    } else {
        cout << "Speedup: Too fast to measure (parallel time < 1ms)\n";
    }

    // Print sample of BFS traversal
    int display_count = min(10, (int)seq_result.size());
    cout << "\nSample of sequential BFS traversal from vertex " << start_vertex << ":\n";
    for (int i = 0; i < display_count; ++i) {
        cout << seq_result[i] << " ";
    }
    if (seq_result.size() > display_count) cout << "... and " << seq_result.size() - display_count << " more\n";
    else cout << "\n";

    cout << "\nSample of parallel BFS traversal from vertex " << start_vertex << ":\n";
    for (int i = 0; i < display_count; ++i) {
        cout << par_result[i] << " ";
    }
    if (par_result.size() > display_count) cout << "... and " << par_result.size() - display_count << " more\n";
    else cout << "\n";

    return 0;
}