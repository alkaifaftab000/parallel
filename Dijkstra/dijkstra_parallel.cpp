#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

// Structure to represent a graph edge
struct Edge {
    int dest;
    int weight;
};

// Structure to represent a graph
class Graph {
public:
    int V; // Number of vertices
    vector<vector<Edge>> adj; // Adjacency list

    Graph(int vertices) : V(vertices) {
        adj.resize(vertices);
    }

    // Add edge to graph
    void addEdge(int src, int dest, int weight) {
        Edge edge = {dest, weight};
        adj[src].push_back(edge);
    }
};

// Generate random graph
Graph generate_graph(int vertices, int edge_density, int min_weight, int max_weight) {
    Graph graph(vertices);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dest_dist(0, vertices - 1);
    uniform_int_distribution<> weight_dist(min_weight, max_weight);
    
    // Each vertex will have approximately edge_density outgoing edges
    #pragma omp parallel for
    for (int i = 0; i < vertices; ++i) {
        int edges_to_create = edge_density;
        
        for (int j = 0; j < edges_to_create; ++j) {
            int dest = dest_dist(gen);
            
            // Avoid self-loops
            if (dest == i) {
                if (dest < vertices - 1) dest++;
                else dest--;
            }
            
            int weight = weight_dist(gen);
            
            #pragma omp critical
            {
                graph.addEdge(i, dest, weight);
            }
        }
    }
    
    return graph;
}

// Sequential Dijkstra's algorithm
vector<int> dijkstra_seq(const Graph& graph, int src) {
    vector<int> dist(graph.V, numeric_limits<int>::max());
    dist[src] = 0;
    
    // Priority queue: pair<distance, vertex>
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, src});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        
        // If distance in queue is greater than known distance, skip
        if (d > dist[u]) continue;
        
        // Check all neighbors of u
        for (const Edge& edge : graph.adj[u]) {
            int v = edge.dest;
            int weight = edge.weight;
            
            // Relaxation step
            if (dist[u] != numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}

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

// Verify results
bool verify_results(const vector<int>& seq_result, const vector<int>& par_result) {
    if (seq_result.size() != par_result.size()) return false;
    
    for (size_t i = 0; i < seq_result.size(); ++i) {
        if (seq_result[i] != par_result[i]) return false;
    }
    
    return true;
}

int main() {
    // User configuration
    int vertices;
    int edge_density;
    int src_vertex;
    int num_threads;
    const int min_weight = 1;
    const int max_weight = 100;

    cout << "PARALLEL DIJKSTRA'S ALGORITHM\n";
    cout << "=============================\n\n";
    
    // Get user input
    cout << "Enter number of vertices: ";
    cin >> vertices;
    cout << "Enter edge density (edges per vertex): ";
    cin >> edge_density;
    cout << "Enter source vertex (0 to " << vertices - 1 << "): ";
    cin >> src_vertex;
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
    if (src_vertex < 0 || src_vertex >= vertices) {
        cerr << "Error: Source vertex must be between 0 and " << vertices - 1 << "!\n";
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
    auto graph = generate_graph(vertices, edge_density, min_weight, max_weight);
    vector<int> seq_result, par_result;

    // Sequential Dijkstra
    cout << "\nSequential Dijkstra's algorithm...\n";
    auto start_seq = high_resolution_clock::now();
    seq_result = dijkstra_seq(graph, src_vertex);
    auto end_seq = high_resolution_clock::now();
    auto seq_time = duration_cast<milliseconds>(end_seq - start_seq).count();
    cout << "Time: " << seq_time << " ms\n";

    // Parallel Dijkstra
    cout << "\nParallel Dijkstra's algorithm (" << num_threads << " threads)...\n";
    auto start_par = high_resolution_clock::now();
    par_result = dijkstra_par(graph, src_vertex);
    auto end_par = high_resolution_clock::now();
    auto par_time = duration_cast<milliseconds>(end_par - start_par).count();
    cout << "Time: " << par_time << " ms\n";

    // Verify results
    cout << "\nVerified: " << (verify_results(seq_result, par_result) ? "Yes" : "No") << "\n";

    // Performance comparison
    cout << "\nPerformance comparison:\n";
    if (par_time > 0) {
        cout << "Speedup: " << fixed << setprecision(2) 
             << (double)seq_time/par_time << "x\n";
    } else {
        cout << "Speedup: Too fast to measure (parallel time < 1ms)\n";
    }

    // Print sample of shortest paths
    int display_count = min(10, vertices);
    cout << "\nSample shortest path distances from vertex " << src_vertex << ":\n";
    cout << "Vertex\tDistance\n";
    cout << "----------------\n";
    for (int i = 0; i < display_count; ++i) {
        cout << i << "\t";
        if (seq_result[i] == numeric_limits<int>::max())
            cout << "INF";
        else
            cout << seq_result[i];
        cout << "\n";
    }
    if (vertices > display_count) cout << "... and " << vertices - display_count << " more\n";

    return 0;
}