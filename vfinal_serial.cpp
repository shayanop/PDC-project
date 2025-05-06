#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <limits>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono> // For performance analysis

using namespace std;
using namespace std::chrono;

// Structure to represent an edge in the graph
struct Edge {
    int src, dest;
    int weight; // Weight of the edge

    Edge(int s, int d, int w) : src(s), dest(d), weight(w) {}
};

// Structure to represent a change in the dynamic graph
struct Change {
    int src, dest;
    bool isInsert; // true for insertion, false for deletion
    int weight;    // Weight of the edge (used for insertion)

    Change(int s, int d, bool ins, int w = 1) 
        : src(s), dest(d), isInsert(ins), weight(w) {}
};

// Class for the dynamic SSSP algorithm
class DynamicSSSP {
private:
    int numVertices;
    vector<vector<pair<int, int>>> graph; // Adjacency list: (neighbor, weight)
    vector<int> distance; // Distance from source
    vector<int> parent;   // Parent in SSSP tree
    vector<bool> affected; // Vertices affected by changes
    vector<bool> affectedDel; // Vertices affected by deletion

public:
    DynamicSSSP(int n) {
        numVertices = n;
        graph.resize(n);
        distance.resize(n, numeric_limits<int>::max());
        parent.resize(n, -1);
        affected.resize(n, false);
        affectedDel.resize(n, false);
    }

    // Add an edge to the graph
    void addEdge(int src, int dest, int weight) {
        // For undirected graph, add edges in both directions
        graph[src].push_back({dest, weight});
        graph[dest].push_back({src, weight});
    }

    // Remove an edge from the graph
    void removeEdge(int src, int dest) {
        // Remove edge from src to dest
        auto& edges = graph[src];
        for (auto it = edges.begin(); it != edges.end(); ) {
            if (it->first == dest) {
                it = edges.erase(it);
            } else {
                ++it;
            }
        }

        // Remove edge from dest to src (for undirected graph)
        auto& rev_edges = graph[dest];
        for (auto it = rev_edges.begin(); it != rev_edges.end(); ) {
            if (it->first == src) {
                it = rev_edges.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Compute initial SSSP using Dijkstra's algorithm
    void computeInitialSSSP(int source) {
        // Reset distances and parents
        fill(distance.begin(), distance.end(), numeric_limits<int>::max());
        fill(parent.begin(), parent.end(), -1);
        
        distance[source] = 0;
        
        // Priority queue for Dijkstra's algorithm
        // Pair of (distance, vertex)
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, source});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            int dist_u = pq.top().first;
            pq.pop();
            
            // Skip if we've already found a better path
            if (dist_u > distance[u])
                continue;
            
            // Check all neighbors of u
            for (auto& edge : graph[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                // If we found a shorter path to v through u
                if (distance[u] != numeric_limits<int>::max() && 
                    distance[u] + weight < distance[v]) {
                    
                    distance[v] = distance[u] + weight;
                    parent[v] = u;
                    pq.push({distance[v], v});
                }
            }
        }
    }

    // Print the SSSP tree
    void printSSSP() {
        cout << "Vertex \t Distance from Source \t Parent" << endl;
        for (int i = 0; i < numVertices; i++) {
            cout << i << " \t ";
            if (distance[i] == numeric_limits<int>::max())
                cout << "INF";
            else
                cout << distance[i];
            cout << " \t\t " << parent[i] << endl;
        }
    }

    // Process a batch of changes
    void processChanges(const vector<Change>& changes) {
        // Process all deletions first
        for (const auto& change : changes) {
            if (!change.isInsert) {
                removeEdge(change.src, change.dest);
            }
        }

        // Process all insertions next
        for (const auto& change : changes) {
            if (change.isInsert) {
                addEdge(change.src, change.dest, change.weight);
            }
        }
    }
};

// Main function
int main() {
    // Read the dataset
    ifstream file("dataset.txt");
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return 1;
    }

    // First, find the maximum vertex ID to determine graph size
    int src, dest, weight, maxVertex = 0;
    string line;

    // Skip the header lines (lines starting with '#')
    while (getline(file, line)) {
        if (line[0] != '#') break;
    }

    // Parse the dataset
    vector<Edge> edges;
    do {
        stringstream ss(line);
        ss >> src >> dest >> weight;
        maxVertex = max(maxVertex, max(src, dest));
        edges.push_back(Edge(src, dest, weight));
    } while (getline(file, line));
    file.close();

    // Create the graph with size maxVertex + 1 (0-indexed)
    DynamicSSSP sssp(maxVertex + 1);

    // Add edges to the graph
    for (const auto& edge : edges) {
        sssp.addEdge(edge.src, edge.dest, edge.weight);
    }

    // Compute initial SSSP from source vertex 0
    const int SOURCE_VERTEX = 0;

    auto start = high_resolution_clock::now();
    sssp.computeInitialSSSP(SOURCE_VERTEX);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Initial SSSP Tree:" << endl;
    sssp.printSSSP();
    cout << "Time taken for initial SSSP computation: " << duration.count() << " ms" << endl;

    // Simulate changes
    vector<Change> changes = {
        Change(0, 10, true, 3),  // Insert edge (0, 10) with weight 3
        Change(2, 5, false),     // Delete edge (2, 5)
        Change(3, 6, true, 2),   // Insert edge (3, 6) with weight 2
        Change(7, 9, true, 4)    // Insert edge (7, 9) with weight 4
    };

    start = high_resolution_clock::now();
    sssp.processChanges(changes);
    sssp.computeInitialSSSP(SOURCE_VERTEX);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    cout << "\nSSSP Tree after changes:" << endl;
    sssp.printSSSP();
    cout << "Time taken for processing changes and recomputing SSSP: " << duration.count() << " ms" << endl;

    return 0;
}