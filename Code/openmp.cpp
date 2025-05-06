#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <limits>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <omp.h> 
#include <sstream>

using namespace std;

// Structure to represent an edge in the graph
struct Edge {
    int src, dest;
    long long timestamp;
    int weight;

    Edge(int s, int d, long long t, int w) : src(s), dest(d), timestamp(t), weight(w) {}
};

// Structure to represent a change in the dynamic graph
struct Change {
    int src, dest;
    long long timestamp;
    bool isInsert; // true for insertion, false for deletion
    int weight;

    Change(int s, int d, long long t, bool ins, int w) 
        : src(s), dest(d), timestamp(t), isInsert(ins), weight(w) {}
};

// Class for the dynamic SSSP algorithm
class ParallelDynamicSSSP {
private:
    int numVertices;
    vector<vector<pair<int, int>>> graph; // Adjacency list: (neighbor, weight)
    vector<int> distance; // Distance from source
    vector<int> parent;   // Parent in SSSP tree
    vector<bool> affected; // Vertices affected by changes
    vector<bool> affectedDel; // Vertices affected by deletion
    int numThreads; // Number of threads for OpenMP
    
    // Store child lists for faster access during deletion
    vector<vector<int>> children;
    
    // Batch size for processing changes
    int batchSize;

public:
    ParallelDynamicSSSP(int n, int threads = 4, int batch = 10000) {
        numVertices = n;
        graph.resize(n);
        distance.resize(n, numeric_limits<int>::max());
        parent.resize(n, -1);
        affected.resize(n, false);
        affectedDel.resize(n, false);
        children.resize(n);
        numThreads = threads;
        batchSize = batch;
        
        // Set the number of threads for OpenMP
        omp_set_num_threads(numThreads);
    }

    // Add an edge to the graph
    void addEdge(int src, int dest, int weight) {
        if(src >= numVertices || dest >= numVertices || src < 0 || dest < 0)
            return;
            
        // For undirected graph, add edges in both directions
        graph[src].push_back({dest, weight});
        graph[dest].push_back({src, weight});
    }

    // Remove an edge from the graph
    void removeEdge(int src, int dest) {
        if(src >= numVertices || dest >= numVertices || src < 0 || dest < 0)
            return;
            
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

    // Compute initial SSSP using parallel Dijkstra's algorithm
    void computeInitialSSSP(int source) {
        // Reset distances, parents, and children
        #pragma omp parallel for
        for (int i = 0; i < numVertices; i++) {
            distance[i] = numeric_limits<int>::max();
            parent[i] = -1;
            children[i].clear();
        }
        
        distance[source] = 0;
        
        // Priority queue for Dijkstra's algorithm
        // We need a sequential priority queue as it's not thread-safe
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, source});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            int dist_u = pq.top().first;
            pq.pop();
            
            // Skip if we've already found a better path
            if (dist_u > distance[u])
                continue;
            
            // Process neighbors in parallel
            vector<tuple<int, int, int>> updates; // (vertex, parent, distance)
            
            #pragma omp parallel
            {
                vector<tuple<int, int, int>> local_updates;
                
                #pragma omp for nowait
                for (int i = 0; i < graph[u].size(); i++) {
                    int v = graph[u][i].first;
                    int weight = graph[u][i].second;
                    
                    // If we found a shorter path to v through u
                    if (distance[u] != numeric_limits<int>::max() && 
                        distance[u] + weight < distance[v]) {
                        
                        local_updates.push_back({v, u, distance[u] + weight});
                    }
                }
                
                #pragma omp critical
                {
                    updates.insert(updates.end(), local_updates.begin(), local_updates.end());
                }
            }
            
            // Apply updates and add to queue
            for (auto& update : updates) {
                int v = get<0>(update);
                int p = get<1>(update);
                int d = get<2>(update);
                
                if (d < distance[v]) {
                    // Remove from old parent's children if it exists
                    if (parent[v] != -1) {
                        auto& oldChildren = children[parent[v]];
                        oldChildren.erase(remove(oldChildren.begin(), oldChildren.end(), v), oldChildren.end());
                    }
                    
                    distance[v] = d;
                    parent[v] = p;
                    
                    // Add to new parent's children
                    children[p].push_back(v);
                    
                    pq.push({d, v});
                }
            }
        }
    }

    // Process edge changes in parallel with batching
    void processChangesInParallel(const vector<Change>& changes, int source) {
        // Reset affected flags
        #pragma omp parallel for
        for (int i = 0; i < numVertices; i++) {
            affected[i] = false;
            affectedDel[i] = false;
        }
        
        // Process changes in batches
        for (int batchStart = 0; batchStart < changes.size(); batchStart += batchSize) {
            int batchEnd = min(batchStart + batchSize, (int)changes.size());
            
            // Process all deletions first in parallel for this batch
            #pragma omp parallel for
            for (int i = batchStart; i < batchEnd; i++) {
                if (!changes[i].isInsert) {
                    #pragma omp critical
                    {
                        processEdgeDeletion(changes[i].src, changes[i].dest);
                    }
                }
            }
            
            // Then process all insertions in parallel for this batch
            #pragma omp parallel for
            for (int i = batchStart; i < batchEnd; i++) {
                if (changes[i].isInsert) {
                    #pragma omp critical
                    {
                        processEdgeInsertion(changes[i].src, changes[i].dest, changes[i].weight);
                    }
                }
            }
            
            // Update affected vertices for this batch
            updateAffectedVerticesParallel();
            
            // Reset affected flags for next batch
            #pragma omp parallel for
            for (int i = 0; i < numVertices; i++) {
                affected[i] = false;
                affectedDel[i] = false;
            }
        }
    }

    // Process a single edge deletion (this is called within a critical section)
    void processEdgeDeletion(int u, int v) {
        if(u >= numVertices || v >= numVertices || u < 0 || v < 0)
            return;
            
        removeEdge(u, v);
        
        // Check if the deleted edge was part of the SSSP tree
        if (parent[u] == v || parent[v] == u) {
            // Determine which vertex is affected
            int affected_vertex = (parent[u] == v) ? u : v;
            
            // Mark the vertex as affected by deletion
            distance[affected_vertex] = numeric_limits<int>::max();
            parent[affected_vertex] = -1; // Make sure to reset parent
            affectedDel[affected_vertex] = true;
            affected[affected_vertex] = true;
            
            // Update the parent's children list
            if (u == affected_vertex && v < numVertices) {
                auto& vChildren = children[v];
                vChildren.erase(remove(vChildren.begin(), vChildren.end(), u), vChildren.end());
            } else if (v == affected_vertex && u < numVertices) {
                auto& uChildren = children[u];
                uChildren.erase(remove(uChildren.begin(), uChildren.end(), v), uChildren.end());
            }
        }
    }

    // Process a single edge insertion (this is called within a critical section)
    void processEdgeInsertion(int u, int v, int weight) {
        if(u >= numVertices || v >= numVertices || u < 0 || v < 0)
            return;
            
        addEdge(u, v, weight);
        
        // Check if the new edge creates a shorter path
        if (distance[u] != numeric_limits<int>::max() && 
            distance[u] + weight < distance[v]) {
            
            // Remove from old parent's children if it exists
            if (parent[v] != -1) {
                auto& oldChildren = children[parent[v]];
                oldChildren.erase(remove(oldChildren.begin(), oldChildren.end(), v), oldChildren.end());
            }
            
            distance[v] = distance[u] + weight;
            parent[v] = u;
            affected[v] = true;
            
            // Add to new parent's children
            children[u].push_back(v);
        } 
        else if (distance[v] != numeric_limits<int>::max() && 
                 distance[v] + weight < distance[u]) {
            
            // Remove from old parent's children if it exists
            if (parent[u] != -1) {
                auto& oldChildren = children[parent[u]];
                oldChildren.erase(remove(oldChildren.begin(), oldChildren.end(), u), oldChildren.end());
            }
            
            distance[u] = distance[v] + weight;
            parent[u] = v;
            affected[u] = true;
            
            // Add to new parent's children
            children[v].push_back(u);
        }
    }

    // Update affected vertices due to deletion in parallel
    void updateDeletionAffectedVerticesParallel() {
        bool hasDeleteAffected = true;
        
        while (hasDeleteAffected) {
            hasDeleteAffected = false;
            vector<int> newlyAffected(numVertices, 0);
            
            #pragma omp parallel
            {
                vector<int> thread_affected;
                
                #pragma omp for reduction(|:hasDeleteAffected)
                for (int v = 0; v < numVertices; v++) {
                    if (affectedDel[v]) {
                        // Reset this flag immediately
                        affectedDel[v] = false;
                        hasDeleteAffected = true;
                        
                        // Use the children vector for faster access
                        for (int c : children[v]) {
                            thread_affected.push_back(c);
                        }
                        
                        // Clear the children list since this vertex is disconnected
                        children[v].clear();
                    }
                }
                
                // Process collected children
                for (int c : thread_affected) {
                    #pragma omp critical
                    {
                        distance[c] = numeric_limits<int>::max();
                        parent[c] = -1;
                        newlyAffected[c] = 1;
                        affected[c] = true;
                    }
                }
            }
            
            // Mark newly affected vertices for deletion effect
            #pragma omp parallel for
            for (int i = 0; i < numVertices; i++) {
                if (newlyAffected[i] == 1) {
                    affectedDel[i] = true;
                }
            }
        }
    }

    // Update all affected vertices in parallel with asynchronous updates
    void updateAffectedVerticesParallel() {
        // First update all deletion affected vertices
        updateDeletionAffectedVerticesParallel();
        
        // The level of asynchrony (distance to explore before synchronizing)
        int ASYNC_LEVEL = 2;
        
        // Then update all affected vertices (both from deletion and insertion)
        bool changes = true;
        int iterations = 0;
        const int MAX_ITERATIONS = 100; // Safety limit to prevent infinite loops
        
        while (changes && iterations < MAX_ITERATIONS) {
            changes = false;
            iterations++;
            
            #pragma omp parallel
            {
                #pragma omp for reduction(|:changes)
                for (int v = 0; v < numVertices; v++) {
                    if (affected[v]) {
                        // Reset this flag
                        affected[v] = false;
                        
                        // Process this vertex asynchronously
                        queue<pair<int, int>> q; // (vertex, level)
                        unordered_set<int> visited; // Track visited vertices to avoid duplicates
                        
                        q.push({v, 0});
                        visited.insert(v);
                        
                        while (!q.empty()) {
                            int current = q.front().first;
                            int level = q.front().second;
                            q.pop();
                            
                            if(current >= numVertices || current < 0)
                                continue;
                                
                            // Check all neighbors of current vertex
                            for (auto& edge : graph[current]) {
                                int n = edge.first;
                                int weight = edge.second;
                                
                                if(n >= numVertices || n < 0)
                                    continue;
                                
                                bool updated = false;
                                
                                // If we can update the neighbor's distance
                                if (distance[current] != numeric_limits<int>::max() && 
                                    distance[current] + weight < distance[n]) {
                                    
                                    // Use critical section instead of atomic
                                    #pragma omp critical
                                    {
                                        if (distance[current] + weight < distance[n]) {
                                            // Remove from old parent's children if it exists
                                            if (parent[n] != -1) {
                                                auto& oldChildren = children[parent[n]];
                                                oldChildren.erase(remove(oldChildren.begin(), oldChildren.end(), n), oldChildren.end());
                                            }
                                            
                                            distance[n] = distance[current] + weight;
                                            parent[n] = current;
                                            
                                            // Add to new parent's children
                                            children[current].push_back(n);
                                            
                                            updated = true;
                                        }
                                    }
                                    
                                    if (updated) {
                                        changes = true;
                                        
                                        // If within async level, continue exploring
                                        if (level < ASYNC_LEVEL && visited.find(n) == visited.end()) {
                                            q.push({n, level + 1});
                                            visited.insert(n);
                                        } else {
                                            affected[n] = true;
                                        }
                                    }
                                }
                                
                                // If the neighbor can update current's distance
                                if (distance[n] != numeric_limits<int>::max() && 
                                    distance[n] + weight < distance[current]) {
                                    
                                    #pragma omp critical
                                    {
                                        if (distance[n] + weight < distance[current]) {
                                            // Remove from old parent's children if it exists
                                            if (parent[current] != -1) {
                                                auto& oldChildren = children[parent[current]];
                                                oldChildren.erase(remove(oldChildren.begin(), oldChildren.end(), current), oldChildren.end());
                                            }
                                            
                                            distance[current] = distance[n] + weight;
                                            parent[current] = n;
                                            
                                            // Add to new parent's children
                                            children[n].push_back(current);
                                            
                                            updated = true;
                                        }
                                    }
                                    
                                    if (updated) {
                                        changes = true;
                                        
                                        // If within async level, continue exploring
                                        if (level < ASYNC_LEVEL) {
                                            // Current node was updated, need to reprocess its neighbors
                                            // We don't need to add it back to the queue because we're currently processing it
                                        } else {
                                            affected[current] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Debug progress
            if (iterations % 10 == 0) {
                cout << "Completed " << iterations << " iterations, continuing: " << (changes ? "Yes" : "No") << endl;
            }
        }
        
        if (iterations >= MAX_ITERATIONS) {
            cout << "Warning: Reached maximum iterations (" << MAX_ITERATIONS << "). Results may not be optimal." << endl;
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

    // Validate the SSSP tree
    bool validateSSSP(int source) {
        // Check that source has distance 0 and no parent
        if (distance[source] != 0 || parent[source] != -1) {
            cout << "Error: Source node has incorrect values" << endl;
            return false;
        }
        
        // Check that all vertices with finite distance have a valid parent
        for (int v = 0; v < numVertices; v++) {
            if (v != source && distance[v] != numeric_limits<int>::max()) {
                if (parent[v] == -1) {
                    cout << "Error: Vertex " << v << " has distance " << distance[v] 
                         << " but no parent" << endl;
                    return false;
                }
                
                // Check that parent distance plus edge weight equals vertex distance
                int p = parent[v];
                bool found = false;
                for (auto& edge : graph[p]) {
                    if (edge.first == v) {
                        if (distance[p] + edge.second != distance[v]) {
                            cout << "Error: Vertex " << v << " distance is not consistent with parent" << endl;
                            return false;
                        }
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    cout << "Error: Vertex " << v << " parent " << p 
                         << " does not have an edge to this vertex" << endl;
                    return false;
                }
            }
        }
            // cout << "Print updated SSSP tree? (y/n): ";
    // cin >> printTree;
    // if (printTree == 'y' || printTree == 'Y') {
    //     cout << "\nSSSP Tree after changes:" << endl;
    //     sssp.printSSSP();
    // }
        // Check that children lists are correct
        for (int v = 0; v < numVertices; v++) {
            for (int c : children[v]) {
                if (parent[c] != v) {
                    cout << "Error: Child list inconsistency for vertex " << v << endl;
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // Get the number of reachable vertices from the source
    int getReachableCount() {
        int count = 0;
        for (int i = 0; i < numVertices; i++) {
            if (distance[i] != numeric_limits<int>::max()) {
                count++;
            }
        }
        return count;
    }
    
    // Get the average distance of reachable vertices
    double getAverageDistance() {
        int count = 0;
        long long sum = 0;
        for (int i = 0; i < numVertices; i++) {
            if (distance[i] != numeric_limits<int>::max()) {
                sum += distance[i];
                count++;
            }
        }
        return count > 0 ? (double)sum / count : 0;
    }
};

// Function to skip comment lines in the input file
void skipComments(ifstream& infile) {
    string line;
    while (infile.peek() == '#') {
        getline(infile, line);
    }
}

// Function to parse the header and get number of nodes and edges
bool parseHeader(ifstream& infile, int& numNodes, int& numEdges) {
    string line;
    while (getline(infile, line)) {
        if (line.empty() || line[0] != '#') {
            // Not a comment line, go back to the beginning of this line
            infile.seekg(-line.length()-1, ios_base::cur);
            return false;
        }
        if (line.find("Nodes:") != string::npos) {
            // Extract the number of nodes
            size_t pos = line.find("Nodes:");
            size_t endPos = line.find("Edges:");
            string nodesStr = line.substr(pos + 6, endPos - pos - 6);
            numNodes = stoi(nodesStr);
        }
        if (line.find("Edges:") != string::npos) {
            // Extract the number of edges
            size_t pos = line.find("Edges:");
            string edgesStr = line.substr(pos + 6);
            numEdges = stoi(edgesStr);
            return true;  // Successfully parsed both nodes and edges
        }
    }
    return false;  // Failed to parse header
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_file> [num_threads] [batch_size]" << endl;
        return 1;
    }

    string datasetFile = argv[1];
    int numThreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    int batchSize = (argc > 3) ? atoi(argv[3]) : 10000;
    
    ifstream infile(datasetFile);
    if (!infile.is_open()) {
        cerr << "Error: Could not open file " << datasetFile << endl;
        return 1;
    }
    
    // Parse header information if available
    int numNodes = 0, numEdges = 0;
    bool headerFound = parseHeader(infile, numNodes, numEdges);
    
    // Skip any remaining comment lines
    skipComments(infile);
    
    // If header wasn't found, find the maximum vertex ID
    if (!headerFound) {
        // First, find the maximum vertex ID to determine graph size
        int src, dest, weight;
        string line;
        
        while (getline(infile, line)) {
            // Skip empty lines or comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            stringstream ss(line);
            if (ss >> src >> dest >> weight) {
                numNodes = max(numNodes, max(src, dest) + 1);
            }
        }
        
        // Reset file pointer to beginning and skip comments again
        infile.clear();
        infile.seekg(0);
        skipComments(infile);
    } else {
        // If we found the header, ensure numNodes is set correctly (for 0-indexed)
        numNodes = max(numNodes, 1); // At least 1 node
    }

    cout << "Number of nodes: " << numNodes << endl;
    if (numEdges > 0) {
        cout << "Number of edges: " << numEdges << endl;
    }
    cout << "Using " << numThreads << " threads with batch size " << batchSize << endl;

    // Create the graph with the determined size
    ParallelDynamicSSSP sssp(numNodes, numThreads, batchSize);

    // Read edges from the dataset
    vector<Edge> initialEdges;
    int src, dest, weight;
    long long timestamp = 0;  // Use sequential timestamp
    string line;
    
    while (getline(infile, line)) {
        // Skip empty lines or comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        stringstream ss(line);
        if (ss >> src >> dest >> weight) {
            initialEdges.push_back(Edge(src, dest, timestamp++, weight));
        }
    }
    infile.close();

    cout << "Total edges read: " << initialEdges.size() << endl;
    
    // Add initial edges to the graph (in parallel)
    #pragma omp parallel
    {
        vector<Edge> thread_edges;
        
        #pragma omp for schedule(dynamic, 1000)
        for (int i = 0; i < initialEdges.size(); i++) {
            thread_edges.push_back(initialEdges[i]);
        }
        
        #pragma omp critical
        {
            for (const auto& edge : thread_edges) {
                sssp.addEdge(edge.src, edge.dest, edge.weight);
            }
        }
    }

    // Compute initial SSSP from source vertex (choose source 0)
    const int SOURCE_VERTEX = 0;
    cout << "Computing SSSP from source vertex " << SOURCE_VERTEX << "..." << endl;
    
    double startTime = omp_get_wtime();
    sssp.computeInitialSSSP(SOURCE_VERTEX);
    double endTime = omp_get_wtime();

    cout << "Initial SSSP computation time: " << (endTime - startTime) << " seconds" << endl;
    
    // Validate the SSSP tree
    if (sssp.validateSSSP(SOURCE_VERTEX)) {
        cout << "Initial SSSP tree is valid." << endl;
    } else {
        cout << "Warning: Initial SSSP tree validation failed!" << endl;
    }

    // Report statistics
    cout << "Reachable vertices: " << sssp.getReachableCount() << " out of " << numNodes << endl;
    cout << "Average distance: " << sssp.getAverageDistance() << endl;

    // Optionally print the SSSP tree (can be large)
    // char printTree;
    // cout << "Print SSSP tree? (y/n): ";
    // cin >> printTree;
    // if (printTree == 'y' || printTree == 'Y') {
    //     sssp.printSSSP();
    // }
    sssp.printSSSP();

    // Create some example changes
    cout << "\nSimulating changes to the graph..." << endl;
    vector<Change> changes;
    
    // For demonstration, randomly remove 10 edges and add 10 edges
    srand(time(NULL));
    
    // Remove 10 random edges that exist in the initialEdges
    for (int i = 0; i < 10 && i < initialEdges.size(); i++) {
        int randomIndex = rand() % initialEdges.size();
        Edge e = initialEdges[randomIndex];
        changes.push_back(Change(e.src, e.dest, timestamp++, false, e.weight));
        cout << "Delete edge: " << e.src << " -> " << e.dest << " (weight: " << e.weight << ")" << endl;
    }
    
    // Add 10 new random edges
    for (int i = 0; i < 10; i++) {
        int u = rand() % numNodes;
        int v = rand() % numNodes;
        int w = 1 + rand() % 10;  // Random weight between 1 and 10
        
        // Don't add self-loops
        if (u != v) {
            changes.push_back(Change(u, v, timestamp++, true, w));
            cout << "Insert edge: " << u << " -> " << v << " (weight: " << w << ")" << endl;
        }
    }

    // Process the changes in parallel
    cout << "\nProcessing changes..." << endl;
    startTime = omp_get_wtime();
    sssp.processChangesInParallel(changes, SOURCE_VERTEX);
    endTime = omp_get_wtime();

    cout << "SSSP update time: " << (endTime - startTime) << " seconds" << endl;
    
    // Validate the updated SSSP tree
    if (sssp.validateSSSP(SOURCE_VERTEX)) {
        cout << "Updated SSSP tree is valid." << endl;
    } else {
        cout << "Warning: Updated SSSP tree validation failed!" << endl;
    }
    
    // Report statistics
    cout << "Updated reachable vertices: " << sssp.getReachableCount() << " out of " << numNodes << endl;
    cout << "Updated average distance: " << sssp.getAverageDistance() << endl;

    // Optionally print the updated SSSP tree
    // cout << "Print updated SSSP tree? (y/n): ";
    // cin >> printTree;
    // if (printTree == 'y' || printTree == 'Y') {
    //     cout << "\nSSSP Tree after changes:" << endl;
    //     sssp.printSSSP();
    // }
    sssp.printSSSP();

    return 0;
}