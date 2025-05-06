#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <limits>
#include <algorithm>
#include <string>
#include <omp.h>
#include <mpi.h>
#include <cstring>
#include <cerrno>
#include <sstream>
using namespace std;

#define DEBUG_PRINT(rank, verbose, msg) \
    if (verbose) { \
        time_t now = time(nullptr); \
        char timeStr[100]; \
        strftime(timeStr, sizeof(timeStr), "%H:%M:%S", localtime(&now)); \
        cout << "[" << timeStr << "] Process " << rank << " [" << __FUNCTION__ << "]: " << msg << endl; \
    }

// Structure to represent an edge in the graph
struct Edge {
    int src, dest;
    long long timestamp;
    int weight;
    Edge() : src(0), dest(0), timestamp(0), weight(1) {}
    Edge(int s, int d, long long t, int w = 1) : src(s), dest(d), timestamp(t), weight(w) {}
};

// Structure to represent a change in the graph
struct Change {
    int src, dest;
    long long timestamp;
    bool isInsert;
    int weight;
    Change(int s, int d, long long t, bool ins, int w = 1)
        : src(s), dest(d), timestamp(t), isInsert(ins), weight(w) {}
};

// Function to create MPI datatype for Edge struct
void createMpiEdgeType(MPI_Datatype* edgeType, bool verbose) {
    const int count = 4;
    int blockLengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_LONG_LONG, MPI_INT};
    MPI_Aint displacements[4];
    Edge edge;
    MPI_Aint baseAddress;
    MPI_Get_address(&edge, &baseAddress);
    MPI_Get_address(&edge.src, &displacements[0]);
    MPI_Get_address(&edge.dest, &displacements[1]);
    MPI_Get_address(&edge.timestamp, &displacements[2]);
    MPI_Get_address(&edge.weight, &displacements[3]);
    for (int i = 0; i < count; i++) {
        displacements[i] -= baseAddress;
    }
    MPI_Type_create_struct(count, blockLengths, displacements, types, edgeType);
    MPI_Type_commit(edgeType);

    if (verbose) {
        cout << "Created MPI datatype for Edge struct." << endl;
    }
}

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
            infile.seekg(-line.length() - 1, ios_base::cur);
            return false;
        }
        if (line.find("Nodes:") != string::npos) {
            size_t pos = line.find("Nodes:");
            size_t endPos = line.find("Edges:");
            string nodesStr = line.substr(pos + 6, endPos - pos - 6);
            numNodes = stoi(nodesStr);
        }
        if (line.find("Edges:") != string::npos) {
            size_t pos = line.find("Edges:");
            string edgesStr = line.substr(pos + 6);
            numEdges = stoi(edgesStr);
            return true;  // Successfully parsed both nodes and edges
        }
    }
    return false;  // Failed to parse header
}

// Class for the dynamic SSSP algorithm with MPI
class MPIDynamicSSSP {
private:
    int numVertices;
    vector<vector<pair<int, int>>> graph;
    vector<int> distance;
    vector<int> parent;
    vector<char> affected;
    vector<char> affectedDel;
    int mpiRank;
    int mpiSize;
    vector<int> localVertices;
    vector<vector<int>> children;
    bool verbose;

    int getOwnerProcess(int vertex) {
        int verticesPerProcess = numVertices / mpiSize;
        return min(vertex / verticesPerProcess, mpiSize - 1);
    }

    void broadcastVertexUpdate(int vertex, int newDistance, int newParent) {
        DEBUG_PRINT(mpiRank, verbose, "Broadcasting update for vertex " << vertex << ": dist=" << newDistance << ", parent=" << newParent);
        for (int p = 0; p < mpiSize; p++) {
            if (p == mpiRank) continue;
            MPI_Send(&vertex, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&newDistance, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
            MPI_Send(&newParent, 1, MPI_INT, p, 2, MPI_COMM_WORLD);
        }
    }

    void processIncomingUpdates() {
        MPI_Status status;
        int count = 0;
        while (true) {
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            int vertex, newDistance, newParent;
            MPI_Recv(&vertex, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&newDistance, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&newParent, 1, MPI_INT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, &status);
            if (distance[vertex] > newDistance) {
                distance[vertex] = newDistance;
                parent[vertex] = newParent;
                affected[vertex] = 1;
                DEBUG_PRINT(mpiRank, verbose, "Received update for vertex " << vertex << ": dist=" << newDistance << ", parent=" << newParent);
            }
            count++;
        }
        if (count > 0) {
            DEBUG_PRINT(mpiRank, verbose, "Processed " << count << " incoming updates.");
        }
    }

public:
    MPIDynamicSSSP(int n, bool verb = false) {
        numVertices = n;
        graph.resize(n);
        distance.resize(n, numeric_limits<int>::max());
        parent.resize(n, -1);
        affected.resize(n, 0);
        affectedDel.resize(n, 0);
        children.resize(n);
        verbose = verb;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        int verticesPerProcess = numVertices / mpiSize;
        int startVertex = mpiRank * verticesPerProcess;
        int endVertex = (mpiRank == mpiSize - 1) ? numVertices : startVertex + verticesPerProcess;
        for (int v = startVertex; v < endVertex; v++) {
            localVertices.push_back(v);
        }
        DEBUG_PRINT(mpiRank, verbose, "Initialized with " << localVertices.size() << " local vertices from " << startVertex << " to " << (endVertex - 1));
    }

    void addEdge(int src, int dest, int weight) {
        if (src >= numVertices || dest >= numVertices || src < 0 || dest < 0) return;
        graph[src].push_back({dest, weight});
        graph[dest].push_back({src, weight});
        DEBUG_PRINT(mpiRank, verbose, "Added edge: " << src << " -> " << dest << " (weight: " << weight << ")");
    }

    void removeEdge(int src, int dest) {
        if (src >= numVertices || dest >= numVertices || src < 0 || dest < 0) return;
        auto& edges = graph[src];
        edges.erase(remove_if(edges.begin(), edges.end(),
                             [&](const pair<int, int>& p) { return p.first == dest; }),
                    edges.end());
        auto& rev_edges = graph[dest];
        rev_edges.erase(remove_if(rev_edges.begin(), rev_edges.end(),
                                  [&](const pair<int, int>& p) { return p.first == src; }),
                        rev_edges.end());
        DEBUG_PRINT(mpiRank, verbose, "Removed edge: " << src << " <-> " << dest);
    }

    void computeInitialSSSP(int source) {
        DEBUG_PRINT(mpiRank, verbose, "Starting initial SSSP computation from source " << source);
        fill(distance.begin(), distance.end(), numeric_limits<int>::max());
        fill(parent.begin(), parent.end(), -1);
        fill(affected.begin(), affected.end(), 0);
        fill(affectedDel.begin(), affectedDel.end(), 0);
        distance[source] = 0;
        bool globalDone = false;
        int iteration = 0;

        time_t startTime = time(nullptr);
        if (mpiRank == 0) {
            char timeStr[100];
            strftime(timeStr, sizeof(timeStr), "%H:%M:%S", localtime(&startTime));
            cout << "[TIME] Starting SSSP computation at: " << timeStr << endl;
        }

        while (!globalDone) {
            DEBUG_PRINT(mpiRank, verbose, "SSSP Iteration " << iteration << " started.");
            // Find minimum distance vertex locally
            int minDist = numeric_limits<int>::max();
            int minVertex = -1;
            for (int v = 0; v < numVertices; v++) {
                if (distance[v] < minDist && affected[v] == 0) {
                    minDist = distance[v];
                    minVertex = v;
                }
            }
            DEBUG_PRINT(mpiRank, verbose, "Local min vertex: " << minVertex << " with distance: " << minDist);

            // Find global minimum distance vertex
            struct { int dist; int vertex; } localMin, globalMin;
            localMin.dist = minDist;
            localMin.vertex = minVertex;
            MPI_Allreduce(&localMin, &globalMin, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
            DEBUG_PRINT(mpiRank, verbose, "Global min vertex: " << globalMin.vertex << " with distance: " << globalMin.dist);

            if (globalMin.dist == numeric_limits<int>::max()) {
                DEBUG_PRINT(mpiRank, verbose, "No more reachable vertices found. Exiting loop.");
                globalDone = true;
                continue;
            }

            int u = globalMin.vertex;
            if (u >= 0 && u < numVertices) {
                affected[u] = 1;
                DEBUG_PRINT(mpiRank, verbose, "Marked vertex " << u << " as processed.");
            }

            vector<pair<int, int>> updates;
            if (u >= 0 && u < numVertices) {
                for (auto& edge : graph[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    if (distance[u] != numeric_limits<int>::max() &&
                        distance[u] + weight < distance[v]) {
                        updates.push_back({v, distance[u] + weight});
                        DEBUG_PRINT(mpiRank, verbose, "Found update for vertex " << v << ": new distance=" << (distance[u] + weight));
                    }
                }
            }

            // Exchange updates between processes
            int numUpdates = updates.size();
            vector<int> allNumUpdates(mpiSize);
            MPI_Allgather(&numUpdates, 1, MPI_INT, allNumUpdates.data(), 1, MPI_INT, MPI_COMM_WORLD);
            DEBUG_PRINT(mpiRank, verbose, "Shared update counts across processes.");

            for (int p = 0; p < mpiSize; p++) {
                if (allNumUpdates[p] > 0) {
                    vector<pair<int, int>> procUpdates;
                    if (p == mpiRank) {
                        procUpdates = updates;
                    } else {
                        procUpdates.resize(allNumUpdates[p]);
                    }

                    // Create MPI datatype for updates
                    MPI_Datatype MPI_PAIR;
                    MPI_Type_contiguous(2 * sizeof(int), MPI_BYTE, &MPI_PAIR);
                    MPI_Type_commit(&MPI_PAIR);
                    MPI_Bcast(procUpdates.data(), allNumUpdates[p], MPI_PAIR, p, MPI_COMM_WORLD);
                    MPI_Type_free(&MPI_PAIR);

                    for (auto& update : procUpdates) {
                        int v = update.first;
                        int newDist = update.second;
                        if (newDist < distance[v]) {
                            distance[v] = newDist;
                            parent[v] = u;
                            DEBUG_PRINT(mpiRank, verbose, "Applied update for vertex " << v << ": new distance=" << newDist);
                        }
                    }
                }
            }

            iteration++;
            DEBUG_PRINT(mpiRank, verbose, "Completed SSSP iteration " << iteration);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (mpiRank == 0) {
            time_t endTime = time(nullptr);
            int totalSeconds = difftime(endTime, startTime);
            int hours = totalSeconds / 3600;
            int minutes = (totalSeconds % 3600) / 60;
            int seconds = totalSeconds % 60;
            cout << "[COMPLETE] Initial SSSP computation finished in " << iteration << " iterations." << endl;
            cout << "[TIME] Total runtime: " << hours << "h:" << minutes << "m:" << seconds << "s" << endl;
        }
    }

    void printSSSP(bool fullOutput = false) {
        vector<int> allDistances(numVertices);
        vector<int> allParents(numVertices);
        MPI_Allreduce(distance.data(), allDistances.data(), numVertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        for (int v = 0; v < numVertices; v++) {
            struct { int dist; int rank; } local, global;
            local.dist = distance[v];
            local.rank = mpiRank;
            MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
            int parentVal = parent[v];
            MPI_Bcast(&parentVal, 1, MPI_INT, global.rank, MPI_COMM_WORLD);
            allParents[v] = parentVal;
        }

        if (mpiRank == 0) {
            cout << "\n[STATS] SSSP Tree Statistics:" << endl;
            int reachable = 0;
            long long totalDist = 0;
            for (int i = 0; i < numVertices; i++) {
                if (allDistances[i] != numeric_limits<int>::max()) {
                    reachable++;
                    totalDist += allDistances[i];
                }
            }
            cout << "[STATS] Reachable vertices: " << reachable << " out of " << numVertices
                 << " (" << (reachable * 100.0 / numVertices) << "%)" << endl;
            if (reachable > 0) {
                cout << "[STATS] Average distance: " << (double)totalDist / reachable << endl;
            }
            if (fullOutput) {
                cout << "\n[TREE] Complete SSSP Tree:" << endl;
                cout << "Vertex \t Distance from Source \t Parent" << endl;
                for (int i = 0; i < numVertices; i++) {
                    cout << i << " \t ";
                    if (allDistances[i] == numeric_limits<int>::max())
                        cout << "INF";
                    else
                        cout << allDistances[i];
                    cout << " \t\t " << allParents[i] << endl;
                }
            }
        }
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) {
        cerr << "Usage: " << argv[0] << " <dataset_file> [source_vertex] [verbose]" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string datasetFile = (argc > 1) ? argv[1] : "as20000102.txt";
    int sourceVertex = (argc > 2) ? atoi(argv[2]) : 0;
    bool verbose = (argc > 3) ? (atoi(argv[3]) != 0) : false;

    if (rank == 0) {
        cout << "MPI Dynamic SSSP program started with " << size << " processes." << endl;
        cout << "Dataset: " << datasetFile << endl;
        cout << "Source vertex: " << sourceVertex << endl;
        cout << "Verbose mode: " << (verbose ? "ON" : "OFF") << endl;
    }

    int numNodes = 0, numEdges = 0;
    vector<Edge> initialEdges;

    if (rank == 0) {
        cout << "Reading edges from dataset file..." << endl;
        ifstream infile(datasetFile);
        if (!infile.is_open()) {
            cerr << "Error opening file: " << datasetFile << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        skipComments(infile);
        parseHeader(infile, numNodes, numEdges);
        string line;
        int src, dest, weight;
        long long timestamp = 0;
        while (getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;
            stringstream ss(line);
            if (ss >> src >> dest >> weight) {
                numNodes = max(numNodes, max(src, dest) + 1);
                initialEdges.push_back(Edge(src, dest, timestamp++, weight));
            }
        }
        infile.close();
        cout << "Finished reading edges. Nodes: " << numNodes << ", Edges: " << initialEdges.size() << endl;
    }

    MPI_Bcast(&numNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG_PRINT(rank, verbose, "Broadcasted number of nodes: " << numNodes);

    MPIDynamicSSSP sssp(numNodes, verbose);

    int edgeCount = initialEdges.size();
    MPI_Bcast(&edgeCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) initialEdges.resize(edgeCount);

    MPI_Datatype MPI_EDGE;
    createMpiEdgeType(&MPI_EDGE, verbose);
    MPI_Bcast(initialEdges.data(), edgeCount, MPI_EDGE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_EDGE);

    int edgesPerProcess = edgeCount / size;
    int startEdge = rank * edgesPerProcess;
    int endEdge = (rank == size - 1) ? edgeCount : startEdge + edgesPerProcess;

    for (int i = startEdge; i < endEdge; i++) {
        const auto& edge = initialEdges[i];
        sssp.addEdge(edge.src, edge.dest, edge.weight);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    DEBUG_PRINT(rank, verbose, "All processes synchronized after edge distribution.");

    // Compute initial SSSP
    double startTime = MPI_Wtime();
    sssp.computeInitialSSSP(sourceVertex);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        cout << "Initial SSSP computation completed in " << (endTime - startTime) << " seconds." << endl;
    }

    // Print SSSP before updates
    if (rank == 0) {
        cout << "\n[BEFORE UPDATES] SSSP Tree:" << endl;
    }
    sssp.printSSSP(true);

    // Simulate updates or deletions
    if (rank == 0) {
        cout << "\nApplying updates and deletions..." << endl;
    }

    // Example updates and deletions
    if (rank == 0) {
        sssp.addEdge(0, 10, 5);  // Add an edge
        sssp.removeEdge(2, 5);   // Remove an edge
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Recompute SSSP after updates
    startTime = MPI_Wtime();
    sssp.computeInitialSSSP(sourceVertex);
    endTime = MPI_Wtime();

    if (rank == 0) {
        cout << "SSSP recomputation after updates completed in " << (endTime - startTime) << " seconds." << endl;
    }

    // Print SSSP after updates
    if (rank == 0) {
        cout << "\n[AFTER UPDATES] SSSP Tree:" << endl;
    }
    sssp.printSSSP(true);

    MPI_Finalize();
    return 0;
}