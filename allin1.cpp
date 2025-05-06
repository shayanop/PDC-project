#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <ctime>
#include <omp.h>
#include <mpi.h>
#include <metis.h>

using namespace std;

// Debug Macro
#define DEBUG_PRINT(rank, msg) \
    if (verbose) { \
        time_t now = time(nullptr); \
        char timeStr[100]; \
        strftime(timeStr, sizeof(timeStr), "%H:%M:%S", localtime(&now)); \
        cout << "[" << timeStr << "] Process " << rank << " [" << __FUNCTION__ << "]: " << msg << endl; \
    }

// Structure for edge in graph
struct Edge {
    int src, dest;
    long long timestamp;
    int weight;
    Edge(int s = 0, int d = 0, long long t = 0, int w = 1) : src(s), dest(d), timestamp(t), weight(w) {}
};

// Structure for dynamic change
struct Change {
    int src, dest;
    long long timestamp;
    bool isInsert;
    int weight;
    Change(int s, int d, long long t, bool ins, int w = 1) : src(s), dest(d), timestamp(t), isInsert(ins), weight(w) {}
};

// Utility: Get current time as string
string getCurrentTimeString() {
    time_t now = time(nullptr);
    char timeStr[100];
    strftime(timeStr, sizeof(timeStr), "%H:%M:%S", localtime(&now));
    return string(timeStr);
}

// Parse header to get node and edge count
bool parseHeader(ifstream& infile, int& numNodes, int& numEdges) {
    string line;
    while (getline(infile, line)) {
        if (line.empty() || line[0] != '#') {
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
            return true;
        }
    }
    return false;
}

// Skip comment lines
void skipComments(ifstream& infile) {
    string line;
    while (infile.peek() == '#') {
        getline(infile, line);
    }
}

// Create MPI type for Edge struct
void createMpiEdgeType(MPI_Datatype* edgeType) {
    const int count = 4;
    int blockLengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_LONG_LONG, MPI_INT};
    Edge edge;
    MPI_Aint displacements[4], baseAddress;
    MPI_Get_address(&edge, &baseAddress);
    MPI_Get_address(&edge.src, &displacements[0]);
    MPI_Get_address(&edge.dest, &displacements[1]);
    MPI_Get_address(&edge.timestamp, &displacements[2]);
    MPI_Get_address(&edge.weight, &displacements[3]);
    for (int i = 0; i < count; i++) displacements[i] -= baseAddress;
    MPI_Type_create_struct(count, blockLengths, displacements, types, edgeType);
    MPI_Type_commit(edgeType);
}

class HybridDynamicSSSP {
private:
    int numVertices;
    vector<vector<pair<int, int>>> graph;
    vector<int> distance;
    vector<int> parent;
    vector<char> affected;
    vector<char> affectedDel;
    vector<vector<int>> children;
    int mpiRank, mpiSize;
    vector<int> localVertices;
    vector<idx_t> vertexPartition;
    bool verbose;
    int numThreads;

    int getOwnerProcess(int vertex) {
        return vertexPartition[vertex];
    }

    void broadcastVertexUpdate(int v, int newDist, int newParent) {
        int updateData[3] = {v, newDist, newParent};
        for (int p = 0; p < mpiSize; p++) {
            if (p == mpiRank) continue;
            MPI_Send(updateData, 3, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
    }

    void processIncomingUpdates() {
        MPI_Status status;
        int flag;
        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            int updateData[3];
            MPI_Recv(updateData, 3, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
            int v = updateData[0];
            int newDist = updateData[1];
            int newParent = updateData[2];
            if (newDist < distance[v]) {
                #pragma omp critical
                {
                    if (newDist < distance[v]) {
                        distance[v] = newDist;
                        parent[v] = newParent;
                        affected[v] = 1;
                        if (newParent != -1) {
                            auto& oldChildren = children[parent[v]];
                            oldChildren.erase(remove(oldChildren.begin(), oldChildren.end(), v), oldChildren.end());
                            children[newParent].push_back(v);
                        }
                    }
                }
            }
        }
    }

public:
    HybridDynamicSSSP(int n, bool verb = false, int threads = 1) {
        numVertices = n;
        graph.resize(n);
        distance.resize(n, numeric_limits<int>::max());
        parent.resize(n, -1);
        affected.resize(n, 0);
        affectedDel.resize(n, 0);
        children.resize(n);
        vertexPartition.resize(n);
        verbose = verb;
        numThreads = threads;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        omp_set_num_threads(numThreads);
    }

    void addEdge(int src, int dest, int weight) {
        if (src >= 0 && src < numVertices && dest >= 0 && dest < numVertices) {
            graph[src].push_back({dest, weight});
            graph[dest].push_back({src, weight});
            DEBUG_PRINT(mpiRank, "Added edge: " << src << " -> " << dest << " (weight: " << weight << ")");
        }
    }

    void removeEdge(int src, int dest) {
        if (src >= 0 && src < numVertices && dest >= 0 && dest < numVertices) {
            graph[src].erase(remove_if(graph[src].begin(), graph[src].end(),
                               [&](const pair<int, int>& p) { return p.first == dest; }), graph[src].end());
            graph[dest].erase(remove_if(graph[dest].begin(), graph[dest].end(),
                               [&](const pair<int, int>& p) { return p.first == src; }), graph[dest].end());
            DEBUG_PRINT(mpiRank, "Removed edge: " << src << " <-> " << dest);
        }
    }

    void partitionGraph() {
        if (mpiRank == 0) cout << "[PARTITION] Starting graph partitioning with METIS..." << endl;
        idx_t nvtxs = numVertices;
        idx_t ncon = 1;
        vector<idx_t> xadj(nvtxs + 1, 0), adjncy, adjwgt;
        for (int i = 0; i < nvtxs; i++) {
            xadj[i+1] = xadj[i] + graph[i].size();
        }
        adjncy.resize(xadj[nvtxs]);
        adjwgt.resize(xadj[nvtxs]);
        for (int i = 0; i < nvtxs; i++) {
            int offset = xadj[i];
            for (size_t j = 0; j < graph[i].size(); j++) {
                adjncy[offset + j] = graph[i][j].first;
                adjwgt[offset + j] = graph[i][j].second;
            }
        }
        idx_t nparts = mpiSize;
        idx_t edgecut;
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
        options[METIS_OPTION_NUMBERING] = 0;
        vector<idx_t> part(nvtxs);
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                    NULL, NULL, adjwgt.data(), &nparts, NULL,
                                    NULL, options, &edgecut, part.data());
        if (ret != METIS_OK) {
            if (mpiRank == 0) cout << "[ERROR] METIS partitioning failed with error code " << ret << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (mpiRank == 0) {
            cout << "[PARTITION] Graph partitioned with edge-cut: " << edgecut << endl;
        }
        MPI_Bcast(part.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < nvtxs; i++) {
            vertexPartition[i] = part[i];
            if (part[i] == mpiRank) localVertices.push_back(i);
        }
    }

    void computeInitialSSSP(int source) {
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
            int minDist = numeric_limits<int>::max(), minVertex = -1;
            for (int v = 0; v < numVertices; v++) {
                if (distance[v] < minDist && affected[v] == 0) {
                    minDist = distance[v];
                    minVertex = v;
                }
            }
            struct { int dist; int vertex; } localMin = {minDist, minVertex}, globalMin;
            MPI_Allreduce(&localMin, &globalMin, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
            if (globalMin.dist == numeric_limits<int>::max()) {
                globalDone = true;
                continue;
            }
            int u = globalMin.vertex;
            if (u >= 0 && u < numVertices) {
                affected[u] = 1;
            }
            vector<pair<int, int>> updates;
            if (u >= 0 && u < numVertices) {
                for (auto& edge : graph[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    if (distance[u] != numeric_limits<int>::max() &&
                        distance[u] + weight < distance[v]) {
                        updates.push_back({v, distance[u] + weight});
                    }
                }
            }
            int numUpdates = updates.size();
            vector<int> allNumUpdates(mpiSize);
            MPI_Allgather(&numUpdates, 1, MPI_INT, allNumUpdates.data(), 1, MPI_INT, MPI_COMM_WORLD);
            for (int p = 0; p < mpiSize; p++) {
                if (allNumUpdates[p] > 0) {
                    vector<pair<int, int>> procUpdates;
                    if (p == mpiRank) {
                        procUpdates = updates;
                    } else {
                        procUpdates.resize(allNumUpdates[p]);
                    }
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
                        }
                    }
                }
            }
            iteration++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (mpiRank == 0) {
            time_t endTime = time(nullptr);
            int totalSeconds = difftime(endTime, startTime);
            int hours = totalSeconds / 3600, minutes = (totalSeconds % 3600) / 60, seconds = totalSeconds % 60;
            cout << "[COMPLETE] Initial SSSP computation finished in " << iteration << " iterations." << endl;
            cout << "[TIME] Total runtime: " << hours << "h:" << minutes << "m:" << seconds << "s" << endl;
        }
    }

    void printSSSP(bool fullOutput = false) {
        vector<int> allDistances(numVertices);
        vector<int> allParents(numVertices);
        MPI_Allreduce(distance.data(), allDistances.data(), numVertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        for (int v = 0; v < numVertices; v++) {
            struct { int dist; int rank; } local = {distance[v], mpiRank}, global;
            MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
            int parentVal;
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
            cout << "[STATS] Reachable vertices: " << reachable << " out of " << numVertices << " (" << (reachable * 100.0 / numVertices) << "%)" << endl;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        cerr << "[ERROR] MPI does not support MPI_THREAD_MULTIPLE" << endl;
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) {
        cerr << "Usage: " << argv[0] << " <dataset_file> [source_vertex] [verbose] [num_threads]" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string datasetFile = (argc > 1) ? argv[1] : "as20000102.txt";
    int sourceVertex = (argc > 2) ? atoi(argv[2]) : 0;
    bool verbose = (argc > 3) ? (atoi(argv[3]) != 0) : false;
    int numThreads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();

    if (rank == 0) {
        cout << "[CONFIG] Hybrid MPI+OpenMP+METIS SSSP implementation" << endl;
        cout << "[CONFIG] Dataset: " << datasetFile << endl;
        cout << "[CONFIG] Source vertex: " << sourceVertex << endl;
        cout << "[CONFIG] Verbose mode: " << (verbose ? "ON" : "OFF") << endl;
        cout << "[CONFIG] OpenMP threads per process: " << numThreads << endl;
    }

    int numNodes = 0, numEdges = 0;
    vector<Edge> initialEdges;

    if (rank == 0) {
        ifstream infile(datasetFile);
        if (!infile.is_open()) {
            cerr << "[ERROR] Could not open file: " << datasetFile << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        parseHeader(infile, numNodes, numEdges);
        skipComments(infile);
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
        cout << "[DATA] Finished reading edges. Nodes: " << numNodes << ", Edges: " << initialEdges.size() << endl;
    }

    MPI_Bcast(&numNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    HybridDynamicSSSP sssp(numNodes, verbose, numThreads);

    int edgeCount = initialEdges.size();
    MPI_Bcast(&edgeCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) initialEdges.resize(edgeCount);
    MPI_Datatype MPI_EDGE;
    createMpiEdgeType(&MPI_EDGE);
    MPI_Bcast(initialEdges.data(), edgeCount, MPI_EDGE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_EDGE);

    sssp.partitionGraph();

    int edgesPerProcess = edgeCount / size;
    int startEdge = rank * edgesPerProcess;
    int endEdge = (rank == size - 1) ? edgeCount : startEdge + edgesPerProcess;

    for (int i = startEdge; i < endEdge; i++) {
        const auto& edge = initialEdges[i];
        sssp.addEdge(edge.src, edge.dest, edge.weight);
    }

    double startTime = MPI_Wtime();
    sssp.computeInitialSSSP(sourceVertex);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        cout << "Initial SSSP computation completed in " << (endTime - startTime) << " seconds." << endl;
    }

    sssp.printSSSP(false);

    MPI_Finalize();
    return 0;
}