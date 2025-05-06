🔄 Hybrid Parallel Dynamic SSSP

This project is an implementation of the parallel dynamic Single Source Shortest Path (SSSP) algorithm based on the framework proposed by Khanda et al. in IEEE TPDS 2022. It efficiently computes and updates shortest paths in large-scale, evolving graphs using MPI, OpenMP, and METIS for hybrid parallelism and distributed partitioning.
👨‍💻 Authors

    Shayan Ahmed - i22-0822

    Saad Mursaleen - i22-0835

    Bilal Raza - i22-1325

    Section: A

📚 Project Overview

This project includes four progressive implementations of the Dynamic SSSP algorithm:
✅ Part 1: Serial Dijkstra-Based Dynamic SSSP

    Basic implementation using Dijkstra’s algorithm.

    Supports dynamic updates (add/remove edges).

    Recomputes shortest paths after each batch of changes.

    Suitable for prototyping and small graphs.

⚙️ Part 2: OpenMP Parallel Dynamic SSSP

    Accelerates Dijkstra’s algorithm using OpenMP.

    Performs parallel relaxation of edges and updates.

    Ensures correctness via thread-safe updates and validation.

🌐 Part 3: MPI + OpenMP Hybrid Parallelism

    Combines distributed (MPI) and shared memory (OpenMP).

    Each process handles a partition of the graph.

    Efficient edge relaxations and communication using MPI_Bcast, MPI_Allreduce.

🧩 Part 4: MPI + OpenMP + METIS (Hybrid with Partitioning)

    Uses METIS to partition the graph for load balancing.

    Employs delta-stepping strategy for fast incremental updates.

    Batch updates and synchronization improve scalability and responsiveness.

🚀 Features

    Edge insertions/deletions with real-time propagation.

    Batch update processing for performance tuning.

    Timestamps for update ordering and validation.

    Delta-stepping approach for efficient incremental recalculation.

    Debugging macros (DEBUG_PRINT) for traceability and hang detection.

🛠️ Build Instructions
Prerequisites

    C++ compiler with OpenMP support

    MPI implementation (OpenMPI or MPICH)

    METIS library (for partitioning)

    CUDA (for GPU variant if applicable)

Compilation Examples
Serial / OpenMP

g++ -fopenmp -O2 -std=c++17 main_serial.cpp -o sssp_serial

Hybrid MPI + OpenMP

mpic++ -fopenmp -O2 -std=c++17 main_mpi.cpp -o sssp_hybrid

METIS Partitioned

mpic++ -fopenmp -O2 -std=c++17 main_mpi_metis.cpp -lmetis -o sssp_distributed

🧪 How to Run

# Serial
./sssp_serial <graph_file> <updates_file> <source_vertex>

# MPI + OpenMP
mpirun -np <num_procs> ./sssp_hybrid <graph_file> <updates_file> <source_vertex>

📁 Input Format

    Graph File:

u v weight

Update File:

    + u v weight  # Edge insertion
    - u v         # Edge deletion

🧠 Use Case Suitability

Ideal for:

    Real-time traffic simulation

    Dynamic social network analysis

    Telecommunication rerouting

    Distributed system dependency updates

📌 Acknowledgement

This project is based on the research work:
“A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks”
IEEE Transactions on Parallel and Distributed Systems, 2022
By: Arindam Khanda, Sriram Srinivasan, Sanjukta Bhowmick, Boyana Norris, Sajal K. Das
