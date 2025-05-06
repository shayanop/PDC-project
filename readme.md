Hybrid Dynamic SSSP - Parallel Update Framework

This project implements the parallel framework described in the paper:
"A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks" by Khanda et al., IEEE TPDS 2022.
🔧 Features

    Dynamic Graph Updates: Supports both edge insertions and deletions in large-scale graphs.

    Parallel Execution:

        MPI for distributed memory processing.

        OpenMP for shared-memory acceleration.

        GPU support (CUDA) via Vertex-Marking Functional Blocks.

    Efficient Subgraph Updates: Avoids full recomputation by updating only the affected parts of the graph.

    Rooted Tree Representation: Maintains SSSP as a tree with parent-child relationships and distance arrays.

📦 Build Instructions
Dependencies

    C++17 compiler

    MPI library (e.g., OpenMPI or MPICH)

    OpenMP

    METIS (for graph partitioning)

    CUDA Toolkit (for GPU version)

Compilation (Shared Memory Version)

g++ -fopenmp -O3 -std=c++17 src/main.cpp -o sssp_update

Compilation (Distributed Memory + METIS)

mpic++ -fopenmp -O3 -std=c++17 src/main_mpi.cpp -lmetis -o hybrid_sssp_update

Compilation (GPU Version)

nvcc -O3 -std=c++17 src/main_cuda.cu -o sssp_gpu_update

🚀 Usage

# Shared memory
./sssp_update graph.txt updates.txt source_vertex

# Distributed MPI version
mpirun -np <num_procs> ./hybrid_sssp_update graph.txt updates.txt source_vertex

# GPU version
./sssp_gpu_update graph.txt updates.txt source_vertex

📁 Input Format

    graph.txt: Edge list with weights, e.g.
    u v weight

    updates.txt: Edge insertions/deletions, e.g.
    + u v w (insert)
    - u v (delete)

🧪 Testing & Validation

    validateSSSP() ensures correctness of SSSP updates after each batch.

    DEBUG_PRINT macros can be enabled to trace execution flow and diagnose issues.

📊 Performance

    Speedups of up to 8.5× (GPU) and 5× (OpenMP) over recomputation.

    Efficient for graphs with frequent updates and partial structure changes.

    Best suited for use cases like:

        Traffic and route planning systems

        Social network dynamics

        Real-time dependency tracking

📄 Reference

If you use this implementation in your research, please cite:

@article{Khanda2022SSSP,
  author    = {Arindam Khanda and Sriram Srinivasan and Sanjukta Bhowmick and Boyana Norris and Sajal K. Das},
  title     = {A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks},
  journal   = {IEEE Trans. Parallel Distrib. Syst.},
  volume    = {33},
  number    = {4},
  year      = {2022},
  pages     = {929--940},
  doi       = {10.1109/TPDS.2021.3084096}
}

Let me know if you'd like a version tailored for GitHub with badges or an example dataset to include.
