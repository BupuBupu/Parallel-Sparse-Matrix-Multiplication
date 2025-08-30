#include <mpi.h>
#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "read_matrix.h"
#include "mul_bsr.cuh"
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace std;

__attribute__ ((constructor))
void set_omp_threads()
{
    omp_set_num_threads(omp_get_num_procs());
}

void send_bsr_matrix(const BSRMatrix& mat, int dest, int k) {
    MPI_Send(&mat.height, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(&mat.width, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(&mat.num_block_rows, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(&mat.num_blocks, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(mat.values, mat.num_blocks * k * k, MPI_UNSIGNED_LONG_LONG, dest, 0, MPI_COMM_WORLD);
    MPI_Send(mat.col_idx, mat.num_blocks, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(mat.row_ptr, mat.num_block_rows + 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
}

BSRMatrix recv_bsr_matrix(int src, int k) {
    BSRMatrix mat;
    MPI_Recv(&mat.height, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&mat.width, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&mat.num_block_rows, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&mat.num_blocks, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mat.values = new uint64_t[mat.num_blocks * k * k];
    mat.col_idx = new int[mat.num_blocks];
    mat.row_ptr = new int[mat.num_block_rows + 1];
    MPI_Recv(mat.values, mat.num_blocks * k * k, MPI_UNSIGNED_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(mat.col_idx, mat.num_blocks, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(mat.row_ptr, mat.num_block_rows + 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return mat;
}

void free_bsr_matrix(BSRMatrix& mat) {
    delete[] mat.values;
    delete[] mat.col_idx;
    delete[] mat.row_ptr;
}

void write_matrix_bsr(const std::string& path, const BSRMatrix& mat, int k) {
    std::ofstream fout(path);
    if (!fout) {
        fprintf(stderr, "[ERROR] Cannot open file %s for writing\n", path.c_str());
        exit(EXIT_FAILURE);
    }

    int height = mat.height;
    int width = mat.width;

    struct BlockInfo { int row, col, idx; };
    std::vector<BlockInfo> blks;
    int block_idx = 0;
    for (int i = 0; i < mat.num_block_rows; ++i) {
        for (int j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j, ++block_idx) {
            bool non_zero = false;
            for (int r = 0; r < k * k; ++r) {
                if (mat.values[block_idx * k * k + r] != 0) {
                    non_zero = true;
                    break;
                }
            }
            if (!non_zero) continue;
            int row_start = i * k;
            int col_start = mat.col_idx[j] * k;
            blks.push_back({row_start, col_start, block_idx});
        }
    }

    std::sort(blks.begin(), blks.end(), [](auto &a, auto &b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });

    fout << height << " " << width << "\n";
    fout << blks.size() << "\n";

    for (auto &b : blks) {
        fout << b.row << " " << b.col << "\n";
        int block_h = std::min(k, height - b.row);
        int block_w = std::min(k, width  - b.col);
        for (int r = 0; r < block_h; ++r) {
            for (int c = 0; c < block_w; ++c) {
                fout << mat.values[b.idx * k * k + r * k + c];
                if (c + 1 < block_w) fout << " ";
            }
            fout << "\n";
        }
    }

    fout.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // auto start = std::chrono::high_resolution_clock::now();
    // auto start1 = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        if (rank == 0) cerr << "[rank " << rank << "] Usage: mpirun -n <procs> ./a4 <folderpath>\n";
        MPI_Finalize();
        return 1;
    }

    string folder = argv[1];
    int N, k;
    if (rank == 0) {
        ifstream szf(folder + "/size");
        szf >> N >> k;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int p = (N / size);
    int chunk = (p > 1 ? p+1 : min(2,N));
    int start_idx = rank * chunk;
    int count = (start_idx < N) ? min(chunk, N - start_idx) : 0;

    BSRMatrix* local_mats = nullptr;
    if (count > 0) {
        local_mats = new BSRMatrix[count];
        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            int idx = start_idx + i + 1;
            // cout<<"Reading matrix for rank "<<rank<<" files\n";
            string path = folder + "/matrix" + to_string(idx);
            local_mats[i] = read_matrix_bsr(path, k);
            
            
        }
    }

    // if (rank==0){
    //     end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //     std::cout << "Time taken at Rank 0 for reading: " << duration.count() << " milliseconds." << std::endl;
    //     start = std::chrono::high_resolution_clock::now();
    // }

    BSRMatrix local_result;
    bool has_local_result = false;
    if (count > 0) {
        local_result = local_mats[0];
        for (int i = 1; i < count; ++i) {
            BSRMatrix tmp = multiply_matrices(local_result, local_mats[i], k);
            free_bsr_matrix(local_result);
            local_result = tmp;
        }
        has_local_result = true;
        // cout << "[rank " << rank << "] Finished local multiplication of " << count << " matrices.\n";
    }

    // if (rank==0){
    //     end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //     std::cout << "Time taken at Rank 0 for multiplication: " << duration.count() << " milliseconds." << std::endl;
    //     start = std::chrono::high_resolution_clock::now();
    // }

    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            int src = rank + step;
            if (src < size) {
                int has_matrix;
                MPI_Recv(&has_matrix, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (has_matrix) {
                    BSRMatrix recv_mat = recv_bsr_matrix(src, k);
                    if (has_local_result) {
                        BSRMatrix tmp = multiply_matrices(local_result, recv_mat, k);
                        free_bsr_matrix(local_result);
                        local_result = tmp;
                    } else {
                        local_result = recv_mat;
                        has_local_result = true;
                    }
                    free_bsr_matrix(recv_mat);
                    // cout << "[rank " << rank << "] Received and merged matrix from rank " << src << "\n";
                }
            }
        } else {
            int dest = rank - step;
            int has_matrix = has_local_result ? 1 : 0;
            MPI_Send(&has_matrix, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            if (has_matrix) send_bsr_matrix(local_result, dest, k);
            if (has_local_result) free_bsr_matrix(local_result);
            // cout << "[rank " << rank << "] Sent matrix to rank " << dest << " and exiting.\n";
            break;
        }
        step *= 2;
    }

    // if (rank==0){
    //     end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //     std::cout << "Time taken at Rank 0 for reduction : " << duration.count() << " milliseconds." << std::endl;
    //     start = std::chrono::high_resolution_clock::now();
    // }

    if (rank == 0 && has_local_result) {
        // cout << "[rank 0] Final result ready. Writing to file...\n";
        write_matrix_bsr("matrix", local_result, k);
        // if (rank==0){
        //     end = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //     std::cout << "Time taken at Rank 0 for writing: " << duration.count() << " milliseconds." << std::endl;
        //     auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start1);
        //     std::cout << "Total Time taken at Rank for writing: " << duration2.count() << " milliseconds." << std::endl;
        // }
    }

    // cout << "[rank " << rank << "] Finalizing MPI.\n";
    MPI_Finalize();
    return 0;
}
