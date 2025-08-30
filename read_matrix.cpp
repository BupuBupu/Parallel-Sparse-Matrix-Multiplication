#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include "read_matrix.h"

using namespace std;

inline void fast_parse_int(const char*& ptr, int& out) {
    out = 0;
    bool neg = false;
    // skip until digit or sign
    while (*ptr && (*ptr != '-' && (*ptr < '0' || *ptr > '9'))) ++ptr;
    if (*ptr == '-' || *ptr == '+') {
        neg = (*ptr == '-');
        ++ptr;
    }
    while (*ptr >= '0' && *ptr <= '9') {
        out = out * 10 + (*ptr - '0');
        ++ptr;
    }
    if (neg) out = -out;
}

inline void fast_parse_uint64(const char*& ptr, uint64_t& out) {
    out = 0;
    bool neg = false;
    while (*ptr && (*ptr != '-' && (*ptr < '0' || *ptr > '9'))) ++ptr;
    if (*ptr == '-' || *ptr == '+') {
        neg = (*ptr == '-');
        ++ptr;
    }
    while (*ptr >= '0' && *ptr <= '9') {
        out = out * 10 + (*ptr - '0');
        ++ptr;
    }
    if (neg) out = -out;
}

BSRMatrix read_matrix_bsr(const string& filepath, int k) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); exit(1); }
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    char* data = (char*)mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); close(fd); exit(1); }

    const char* ptr = data;
    int height, width, num_blocks;
    fast_parse_int(ptr, height);
    fast_parse_int(ptr, width);
    fast_parse_int(ptr, num_blocks);

    using Block = tuple<int,int,uint64_t*>;
    Block* blocks = new Block[num_blocks];
    for (int i = 0; i < num_blocks; ++i) {
        int br, bc;
        fast_parse_int(ptr, br);
        fast_parse_int(ptr, bc);
        int bh = min(k, height - br);
        int bw = min(k, width - bc);
        uint64_t* vals = new uint64_t[k*k](); // zero initalised
        for (int r = 0; r < bh; ++r) {
            for (int c = 0; c < bw; ++c) {
                uint64_t val;
                fast_parse_uint64(ptr, val);
                vals[r * k + c] = val;
            }
        }
        blocks[i] = make_tuple(br, bc, vals);
    }

    sort(blocks, blocks + num_blocks,
         [](auto &A, auto &B) {
           int ar = get<0>(A), ac = get<1>(A);
           int br = get<0>(B), bc = get<1>(B);
           if (ar==br){
            return ac < bc;
           }
           return (ar < br);
         });

    BSRMatrix M;
    M.height         = height;
    M.width          = width;
    M.num_blocks     = num_blocks;
    M.num_block_rows = (height + k - 1) / k;
    M.values    = new uint64_t[num_blocks * k * k];
    M.col_idx   = new int[num_blocks];
    M.row_ptr   = new int[M.num_block_rows + 1]();

    for (int i = 0; i < num_blocks; ++i) {
        auto [br, bc, vals] = blocks[i];
        int br_idx = br / k;
        int bc_idx = bc / k;
        M.col_idx[i] = bc_idx;
        memcpy(M.values + i * (k*k), vals, sizeof(uint64_t)*(k*k));
        M.row_ptr[br_idx + 1]++;
    }

    for (int i = 1; i <= M.num_block_rows; ++i) {
        M.row_ptr[i] += M.row_ptr[i-1];
    }

    for (int i = 0; i < num_blocks; ++i) {
        delete[] get<2>(blocks[i]);
    }
    delete[] blocks;
    munmap(data, sz);
    close(fd);

    return M;
}


// int main() {
//     string filepath = "ultra_small/matrix1"; // Replace with your actual file path
//     int k = 3;

//     BSRMatrix M = read_matrix_bsr(filepath, k);

//     cout << "Matrix dimensions: " << M.height << "x" << M.width << endl;
//     cout << "Block size: " << k << "x" << k << endl;
//     cout << "Number of blocks: " << M.num_blocks << endl;
//     cout << "Number of block rows: " << M.num_block_rows << endl;

//     cout << "row_ptr: ";
//     for (int i = 0; i <= M.num_block_rows; ++i)
//         cout << M.row_ptr[i] << " ";
//     cout << endl;

//     cout << "col_idx: ";
//     for (int i = 0; i < M.num_blocks; ++i)
//         cout << M.col_idx[i] << " ";
//     cout << endl;

//     cout << "Values of first block: ";
//     for (int i = 0; i < k*k && i < M.num_blocks * k * k; ++i)
//         cout << M.values[i] << " ";
//     cout << endl;

//     delete[] M.values;
//     delete[] M.col_idx;
//     delete[] M.row_ptr;

//     return 0;
// }
