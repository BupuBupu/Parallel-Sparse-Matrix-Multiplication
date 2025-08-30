#ifndef READ_MATRIX_H
#define READ_MATRIX_H

#include <cstdint>
#include <string>

struct BSRMatrix {
    int height;
    int width;
    int num_block_rows;
    int num_blocks;
    uint64_t* values;    // length = num_blocks * k * k
    int*      col_idx;   // length = num_blocks
    int*      row_ptr;   // length = num_block_rows + 1
};

BSRMatrix read_matrix_bsr(const std::string& filepath, int k);

#endif // READ_MATRIX_H
