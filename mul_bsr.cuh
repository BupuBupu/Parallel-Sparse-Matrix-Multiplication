#ifndef MULTIPLY_BSR_H
#define MULTIPLY_BSR_H
#include "read_matrix.h"

#include <cstdint>

BSRMatrix multiply_matrices(const BSRMatrix& A, const BSRMatrix& B, int k);

#endif // MULTIPLY_BSR_H
