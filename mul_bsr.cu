#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <omp.h>
#include <algorithm>
#include "mul_bsr.cuh"
#include "read_matrix.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

typedef uint64_t ll;

__global__ void bsrNumericKernel(
    int bs,
    int numBlocksC,
    const int* __restrict__ blkPairPtr,
    const int* __restrict__ blkPairA,
    const int* __restrict__ blkPairB,
    const ll*  __restrict__ Aval,
    const ll*  __restrict__ Bval,
          ll*  __restrict__ Cval)
{
    int bc = blockIdx.x;
    int tid = threadIdx.x;
    int blockArea = bs*bs;
    if (bc >= numBlocksC || tid >= blockArea) return;

    int i0 = tid / bs, j0 = tid % bs;
    ll sum = 0;
    int start = blkPairPtr[bc], end = blkPairPtr[bc+1];
    for (int p = start; p < end; ++p) {
        int aidx = blkPairA[p];
        int bidx = blkPairB[p];
        const ll* Ablk = Aval + size_t(aidx)*blockArea;
        const ll* Bblk = Bval + size_t(bidx)*blockArea;
        for (int w = 0; w < bs; ++w)
            sum += Ablk[i0*bs + w] * Bblk[w*bs + j0];
    }
    Cval[size_t(bc)*blockArea + tid] = sum;
}

BSRMatrix multiply_matrices(const BSRMatrix& A, const BSRMatrix& B, int bs) {
    int R = A.num_block_rows;
    int Cn = B.width/bs;
    BSRMatrix C;
    C.height = A.height;
    C.width = B.width;
    C.num_block_rows = R;

    C.row_ptr = (int*)malloc((R+1)*sizeof(int));
    C.row_ptr[0] = 0;
    
    #pragma omp parallel
    {
        int* private_marker = (int*)malloc(Cn * sizeof(int));
        for(int i = 0; i < Cn; i++) private_marker[i] = -1;
        
        #pragma omp for schedule(dynamic,64)
        for(int i = 0; i < R; i++) {
            int cnt = 0;
            for(int ap = A.row_ptr[i]; ap < A.row_ptr[i+1]; ap++) {
                int ac = A.col_idx[ap];
                for(int bp = B.row_ptr[ac]; bp < B.row_ptr[ac+1]; bp++) {
                    int bc = B.col_idx[bp];
                    if(private_marker[bc] != i) {
                        private_marker[bc] = i;
                        cnt++;
                    }
                }
            }
            C.row_ptr[i+1] = cnt;
        }
        free(private_marker);
    }
    
    for(int i = 1; i <= R; i++) C.row_ptr[i] += C.row_ptr[i-1];
    C.num_blocks = C.row_ptr[R];
    C.col_idx = (int*)malloc(C.num_blocks * sizeof(int));
    
    #pragma omp parallel
    {
        int* private_marker = (int*)malloc(Cn * sizeof(int));
        int* private_next = (int*)malloc(R * sizeof(int));
        for(int i = 0; i < Cn; i++) private_marker[i] = -1;
        for(int i = 0; i < R; i++) private_next[i] = 0;
        
        #pragma omp for schedule(dynamic,64)
        for(int i = 0; i < R; i++) {
            for(int ap = A.row_ptr[i]; ap < A.row_ptr[i+1]; ap++) {
                int ac = A.col_idx[ap];
                for(int bp = B.row_ptr[ac]; bp < B.row_ptr[ac+1]; bp++) {
                    int bc = B.col_idx[bp];
                    if(private_marker[bc] != i) {
                        private_marker[bc] = i;
                        int cpos = C.row_ptr[i] + private_next[i]++;
                        C.col_idx[cpos] = bc;
                    }
                }
            }
        }
        free(private_marker);
        free(private_next);
    }

    int* pairsPerBlock = (int*)calloc(C.num_blocks, sizeof(int));
    
    #pragma omp parallel for schedule(dynamic,64)
    for(int cbi = 0; cbi < C.num_blocks; cbi++) {
        int r = std::upper_bound(C.row_ptr, C.row_ptr+R+1, cbi) - C.row_ptr - 1;
        int cc = C.col_idx[cbi];
        int count = 0;
        for(int ap = A.row_ptr[r]; ap < A.row_ptr[r+1]; ap++) {
            int ac = A.col_idx[ap];
            for(int bp = B.row_ptr[ac]; bp < B.row_ptr[ac+1]; bp++) {
                if(B.col_idx[bp] == cc) {
                    count++;
                }
            }
        }
        pairsPerBlock[cbi] = count;
    }
    
    int* blkPairPtr = (int*)malloc((C.num_blocks+1) * sizeof(int));
    blkPairPtr[0] = 0;
    for(int i = 0; i < C.num_blocks; i++)
        blkPairPtr[i+1] = blkPairPtr[i] + pairsPerBlock[i];
    
    size_t totalPairs = blkPairPtr[C.num_blocks];
    int* blkPairA = (int*)malloc(totalPairs * sizeof(int));
    int* blkPairB = (int*)malloc(totalPairs * sizeof(int));
    
    #pragma omp parallel for schedule(dynamic,64)
    for(int cbi = 0; cbi < C.num_blocks; cbi++) {
        int r = std::upper_bound(C.row_ptr, C.row_ptr+R+1, cbi) - C.row_ptr - 1;
        int cc = C.col_idx[cbi];
        int pairIdx = blkPairPtr[cbi];
        
        for(int ap = A.row_ptr[r]; ap < A.row_ptr[r+1]; ap++) {
            int ac = A.col_idx[ap];
            for(int bp = B.row_ptr[ac]; bp < B.row_ptr[ac+1]; bp++) {
                if(B.col_idx[bp] == cc) {
                    blkPairA[pairIdx] = ap;
                    blkPairB[pairIdx] = bp;
                    pairIdx++;
                }
            }
        }
    }

    int blockArea = bs*bs;
    C.values = (ll*)calloc(size_t(C.num_blocks) * blockArea, sizeof(ll));

    int *d_blkPairPtr, *d_blkPairA, *d_blkPairB;
    ll *d_Aval, *d_Bval, *d_Cval;
    
    CUDA_CHECK(cudaMalloc(&d_blkPairPtr, (C.num_blocks+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blkPairA, totalPairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blkPairB, totalPairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Aval, (size_t)A.num_blocks*blockArea*sizeof(ll)));
    CUDA_CHECK(cudaMalloc(&d_Bval, (size_t)B.num_blocks*blockArea*sizeof(ll)));
    CUDA_CHECK(cudaMalloc(&d_Cval, (size_t)C.num_blocks*blockArea*sizeof(ll)));
    
    CUDA_CHECK(cudaMemcpy(d_blkPairPtr, blkPairPtr, 
              (C.num_blocks+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blkPairA, blkPairA,
              totalPairs*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blkPairB, blkPairB,
              totalPairs*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Aval, A.values,
              (size_t)A.num_blocks*blockArea*sizeof(ll),
              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bval, B.values,
              (size_t)B.num_blocks*blockArea*sizeof(ll),
              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Cval, 0, (size_t)C.num_blocks*blockArea*sizeof(ll)));
    
    int threads = blockArea;
    if (threads > 1024) threads = 1024;
    int blocks = C.num_blocks;
    
    bsrNumericKernel<<<blocks, threads>>>(
        bs, C.num_blocks,
        d_blkPairPtr, d_blkPairA, d_blkPairB,
        d_Aval, d_Bval, d_Cval
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(C.values, d_Cval,
              (size_t)C.num_blocks*blockArea*sizeof(ll),
              cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_blkPairPtr));
    CUDA_CHECK(cudaFree(d_blkPairA));
    CUDA_CHECK(cudaFree(d_blkPairB));
    CUDA_CHECK(cudaFree(d_Aval));
    CUDA_CHECK(cudaFree(d_Bval));
    CUDA_CHECK(cudaFree(d_Cval));
    
    free(blkPairPtr);
    free(blkPairA);
    free(blkPairB);
    free(pairsPerBlock);

    return C;
}
