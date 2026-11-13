#pragma once
// Interface to the thrust-based cell-list sort (esp/sort.cu). Kept free of DMK types: nvcc cannot
// parse SCTL, which every dmk header pulls in.

#include <cuda_runtime.h>

namespace dmk::cuda::esp {

void sort_cell_keys(int *keys, int *orig, int n, int ncells, int buckets, int *cell_start, cudaStream_t stream);
void sort_cell_keys(unsigned long long *keys, int *orig, int n, int ncells, unsigned long long buckets, int *cell_start,
                    cudaStream_t stream);

int max_cell_population(const int *cell_start, int ncells, cudaStream_t stream);

} // namespace dmk::cuda::esp
