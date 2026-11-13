// The only nvcc-compiled TU in the ESP GPU path: thrust's device algorithms need nvcc, but nvcc
// cannot parse SCTL (it crashes in cudafe++), which every dmk header pulls in. Hence no DMK headers
// here. Every __global__ in the ESP path is NVRTC-compiled; this is host-side orchestration only.

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace dmk::cuda::esp {
namespace {

// Functors rather than lambdas: avoids depending on extended-lambda support.
template <typename Key>
struct MulByBuckets {
    Key buckets;
    __host__ __device__ Key operator()(int c) const { return static_cast<Key>(c) * buckets; }
};

struct CellPopulation {
    const int *cell_start;
    __host__ __device__ int operator()(int i) const { return cell_start[i + 1] - cell_start[i]; }
};

// cell_start[c] is the first sorted position whose composite key belongs to cell c; the sub-cell bin
// part of the key never perturbs that, being always < buckets.
template <typename Key>
void sort_and_bucket(Key *keys, int *orig, int n, int ncells, Key buckets, int *cell_start,
                     cudaStream_t stream) {
    const auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<Key> keys_ptr(keys);
    thrust::device_ptr<int> orig_ptr(orig);
    thrust::sequence(policy, orig_ptr, orig_ptr + n);
    thrust::sort_by_key(policy, keys_ptr, keys_ptr + n, orig_ptr);

    thrust::device_ptr<int> cell_start_ptr(cell_start);
    auto search_begin =
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), MulByBuckets<Key>{buckets});
    thrust::lower_bound(policy, keys_ptr, keys_ptr + n, search_begin, search_begin + ncells + 1, cell_start_ptr);
}

} // namespace

void sort_cell_keys(int *keys, int *orig, int n, int ncells, int buckets, int *cell_start, cudaStream_t stream) {
    sort_and_bucket<int>(keys, orig, n, ncells, buckets, cell_start, stream);
}

void sort_cell_keys(unsigned long long *keys, int *orig, int n, int ncells, unsigned long long buckets,
                    int *cell_start, cudaStream_t stream) {
    sort_and_bucket<unsigned long long>(keys, orig, n, ncells, buckets, cell_start, stream);
}

int max_cell_population(const int *cell_start, int ncells, cudaStream_t stream) {
    auto begin = thrust::counting_iterator<int>(0);
    return thrust::transform_reduce(thrust::cuda::par.on(stream), begin, begin + ncells,
                                    CellPopulation{cell_start}, 0, thrust::maximum<int>());
}

} // namespace dmk::cuda::esp
