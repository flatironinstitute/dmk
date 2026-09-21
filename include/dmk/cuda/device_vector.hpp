#ifndef DMK_CUDA_DEVICE_VECTOR_HPP
#define DMK_CUDA_DEVICE_VECTOR_HPP

// Device storage for SCTL's gpu_tree, which is generic over its container template. Compiled by
// nvcc only: thrust is in the interface.

#include <cstddef>

#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "dmk/cuda/helpers.hpp"

namespace dmk::cuda {

// thrust::device_allocator whose construct is a no-op, so resize leaves trivial elements
// uninitialized (thrust's uninitialized_vector idiom).
template <class T>
struct DeviceUninitAllocator : thrust::device_allocator<T> {
    using pointer = thrust::device_ptr<T>;
    using size_type = std::size_t;
    template <class U>
    struct rebind {
        using other = DeviceUninitAllocator<U>;
    };
    __host__ __device__ void construct(T *) {}
    // gpu_tree runs every thrust call on the legacy default stream, so that is the stream its
    // storage is ordered in.
    pointer allocate(size_type n) { return pointer(static_cast<T *>(cuda_helpers::pool_alloc(n * sizeof(T), 0))); }
    void deallocate(pointer p, size_type n) { cuda_helpers::pool_free(thrust::raw_pointer_cast(p), n * sizeof(T), 0); }
};

// The container gpu_tree::PtTree is instantiated with. Its data() returns a thrust::device_ptr,
// which is what the tree's device-backend check probes for.
template <class T>
class DeviceVector : public thrust::device_vector<T, DeviceUninitAllocator<T>> {
  public:
    using thrust::device_vector<T, DeviceUninitAllocator<T>>::device_vector;
};

} // namespace dmk::cuda

#endif
