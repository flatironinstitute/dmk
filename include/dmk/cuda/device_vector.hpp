#ifndef DMK_CUDA_DEVICE_VECTOR_HPP
#define DMK_CUDA_DEVICE_VECTOR_HPP

// Device storage for SCTL's gpu_tree, which is generic over its container template. Compiled by
// nvcc only: thrust is in the interface.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace dmk::cuda {

// Device blocks come from the driver's stream-ordered pool: cudaMallocAsync reuses a freed block
// without a driver round trip, and cudaFreeAsync does not synchronize the whole device the way
// cudaFree does. One tree build makes dozens of allocations at sizes that repeat exactly from one
// build to the next. Everything is ordered on the default stream, so a consumer on another stream
// must synchronize before touching the memory.
inline void *device_block_alloc(std::size_t bytes) {
    static const bool configured = [] {
        int device = 0;
        cudaMemPool_t pool;
        if (cudaGetDevice(&device) != cudaSuccess)
            return false;
        if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess)
            return false;
        // The pool otherwise hands everything back to the OS at each synchronization.
        std::uint64_t retain_max = std::uint64_t(1) << 30;
        return cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &retain_max) == cudaSuccess;
    }();
    (void)configured;
    void *p = nullptr;
    if (cudaMallocAsync(&p, bytes, 0) != cudaSuccess)
        return nullptr;
    return p;
}

inline void device_block_free(void *p) {
    if (p)
        cudaFreeAsync(p, 0);
}

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
    pointer allocate(size_type n) {
        void *const p = device_block_alloc(n * sizeof(T));
        if (!p && n)
            throw thrust::system::detail::bad_alloc("DeviceUninitAllocator::allocate");
        return pointer(static_cast<T *>(p));
    }
    void deallocate(pointer p, size_type) { device_block_free(thrust::raw_pointer_cast(p)); }
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
