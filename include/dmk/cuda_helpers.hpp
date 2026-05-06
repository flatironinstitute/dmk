#ifndef DMK_CUDA_HELPERS_HPP
#define DMK_CUDA_HELPERS_HPP

// Tiny utilities used by every cuda_*.cpp orchestrator. Header-only so each
// translation unit gets its own copies — no ODR concerns since everything is
// inline / template / static.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda_helpers {

#define DMK_CHECK_CUDA(expr)                                                                                           \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        if (_e != cudaSuccess)                                                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e));                            \
    } while (0)

template <typename T>
inline T *device_alloc(std::size_t n) {
    if (n == 0)
        return nullptr;
    T *d = nullptr;
    DMK_CHECK_CUDA(cudaMalloc(&d, n * sizeof(T)));
    return d;
}

template <typename T>
inline T *device_alloc_and_zero(std::size_t n) {
    T *d = device_alloc<T>(n);
    if (d)
        DMK_CHECK_CUDA(cudaMemsetAsync(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
inline T *device_upload(const T *src_host, std::size_t n) {
    T *d = device_alloc<T>(n);
    if (d)
        DMK_CHECK_CUDA(cudaMemcpy(d, src_host, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

inline void device_free(void *p) {
    if (p)
        cudaFree(p);
}

template <typename SctlVec>
inline std::vector<int> sctl_int_vec_to_std(const SctlVec &v) {
    std::vector<int> out(v.Dim());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = (int)v[i];
    return out;
}

// Download a device buffer and write it to disk in the SCTL binary layout
// (16-byte header + flat T data) used by DMKPtTree::dump. Caller creates dirs.
template <typename T>
inline void dump_device_buffer_to_file(const std::string &filepath, const T *d_ptr, std::size_t n) {
    if (!d_ptr || n == 0)
        return;
    std::vector<T> host(n);
    DMK_CHECK_CUDA(cudaMemcpy(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::ofstream fout(filepath, std::ios::binary);
    const int64_t dimensions = 1;
    const uint64_t n_elems = n;
    fout.write(reinterpret_cast<const char *>(&dimensions), sizeof(int64_t));
    fout.write(reinterpret_cast<const char *>(&n_elems), sizeof(uint64_t));
    fout.write(reinterpret_cast<const char *>(host.data()), n * sizeof(T));
}

} // namespace dmk::cuda_helpers

#endif // DMK_CUDA_HELPERS_HPP
