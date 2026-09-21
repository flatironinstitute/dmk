#ifndef DMK_CUDA_DEVICE_VECTOR_HPP
#define DMK_CUDA_DEVICE_VECTOR_HPP

// Device storage for SCTL's gpu_tree, which is generic over its container template. New elements
// are left uninitialized and no bookkeeping launches or joins the stream: thrust::device_vector
// runs a construct kernel and synchronizes on every resize, and a tree build resizes a few dozen
// times. Blocks come from the pool on the legacy stream, where gpu_tree runs every thrust call, so
// storage is reused only behind the work that last touched it. Compiled by nvcc only: thrust is in
// the interface.

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>

#include "dmk/cuda/helpers.hpp"

namespace dmk::cuda {

template <class T>
class DeviceVector {
  public:
    using value_type = T;
    using iterator = thrust::device_ptr<T>;
    using const_iterator = thrust::device_ptr<const T>;

    DeviceVector() = default;
    explicit DeviceVector(std::size_t n) { resize(n); }
    template <class It>
    DeviceVector(It first, It last) {
        assign(first, last);
    }
    DeviceVector(const DeviceVector &o) { assign(o.begin(), o.end()); }
    DeviceVector(DeviceVector &&o) noexcept { swap(o); }
    DeviceVector &operator=(const DeviceVector &o) {
        if (this != &o) {
            clear();
            assign(o.begin(), o.end());
        }
        return *this;
    }
    DeviceVector &operator=(DeviceVector &&o) noexcept {
        if (this != &o) {
            release();
            swap(o);
        }
        return *this;
    }
    ~DeviceVector() { release(); }

    /// The caller's device memory as a vector: read in place, never freed. Growing it past `n`
    /// moves the contents into storage of its own.
    static DeviceVector view(T *p, std::size_t n) {
        DeviceVector v;
        v.p_ = p;
        v.n_ = v.cap_ = n;
        v.owned_ = false;
        return v;
    }

    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    iterator data() { return iterator(p_); }
    const_iterator data() const { return const_iterator(p_); }
    iterator begin() { return data(); }
    iterator end() { return data() + n_; }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data() + n_; }
    thrust::device_reference<T> operator[](std::size_t i) { return *(data() + i); }
    thrust::device_reference<const T> operator[](std::size_t i) const { return *(data() + i); }

    void clear() { n_ = 0; }
    void resize(std::size_t n) {
        reserve(n);
        n_ = n;
    }
    void reserve(std::size_t n) {
        if (n <= cap_)
            return;
        const std::size_t cap = std::max(n, 2 * cap_);
        T *const p = static_cast<T *>(cuda_helpers::pool_alloc(cap * sizeof(T), 0));
        if (n_)
            DMK_CHECK_CUDA(cudaMemcpyAsync(p, p_, n_ * sizeof(T), cudaMemcpyDeviceToDevice, 0));
        const std::size_t n_keep = n_;
        release();
        p_ = p;
        n_ = n_keep;
        cap_ = cap;
    }
    void swap(DeviceVector &o) noexcept {
        std::swap(p_, o.p_);
        std::swap(n_, o.n_);
        std::swap(cap_, o.cap_);
        std::swap(owned_, o.owned_);
    }

  private:
    // A device range copies on the stream; a host pointer crosses as a staged copy, which the runtime
    // completes before returning for pageable memory and orders on the stream for page-locked memory;
    // anything else goes through thrust.
    template <class It>
    void assign(It first, It last) {
        resize(last - first);
        if (!n_)
            return;
        if constexpr (std::is_same_v<It, iterator> || std::is_same_v<It, const_iterator>)
            DMK_CHECK_CUDA(
                cudaMemcpyAsync(p_, thrust::raw_pointer_cast(first), n_ * sizeof(T), cudaMemcpyDeviceToDevice, 0));
        else if constexpr (std::is_pointer_v<It>)
            DMK_CHECK_CUDA(cudaMemcpyAsync(p_, first, n_ * sizeof(T), cudaMemcpyHostToDevice, 0));
        else
            thrust::copy(first, last, begin());
    }
    void release() {
        if (p_ && owned_)
            cuda_helpers::pool_free(p_, cap_ * sizeof(T), 0);
        p_ = nullptr;
        n_ = 0;
        cap_ = 0;
        owned_ = true;
    }

    T *p_ = nullptr;
    std::size_t n_ = 0;
    std::size_t cap_ = 0;
    bool owned_ = true;
};

} // namespace dmk::cuda

#endif
