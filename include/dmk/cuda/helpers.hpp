#ifndef DMK_CUDA_HELPERS_HPP
#define DMK_CUDA_HELPERS_HPP

// Tiny utilities used by every cuda_*.cpp orchestrator. Header-only so each
// translation unit gets its own copies — no ODR concerns since everything is
// inline / template / static.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmk::cuda_helpers {

#define DMK_CHECK_CUDA(expr)                                                                                           \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        if (_e != cudaSuccess)                                                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e));                            \
    } while (0)

// Makes `device` current for the lifetime of the guard and restores the caller's device
// afterwards, so a DMK call never leaves the ambient device changed. Non-throwing: it runs
// in destructors, and the device id is range-checked at tree creation.
class ScopedDevice {
  public:
    explicit ScopedDevice(int device) noexcept {
        if (cudaGetDevice(&previous_) == cudaSuccess && device != previous_)
            restore_ = cudaSetDevice(device) == cudaSuccess;
    }
    ~ScopedDevice() {
        if (restore_)
            cudaSetDevice(previous_);
    }
    ScopedDevice(const ScopedDevice &) = delete;
    ScopedDevice &operator=(const ScopedDevice &) = delete;

  private:
    int previous_ = 0;
    bool restore_ = false;
};

// Runtime-API launches record a fault here instead of returning it; call from an entry
// point that has already synced.
inline void check_device_errors(const char *where) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error at ") + where + ": " + cudaGetErrorString(err));
}

// Device allocations are cached host-side and only fall through to the driver on a miss, because
// even a `cudaMallocAsync` hit is a driver round trip (~3 us) against ~20 ns for the scan below,
// and one build makes dozens. The driver calls underneath are the stream-ordered ones: `cudaFree`
// drains the device -- called with a kernel in flight it blocks the host for that kernel's full
// duration -- while `cudaFreeAsync` returns immediately.
//
// A block is reused only for the same size on the same stream, so its next use is ordered after
// its last one. Freeing a block another stream is still reading remains the caller's to avoid.
// Not thread-safe, and single-device by construction (bind_gpu_device pins the process).
struct DeviceBlock {
    void *p;
    std::size_t bytes;
    cudaStream_t stream;
};

inline std::vector<DeviceBlock> &device_pool() {
    static std::vector<DeviceBlock> pool;
    return pool;
}

inline std::size_t &pool_held() {
    static std::size_t held = 0;
    return held;
}

// Stream for allocations whose consumer stream is not known at the call site. Non-blocking, so
// waiting on it never waits on work queued elsewhere.
inline cudaStream_t alloc_stream() {
    static cudaStream_t stream = [] {
        cudaStream_t s = nullptr;
        DMK_CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        int device = 0;
        cudaMemPool_t pool;
        if (cudaGetDevice(&device) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
            // The pool otherwise hands every block back to the OS at each synchronization.
            std::uint64_t retain_max = std::uint64_t(4) << 30;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &retain_max);
        }
        return s;
    }();
    return stream;
}

inline void *pool_take(std::size_t bytes, cudaStream_t stream) {
    auto &pool = device_pool();
    for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (it->bytes != bytes || it->stream != stream)
            continue;
        void *const p = it->p;
        pool_held() -= it->bytes;
        pool.erase(it);
        return p;
    }
    return nullptr;
}

inline void *pool_alloc(std::size_t bytes, cudaStream_t stream) {
    if (!bytes)
        return nullptr;
    if (void *const cached = pool_take(bytes, stream))
        return cached;
    void *p = nullptr;
    DMK_CHECK_CUDA(cudaMallocAsync(&p, bytes, stream));
    return p;
}

// For memory whose consumer stream is not known at the call site: a fresh block is allocated on a
// stream of its own and waited for, so it is valid on any stream afterwards. The wait is for the
// allocation alone and never for work queued on other streams -- about 2 us, and only on a miss.
inline void *pool_alloc(std::size_t bytes) {
    if (!bytes)
        return nullptr;
    if (void *const cached = pool_take(bytes, alloc_stream()))
        return cached;
    void *p = nullptr;
    DMK_CHECK_CUDA(cudaMallocAsync(&p, bytes, alloc_stream()));
    DMK_CHECK_CUDA(cudaStreamSynchronize(alloc_stream()));
    return p;
}

inline void pool_free(void *p, std::size_t bytes, cudaStream_t stream) {
    if (!p)
        return;
    // Retaining one problem's working set is the point; retaining every size a long-running process
    // has ever asked for is not, so past the cap blocks go back to the driver.
    constexpr std::size_t retain_max = std::size_t(4) << 30;
    if (pool_held() + bytes > retain_max) {
        cudaFreeAsync(p, stream);
        return;
    }
    pool_held() += bytes;
    device_pool().push_back({p, bytes, stream});
}

// Page-locked host staging, pooled for the same reason: page-locking a few megabytes costs far more
// than the transfer it serves.
inline std::vector<std::pair<void *, std::size_t>> &pinned_pool() {
    static std::vector<std::pair<void *, std::size_t>> pool;
    return pool;
}

inline std::pair<char *, std::size_t> pinned_alloc(std::size_t bytes) {
    auto &pool = pinned_pool();
    for (auto it = pool.begin(); it != pool.end(); ++it)
        if (it->second >= bytes) {
            const auto block = *it;
            pool.erase(it);
            return {static_cast<char *>(block.first), block.second};
        }
    void *p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess)
        return {nullptr, 0};
    return {static_cast<char *>(p), bytes};
}

inline void pinned_free(char *p, std::size_t bytes) {
    if (!p)
        return;
    // Tighter cap than the device pool: page-locked pages come out of the machine's pool, not the
    // card's.
    constexpr std::size_t retain_max = std::size_t(256) << 20;
    auto &pool = pinned_pool();
    std::size_t held = bytes;
    for (const auto &b : pool)
        held += b.second;
    if (held > retain_max) {
        cudaFreeHost(p);
        return;
    }
    pool.emplace_back(p, bytes);
}

// RAII wrapper around a pooled device region. Move-only. Must not have static storage duration:
// the destructor frees, and a free from a static destructor reaches a CUDA context that may
// already be gone. `resize()` is a
// no-op if the requested size matches what's already allocated, otherwise it
// frees and re-allocates (no realloc — caller's responsibility if old data
// matters).
template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(DeviceBuffer &&o) noexcept : p_(o.p_), n_(o.n_), stream_(o.stream_), owned_(o.owned_) {
        o.p_ = nullptr;
        o.n_ = 0;
        o.stream_ = nullptr;
        o.owned_ = true;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            reset();
            p_ = o.p_;
            n_ = o.n_;
            stream_ = o.stream_;
            owned_ = o.owned_;
            o.p_ = nullptr;
            o.n_ = 0;
            o.stream_ = nullptr;
            o.owned_ = true;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void resize(std::size_t n) {
        // An adopted region is someone else's, so a resize must allocate rather than reuse it,
        // even when the size already matches.
        if (n == n_ && owned_)
            return;
        reset();
        if (n) {
            p_ = static_cast<T *>(pool_alloc(n * sizeof(T)));
            stream_ = alloc_stream();
            n_ = n;
        }
    }

    // For a buffer used on one known stream: the allocation is ordered in it, and so is the free.
    void resize(std::size_t n, cudaStream_t stream) {
        if (n == n_ && owned_)
            return;
        reset();
        if (n) {
            p_ = static_cast<T *>(pool_alloc(n * sizeof(T), stream));
            stream_ = stream;
            n_ = n;
        }
    }

    void reset() {
        if (p_ && owned_)
            pool_free(p_, n_ * sizeof(T), stream_);
        p_ = nullptr;
        n_ = 0;
        stream_ = nullptr;
        owned_ = true;
    }

    // Point at device memory owned elsewhere, so `data()` and `size()` read as usual but nothing
    // is freed: the device tree owns its metadata and outlives the State that reads it.
    void adopt(const T *p, std::size_t n) {
        reset();
        p_ = const_cast<T *>(p);
        n_ = n;
        owned_ = false;
    }

    void upload(const T *src, std::size_t n) {
        resize(n);
        if (n)
            DMK_CHECK_CUDA(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload_async(const T *src, std::size_t n, cudaStream_t stream) {
        resize(n, stream);
        if (n)
            DMK_CHECK_CUDA(cudaMemcpyAsync(p_, src, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    // Grow-only upload for reused scratch whose element count varies call-to-call
    // (e.g. per-launch arg arrays): grows the allocation to fit but never shrinks,
    // so a size that oscillates does not free/realloc every call.
    // size() reflects capacity after this, not the last upload's count.
    void upload_async_grow(const T *src, std::size_t n, cudaStream_t stream) {
        if (n > n_)
            resize(n, stream);
        if (n)
            DMK_CHECK_CUDA(cudaMemcpyAsync(p_, src, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void zero_async(cudaStream_t stream = 0) {
        if (n_)
            DMK_CHECK_CUDA(cudaMemsetAsync(p_, 0, n_ * sizeof(T), stream));
    }

    T *data() { return p_; }
    const T *data() const { return p_; }
    std::size_t size() const { return n_; }
    std::size_t size_bytes() const { return n_ * sizeof(T); }
    explicit operator bool() const { return p_ != nullptr; }

  private:
    T *p_ = nullptr;
    std::size_t n_ = 0;
    cudaStream_t stream_ = nullptr; ///< the stream the allocation is ordered in, and the free will be
    bool owned_ = true;             ///< false after adopt(): the region belongs to someone else
};

// RAII wrapper around a cudaStream. Default-constructed instance carries no
// stream (data() returns 0, the default-stream sentinel). Use the named
// factory to create an owned non-blocking stream.
class DeviceStream {
  public:
    DeviceStream() = default;
    ~DeviceStream() { reset(); }

    DeviceStream(DeviceStream &&o) noexcept : s_(o.s_) { o.s_ = nullptr; }
    DeviceStream &operator=(DeviceStream &&o) noexcept {
        if (this != &o) {
            reset();
            s_ = o.s_;
            o.s_ = nullptr;
        }
        return *this;
    }
    DeviceStream(const DeviceStream &) = delete;
    DeviceStream &operator=(const DeviceStream &) = delete;

    static DeviceStream non_blocking() {
        DeviceStream s;
        DMK_CHECK_CUDA(cudaStreamCreateWithFlags(&s.s_, cudaStreamNonBlocking));
        return s;
    }

    // Priority only biases which of the *ready* blocks the scheduler picks next; it never preempts.
    static DeviceStream non_blocking_priority() {
        DeviceStream s;
        int least = 0, greatest = 0;
        DMK_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        DMK_CHECK_CUDA(cudaStreamCreateWithPriority(&s.s_, cudaStreamNonBlocking, greatest));
        return s;
    }

    void sync() {
        if (s_)
            DMK_CHECK_CUDA(cudaStreamSynchronize(s_));
    }

    void reset() {
        if (s_) {
            cudaStreamDestroy(s_);
            s_ = nullptr;
        }
    }

    cudaStream_t get() const { return s_; }
    operator cudaStream_t() const { return s_; }

  private:
    cudaStream_t s_ = nullptr;
};

// RAII wrapper around a cudaEvent. Default-constructed = no event.
class DeviceEvent {
  public:
    DeviceEvent() = default;
    ~DeviceEvent() { reset(); }

    DeviceEvent(DeviceEvent &&o) noexcept : e_(o.e_) { o.e_ = nullptr; }
    DeviceEvent &operator=(DeviceEvent &&o) noexcept {
        if (this != &o) {
            reset();
            e_ = o.e_;
            o.e_ = nullptr;
        }
        return *this;
    }
    DeviceEvent(const DeviceEvent &) = delete;
    DeviceEvent &operator=(const DeviceEvent &) = delete;

    static DeviceEvent disable_timing() {
        DeviceEvent e;
        DMK_CHECK_CUDA(cudaEventCreateWithFlags(&e.e_, cudaEventDisableTiming));
        return e;
    }

    void reset() {
        if (e_) {
            cudaEventDestroy(e_);
            e_ = nullptr;
        }
    }

    cudaEvent_t get() const { return e_; }
    operator cudaEvent_t() const { return e_; }

  private:
    cudaEvent_t e_ = nullptr;
};

// Typed complex used inside device kernels. ABI-compatible with float2/double2
// so reinterpret_cast'ing a Real* buffer of interleaved (re, im) pairs to
// complx<Real>* gives a single hardware vector load/store
template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

template <typename Real>
__device__ __forceinline__ complx<Real> complx_zero() {
    return complx<Real>{Real{0}, Real{0}};
}

template <typename Real>
__device__ __forceinline__ complx<Real> complx_load(const Real *__restrict__ p) {
    return complx<Real>{p[0], p[1]};
}

template <typename Real>
__device__ __forceinline__ void complx_madd(complx<Real> &acc, const complx<Real> a, const complx<Real> b) {
    acc.r = fma(a.r, b.r, acc.r);
    acc.r = fma(-a.i, b.i, acc.r);
    acc.i = fma(a.r, b.i, acc.i);
    acc.i = fma(a.i, b.r, acc.i);
}

template <typename Real>
__device__ __forceinline__ Real complx_real_madd(Real acc, const complx<Real> a, const complx<Real> b) {
    acc = fma(a.r, b.r, acc);
    acc = fma(-a.i, b.i, acc);
    return acc;
}

template <typename SctlVec>
inline std::vector<int> sctl_int_vec_to_std(const SctlVec &v) {
    std::vector<int> out(v.Dim());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = v[i];
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
