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

// RAII wrapper around a `cudaMalloc`'d region. Move-only. `resize()` is a
// no-op if the requested size matches what's already allocated, otherwise it
// frees and re-allocates (no realloc — caller's responsibility if old data
// matters).
template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(DeviceBuffer &&o) noexcept : p_(o.p_), n_(o.n_) {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            reset();
            p_ = o.p_;
            n_ = o.n_;
            o.p_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void resize(std::size_t n) {
        if (n == n_)
            return;
        reset();
        if (n) {
            DMK_CHECK_CUDA(cudaMalloc(&p_, n * sizeof(T)));
            n_ = n;
        }
    }

    void reset() {
        if (p_) {
            cudaFree(p_);
            p_ = nullptr;
        }
        n_ = 0;
    }

    void upload(const T *src, std::size_t n) {
        resize(n);
        if (n)
            DMK_CHECK_CUDA(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload_async(const T *src, std::size_t n, cudaStream_t stream) {
        resize(n);
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
