#pragma once 
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <string>

namespace dmk::cuda::jit {

class JitKernel {
public:
    JitKernel() = default;
    JitKernel(CUmodule module, CUfunction function)
        : module_(module), function_(function) {}

    JitKernel(const JitKernel&) = delete;
    JitKernel& operator=(const JitKernel&) = delete;

    ~JitKernel() {
        unload();
    }

    JitKernel(JitKernel&& other) noexcept
        : module_(std::exchange(other.module_, nullptr)),
        function_(std::exchange(other.function_, nullptr)) {}

    JitKernel& operator=(JitKernel&& other) noexcept {
        if (this != &other) {
            unload();
            module_ = std::exchange(other.module_, nullptr);
            function_ = std::exchange(other.function_, nullptr);
        }
        return *this;
    }

    CUfunction function () const {
        return function_;
    }

    template <class... Args>
    void launch(dim3 grid, dim3 block, size_t shared_bytes,
        cudaStream_t stream, Args&... args) const {
            if (!function_) {
                throw std::runtime_error("JitKernel::launch: function_ is nullptr!");
            }

            void* kernel_args[] = {
                const_cast<void*>(
                    static_cast<const void*>(&args)
                )...
            };

            CUresult res = cuLaunchKernel(
                function_,
                grid.x, grid.y, grid.z,
                block.x, block.y, block.z,
                static_cast<unsigned int>(shared_bytes),
                reinterpret_cast<CUstream>(stream),
                kernel_args,
                nullptr
            );

            if (res != CUDA_SUCCESS) {
                const char* name = nullptr;
                const char* msg = nullptr;

                cuGetErrorName(res, &name);
                cuGetErrorString(res, &msg);

                throw std::runtime_error(
                    std::string("cuLaunchKernel failed: ") +
                    (name ? name : "<unkown>") + ": " +
                    (msg ? msg : "<no message>")
                );
            }
        }

    
private:
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;

    void unload();
    
};


} // namespace dmk::cuda::jit